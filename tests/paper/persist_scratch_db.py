"""A throwaway PostgreSQL database for the persist characterization tests.

`scratch_db()` creates `ggt_scratch_<8 hex>` on the server named by the process environment
variable `GGT_SCRATCH_DB_ADMIN_URL`, applies only the paper tables' DDL (`persist._SCHEMA`), yields
an engine bound to that database, and drops it in `finally`.

The administrator URL is a superuser credential, so the fixture fails closed:

* The administrator connection runs exactly three statements, all on one validated name that matches
  `SCRATCH_NAME`: `SELECT 1 FROM pg_database WHERE datname = <name>` (is the name free?),
  `CREATE DATABASE <name>` and `DROP DATABASE <name> WITH (FORCE)`. The name is checked before every
  statement and before any connection is opened, so a bad name raises first.
* Ownership: the fixture drops only what it created. The name is `uuid4().hex[:8]`, checked free
  first; the "created" flag is set only after `execute()` of CREATE has returned. If CREATE fails (a
  name collision, SQLSTATE 42P04) or its outcome is unknown (the connection is lost while it runs),
  nothing is dropped and a redacted error asks the operator to look for a leftover `ggt_scratch_%`
  database by hand.
* One administrator session lives for the whole fixture: CREATE and DROP run on the same connection.
  If that session is lost before the DROP, one retry on a fresh connection to the same single host
  follows; if it fails too the fixture fails loudly (a leftover) and never skips the DROP.
* The URL comes only from the environment: never from `.env`, never printed, logged or written. It
  lives inside `_Secret` (its `repr` and `str` are `<redacted>`) and is unwrapped only in the call
  that hands it to the driver. Every place that opens a connection or builds an engine catches the
  driver's exception and raises a NEW `ScratchDbError` (`from None`) with a fixed message, so no
  psycopg2 or SQLAlchemy frame (and none of its locals or arguments) reaches pytest's output, even
  with `--showlocals`. The test engine connects through a factory that does the same, so a failed
  connection inside a test body is redacted too.
* An administrator URL with any query string (a password in the query, `target_session_attrs`,
  `dbname`, ...) or with more than one host is refused before any connection, with a message that
  quotes none of it: one host and one database make CREATE and DROP reach the same server.
* The scratch URL is the administrator URL with the driver and database name replaced and the
  query dropped. It is checked (database equals the validated name, same host, port and user) before
  the engine exists.
* The test skips ONLY for a positively identified connection failure (connection refused, timed out,
  host name not resolved, no route, network unreachable, or a SQLSTATE of class 08). An
  authentication failure (SQLSTATE class 28, "password authentication failed") and every
  unclassified error fail with a redacted message.
* Nothing here touches `ohlcv`, an existing `paper_*` table or an existing database.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import psycopg2
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool

ADMIN_URL_ENV = "GGT_SCRATCH_DB_ADMIN_URL"
SCRATCH_NAME = re.compile(r"^ggt_scratch_[0-9a-f]{8}$")
SCRATCH_DRIVER = "postgresql+psycopg2"
EXISTS_SQL = "SELECT 1 FROM pg_database WHERE datname = %s"
# Driver-message fragments that positively identify "the server cannot be reached".
UNREACHABLE_MESSAGES = (
    "Connection refused",
    "timeout expired",
    "timed out",
    "could not translate host name",
    "No route to host",
    "Network is unreachable",
)
LEFTOVER_HINT = "check for a leftover ggt_scratch_% database manually"
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")
_T = TypeVar("_T")


class ScratchNameError(ValueError):
    """A database name that is not a scratch name was about to reach the server."""


class ScratchConfigError(ValueError):
    """The administrator URL is unusable or unsafe. Messages never quote the URL."""


class ScratchDbError(RuntimeError):
    """A scratch-database step failed. The message is fixed text: it never quotes driver output."""


class _Secret:
    """Holds the administrator URL; every rendering of it is `<redacted>`."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        object.__setattr__(self, "_value", value)

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return "<redacted>"

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a _Secret cannot be changed")

    def __reduce__(self) -> Any:
        raise TypeError("a _Secret cannot be pickled")

    def reveal(self) -> str:
        """The raw URL. Call it only in the expression that hands it to the driver."""
        return self._value


def read_admin_secret() -> _Secret | None:
    """The administrator URL from the environment, wrapped at once; `None` when not set."""
    if not os.environ.get(ADMIN_URL_ENV):
        return None
    return _Secret(os.environ[ADMIN_URL_ENV])


def new_scratch_name() -> str:
    """A fresh scratch database name."""
    return f"ggt_scratch_{uuid.uuid4().hex[:8]}"


def assert_scratch_name(name: str) -> str:
    """Return `name` if it is a scratch name; otherwise raise before anything runs."""
    if not isinstance(name, str) or SCRATCH_NAME.fullmatch(name) is None:
        raise ScratchNameError("not a scratch database name")
    return name


def create_sql(name: str) -> str:
    """The one CREATE statement, for a checked name."""
    return f'CREATE DATABASE "{assert_scratch_name(name)}"'


def drop_sql(name: str) -> str:
    """The one DROP statement, for a checked name."""
    return f'DROP DATABASE "{assert_scratch_name(name)}" WITH (FORCE)'


def exists_query(name: str) -> tuple[str, tuple[str]]:
    """The one SELECT: is a database of this name already there? The name is a bound parameter."""
    return EXISTS_SQL, (assert_scratch_name(name),)


def psycopg_dsn(url: str) -> str:
    """The URL without a SQLAlchemy driver suffix, as libpq reads it."""
    return re.sub(r"^postgresql\+\w+://", "postgresql://", url)


def _describe(exc: BaseException) -> str:
    """A safe fragment for a message: the SQLSTATE if the driver gave one, never any text."""
    code = getattr(exc, "pgcode", None) or getattr(getattr(exc, "orig", None), "pgcode", None)
    return f" (SQLSTATE {code})" if isinstance(code, str) and _SQLSTATE.fullmatch(code) else ""


def is_unreachable(exc: BaseException) -> bool:
    """True only for a driver error that positively says the server could not be reached."""
    if not isinstance(exc, psycopg2.OperationalError):
        return False
    code = getattr(exc, "pgcode", None)
    text = str(exc)
    if (
        isinstance(code, str) and code.startswith("28")
    ) or "password authentication failed" in text:
        return False  # a wrong credential is a failure to fix, not a reason to skip
    if isinstance(code, str) and code.startswith("08"):
        return True
    return any(fragment in text for fragment in UNREACHABLE_MESSAGES)


def _guarded(what: str, call: Callable[[], _T]) -> _T:
    """Run `call`; turn any failure into a fresh `ScratchDbError` that carries no driver frame.

    The new error is raised outside the `except`, with `from None`, so the driver's exception, its
    traceback and the locals of its frames are not part of the chain pytest prints.
    """
    failure: str | None = None
    try:
        return call()
    except ScratchDbError:
        raise
    except Exception as exc:
        failure = _describe(exc)
        del exc
    raise ScratchDbError(f"{what} failed{failure}, details withheld") from None


def _parse_admin(admin: _Secret) -> Any:
    """Parse the administrator URL, refusing a bad one without quoting any of it."""
    if not isinstance(admin, _Secret):
        raise ScratchConfigError("the administrator URL must be held in a _Secret")
    # Checked on the raw text, before any URL object exists: a parsed URL's repr shows its query.
    if "?" in admin.reveal():
        raise ScratchConfigError("the administrator URL carries a query string; refusing to use it")
    url = None
    try:
        url = make_url(psycopg_dsn(admin.reveal()))
    except Exception:
        pass
    if url is None:
        raise ScratchConfigError("the administrator URL is not a parseable URL") from None
    if not url.drivername.startswith("postgresql"):
        raise ScratchConfigError("the administrator URL is not a PostgreSQL URL")
    if len(url.query) != 0:
        raise ScratchConfigError("the administrator URL carries a query string; refusing to use it")
    if "," in (url.host or ""):
        raise ScratchConfigError("the administrator URL names more than one host; refusing it")
    return url


def validate_admin_url(admin: _Secret) -> None:
    """Raise `ScratchConfigError` unless the administrator URL is safe to use."""
    _parse_admin(admin)


class _Session:
    """The one administrator connection of a fixture run; its repr shows nothing."""

    __slots__ = ("conn",)

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def __repr__(self) -> str:
        return "<administrator session>"


def _close_quietly(conn: Any) -> None:
    """Close a connection that may already be broken; a failure here leaves nothing behind."""
    try:
        conn.close()
    except Exception:
        pass


def _open_session(admin: _Secret, connect: Callable[..., Any], *, first: bool) -> _Session:
    """Open an AUTOCOMMIT administrator connection.

    Only the first connection may skip the test, and only for a positively identified
    "server unreachable" error; anything else fails with a redacted message.
    """
    conn = None
    failure: str | None = None
    unreachable = False
    try:
        conn = connect(psycopg_dsn(admin.reveal()))
        conn.autocommit = True
    except Exception as exc:
        failure = _describe(exc)
        unreachable = is_unreachable(exc)
        del exc
    if failure is None:
        return _Session(conn)
    if conn is not None:
        _close_quietly(conn)
        conn = None
    if unreachable and first:
        pytest.skip("the scratch-database server is not reachable")
    raise ScratchDbError(
        f"opening the administrator connection failed{failure}, details withheld"
    ) from None


def _run(session: _Session, statement: str, params: tuple[str, ...] | None = None) -> Any:
    """Run one statement on the administrator session; return the first row, if any."""
    with session.conn.cursor() as cur:
        if params is None:
            cur.execute(statement)
        else:
            cur.execute(statement, params)
            return cur.fetchone()
    return None


def _drop_with_retry(
    session: _Session, admin: _Secret, connect: Callable[..., Any], drop: str
) -> None:
    """DROP on the open session; if it is lost, retry once on a fresh connection, then fail."""
    for attempt in range(2):
        try:
            if attempt:
                _close_quietly(session.conn)
                session.conn = _open_session(admin, connect, first=False).conn
            _run(session, drop)
            return
        except Exception as exc:
            del exc
    raise ScratchDbError(
        f"DROP DATABASE failed twice; a scratch database may be left behind: {LEFTOVER_HINT}"
    ) from None


@contextmanager
def scratch_database_name(
    admin: _Secret, connect: Callable[..., Any] = psycopg2.connect
) -> Iterator[str]:
    """Create a scratch database, yield its name, and drop it (only if this run created it).

    Nothing is opened until the URL and the generated name have both been checked. `created` is set
    only after `execute()` of CREATE returned, so a collision or an unknown outcome never reaches
    the DROP; once it is set, one `finally` always attempts the DROP, whatever else fails.
    """
    validate_admin_url(admin)
    name = new_scratch_name()
    create = create_sql(name)
    drop = drop_sql(name)
    check, check_params = exists_query(name)
    session = _open_session(admin, connect, first=True)
    created = False
    try:
        taken = _guarded("the scratch-name check", lambda: _run(session, check, check_params))
        if taken is not None:
            raise ScratchDbError("the generated scratch name already exists; nothing was created")
        failure: str | None = None
        try:
            with session.conn.cursor() as cur:
                cur.execute(create)
                created = True
        except Exception as exc:
            failure = _describe(exc)
            del exc
        if failure is not None and created:
            raise ScratchDbError(f"cleaning up after CREATE DATABASE failed{failure}") from None
        if failure is not None:
            raise ScratchDbError(
                f"CREATE DATABASE failed or its outcome is unknown{failure}; no DROP was issued: "
                f"{LEFTOVER_HINT}, details withheld"
            ) from None
        yield name
    finally:
        try:
            if created:
                _drop_with_retry(session, admin, connect, drop)
        finally:
            _close_quietly(session.conn)


def scratch_engine_url(admin: _Secret, name: str) -> Any:
    """The administrator URL with the driver and database replaced and the query dropped.

    The driver is named because SQLAlchemy 2.1 picks psycopg 3 for a bare `postgresql://` URL and
    this project installs psycopg2. The result is checked before it is returned.
    """
    checked = assert_scratch_name(name)
    parsed = _parse_admin(admin)
    url = parsed.set(drivername=SCRATCH_DRIVER, database=checked, query={})
    same_server = (url.host, url.port, url.username) == (parsed.host, parsed.port, parsed.username)
    if url.database != checked or len(url.query) != 0 or not same_server:
        raise ScratchConfigError("the scratch URL does not match the validated name")
    return url


def _connection_factory(dsn: _Secret, connect: Callable[..., Any]) -> Callable[[], Any]:
    """A pool `creator` whose failures are redacted, so a test body cannot leak the password."""

    def creator() -> Any:
        return _guarded("a scratch-database connection", lambda: connect(dsn.reveal()))

    return creator


def build_scratch_engine(
    admin: _Secret, name: str, connect: Callable[..., Any] = psycopg2.connect
) -> Engine:
    """An engine on the scratch database. Its URL holds no password: the factory holds the DSN."""
    dsn = _Secret(
        psycopg_dsn(scratch_engine_url(admin, name).render_as_string(hide_password=False))
    )
    return _guarded(
        "building the scratch engine",
        lambda: create_engine(
            f"{SCRATCH_DRIVER}://",
            creator=_connection_factory(dsn, connect),
            poolclass=NullPool,
        ),
    )


def apply_paper_schema(engine: Engine) -> None:
    """Create the paper tables from the DDL the code itself carries."""
    from ggTrader.paper import persist

    with engine.begin() as conn:
        for stmt in persist._SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))


@contextmanager
def scratch_db(
    *,
    schema: bool = True,
    connect: Callable[..., Any] = psycopg2.connect,
    engine_connect: Callable[..., Any] = psycopg2.connect,
) -> Iterator[Engine]:
    """Yield an engine on a fresh scratch database; skip when no usable server is named.

    `NullPool` makes every connection a real one, so a transaction left open at block exit
    reaches the server as a rollback exactly as it would in production. `connect` opens the
    administrator session, `engine_connect` the engine's connections (both are for tests).
    """
    admin = read_admin_secret()
    if admin is None:
        pytest.skip(f"{ADMIN_URL_ENV} is not set")
    with scratch_database_name(admin, connect) as name:
        engine = build_scratch_engine(admin, name, engine_connect)
        try:
            if schema:
                _guarded("applying the paper schema", lambda: apply_paper_schema(engine))
            yield engine
        finally:
            engine.dispose()
