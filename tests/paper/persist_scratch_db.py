"""A throwaway PostgreSQL database for the persist characterization tests.

WARNING: four independent security reviews (2026-11-20) still rated this fixture NOT SAFE when
it is given a superuser URL (for example, a database renamed or replaced by another session after
our CREATE would be force-dropped; libpq service files and environment defaults can redirect the
connection). It is merged at the owner's request and is inert unless `GGT_SCRATCH_DB_ADMIN_URL` is
set. Use it only against a disposable PostgreSQL cluster with a dedicated CREATEDB role, never a
superuser URL for a server that holds real data.

`scratch_db()` creates `ggt_scratch_<8 hex>` on the server named by the process environment
variable `GGT_SCRATCH_DB_ADMIN_URL`, applies only the paper tables' DDL (`persist._SCHEMA`), yields
an engine bound to that database, and drops it at the end.

The administrator URL is a superuser credential, so the fixture is deliberately small and fails
closed:

* One host. The URL's host must be exactly `localhost`, `127.0.0.1` or `[::1]`; no query, no
  fragment, no `%` in the host, no comma anywhere in the authority, and a user is required. The port
  is the URL's or, when it has none, 5433, always passed explicitly. The driver receives keyword
  arguments (host, port, user, password, database), never a URL or DSN for libpq to interpret, and
  the run is refused when `PGHOSTADDR` or `PGSERVICE` could re-route it. All of this is checked
  before any socket is opened.
* Skip only when the server cannot be reached: a plain TCP connect to host:port (3 s, no
  credentials) decides. Once it succeeds every later failure (authentication, unknown database,
  anything) FAILS with a redacted message.
* One administrator session for the fixture's lifetime and exactly three statements, all on one
  validated name: `SELECT 1 FROM pg_database WHERE datname = <name>`, `CREATE DATABASE <name>` and
  `DROP DATABASE <name> WITH (FORCE)`. There is no retry and no second administrator connection.
* Ownership. `maybe_created` is set before CREATE is sent and `created` after it returned. The DROP
  runs once, only if `created`, and only in the process that created the database (a forked child
  that unwinds the fixture does nothing at all). If the DROP fails, or CREATE's outcome is unknown,
  the fixture raises a loud error that names the database (the name is not secret) and gives the SQL
  to check or remove it by hand.
* Secrets. The URL comes only from the environment, lives inside `_Secret` (every rendering is
  `<redacted>`) and is unwrapped only inside small helpers that set `__tracebackhide__`. Every
  driver or engine call goes through `_attempt`, which catches `BaseException`, keeps at most a
  SQLSTATE, and lets the caller raise a NEW exception (`from None`): `ScratchDbError`, or
  `KeyboardInterrupt` / `SystemExit` as themselves. No driver frame, message or local reaches
  pytest's output, even with `--showlocals`.
* Nothing here touches `ohlcv`, an existing `paper_*` table or an existing database.
"""

from __future__ import annotations

import os
import re
import socket
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NoReturn

import psycopg2
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool

ADMIN_URL_ENV = "GGT_SCRATCH_DB_ADMIN_URL"
SCRATCH_NAME = re.compile(r"^ggt_scratch_[0-9a-f]{8}$")
SCRATCH_DRIVER = "postgresql+psycopg2"
EXISTS_SQL = "SELECT 1 FROM pg_database WHERE datname = %s"
DEFAULT_PORT = 5433
DEFAULT_ADMIN_DATABASE = "postgres"
TCP_TIMEOUT_S = 3.0
CONNECT_TIMEOUT_S = 10
ALLOWED_HOSTS = {"localhost": "localhost", "127.0.0.1": "127.0.0.1", "[::1]": "::1"}
ROUTING_ENV = ("PGHOSTADDR", "PGSERVICE")  # libpq variables that can re-route a connection
_URL_SHAPE = re.compile(r"postgres(?:ql)?(?:\+[a-z0-9]+)?://(?P<authority>[^/]*)(?:/[^/]*)?")
_HOST_PORT = re.compile(r"(?P<host>localhost|127\.0\.0\.1|\[::1\])(?::(?P<port>[0-9]{1,5}))?")
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")


class ScratchNameError(ValueError):
    """A database name that is not a scratch name was about to reach the server."""


class ScratchConfigError(ValueError):
    """The administrator URL is unusable or unsafe. Messages never quote the URL."""


class ScratchDbError(RuntimeError):
    """A scratch-database step failed. The message is fixed text: it never quotes driver output."""


class _Secret:
    """Holds a credential (the URL, or the login fields); every rendering is `<redacted>`."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
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

    def reveal(self) -> Any:
        """The raw value. Call it only inside a helper that sets `__tracebackhide__`."""
        __tracebackhide__ = True
        return self._value


class _Endpoint:
    """The validated administrator endpoint: one loopback host, an explicit port, a hidden login."""

    __slots__ = ("host", "port", "_login")

    def __init__(self, host: str, port: int, login: _Secret) -> None:
        self.host = host
        self.port = port
        self._login = login  # (user, password or None, administrator database)

    def __repr__(self) -> str:
        return f"<scratch endpoint {self.host}:{self.port}>"

    def __reduce__(self) -> Any:
        raise TypeError("an endpoint cannot be pickled")


class _Failure:
    """What `_attempt` keeps of an exception: its kind and, at most, a SQLSTATE."""

    __slots__ = ("kind", "sqlstate", "code")

    def __init__(self, kind: str, sqlstate: str = "", code: int = 1) -> None:
        self.kind = kind  # "interrupt", "exit" or "error"
        self.sqlstate = sqlstate
        self.code = code

    @property
    def detail(self) -> str:
        return f" (SQLSTATE {self.sqlstate})" if self.sqlstate else ""


def _sqlstate(exc: BaseException) -> str:
    code = getattr(exc, "pgcode", None) or getattr(getattr(exc, "orig", None), "pgcode", None)
    return code if isinstance(code, str) and _SQLSTATE.fullmatch(code) else ""


def _attempt(call: Callable[[], Any]) -> tuple[Any, _Failure | None]:
    """Run `call`; on ANY exception return a `_Failure` instead, dropping the exception.

    The caller raises a fresh exception outside any `except`, so the original exception, its
    traceback and the locals of the driver's frames are never shown.
    """
    __tracebackhide__ = True
    try:
        return call(), None
    except KeyboardInterrupt:
        return None, _Failure("interrupt")
    except SystemExit as exc:
        return None, _Failure("exit", code=exc.code if isinstance(exc.code, int) else 1)
    except BaseException as exc:  # every failure becomes a redacted one
        return None, _Failure("error", _sqlstate(exc))


def _raise(failure: _Failure, message: str) -> NoReturn:
    """Raise the fresh exception for `failure`: an interrupt or exit as itself, else redacted."""
    __tracebackhide__ = True
    if failure.kind == "interrupt":
        raise KeyboardInterrupt() from None
    if failure.kind == "exit":
        raise SystemExit(failure.code) from None
    raise ScratchDbError(f"{message}{failure.detail}, details withheld") from None


def _pid() -> int:
    return os.getpid()


def read_admin_secret() -> _Secret | None:
    """The administrator URL from the environment, wrapped at once; `None` when not set."""
    __tracebackhide__ = True
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


def _parse_endpoint(admin: _Secret) -> _Endpoint | str:
    """The endpoint, or the fixed reason the URL is refused. Raw text never leaves this frame."""
    __tracebackhide__ = True
    shape = _URL_SHAPE.fullmatch(admin.reveal())
    if shape is None or "?" in admin.reveal() or "#" in admin.reveal():
        return "the administrator URL is not a plain PostgreSQL URL without a query or fragment"
    if "," in shape["authority"]:
        return "the administrator URL names more than one host; refusing it"
    if "@" not in shape["authority"]:
        return "the administrator URL names no user; refusing it"
    host_port = shape["authority"].rpartition("@")[2]
    if "%" in host_port:
        return "the administrator URL's host is percent-encoded; refusing it"
    allowed = _HOST_PORT.fullmatch(host_port)
    if allowed is None:
        return "the administrator URL's host must be exactly localhost, 127.0.0.1 or [::1]"
    port = int(allowed["port"]) if allowed["port"] else DEFAULT_PORT
    if not 1 <= port <= 65535:
        return "the administrator URL's port is out of range"
    url = make_url(re.sub(r"^postgres(?:ql)?(?:\+[a-z0-9]+)?://", "postgresql://", admin.reveal()))
    host = ALLOWED_HOSTS[allowed["host"]]
    explicit_port = int(allowed["port"]) if allowed["port"] else None
    if url.host != host or url.port != explicit_port or len(url.query) != 0 or not url.username:
        return "the administrator URL does not parse to the one host it names; refusing it"
    login = _Secret((url.username, url.password, url.database or DEFAULT_ADMIN_DATABASE))
    return _Endpoint(host, port, login)


def admin_endpoint(admin: _Secret) -> _Endpoint:
    """The validated endpoint; raise `ScratchConfigError` (quoting nothing) before any socket."""
    __tracebackhide__ = True
    if not isinstance(admin, _Secret):
        del admin  # a raw string must not stay in this frame's locals
        raise ScratchConfigError("the administrator URL must be held in a _Secret")
    for variable in ROUTING_ENV:
        if os.environ.get(variable):
            raise ScratchConfigError(f"{variable} is set and could re-route the connection")
    parsed, failure = _attempt(lambda: _parse_endpoint(admin))
    if failure is not None and failure.kind != "error":
        _raise(failure, "")
    if failure is not None:
        raise ScratchConfigError("the administrator URL is not a parseable URL") from None
    if isinstance(parsed, str):
        raise ScratchConfigError(parsed) from None
    return parsed


def _connect_kwargs(endpoint: _Endpoint, database: str | None = None) -> dict[str, Any]:
    """Keyword arguments for the driver: every routing field explicit, nothing for libpq to guess.

    Call it only inside the expression handed to the driver: the result holds the password.
    """
    __tracebackhide__ = True
    user, password, admin_database = endpoint._login.reveal()
    kwargs = {
        "host": endpoint.host,
        "port": endpoint.port,
        "user": user,
        "dbname": admin_database if database is None else assert_scratch_name(database),
        "connect_timeout": CONNECT_TIMEOUT_S,
    }
    if password is not None:
        kwargs["password"] = password
    return kwargs


def tcp_reachable(host: str, port: int, timeout: float = TCP_TIMEOUT_S) -> bool:
    """True when a plain TCP connection to host:port succeeds (no credentials are involved)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class _Session:
    """The one administrator connection of a fixture run; its repr shows nothing."""

    __slots__ = ("conn",)

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def __repr__(self) -> str:
        return "<administrator session>"


def _open_session(endpoint: _Endpoint, connect: Callable[..., Any]) -> _Session:
    """Open the AUTOCOMMIT administrator connection; any failure is loud (never a skip)."""
    __tracebackhide__ = True

    def call() -> Any:
        conn = connect(**_connect_kwargs(endpoint))
        try:
            conn.autocommit = True
        except BaseException:
            conn.close()
            raise
        return conn

    conn, failure = _attempt(call)
    if failure is not None:
        _raise(failure, "opening the administrator connection failed")
    return _Session(conn)


def _execute(
    session: _Session, statement: str, params: tuple[str, ...] | None = None
) -> tuple[Any, _Failure | None, bool]:
    """Run one statement; return (first row for a SELECT, failure, whether execute() returned)."""
    __tracebackhide__ = True
    executed: list[bool] = []

    def call() -> Any:
        cur = session.conn.cursor()
        try:
            if params is None:
                cur.execute(statement)
            else:
                cur.execute(statement, params)
            executed.append(True)
            return cur.fetchone() if params is not None else None
        finally:
            cur.close()

    row, failure = _attempt(call)
    return row, failure, bool(executed)


def _loud(message: str, interrupted: bool) -> NoReturn:
    """A leftover or an unknown outcome: an interrupt stays an interrupt, but says so."""
    __tracebackhide__ = True
    if interrupted:
        raise KeyboardInterrupt(message)
    raise ScratchDbError(message)


def _teardown(
    session: _Session,
    name: str,
    *,
    owner_pid: int,
    maybe_created: bool,
    created: bool,
    pending: type[BaseException] | None,
) -> None:
    """DROP once on the same session if this process created the database; then close."""
    __tracebackhide__ = True
    if _pid() != owner_pid:
        return  # a forked child: the session and the database belong to the parent
    interrupted = pending is not None and issubclass(pending, (KeyboardInterrupt, SystemExit))
    message = None
    if created:
        _, failure, executed = _execute(session, drop_sql(name))
        if not executed:
            interrupted = interrupted or failure.kind != "error"
            message = (
                f"DROP DATABASE failed{failure.detail}; the scratch database {name} may be left "
                f'behind. Check and remove it by hand: DROP DATABASE "{name}" WITH (FORCE);'
            )
    elif maybe_created:
        message = (
            f"CREATE DATABASE failed or its outcome is unknown; no DROP was issued because this "
            f"run cannot prove it created {name}. Check by hand (manual cleanup): SELECT 1 FROM "
            f"pg_database WHERE datname = '{name}'; and drop it only if no other run owns it."
        )
    _attempt(session.conn.close)  # the database is already dealt with; a close failure is moot
    if message is not None:
        _loud(message, interrupted)


@contextmanager
def scratch_database_name(
    admin: _Secret,
    connect: Callable[..., Any] = psycopg2.connect,
    probe: Callable[[str, int], bool] = tcp_reachable,
) -> Iterator[str]:
    """Create a scratch database, yield its name, and drop it (only if this run created it)."""
    __tracebackhide__ = True
    endpoint = admin_endpoint(admin)
    name = new_scratch_name()
    create = create_sql(name)
    check, check_params = exists_query(name)
    if not probe(endpoint.host, endpoint.port):
        pytest.skip("the scratch-database server is not reachable")
    session = _open_session(endpoint, connect)
    owner_pid = _pid()
    maybe_created = created = False
    pending: type[BaseException] | None = None
    try:
        row, failure, _ = _execute(session, check, check_params)
        if failure is not None:
            _raise(failure, "the scratch-name check failed")
        if row is not None:
            raise ScratchDbError("the generated scratch name already exists; nothing was created")
        maybe_created = True
        _, failure, created = _execute(session, create)
        if failure is not None:
            _raise(failure, f"CREATE DATABASE {name} failed")
        yield name
    except BaseException as exc:
        pending = type(exc)
        raise
    finally:
        _teardown(
            session,
            name,
            owner_pid=owner_pid,
            maybe_created=maybe_created,
            created=created,
            pending=pending,
        )


def _connection_factory(
    endpoint: _Endpoint, name: str, connect: Callable[..., Any]
) -> Callable[[], Any]:
    """A pool `creator` on the scratch database whose failures are redacted."""
    checked = assert_scratch_name(name)

    def creator() -> Any:
        __tracebackhide__ = True
        conn, failure = _attempt(lambda: connect(**_connect_kwargs(endpoint, checked)))
        if failure is not None:
            _raise(failure, "a scratch-database connection failed")
        return conn

    return creator


def build_scratch_engine(
    endpoint: _Endpoint, name: str, connect: Callable[..., Any] = psycopg2.connect
) -> Engine:
    """An engine on the scratch database. Its URL is empty: only the factory knows the login."""
    __tracebackhide__ = True
    engine, failure = _attempt(
        lambda: create_engine(
            f"{SCRATCH_DRIVER}://",
            creator=_connection_factory(endpoint, name, connect),
            poolclass=NullPool,
        )
    )
    if failure is not None:
        _raise(failure, "building the scratch engine failed")
    return engine


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
    probe: Callable[[str, int], bool] = tcp_reachable,
) -> Iterator[Engine]:
    """Yield an engine on a fresh scratch database; skip only when the server is unreachable.

    `NullPool` makes every connection a real one, so a transaction left open at block exit
    reaches the server as a rollback exactly as it would in production. `connect` opens the
    administrator session, `engine_connect` the engine's connections, `probe` is the TCP check
    (all three are for tests).
    """
    __tracebackhide__ = True
    admin = read_admin_secret()
    if admin is None:
        pytest.skip(f"{ADMIN_URL_ENV} is not set")
    endpoint = admin_endpoint(admin)
    with scratch_database_name(admin, connect, probe) as name:
        engine = build_scratch_engine(endpoint, name, engine_connect)
        owner_pid = _pid()
        try:
            if schema:
                _, failure = _attempt(lambda: apply_paper_schema(engine))
                if failure is not None:
                    _raise(failure, "applying the paper schema failed")
            yield engine
        finally:
            if _pid() == owner_pid:
                engine.dispose()
