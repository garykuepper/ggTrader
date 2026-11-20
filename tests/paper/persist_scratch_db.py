"""A throwaway PostgreSQL database for the persist characterization tests.

`scratch_db()` creates `ggt_scratch_<8 hex>` on the server named by the process environment
variable `GGT_SCRATCH_DB_ADMIN_URL`, applies only the paper tables' DDL (`persist._SCHEMA`), yields
an engine bound to that database, and drops it in `finally`.

The administrator URL is a superuser credential, so the fixture fails closed:

* The administrator connection issues exactly two kinds of statement, `CREATE DATABASE <name>` and
  `DROP DATABASE <name> WITH (FORCE)`, on a name that matches `SCRATCH_NAME`. The name is checked
  before every statement and before any connection is opened, so a bad name raises first.
* The URL comes only from the environment: never from `.env`, never printed, logged or written. It
  lives inside `_Secret` (its `repr` and `str` are `<redacted>`) and is unwrapped only in the call
  that hands it to the driver, so a traceback with locals shown cannot expose it. No message the
  fixture raises or skips with contains any part of the URL.
* An administrator URL whose query names `dbname`, `database`, `host` (or `hostaddr`, `service`,
  `port`, `user`, `password`) is refused before any connection: a query option is applied after
  the path and would silently re-point the connection.
* The scratch URL is the administrator URL with the driver and database name replaced and the
  WHOLE query dropped (an exotic option fails closed rather than carrying over). It is checked
  (database equals the validated name, query empty, same host, port and user) before the engine
  exists.
* If the server cannot be reached for the first connection the test skips. Once CREATE has been
  sent, one `finally` always attempts the DROP, whatever else fails; every error then fails loudly.
* Nothing here touches `ohlcv`, an existing `paper_*` table or an existing database.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import psycopg2
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool

ADMIN_URL_ENV = "GGT_SCRATCH_DB_ADMIN_URL"
SCRATCH_NAME = re.compile(r"^ggt_scratch_[0-9a-f]{8}$")
SCRATCH_DRIVER = "postgresql+psycopg2"
# Query keys that can re-point or re-authenticate a connection after the URL path was read.
FORBIDDEN_QUERY_KEYS = frozenset(
    {"dbname", "database", "host", "hostaddr", "service", "port", "user", "password"}
)


class ScratchNameError(ValueError):
    """A database name that is not a scratch name was about to reach the server."""


class ScratchConfigError(ValueError):
    """The administrator URL is unusable or unsafe. Messages never quote the URL."""


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


def psycopg_dsn(url: str) -> str:
    """The URL without a SQLAlchemy driver suffix, as libpq reads it."""
    return re.sub(r"^postgresql\+\w+://", "postgresql://", url)


def _parse_admin(admin: _Secret) -> Any:
    """Parse the administrator URL, refusing a bad one without quoting any of it."""
    if not isinstance(admin, _Secret):
        raise ScratchConfigError("the administrator URL must be held in a _Secret")
    try:
        url = make_url(psycopg_dsn(admin.reveal()))
    except Exception:
        raise ScratchConfigError("the administrator URL is not a parseable URL") from None
    if not url.drivername.startswith("postgresql"):
        raise ScratchConfigError("the administrator URL is not a PostgreSQL URL")
    forbidden = sorted({str(k).lower() for k in url.query} & FORBIDDEN_QUERY_KEYS)
    if forbidden:
        raise ScratchConfigError(
            f"the administrator URL query names {', '.join(forbidden)}; refusing to use it"
        )
    return url


def validate_admin_url(admin: _Secret) -> None:
    """Raise `ScratchConfigError` unless the administrator URL is safe to use."""
    _parse_admin(admin)


def _open_admin(admin: _Secret, connect: Callable[..., Any]) -> Any:
    """Open the first administrator connection; an unreachable server skips the test."""
    try:
        return connect(psycopg_dsn(admin.reveal()))
    except psycopg2.OperationalError:
        pass
    # Skipped outside the `except` so the driver's message cannot ride along as context.
    pytest.skip("the scratch-database server is not reachable")


def _run_admin(admin: _Secret, statement: str, connect: Callable[..., Any]) -> None:
    """Run one statement on an AUTOCOMMIT administrator connection, then close it."""
    conn = connect(psycopg_dsn(admin.reveal()))
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(statement)
    finally:
        conn.close()


@contextmanager
def scratch_database_name(
    admin: _Secret, connect: Callable[..., Any] = psycopg2.connect
) -> Iterator[str]:
    """Create a scratch database, yield its name, and drop it in one `finally`.

    Nothing is opened until the URL and the generated name have both been checked. `created` is
    set before CREATE is sent, so even a failure while sending it, or in the cleanup that follows
    it, still reaches the DROP.
    """
    validate_admin_url(admin)
    name = new_scratch_name()
    create = create_sql(name)
    drop = drop_sql(name)
    conn = _open_admin(admin, connect)
    created = False
    try:
        try:
            conn.autocommit = True
            created = True
            with conn.cursor() as cur:
                cur.execute(create)
        finally:
            conn.close()
        yield name
    finally:
        if created:
            _run_admin(admin, drop, connect)


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
    *, schema: bool = True, connect: Callable[..., Any] = psycopg2.connect
) -> Iterator[Engine]:
    """Yield an engine on a fresh scratch database; skip when no usable server is named.

    `NullPool` makes every connection a real one, so a transaction left open at block exit
    reaches the server as a rollback exactly as it would in production.
    """
    admin = read_admin_secret()
    if admin is None:
        pytest.skip(f"{ADMIN_URL_ENV} is not set")
    with scratch_database_name(admin, connect) as name:
        engine = create_engine(scratch_engine_url(admin, name), poolclass=NullPool)
        try:
            if schema:
                apply_paper_schema(engine)
            yield engine
        finally:
            engine.dispose()
