"""A throwaway PostgreSQL database for the persist characterization tests.

`scratch_db()` creates `ggt_scratch_<8 hex>` on the server named by the process environment
variable `GGT_SCRATCH_DB_ADMIN_URL`, applies only the paper tables' DDL (`persist._SCHEMA`), yields
an engine bound to that database, and drops it in `finally`.

Safety rules, each pinned by a test that runs without a database (`test_persist_scratch_db.py`):

* The administrator connection issues exactly two kinds of statement, `CREATE DATABASE <name>` and
  `DROP DATABASE <name> WITH (FORCE)`, on a name that matches `SCRATCH_NAME`. The name is checked
  before every statement and before any connection is opened, so a bad name raises first.
* The URL comes only from the environment: never from `.env`, never printed, logged or written.
  The yielded engine's own URL is the administrator URL with only the database name replaced.
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


class ScratchNameError(ValueError):
    """A database name that is not a scratch name was about to reach the server."""


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


def _run_admin(admin_url: str, statement: str, connect: Callable[..., Any]) -> None:
    """Run one statement on an AUTOCOMMIT administrator connection, then close it."""
    conn = connect(psycopg_dsn(admin_url))
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(statement)
    finally:
        conn.close()


@contextmanager
def scratch_database_name(
    admin_url: str, connect: Callable[..., Any] = psycopg2.connect
) -> Iterator[str]:
    """Create a scratch database, yield its name, and drop it in `finally`."""
    name = new_scratch_name()
    statement = create_sql(name)  # the name is checked before any connection is opened
    _run_admin(admin_url, statement, connect)
    try:
        yield name
    finally:
        _run_admin(admin_url, drop_sql(name), connect)


def scratch_engine_url(admin_url: str, name: str) -> Any:
    """The administrator URL with only the database name replaced.

    The driver is named because SQLAlchemy 2.1 picks psycopg 3 for a bare `postgresql://` URL and
    this project installs psycopg2.
    """
    return make_url(psycopg_dsn(admin_url)).set(
        drivername="postgresql+psycopg2", database=assert_scratch_name(name)
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
def scratch_db(*, schema: bool = True) -> Iterator[Engine]:
    """Yield an engine on a fresh scratch database; skip when the environment names no server.

    `NullPool` makes every connection a real one, so a transaction left open at block exit
    reaches the server as a rollback exactly as it would in production.
    """
    admin_url = os.environ.get(ADMIN_URL_ENV)
    if not admin_url:
        pytest.skip(f"{ADMIN_URL_ENV} is not set")
    with scratch_database_name(admin_url) as name:
        engine = create_engine(scratch_engine_url(admin_url, name), poolclass=NullPool)
        try:
            if schema:
                apply_paper_schema(engine)
            yield engine
        finally:
            engine.dispose()
