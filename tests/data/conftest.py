"""Fixtures for the database-backed membership tests (scratch database per test)."""

from __future__ import annotations

from collections.abc import Iterator

import psycopg2
import pytest

from ggTrader.data.universe_db import connect
from tests.data.scratch_db import scratch_database


@pytest.fixture
def db_url() -> Iterator[str]:
    """A fresh scratch database with the real DDL applied; dropped (with its roles) afterwards."""
    with scratch_database() as url:
        yield url


@pytest.fixture
def conn(db_url: str) -> Iterator[psycopg2.extensions.connection]:
    """A read-write connection to the scratch database."""
    connection = connect(db_url)
    try:
        yield connection
    finally:
        connection.close()
