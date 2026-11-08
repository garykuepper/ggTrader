"""Throwaway-database helpers shared by the integration tests of the membership schema.

Nothing here runs unless a test asks for the ``db_url`` fixture (``tests/data/conftest.py``), and
those tests are marked ``integration`` and ``db``, so no CI or default tier reaches this module's
network calls. See ``test_universe_db_integration.py`` for how to run them.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse

import psycopg2
import pytest

from ggTrader.data.universe_db import apply_ddl, connect, to_psycopg_url
from ggTrader.utils.config import get_db_connection_string

#: The NOLOGIN group roles the DDL creates; dropped afterwards only if this run created them.
ROLES = ("universe_reader", "universe_loader")


def server_url() -> str:
    """The server to create scratch databases on, or skip the test."""
    url = os.environ.get("GGT_TEST_DB_URL")
    if not url:
        pytest.skip("GGT_TEST_DB_URL is not set")
    url = to_psycopg_url(url)
    try:
        production = to_psycopg_url(get_db_connection_string())
    except ValueError:
        production = ""
    shared = url == production or urlparse(url).path.lstrip("/") == "ggtrader"
    if shared and os.environ.get("GGT_TEST_DB_ALLOW_SHARED") != "1":
        pytest.skip("GGT_TEST_DB_URL is the production database; set GGT_TEST_DB_ALLOW_SHARED=1")
    return url


class ScratchUrl(str):
    """A connection string whose repr hides it, so a failing test never prints credentials."""

    def __repr__(self) -> str:
        return "<scratch database url>"


@contextmanager
def scratch_database() -> Iterator[str]:
    """A fresh database ``ggt_test_<uuid>`` with the real DDL applied; dropped on exit."""
    server = server_url()
    name = f"ggt_test_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(server)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)", (list(ROLES),))
        roles_before = {row[0] for row in cur.fetchall()}
        cur.execute(f'CREATE DATABASE "{name}"')
    scratch = urlunparse(urlparse(server)._replace(path=f"/{name}"))
    try:
        conn = connect(scratch)
        try:
            apply_ddl(conn)
        finally:
            conn.close()
        yield ScratchUrl(scratch)
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            for role in ROLES:
                if role not in roles_before:
                    cur.execute(f'DROP ROLE IF EXISTS "{role}"')
        admin.close()
