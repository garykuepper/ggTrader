"""Exercise forward DDL only in an isolated disposable Postgres container."""

import os
import secrets
import subprocess
import time
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

SCHEMA = Path(__file__).resolve().parents[2] / "sql" / "forward_schema.sql"
TABLES = ("loads", "listing_snapshots", "daily_bars", "gaps", "symbol_status", "shares_outstanding")


@pytest.fixture(scope="module")
def database() -> object:
    """Create a separate ephemeral container, bound to a random loopback port."""
    name = "ggt_forward_test_" + secrets.token_hex(12)
    password = secrets.token_urlsafe(24)
    environment = os.environ.copy()
    environment["POSTGRES_PASSWORD"] = password
    started = False
    conn = None
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-p",
                "127.0.0.1::5432",
                "--tmpfs",
                "/var/lib/postgresql/data:rw,size=128m",
                "-e",
                "POSTGRES_PASSWORD",
                "-e",
                "POSTGRES_USER=ggtrader",
                "-e",
                "POSTGRES_DB=postgres",
                "postgres:16",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        started = True
        address = subprocess.run(
            ["docker", "port", name, "5432/tcp"], check=True, capture_output=True, text=True
        ).stdout.strip()
        host, port = address.rsplit(":", 1)
        assert host == "127.0.0.1" and port.isdecimal() and int(port) != 5433
        for attempt in range(60):
            try:
                conn = psycopg2.connect(
                    host=host,
                    port=int(port),
                    user="ggtrader",
                    password=password,
                    dbname="postgres",
                    connect_timeout=2,
                )
                break
            except psycopg2.OperationalError:
                if attempt == 59:
                    raise
                time.sleep(1)
        assert conn is not None
        conn.autocommit = True
        yield conn
    finally:
        if conn is not None:
            conn.close()
        if started:
            subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


@pytest.fixture(scope="module")
def seeded(database: object) -> object:
    """Apply schema and add a row to each table in the throwaway database."""
    with database.cursor() as cur:
        cur.execute(SCHEMA.read_text())
        cur.execute(
            "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
            "code_commit, actor) VALUES ('2026-11-25', 'test', %s, %s, 'abc', 'test') "
            "RETURNING load_id",
            ("a" * 64, "b" * 64),
        )
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.listing_snapshots "
            "(load_id, day, symbol, exchange, security_name, is_etf, is_test) "
            "VALUES (%s, '2026-11-25', 'AAA', 'Q', 'A common', false, false)",
            (load,),
        )
        cur.execute(
            "INSERT INTO forward.daily_bars (load_id, day, symbol, feed, open, high, low, "
            "close, volume) VALUES (%s, '2026-11-25', 'AAA', 'sip', 1, 2, 1, 2, 10)",
            (load,),
        )
        cur.execute(
            "INSERT INTO forward.gaps (load_id, day, symbol, reason) "
            "VALUES (%s, '2026-11-25', 'BBB', 'missing')",
            (load,),
        )
        cur.execute(
            "INSERT INTO forward.symbol_status (symbol, first_seen, last_seen) "
            "VALUES ('AAA', '2026-11-25', '2026-11-25')"
        )
        cur.execute(
            "INSERT INTO forward.shares_outstanding (load_id, day, symbol, shares, source) "
            "VALUES (%s, '2026-11-25', 'AAA', 100, 'yahoo')",
            (load,),
        )
        cur.execute("CREATE ROLE forward_test_writer NOLOGIN")
        cur.execute("GRANT USAGE ON SCHEMA forward TO forward_test_writer")
        cur.execute(
            "GRANT SELECT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA forward "
            "TO forward_test_writer"
        )
    return database


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("role", ["owner", "writer", "replica"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE", "TRUNCATE"])
def test_changes_refused_for_every_role(
    seeded: object, table: str, role: str, operation: str
) -> None:
    """Even privileged writers and replica mode cannot alter stored rows."""
    with seeded.cursor() as cur:
        try:
            if role == "writer":
                cur.execute("SET ROLE forward_test_writer")
            if role == "replica":
                cur.execute("SET session_replication_role = replica")
            stmt = (
                sql.SQL("UPDATE forward.{} SET {} = {}").format(
                    sql.Identifier(table),
                    sql.Identifier("day" if table != "symbol_status" else "symbol"),
                    sql.Identifier("day" if table != "symbol_status" else "symbol"),
                )
                if operation == "UPDATE"
                else sql.SQL("{} FROM forward.{}").format(sql.SQL(operation), sql.Identifier(table))
                if operation == "DELETE"
                else sql.SQL("TRUNCATE forward.{} CASCADE").format(sql.Identifier(table))
            )
            with pytest.raises(psycopg2.Error, match="append-only"):
                cur.execute(stmt)
        finally:
            cur.execute("RESET ROLE")
            cur.execute("SET session_replication_role = origin")


def test_ddl_is_idempotent(seeded: object) -> None:
    """Applying the DDL twice preserves the existing rows."""
    with seeded.cursor() as cur:
        cur.execute(SCHEMA.read_text())
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 1


def test_bars_unique_per_day_symbol_feed_load(seeded: object) -> None:
    """The same daily bar cannot appear twice within a load and feed."""
    with seeded.cursor() as cur:
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute("INSERT INTO forward.daily_bars SELECT * FROM forward.daily_bars")


def test_disposable_database_is_not_live(database: object) -> None:
    """The database identity and connection cannot target the live service."""
    assert database.info.host == "127.0.0.1"
    assert database.info.port != 5433
    assert database.info.dbname == "postgres"
    with database.cursor() as cur:
        cur.execute("SELECT current_database(), inet_server_port()")
        dbname, port = cur.fetchone()
        assert dbname == "postgres"
        assert port == 5432


def test_same_bar_can_be_recorded_in_a_different_load(seeded: object) -> None:
    """Uniqueness is scoped to the load as well as the date, symbol and feed."""
    with seeded.cursor() as cur:
        cur.execute(
            "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
            "code_commit, actor) VALUES ('2026-11-25', 'test', %s, %s, 'abc', 'test') "
            "RETURNING load_id",
            ("c" * 64, "d" * 64),
        )
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.daily_bars (load_id, day, symbol, feed, open, high, low, "
            "close, volume) VALUES (%s, '2026-11-25', 'AAA', 'sip', 1, 2, 1, 2, 10)",
            (load,),
        )
