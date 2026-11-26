"""Kaggle schema tests on a throwaway database."""

import argparse

import psycopg2
import pytest

from ggTrader.cli.cmd_db import register_db_parser, run_db
from ggTrader.data.kaggle_import.ddl import MCGILL_COLUMNS, apply_ddl
from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def conn():
    with scratch_database() as url, psycopg2.connect(url) as connection:
        apply_ddl(connection)
        yield connection


def test_ddl_is_idempotent(conn):
    apply_ddl(conn)


def test_no_timestamp_without_time_zone_columns(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema IN ('ref','market','kaggle_crsp','kaggle_chars') "
            "AND data_type='timestamp without time zone'"
        )
        assert cur.fetchone()[0] == 0


def test_complete_load_is_immutable(conn):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ref.loads(status) VALUES ('running') RETURNING load_id")
        load_id = cur.fetchone()[0]
        cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (load_id,))
    conn.commit()
    for statement in (
        "UPDATE ref.loads SET rows_parsed=1",
        "DELETE FROM ref.loads",
        "TRUNCATE ref.loads",
    ):
        with pytest.raises(psycopg2.Error), conn.cursor() as cur:
            cur.execute(statement)
        conn.rollback()
    with conn.cursor() as cur:
        cur.execute("UPDATE ref.loads SET status='withdrawn' WHERE load_id=%s", (load_id,))
        assert cur.rowcount == 1


def test_hypertables_exist(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT hypertable_schema, hypertable_name, compression_enabled "
            "FROM timescaledb_information.hypertables "
            "WHERE hypertable_schema IN ('market','kaggle_crsp')"
        )
        assert set(cur.fetchall()) == {
            ("market", "daily_bars", True),
            ("kaggle_crsp", "daily", True),
        }
        cur.execute(
            "SELECT hypertable_schema, hypertable_name, time_interval = INTERVAL '1 year' "
            "FROM timescaledb_information.dimensions "
            "WHERE hypertable_schema IN ('market','kaggle_crsp')"
        )
        assert set(cur.fetchall()) == {
            ("market", "daily_bars", True),
            ("kaggle_crsp", "daily", True),
        }
        cur.execute(
            "SELECT hypertable_schema, hypertable_name, attname "
            "FROM timescaledb_information.compression_settings "
            "WHERE segmentby_column_index IS NOT NULL "
            "AND hypertable_schema IN ('market','kaggle_crsp')"
        )
        assert set(cur.fetchall()) == {
            ("market", "daily_bars", "permno"),
            ("market", "daily_bars", "load_id"),
            ("kaggle_crsp", "daily", "permno"),
            ("kaggle_crsp", "daily", "load_id"),
        }


@pytest.mark.parametrize("table", ["symbol_map", "splice_links"])
def test_frozen_tables_reject_update_delete_truncate(conn, table):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ref.releases(status) VALUES ('building') RETURNING release_id")
        rid = cur.fetchone()[0]
        if table == "splice_links":
            cur.execute("INSERT INTO ref.splice_links(release_id, permno) VALUES (%s, 1)", (rid,))
        else:
            cur.execute(f"INSERT INTO ref.{table}(release_id) VALUES (%s)", (rid,))
        cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=%s", (rid,))
    conn.commit()
    for statement in (
        f"UPDATE ref.{table} SET symbol=NULL",
        f"DELETE FROM ref.{table}",
        f"TRUNCATE ref.{table}",
    ):
        with pytest.raises(psycopg2.Error), conn.cursor() as cur:
            cur.execute(statement)
        conn.rollback()


def test_mcgill_columns_match_constant(conn):
    types = dict(MCGILL_COLUMNS)
    assert {name: types[name] for name in ("permno", "shrcd", "exchcd", "year", "month")} == {
        "permno": "bigint",
        "shrcd": "integer",
        "exchcd": "integer",
        "year": "integer",
        "month": "integer",
    }
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='kaggle_chars' AND table_name='mcgill' "
            "ORDER BY ordinal_position"
        )
        assert cur.fetchall() == [("load_id", "bigint"), *MCGILL_COLUMNS]


def test_cli_apply_ddl_parses_and_runs(monkeypatch):
    parser = argparse.ArgumentParser(prog="ggt")
    register_db_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        ["db", "import", "apply-ddl", "--db-url-env", "TEST_IMPORT_URL", "--yes"]
    )
    assert (args.db_command, args.import_command, args.db_url_env) == (
        "import",
        "apply-ddl",
        "TEST_IMPORT_URL",
    )
    with scratch_database() as url:
        monkeypatch.setenv("TEST_IMPORT_URL", url)
        run_db(args)
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('market.daily_bars'), to_regclass('sec.death_labels')")
            assert cur.fetchone() == ("market.daily_bars", "sec.death_labels")


def test_release_withdrawal_and_links_are_immutable(conn):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ref.releases(status) VALUES ('building') RETURNING release_id")
        release_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.release_loads(release_id, load_id) VALUES (%s, 1)", (release_id,)
        )
        cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=%s", (release_id,))
    conn.commit()
    for statement in (
        "UPDATE ref.releases SET settings='{}'::jsonb",
        "UPDATE ref.releases SET status='withdrawn', settings='{}'::jsonb",
        "DELETE FROM ref.releases",
        "DELETE FROM ref.release_loads",
        "UPDATE ref.release_loads SET role='other'",
        "TRUNCATE ref.releases",
        "TRUNCATE ref.release_loads",
    ):
        with pytest.raises(psycopg2.Error), conn.cursor() as cur:
            cur.execute(statement)
        conn.rollback()
    with conn.cursor() as cur:
        cur.execute("UPDATE ref.releases SET status='withdrawn' WHERE release_id=%s", (release_id,))
    conn.commit()
    with pytest.raises(psycopg2.Error), conn.cursor() as cur:
        cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=%s", (release_id,))
    conn.rollback()


def test_release_columns_and_frozen_spans(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='ref' AND "
            "table_name='splice_links'"
        )
        assert {"common_day", "join_returns"} <= {row[0] for row in cur.fetchall()}
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='market' AND "
            "table_name='stock_spans'"
        )
        assert {"kind", "crsp_until", "first_yf_date", "possible_double"} <= {
            row[0] for row in cur.fetchall()
        }
        cur.execute("INSERT INTO ref.releases(status) VALUES ('building') RETURNING release_id")
        rid = cur.fetchone()[0]
        cur.execute("INSERT INTO market.stock_spans(release_id, symbol) VALUES (%s, 'X')", (rid,))
        cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=%s", (rid,))
    conn.commit()
    for sql in (
        "UPDATE market.stock_spans SET symbol='Y'",
        "DELETE FROM market.stock_spans",
        "TRUNCATE market.stock_spans",
    ):
        with pytest.raises(psycopg2.Error), conn.cursor() as cur:
            cur.execute(sql)
        conn.rollback()


def test_closing_check_required_indexes_exist_on_repository_ddl(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tablename,indexname,indexdef FROM pg_indexes "
            "WHERE schemaname IN ('ref', 'market')"
        )
        indexes = {name: (table, definition.lower()) for table, name, definition in cur.fetchall()}
    assert "release_loads_role_unique" in indexes
    assert indexes["symbol_map_release_symbol_dates_idx"][0] == "symbol_map"
    definition = indexes["symbol_map_release_symbol_dates_idx"][1]
    assert all(
        column in definition for column in ("release_id", "symbol", "valid_from", "valid_to")
    )
    assert any(
        table == "daily_bars" and all(col in definition for col in ("load_id", "permno", "date"))
        for table, definition in indexes.values()
    )


def test_old_release_constraint_and_guard_migrate_idempotently(conn):
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE ref.releases DROP CONSTRAINT releases_status_check")
        cur.execute(
            "ALTER TABLE ref.releases ADD CONSTRAINT releases_status_check "
            "CHECK (status IN ('published', 'withdrawn'))"
        )
        cur.execute("""CREATE OR REPLACE FUNCTION ref.guard_release_insert() RETURNS trigger
                    LANGUAGE plpgsql AS $$ BEGIN
                    IF current_setting('ggt.release_build', true) IS DISTINCT FROM 'on' THEN
                        RAISE EXCEPTION 'release build setting required';
                    END IF; RETURN NEW; END $$""")
    conn.commit()
    apply_ddl(conn)
    apply_ddl(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ref.releases(status) VALUES ('building') RETURNING release_id")
        rid = cur.fetchone()[0]
        cur.execute("INSERT INTO ref.symbol_map(release_id,symbol) VALUES (%s,'TEST')", (rid,))
        cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=%s", (rid,))
        cur.execute("SAVEPOINT guard_probe")
        with pytest.raises(psycopg2.Error):
            cur.execute("INSERT INTO ref.symbol_map(release_id,symbol) VALUES (%s,'LATE')", (rid,))
        cur.execute("ROLLBACK TO SAVEPOINT guard_probe")
    conn.rollback()
