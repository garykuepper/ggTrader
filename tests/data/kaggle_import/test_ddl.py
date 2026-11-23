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
        if table == "splice_links":
            cur.execute("INSERT INTO ref.splice_links(release_id, permno) VALUES (1, 1)")
        else:
            cur.execute(f"INSERT INTO ref.{table} DEFAULT VALUES")
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


def test_cli_apply_ddl_parses_and_runs():
    parser = argparse.ArgumentParser(prog="ggt")
    register_db_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["db", "import", "apply-ddl", "--db-url", "x"])
    assert (args.db_command, args.import_command, args.db_url) == ("import", "apply-ddl", "x")
    with scratch_database() as url:
        args.db_url = url
        run_db(args)
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('market.daily_bars')")
            assert cur.fetchone()[0] == "market.daily_bars"


def test_release_withdrawal_and_links_are_immutable(conn):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ref.releases(status) VALUES ('published') RETURNING release_id")
        release_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.release_loads(release_id, load_id) VALUES (%s, 1)", (release_id,)
        )
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
