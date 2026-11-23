"""Kaggle schema tests on a throwaway database."""

import psycopg2
import pytest

from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


def test_schema():
    from ggTrader.data.kaggle_import.ddl import MCGILL_COLUMNS, apply_ddl

    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_ddl(conn)
        apply_ddl(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='kaggle_chars' AND table_name='mcgill' "
                "ORDER BY ordinal_position"
            )
            assert [r[0] for r in cur.fetchall()] == ["load_id", *MCGILL_COLUMNS]
            cur.execute(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_schema IN ('ref','market','kaggle_crsp','kaggle_chars') "
                "AND data_type='timestamp without time zone'"
            )
            assert cur.fetchone()[0] == 0
            cur.execute(
                "SELECT hypertable_schema, hypertable_name "
                "FROM timescaledb_information.hypertables "
                "WHERE hypertable_schema IN ('market','kaggle_crsp')"
            )
            assert set(cur.fetchall()) == {("market", "daily_bars"), ("kaggle_crsp", "daily")}
            cur.execute("INSERT INTO ref.loads(status) VALUES ('running') RETURNING load_id")
            load_id = cur.fetchone()[0]
            cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (load_id,))
        conn.commit()
        with pytest.raises(psycopg2.Error), conn.cursor() as cur:
            cur.execute("UPDATE ref.loads SET rows_parsed=1 WHERE load_id=%s", (load_id,))
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("UPDATE ref.loads SET status='withdrawn' WHERE load_id=%s", (load_id,))


def test_cli_apply_ddl():
    import argparse

    from ggTrader.cli.cmd_db import run_db

    with scratch_database() as url:
        run_db(argparse.Namespace(db_command="import", import_command="apply-ddl", db_url=url))
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('market.daily_bars')")
            assert cur.fetchone()[0] == "market.daily_bars"
