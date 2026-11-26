"""Consolidation schema tests on a throwaway database."""

from collections.abc import Iterator
from pathlib import Path

import psycopg2
import pytest
from psycopg2.extensions import connection

from ggTrader.data.kaggle_import.ddl import apply_ddl as apply_kaggle_ddl
from ggTrader.data.kaggle_import.loadlog import (
    LoadFailed,
    begin_load,
    complete_load,
    record_files,
)
from ggTrader.data.sec_import.ddl import apply_ddl
from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def kaggle_conn() -> Iterator[connection]:
    """Apply the prerequisite schema before the consolidation migration."""
    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_kaggle_ddl(conn)
        yield conn


@pytest.fixture
def conn(kaggle_conn: connection) -> connection:
    """Return a connection with the consolidation schema applied."""
    apply_ddl(kaggle_conn)
    return kaggle_conn


def _begin(conn: connection, dataset: str = "sec-index") -> int:
    return begin_load(
        conn,
        dataset=dataset,
        kaggle_slug="fixture",
        file_path=Path("fixture"),
        file_sha256=dataset,
        parser_version="1",
        content_sha256=dataset,
    )


def test_apply_ddl_is_idempotent(conn: connection) -> None:
    """Reapplying the schema leaves the bookkeeping rows in place."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ref.integrity_checks(mode, ok, report) VALUES ('ordinary', true, '{}')"
        )
    conn.commit()

    apply_ddl(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT mode, ok, report FROM ref.integrity_checks")
        assert cur.fetchall() == [("ordinary", True, {})]


@pytest.mark.parametrize(
    ("table", "columns", "values"),
    [
        ("ref.load_files", "load_id, path", "%s, 'fixture'"),
        ("ref.load_coverage", "load_id, coverage, note", "%s, 'not_a_stock_source', 'fixture'"),
        (
            "ref.integrity_checks",
            "mode, ok, report",
            "'ordinary', true, jsonb_build_object('load_id', %s)",
        ),
        ("sec.fetched_files", "load_id, url_hash", "%s, 'hash'"),
        ("sec.filing_headers", "load_id, accession, cik", "%s, 'fixture', 1"),
        (
            "sec.death_labels",
            "load_id, study, method_version, permno, last_date",
            "%s, 'fixture', 'v1', 1, '2000-01-01'",
        ),
    ],
)
def test_bookkeeping_tables_reject_update_delete_truncate(
    conn: connection, table: str, columns: str, values: str
) -> None:
    """Reject writes that would overwrite or remove an existing row."""
    load_id = _begin(conn)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {table} ({columns}) VALUES ({values})", (load_id,))
    conn.commit()

    first_column = columns.split(",")[0]
    for statement in (
        f"UPDATE {table} SET {first_column}={first_column}",
        f"DELETE FROM {table}",
        f"TRUNCATE {table}",
    ):
        with pytest.raises(psycopg2.Error, match="append-only table"), conn.cursor() as cur:
            cur.execute(statement)
        conn.rollback()


def test_index_entries_reject_truncate(conn: connection) -> None:
    """Protect the large index table from truncation."""
    with pytest.raises(psycopg2.Error, match="append-only table"), conn.cursor() as cur:
        cur.execute("TRUNCATE sec.index_entries")
    conn.rollback()


def test_complete_without_coverage_fails_for_new_dataset(conn: connection) -> None:
    """The database blocks an unlabeled non-legacy load from completing."""
    load_id = _begin(conn)
    with pytest.raises(psycopg2.Error, match="coverage label required"):
        complete_load(conn, load_id, {})
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM ref.loads WHERE load_id=%s", (load_id,))
        assert cur.fetchone() == ("running",)


def test_complete_with_coverage_writes_row_in_same_transaction(conn: connection) -> None:
    """If completing the load fails, its coverage row rolls back with it."""
    load_id = _begin(conn)
    coverage = ("not_a_stock_source", "filing index")
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE ref.loads ADD CONSTRAINT reject_complete CHECK (status <> 'complete')"
        )
    conn.commit()

    with pytest.raises(psycopg2.errors.CheckViolation, match="reject_complete"):
        complete_load(conn, load_id, {}, coverage=coverage)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.load_coverage WHERE load_id=%s", (load_id,))
        assert cur.fetchone() == (0,)
        cur.execute("ALTER TABLE ref.loads DROP CONSTRAINT reject_complete")
    conn.commit()

    complete_load(conn, load_id, {}, coverage=coverage)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, coverage, note FROM ref.loads JOIN ref.load_coverage USING (load_id) "
            "WHERE load_id=%s",
            (load_id,),
        )
        assert cur.fetchone() == ("complete", *coverage)


@pytest.mark.parametrize(
    "dataset", ["crsp-daily", "crsp-daily-pilot", "mcgill-fiam", "crsp-derived-bars"]
)
def test_no_dataset_completes_without_coverage(conn: connection, dataset: str) -> None:
    """The older market-data datasets are held to the same label rule as every other load."""
    load_id = _begin(conn, dataset)
    with pytest.raises(psycopg2.errors.RaiseException, match="coverage label required"):
        complete_load(conn, load_id, {})
    conn.rollback()
    complete_load(conn, load_id, {}, coverage=("includes_dead_stocks", "fixture"))
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM ref.loads WHERE load_id=%s", (load_id,))
        assert cur.fetchone() == ("complete",)


def test_coverage_rejects_unknown_value_and_second_row(conn: connection) -> None:
    """A load can receive one coverage label drawn from the fixed list."""
    load_id = _begin(conn)
    insert = "INSERT INTO ref.load_coverage(load_id, coverage, note) VALUES (%s, %s, 'fixture')"
    with pytest.raises(psycopg2.errors.CheckViolation), conn.cursor() as cur:
        cur.execute(insert, (load_id, "unknown"))
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(insert, (load_id, "not_a_stock_source"))
    conn.commit()
    with pytest.raises(psycopg2.errors.UniqueViolation), conn.cursor() as cur:
        cur.execute(insert, (load_id, "survivors_only"))
    conn.rollback()


def test_record_files_is_idempotent(conn: connection) -> None:
    """Replaying a file list preserves exactly one record per path."""
    load_id = _begin(conn)
    files = [("a", "a" * 64, 10), ("b", "b" * 64, 20)]
    record_files(conn, load_id, files)
    record_files(conn, load_id, files)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT path, sha256, bytes FROM ref.load_files WHERE load_id=%s ORDER BY path",
            (load_id,),
        )
        assert cur.fetchall() == files


def test_record_files_rejects_different_contents_for_same_path(conn: connection) -> None:
    """A changed hash or size for an already recorded path is an error and records nothing new."""
    load_id = _begin(conn)
    record_files(conn, load_id, [("a", "a" * 64, 10)])
    for changed in (("a", "b" * 64, 10), ("a", "a" * 64, 11)):
        with pytest.raises(LoadFailed, match="different contents"):
            record_files(conn, load_id, [("z", "z" * 64, 1), changed])
    with pytest.raises(LoadFailed, match="Conflicting"):
        record_files(conn, load_id, [("q", "1" * 64, 1), ("q", "2" * 64, 1)])
    with conn.cursor() as cur:
        cur.execute("SELECT path FROM ref.load_files WHERE load_id=%s", (load_id,))
        assert cur.fetchall() == [("a",)]


def test_record_files_requires_a_running_load(conn: connection) -> None:
    """Files cannot be attached to a completed or unknown load."""
    load_id = _begin(conn)
    complete_load(conn, load_id, {}, coverage=("not_a_stock_source", "test"))
    with pytest.raises(LoadFailed, match="not running"):
        record_files(conn, load_id, [("a", "a" * 64, 1)])
    with pytest.raises(LoadFailed, match="not running"):
        record_files(conn, 999999, [("a", "a" * 64, 1)])


def test_triggers_are_always_enabled_and_indexes_exist(conn: connection) -> None:
    """Replica-mode loads cannot bypass the protections; the filing index keeps its two indexes."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.oid::regclass::text, t.tgname, t.tgenabled FROM pg_trigger t "
            "JOIN pg_class c ON c.oid = t.tgrelid "
            "WHERE t.tgname IN ('frozen', 'no_truncate', 'require_coverage') AND NOT t.tgisinternal"
        )
        found = {(table, name): enabled for table, name, enabled in cur.fetchall()}
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname='sec' AND tablename='index_entries'"
        )
        indexes = {row[0] for row in cur.fetchall()}
    price_tables = ["kaggle_crsp.daily", "market.daily_bars", "kaggle_chars.mcgill"]
    # TimescaleDB refuses "always enabled" on compressed tables, so these three carry plain ones.
    assert {v for (table, _), v in found.items() if table not in price_tables} == {"A"}
    for table in price_tables:
        assert found[(table, "frozen")] == "O"
        assert found[(table, "no_truncate")] == "O"
    protected = [
        "ref.load_files",
        "ref.load_coverage",
        "ref.integrity_checks",
        "sec.fetched_files",
        "sec.index_entries",
        "sec.filing_headers",
        "sec.death_labels",
        *price_tables,
    ]
    for table in protected:
        assert (table, "frozen") in found
        assert (table, "no_truncate") in found
    assert ("ref.loads", "require_coverage") in found
    assert {
        "index_entries_load_cik_date_idx",
        "index_entries_load_form_date_idx",
        "index_entries_accession_idx",
    } <= indexes


def test_backfill_skips_loads_that_are_not_complete(kaggle_conn: connection) -> None:
    """A running or failed legacy load gets no coverage row, so completing it later still works."""
    running = _begin(kaggle_conn, "crsp-daily")
    kaggle_conn.commit()
    apply_ddl(kaggle_conn)
    with kaggle_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.load_coverage WHERE load_id=%s", (running,))
        assert cur.fetchone()[0] == 0
    complete_load(kaggle_conn, running, {}, coverage=("includes_dead_stocks", "late label"))


def test_backfill_adds_rows_for_existing_loads_and_leaves_loads_unchanged(
    kaggle_conn: connection,
) -> None:
    """Backfill the three legacy labels without altering their original records."""
    expected = [
        ("crsp-daily", "includes_dead_stocks", "CRSP; no delisting returns"),
        ("crsp-derived-bars", "includes_dead_stocks", "derived from the CRSP load"),
        ("mcgill-fiam", "large_stocks_only", "McGill-FIAM, 2000 to 2023, large stocks"),
    ]
    for dataset, _, _ in expected:
        complete_load(kaggle_conn, _begin(kaggle_conn, dataset), {})
    _begin(kaggle_conn)
    with kaggle_conn.cursor() as cur:
        cur.execute("SELECT to_jsonb(l) FROM ref.loads l ORDER BY load_id")
        before = cur.fetchall()
    kaggle_conn.commit()

    apply_ddl(kaggle_conn)
    apply_ddl(kaggle_conn)

    with kaggle_conn.cursor() as cur:
        cur.execute(
            "SELECT dataset, coverage, note FROM ref.load_coverage JOIN ref.loads USING (load_id) "
            "ORDER BY dataset"
        )
        assert cur.fetchall() == expected
        cur.execute("SELECT to_jsonb(l) FROM ref.loads l ORDER BY load_id")
        assert cur.fetchall() == before


def test_fail_load_command_abandons_only_a_running_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck load can be closed from the command line, and a finished one cannot."""
    import argparse

    from ggTrader.cli.cmd_db import register_db_parser, run_db

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    with scratch_database() as url, psycopg2.connect(url) as target:
        apply_kaggle_ddl(target)
        apply_ddl(target)
        running = _begin(target)
        finished = _begin(target, "sec-cache")
        complete_load(target, finished, {}, coverage=("not_a_stock_source", "test"))
        monkeypatch.setenv("FAIL_LOAD_URL", url)
        base = ["db", "import", "fail-load", "--db-url-env", "FAIL_LOAD_URL", "--reason", "test"]
        run_db(parser.parse_args([*base, "--load-id", str(running)]))  # preview only
        with target.cursor() as cur:
            cur.execute("SELECT status FROM ref.loads WHERE load_id=%s", (running,))
            assert cur.fetchone() == ("running",)
        target.rollback()
        run_db(parser.parse_args([*base, "--load-id", str(running), "--yes"]))
        with target.cursor() as cur:
            cur.execute("SELECT status FROM ref.loads WHERE load_id=%s", (running,))
            assert cur.fetchone() == ("failed",)
        with pytest.raises(SystemExit, match="not running"):
            run_db(parser.parse_args([*base, "--load-id", str(finished), "--yes"]))


@pytest.mark.parametrize(
    "reader",
    ["fetched_files_of", "index_entries_of", "filing_headers_of", "death_labels_of"],
)
def test_sec_readers_refuse_loads_that_are_not_complete(conn: connection, reader: str) -> None:
    running = _begin(conn)
    done = _begin(conn, "sec-cache")
    complete_load(conn, done, {}, coverage=("not_a_stock_source", "test"))
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM sec.{reader}(%s)", (done,))
        assert cur.fetchone() == (0,)
    conn.rollback()
    with pytest.raises(psycopg2.errors.RaiseException, match="not complete"), conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM sec.{reader}(%s)", (running,))
    conn.rollback()


def test_filing_index_rows_cannot_be_changed_and_the_view_shows_only_complete_loads(
    conn: connection,
) -> None:
    """Index rows are append-only; the complete-loads view hides rows of unfinished loads."""
    running = _begin(conn, "sec-headers")
    done = _begin(conn, "sec-index")
    with conn.cursor() as cur:
        for load_id in (running, done):
            cur.execute(
                "INSERT INTO sec.index_entries (load_id, quarter, cik, company_name, form_type,"
                " date_filed, filename) VALUES (%s, '2005Q2', 1, 'A', '8-K', '2005-04-01',"
                " 'edgar/data/1/0000000001-05-000001.txt')",
                (load_id,),
            )
    complete_load(conn, done, {}, coverage=("not_a_stock_source", "test"))
    with conn.cursor() as cur:
        cur.execute("SELECT load_id FROM sec.index_entries_complete")
        assert cur.fetchall() == [(done,)]
    conn.commit()
    with pytest.raises(psycopg2.errors.RaiseException), conn.cursor() as cur:
        cur.execute("UPDATE sec.index_entries SET quarter='2005Q3'")
    conn.rollback()
    with pytest.raises(psycopg2.errors.RaiseException), conn.cursor() as cur:
        cur.execute("DELETE FROM sec.index_entries")
    conn.rollback()


def _stage_price_rows(conn: connection, *, with_yahoo: bool) -> tuple[int, int]:
    """Insert one CRSP row and one derived row for a complete load and for a running load."""
    done = _begin(conn, "crsp-daily")
    running = _begin(conn, "crsp-derived-bars")
    with conn.cursor() as cur:
        for load_id in (done, running):
            cur.execute(
                "INSERT INTO kaggle_crsp.daily (load_id, permno, date, prc, ret)"
                " VALUES (%s, 14593, '2000-01-03', -111.9375, 0.01)",
                (load_id,),
            )
            cur.execute(
                "INSERT INTO market.daily_bars (load_id, permno, spell, date, close, abs_prc,"
                " midpoint_flag) VALUES (%s, 14593, 1, '2000-01-03', 0.843, 111.9375, true)",
                (load_id,),
            )
    complete_load(conn, done, {}, coverage=("includes_dead_stocks", "fixture"))
    conn.commit()
    return done, running


def test_price_sources_view_labels_sources_and_hides_unfinished_loads(conn: connection) -> None:
    """Each source keeps its own rows; raw and adjusted prices stay in separate columns."""
    done, _running = _stage_price_rows(conn, with_yahoo=False)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, load_id, coverage, raw_close, adj_close, is_midpoint, adj_type"
            " FROM market.prices_daily_sources WHERE permno = 14593 ORDER BY source"
        )
        rows = cur.fetchall()
    assert rows == [
        ("crsp_daily", done, "includes_dead_stocks", 111.9375, None, True, None),
        (
            "crsp_derived_bars",
            done,
            "includes_dead_stocks",
            111.9375,
            0.843,
            True,
            "crsp_spell_chained_total_return",
        ),
    ]


def test_price_sources_view_adds_yahoo_when_the_shared_table_exists(
    kaggle_conn: connection,
) -> None:
    """The Yahoo branch appears only where public.ohlcv exists, with its vintage marked unknown."""
    with kaggle_conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE public.ohlcv (timestamp timestamp, symbol varchar(20),"
            " interval varchar(5), open double precision, high double precision,"
            " low double precision, close double precision, volume double precision,"
            " trades integer, venue varchar(20))"
        )
        cur.execute(
            "INSERT INTO public.ohlcv (timestamp, symbol, interval, open, close, volume, venue)"
            " VALUES ('2000-01-03', 'AAPL', '1d', 0.78, 0.84, 5.0, 'yfinance'),"
            " ('2000-01-03', 'BTC', '1m', 1, 1, 1, 'kraken_spot')"
        )
    kaggle_conn.commit()
    apply_ddl(kaggle_conn)
    with kaggle_conn.cursor() as cur:
        cur.execute(
            "SELECT source, symbol, load_id, coverage, adj_open, adj_close, raw_close, adj_type"
            " FROM market.prices_daily_sources WHERE source = 'yahoo'"
        )
        assert cur.fetchall() == [
            ("yahoo", "AAPL", None, None, 0.78, 0.84, None, "yahoo_adjusted_vintage_unrecorded")
        ]
    apply_ddl(kaggle_conn)  # a rerun replaces the view cleanly


@pytest.mark.parametrize(
    ("table", "change"),
    [
        ("kaggle_crsp.daily", "UPDATE kaggle_crsp.daily SET prc = 1"),
        ("kaggle_crsp.daily", "DELETE FROM kaggle_crsp.daily"),
        ("market.daily_bars", "UPDATE market.daily_bars SET close = 1"),
        ("market.daily_bars", "DELETE FROM market.daily_bars"),
        ("kaggle_crsp.daily", "TRUNCATE kaggle_crsp.daily"),
        ("market.daily_bars", "TRUNCATE market.daily_bars"),
    ],
)
def test_price_tables_refuse_edits_deletes_and_truncate(
    conn: connection, table: str, change: str
) -> None:
    """The big price tables are append-only like the rest of the new tables."""
    _stage_price_rows(conn, with_yahoo=False)
    with pytest.raises(psycopg2.errors.RaiseException, match="append-only"), conn.cursor() as cur:
        cur.execute(change)
    conn.rollback()
