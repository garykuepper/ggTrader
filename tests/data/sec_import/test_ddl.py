"""Consolidation schema tests on a throwaway database."""

from collections.abc import Iterator
from pathlib import Path

import psycopg2
import pytest
from psycopg2.extensions import connection

from ggTrader.data.kaggle_import.ddl import apply_ddl as apply_kaggle_ddl
from ggTrader.data.kaggle_import.loadlog import begin_load, complete_load, record_files
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


@pytest.mark.parametrize("dataset", ["crsp-daily", "mcgill-fiam", "crsp-derived-bars"])
def test_legacy_datasets_complete_without_coverage(conn: connection, dataset: str) -> None:
    """Existing importers can still complete without a label argument."""
    load_id = _begin(conn, dataset)
    complete_load(conn, load_id, {})
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
