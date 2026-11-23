"""Load-log integration tests on a throwaway database."""

import getpass
from pathlib import Path

import psycopg2
import pytest

from ggTrader.data.kaggle_import.ddl import apply_ddl
from ggTrader.data.kaggle_import.loadlog import (
    AlreadyLoaded,
    LoadFailed,
    RetentionLimit,
    begin_load,
    can_delete_load,
    commit_chunk,
    complete_load,
)
from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def conn():
    with scratch_database() as url, psycopg2.connect(url) as connection:
        apply_ddl(connection)
        yield connection


def begin(conn, digest="a", version="1", **kwargs):
    return begin_load(
        conn,
        dataset="d",
        kaggle_slug="s",
        file_path=Path("x"),
        file_sha256=digest,
        parser_version=version,
        content_sha256="c",
        **kwargs,
    )


def test_resume_counts_and_completion(conn):
    load = begin(conn, expected_chunks=["1999", "2000"])

    def writer(cur):
        cur.execute("INSERT INTO ref.quarantine VALUES (%s, 'test', 'x', '{}', 'test')", (load,))
        return 1

    assert commit_chunk(conn, load, "1999", 1, writer)
    assert begin(conn) == load
    assert not commit_chunk(conn, load, "1999", 1, writer)
    with pytest.raises(LoadFailed, match="2000"):
        complete_load(conn, load, {})
    assert commit_chunk(conn, load, "2000", 0, lambda cur: 0, [("r", {"x": 1}, "bad")])
    complete_load(conn, load, {"ok": True})
    with pytest.raises(AlreadyLoaded):
        begin(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rows_parsed, rows_accepted, rows_quarantined FROM ref.loads WHERE load_id=%s",
            (load,),
        )
        assert cur.fetchone() == (2, 1, 1)


def test_rollback_and_counter_mismatch(conn):
    load = begin(conn)

    def bad(cur):
        cur.execute("INSERT INTO ref.quarantine VALUES (%s, 'x', 'x', '{}', 'x')", (load,))
        return 0

    with pytest.raises(LoadFailed):
        commit_chunk(conn, load, "x", 1, bad)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.quarantine")
        assert cur.fetchone()[0] == 0
        cur.execute("UPDATE ref.loads SET rows_parsed=1 WHERE load_id=%s", (load,))
    conn.commit()
    with pytest.raises(LoadFailed):
        complete_load(conn, load, {})


def test_retention_and_published_release(conn):
    first = begin(conn)
    complete_load(conn, first, {})
    second = begin(conn, version="2")
    complete_load(conn, second, {})
    with pytest.raises(RetentionLimit):
        begin(conn, digest="b")
    assert begin(conn, digest="b", allow_third=True) != second
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ref.releases(status) VALUES ('published') RETURNING release_id")
        release = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.release_loads(release_id, load_id) VALUES (%s, %s)", (release, first)
        )
    conn.commit()
    assert not can_delete_load(conn, first)
    assert can_delete_load(conn, second)


def test_quarantine_commits_with_chunk_and_retry_does_not_duplicate(conn):
    load = begin(conn)
    rejected = [("1", {"v": 1}, "bad"), ("2", {"v": 2}, "bad")]
    assert commit_chunk(conn, load, "ok", 0, lambda cur: 0, rejected)
    assert not commit_chunk(conn, load, "ok", 0, lambda cur: 0, rejected)
    # A database failure after quarantine insertion must roll back the entire chunk.
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE ref.load_chunks ADD CONSTRAINT reject_key CHECK (chunk_key <> 'fail')"
        )
    conn.commit()
    with pytest.raises(psycopg2.Error):
        commit_chunk(conn, load, "fail", 0, lambda cur: 0, rejected)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.quarantine WHERE load_id=%s", (load,))
        assert cur.fetchone()[0] == 2
        cur.execute(
            "SELECT rows_parsed, rows_accepted, rows_quarantined FROM ref.loads WHERE load_id=%s",
            (load,),
        )
        assert cur.fetchone() == (2, 0, 2)
        cur.execute(
            "SELECT count(*) FROM ref.load_chunks WHERE load_id=%s AND chunk_key='fail'", (load,)
        )
        assert cur.fetchone()[0] == 0


def test_new_parser_version_same_bytes_is_a_new_load(conn):
    first = begin(conn)
    complete_load(conn, first, {})
    with pytest.raises(AlreadyLoaded):
        begin(conn)
    assert begin(conn, version="2") != first


def test_missing_expected_chunk_blocks_complete(conn):
    load = begin(conn, expected_chunks=["missing"])
    with pytest.raises(LoadFailed, match="missing"):
        complete_load(conn, load, {})


def test_running_load_resumes_same_id(conn):
    load = begin(conn)
    assert begin(conn) == load
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.loads")
        assert cur.fetchone()[0] == 1


def test_third_load_refused_without_allow_third(conn):
    for version in ("1", "2"):
        complete_load(conn, begin(conn, version=version), {})
    with pytest.raises(RetentionLimit):
        begin(conn, digest="other")
    assert begin(conn, digest="other", allow_third=True)


def test_failed_load_is_terminal_and_retry_starts_new(conn):
    from ggTrader.data.kaggle_import.loadlog import fail_load

    load = begin(conn)
    fail_load(conn, load, "bad data")
    with conn.cursor() as cur:
        cur.execute("SELECT status, checks->>'failure' FROM ref.loads WHERE load_id=%s", (load,))
        assert cur.fetchone() == ("failed", "bad data")
    with pytest.raises(LoadFailed):
        complete_load(conn, load, {})
    with pytest.raises(LoadFailed):
        fail_load(conn, load, "again")
    with pytest.raises(psycopg2.Error), conn.cursor() as cur:
        cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (load,))
    conn.rollback()
    with pytest.raises(psycopg2.Error), conn.cursor() as cur:
        cur.execute("UPDATE ref.loads SET checks='{}' WHERE load_id=%s", (load,))
    conn.rollback()
    assert begin(conn) != load


def test_running_load_counts_toward_retention_and_can_resume(conn):
    complete_load(conn, begin(conn), {})
    running = begin(conn, version="2")
    assert begin(conn, version="2") == running
    with pytest.raises(RetentionLimit):
        begin(conn, version="3")


@pytest.mark.parametrize(
    "function,table",
    [
        ("kaggle_crsp.daily_of", "kaggle_crsp.daily"),
        ("kaggle_chars.mcgill_of", "kaggle_chars.mcgill"),
    ],
)
def test_complete_only_read_helpers(conn, function, table):
    from ggTrader.data.kaggle_import.loadlog import fail_load

    running = begin(conn)
    failed = begin(conn, version="2")
    fail_load(conn, failed, "bad")
    for load in (running, failed):
        with pytest.raises(psycopg2.Error, match="not complete"), conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {function}(%s)", (load,))
        conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {table} (load_id) VALUES (%s)", (running,)) if table.endswith(
            "mcgill"
        ) else cur.execute(
            f"INSERT INTO {table} (load_id, permno, date) VALUES (%s, 1, '2000-01-01')", (running,)
        )
    conn.commit()
    complete_load(conn, running, {})
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {function}(%s)", (running,))
        assert cur.fetchone()[0] == 1


def test_load_provenance(conn):
    load = begin(conn, slug_version="v1", download_date="2026-11-22")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT loader_git_sha, loaded_by, checks, current_user "
            "FROM ref.loads WHERE load_id=%s",
            (load,),
        )
        sha, loaded_by, checks, db_user = cur.fetchone()
    assert sha
    assert loaded_by == f"{getpass.getuser()}@{db_user}"
    assert checks["slug_version"] == "v1"
    assert checks["download_date"] == "2026-11-22"
