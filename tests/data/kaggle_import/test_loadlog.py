"""Load-log integration tests on a throwaway database."""

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
    quarantine_rows,
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
    quarantine_rows(conn, load, "2000", [("r", {"x": 1}, "bad")])
    assert commit_chunk(conn, load, "2000", 0, lambda cur: 0)
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
