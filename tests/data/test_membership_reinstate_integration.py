"""Integration tests of ``reinstate`` and ``export`` against throwaway databases on a real server.

Never part of the fast, full or CI tiers (marked ``integration`` and ``db``; skipped unless
``GGT_TEST_DB_URL`` names a server). Every test works in its own ``ggt_test_<uuid>`` database and
nothing here runs against the production ``universe`` schema. Run (the main database is only the
door to the server, hence the explicit consent flag)::

    GGT_TEST_DB_URL=... GGT_TEST_DB_ALLOW_SHARED=1 \\
        scripts/run_tests.sh -m integration tests/data/test_membership_reinstate_integration.py
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse, urlunparse

import psycopg2
import pytest

from ggTrader.data.core.index_constituents import SP500_HISTORY_CSV
from ggTrader.data.membership_export import export_universe
from ggTrader.data.universe_algebra import content_sha256, parse_history_csv, raw_sha256
from ggTrader.data.universe_db import (
    LoadRefused,
    connect,
    diff_vintages,
    load_vintage,
    read_vintage,
    reinstate_vintage,
    verify_vintage,
)
from tests.data.scratch_db import server_url
from tests.data.test_universe_db_integration import V2_AT, _count, _request, _v1_bytes, _v2_bytes

pytestmark = [pytest.mark.db, pytest.mark.integration]

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def _load_two(conn) -> tuple[bytes, bytes]:
    """Vintage 1 (the committed file) then a forced vintage 2 (7 re-dated joins plus a tail)."""
    v1, v2 = _v1_bytes(), _v2_bytes()
    load_vintage(conn, _request(v1, parent="none"))
    load_vintage(conn, _request(v2, retrieved_at=V2_AT, force=True, force_reason="test"))
    return v1, v2


def _sizes(conn) -> dict[str, int]:
    tables = ("loads", "load_blobs", "snapshot_dates", "membership", "checks")
    return {t: _count(conn, t) for t in tables}


def _blob(conn, load_id: int) -> bytes:
    with conn.cursor() as cur:
        cur.execute("SELECT raw_gzip FROM universe.load_blobs WHERE load_id = %s", (load_id,))
        return bytes(cur.fetchone()[0])


def test_a_dry_run_reports_and_writes_nothing(conn) -> None:
    _load_two(conn)
    before = _sizes(conn)
    report = reinstate_vintage(conn, load_id=1, reason="undo the refresh", now=NOW)
    assert not report.committed and report.load_id is None and report.parent_load_id == 2
    assert report.refusal_reasons == []  # a rollback needs no --force
    assert report.rows_superseded > 0 and report.rows_added > 0
    assert _sizes(conn) == before


def test_reinstating_vintage_one_creates_vintage_three_with_the_same_content(conn) -> None:
    v1, v2 = _load_two(conn)
    report = reinstate_vintage(
        conn,
        load_id=1,
        reason="undo the refresh",
        now=NOW,
        parent=2,
        loader_git_sha="test",
        commit=True,
    )
    assert report.committed and report.load_id == 3 and report.parent_load_id == 2

    with conn.cursor() as cur:
        one, two, three = (read_vintage(cur, n) for n in (1, 2, 3))
        cur.execute(
            "SELECT source, source_url, parent_load_id, reinstates_load_id, forced, force_reason, "
            "retrieved_at, upstream_commit, note, rows_added, rows_superseded, "
            "raw_sha256, content_sha256 FROM universe.loads WHERE load_id = 3"
        )
        row = cur.fetchone()
    assert row[:6] == ("reinstate", "reinstate:load/1", 2, 1, False, None)
    assert row[6] == NOW and row[7] is None
    assert row[8] == "reinstates load 1: undo the refresh"
    assert (row[9], row[10]) == (report.rows_added, report.rows_superseded)
    assert row[11] == raw_sha256(v1) == one.load["raw_sha256"]
    assert row[12] == content_sha256(parse_history_csv(v1)) == one.load["content_sha256"]

    # Content equals vintage 1 on every snapshot date, and the byte store is a verbatim copy.
    assert three.history() == one.history() == parse_history_csv(v1)
    assert three.dates == one.dates
    assert _blob(conn, 3) == _blob(conn, 1)
    assert gzip.decompress(_blob(conn, 3)) == v1
    assert diff_vintages(one, three)["same_content"] is True
    assert diff_vintages(two, three)["same_content"] is False

    # Nothing before it moved: vintage 1 and 2 still read exactly as they did.
    assert one.history() == parse_history_csv(v1) and two.history() == parse_history_csv(v2)
    assert verify_vintage(conn, 1, SP500_HISTORY_CSV).ok
    assert verify_vintage(conn, 3, SP500_HISTORY_CSV).ok


def test_the_live_rows_of_the_new_vintage_are_exactly_the_old_content(conn) -> None:
    v1, _ = _load_two(conn)
    reinstate_vintage(conn, load_id=1, reason="undo", now=NOW, commit=True)
    with conn.cursor() as cur:
        # Q1 at the latest vintage returns vintage 1's membership on a date vintage 1 knows.
        cur.execute(
            "SELECT symbol_raw FROM universe.membership WHERE index_name = 'sp500' "
            "AND superseded_by_load_id IS NULL AND valid_from <= %s::date "
            "AND (valid_to IS NULL OR valid_to > %s::date) ORDER BY symbol_raw",
            ("2026-06-02", "2026-06-02"),
        )
        live = tuple(r[0] for r in cur.fetchall())
        # The exclusion constraint would have refused any overlap among live rows.
        cur.execute(
            "SELECT count(*) FROM universe.membership WHERE load_id = 3 AND superseded_by_load_id "
            "IS NOT NULL"
        )
        assert cur.fetchone()[0] == 0
    history = parse_history_csv(v1)
    assert live == history[max(d for d in history if d <= datetime(2026, 6, 2).date())]


def test_every_earlier_row_keeps_its_content_and_only_gains_a_superseded_marker(conn) -> None:
    _load_two(conn)

    def rows():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT interval_id, index_name, symbol_raw, symbol_yf, valid_from, valid_to, "
                "load_id, superseded_by_load_id FROM universe.membership ORDER BY interval_id"
            )
            return cur.fetchall()

    before = rows()
    reinstate_vintage(conn, load_id=1, reason="undo", now=NOW, commit=True)
    after = rows()
    assert after[: len(before)] == [
        b[:7] + (a[7],) for b, a in zip(before, after[: len(before)], strict=True)
    ]
    for old, new in zip(before, after, strict=False):
        if old[7] is None and new[7] is not None:
            assert new[7] == 3  # only the new load may supersede, and only once
        else:
            assert old[7] == new[7]


def test_reinstating_the_vintage_before_last_again_and_a_later_vintage(conn) -> None:
    v1, v2 = _load_two(conn)
    reinstate_vintage(conn, load_id=1, reason="undo", now=NOW, commit=True)  # -> 3, content of 1
    redo = reinstate_vintage(
        conn, load_id=2, reason="redo the refresh", now=NOW, commit=True
    )  # -> 4
    assert redo.load_id == 4 and redo.parent_load_id == 3
    with conn.cursor() as cur:
        four, three = read_vintage(cur, 4), read_vintage(cur, 3)
    assert four.history() == parse_history_csv(v2)
    assert three.history() == parse_history_csv(v1)  # vintage 3 is still readable as of itself
    assert four.load["reinstates_load_id"] == 2


def test_refusals_leave_the_database_untouched(conn) -> None:
    _load_two(conn)
    before = _sizes(conn)
    cases = [
        (dict(load_id=2), "already has the content"),  # the latest is vintage 2
        (dict(load_id=99), "no load 99"),
        (dict(load_id=1, parent=1), "pinned to 1"),
        (dict(load_id=1, reason="  "), "--reason is required"),
    ]
    for kwargs, message in cases:
        kwargs.setdefault("reason", "x")
        with pytest.raises(LoadRefused, match=message):
            reinstate_vintage(conn, now=NOW, commit=True, **kwargs)
    assert _sizes(conn) == before


def test_nothing_loaded_means_nothing_to_reinstate(conn) -> None:
    with pytest.raises(LoadRefused, match="nothing is loaded"):
        reinstate_vintage(conn, load_id=1, reason="x", now=NOW, commit=True)


def test_a_failure_during_the_write_rolls_the_whole_reinstate_back(conn, monkeypatch) -> None:
    from ggTrader.data import universe_db

    _load_two(conn)
    before = _sizes(conn)

    def broken(*_a, **_k) -> None:
        raise LoadRefused(["simulated read-back failure"])

    monkeypatch.setattr(universe_db, "_verify_readback", broken)
    with pytest.raises(LoadRefused, match="simulated"):
        reinstate_vintage(conn, load_id=1, reason="x", now=NOW, commit=True)
    assert _sizes(conn) == before
    with conn.cursor() as cur:  # no superseded marker survived either
        cur.execute("SELECT count(*) FROM universe.membership WHERE superseded_by_load_id = 3")
        assert cur.fetchone()[0] == 0


# --------------------------------------------------------------------------- export


@pytest.mark.skipif(shutil.which("pg_dump") is None, reason="pg_dump is not installed")
def test_export_dumps_the_schema_and_it_restores_into_a_bare_database(db_url, tmp_path) -> None:
    conn = connect(db_url)
    try:
        _load_two(conn)
        reinstate_vintage(conn, load_id=1, reason="undo", now=NOW, commit=True)
        expected = _sizes(conn) | {"check_blobs": _count(conn, "check_blobs")}
    finally:
        conn.close()

    result = export_universe(db_url, out_dir=tmp_path / "x")
    assert result.path.name.startswith("universe_") and result.path.suffix == ".dump"
    assert result.size_bytes == result.path.stat().st_size > 0
    assert len(result.sha256) == 64

    server = server_url()
    name = f"ggt_test_restore_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(server)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{name}"')
    try:
        target = urlunparse(urlparse(server)._replace(path=f"/{name}"))
        bare = psycopg2.connect(target)
        bare.autocommit = True
        with bare.cursor() as cur:
            cur.execute("CREATE EXTENSION btree_gist")  # the dump does not carry the extension
        bare.close()
        parts = urlparse(target)
        env = {**os.environ}
        if parts.password:
            env["PGPASSWORD"] = unquote(parts.password)
        restore = ["pg_restore", f"--dbname={name}", "--no-owner", str(result.path)]
        if parts.hostname:
            restore.insert(1, f"--host={parts.hostname}")
        if parts.port:
            restore.insert(2, f"--port={parts.port}")
        if parts.username:
            restore.insert(3, f"--username={unquote(parts.username)}")
        done = subprocess.run(restore, env=env, capture_output=True, text=True, check=False)
        assert done.returncode == 0, "pg_restore failed"

        restored = connect(target)
        try:
            assert _sizes(restored) | {"check_blobs": _count(restored, "check_blobs")} == expected
            with restored.cursor() as cur:
                one, three = read_vintage(cur, 1), read_vintage(cur, 3)
                assert one.history() == three.history()
                # The append-only guard travelled with the schema.
                with pytest.raises(psycopg2.Error, match="append-only"):
                    cur.execute("DELETE FROM universe.loads WHERE load_id = 1")
            restored.rollback()
        finally:
            restored.close()
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()
