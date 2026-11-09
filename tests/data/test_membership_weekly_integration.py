"""Integration tests of the weekly job against a throwaway database on a real server.

Never part of the fast, full or CI tiers (marked ``integration`` and ``db``); they skip unless
``GGT_TEST_DB_URL`` names a server, and work only inside their own ``ggt_test_<uuid>`` database
(``tests/data/scratch_db.py``). The network and Telegram are fakes; only the database is real.
Run::

    GGT_TEST_DB_URL=... GGT_TEST_DB_ALLOW_SHARED=1 \\
        scripts/run_tests.sh -m integration tests/data/test_membership_weekly_integration.py
"""

from __future__ import annotations

import gzip
import hashlib
from datetime import date, datetime, timezone

import psycopg2
import pytest
from psycopg2 import errors

from ggTrader.data.membership import DbMembershipSource, PgMembershipStore
from ggTrader.data.membership_gate import decide_gate
from ggTrader.data.membership_holdings import HoldingsFetchError, failed_holdings
from ggTrader.data.membership_weekly import PgWeeklyStore, SnapshotRow, run_weekly
from ggTrader.data.universe_db import LoadRequest, apply_ddl, load_vintage
from tests.data.membership_fakes import synthetic_history
from tests.data.membership_weekly_fakes import NOW, RecordingNotifier, World
from tests.data.test_membership_upstream import csv_bytes
from tests.data.xlsx_builder import holdings_xlsx

pytestmark = [pytest.mark.db, pytest.mark.integration]

V1_AT = datetime(2026, 6, 9, tzinfo=timezone.utc)
PROTECTED = ("loads", "membership", "snapshot_dates", "load_blobs", "load_evidence")


def _counts(conn, tables=PROTECTED) -> dict[str, int]:
    out = {}
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"SELECT count(*) FROM universe.{table}")
            out[table] = cur.fetchone()[0]
    conn.rollback()
    return out


def _load_synthetic(conn) -> list[str]:
    history = synthetic_history(3, n_dates=6)
    load_vintage(
        conn,
        LoadRequest(
            raw=csv_bytes(history),
            source="fixture-import",
            source_url="file:test",
            retrieved_at=V1_AT,
            commit=True,
            loader_git_sha="test",
            parent="none",
        ),
    )
    return list(history[max(history)])


def _weekly(db_url, world, tmp_path):
    notifier = RecordingNotifier()
    run = run_weekly(
        PgWeeklyStore(db_url),
        notifier,
        get=world,
        now=lambda: NOW,
        proposal_dir=tmp_path / "proposals",
    )
    return run, notifier


def test_the_ddl_adds_source_snapshots_to_an_older_schema_and_is_idempotent(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE universe.source_snapshots")  # a schema from before M6
    conn.commit()
    apply_ddl(conn)
    apply_ddl(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('universe.source_snapshots')")
        assert cur.fetchone()[0] == "universe.source_snapshots"
        cur.execute(
            "SELECT count(*) FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            "WHERE c.relname = 'source_snapshots' AND NOT t.tgisinternal"
        )
        assert cur.fetchone()[0] == 2  # frozen (update/delete) + truncate


def test_a_clean_run_records_everything_and_leaves_the_list_alone(conn, db_url, tmp_path) -> None:
    ours = _load_synthetic(conn)
    before = _counts(conn)
    store = PgMembershipStore(db_url)
    assert (
        decide_gate(DbMembershipSource("latest", store=store).vintage(), date(2026, 11, 9)).level
        == "refuse"
    )
    world = World(ours)
    run, notifier = _weekly(db_url, world, tmp_path)
    assert run.status == "ok" and run.check_id is not None and len(notifier.messages) == 1
    assert _counts(conn) == before  # not one row of the list changed
    with conn.cursor() as cur:
        cur.execute(
            "SELECT outcome, matched_load_id FROM universe.checks WHERE check_id = %s",
            (run.check_id,),
        )
        assert cur.fetchone() == ("matched", 1)
        cur.execute(
            "SELECT source, status, last_change, raw_sha256, raw_gzip "
            "FROM universe.source_snapshots WHERE check_id = %s ORDER BY source",
            (run.check_id,),
        )
        rows = {r[0]: r for r in cur.fetchall()}
        assert set(rows) == {
            "chinobing_changes",
            "chinobing_components",
            "spy_holdings",
            "wikipedia_current",
            "wikipedia_ledger",
        }
        ledger = rows["wikipedia_ledger"]
        assert ledger[1] == "ok" and ledger[2] == date(2026, 11, 3)
        assert gzip.decompress(bytes(ledger[4])) == world.ledger
        assert ledger[3] == hashlib.sha256(world.ledger).hexdigest()
        assert rows["spy_holdings"][4] is None
        cur.execute(
            "SELECT raw_gzip FROM universe.check_blobs WHERE check_id = %s", (run.check_id,)
        )
        assert gzip.decompress(bytes(cur.fetchone()[0])) == world.spy
    vintage = DbMembershipSource("latest", store=store).vintage()
    assert vintage.known_through == date(2026, 11, 5) and vintage.matched_at is not None
    assert decide_gate(vintage, date(2026, 11, 9)).level == "pass"
    assert PgWeeklyStore(db_url).known_through(1, "sp500") == date(2026, 11, 5)


def test_a_mismatch_stores_the_proposal_bytes_and_still_loads_nothing(
    conn, db_url, tmp_path
) -> None:
    ours = _load_synthetic(conn)
    before = _counts(conn)
    world = World(ours)
    world.spy = holdings_xlsx([t for t in ours if t != ours[7]] + ["ZZNEW"], "As of 06-Nov-2026")
    run, _ = _weekly(db_url, world, tmp_path)
    assert run.status == "refuse" and run.proposal_path is not None
    assert _counts(conn) == before
    with conn.cursor() as cur:
        cur.execute(
            "SELECT outcome, matched_load_id FROM universe.checks WHERE check_id = %s",
            (run.check_id,),
        )
        assert cur.fetchone() == ("mismatch", None)
        cur.execute(
            "SELECT raw_gzip, raw_sha256 FROM universe.source_snapshots "
            "WHERE check_id = %s AND source = 'proposal'",
            (run.check_id,),
        )
        blob, sha = cur.fetchone()
        assert gzip.decompress(bytes(blob)) == run.proposal_path.read_bytes()
        assert sha == hashlib.sha256(run.proposal_path.read_bytes()).hexdigest()
    vintage = DbMembershipSource("latest", store=PgMembershipStore(db_url)).vintage()
    assert vintage.matched_at is None  # a mismatch never advances the gate


def test_a_failed_snapshot_row_rolls_the_whole_record_back(conn, db_url) -> None:
    before = _counts(conn, ("checks", "check_blobs", "source_snapshots"))
    spy = failed_holdings(HoldingsFetchError("not_xlsx", "html", b"<html>blocked</html>"))
    bad = SnapshotRow("x", "u", "bogus", NOW, None, None, {})  # status violates the CHECK
    with pytest.raises(psycopg2.Error):
        PgWeeklyStore(db_url).record_run("sp500", spy, [bad])
    assert _counts(conn, ("checks", "check_blobs", "source_snapshots")) == before


def test_source_snapshots_are_append_only(conn, db_url, tmp_path) -> None:
    ours = _load_synthetic(conn)
    run, _ = _weekly(db_url, World(ours), tmp_path)
    assert run.check_id is not None
    for sql in (
        "UPDATE universe.source_snapshots SET status = 'failed'",
        "DELETE FROM universe.source_snapshots",
        "TRUNCATE universe.source_snapshots",
    ):
        with conn.cursor() as cur, pytest.raises(psycopg2.Error, match="append-only"):
            cur.execute(sql)
        conn.rollback()
    with conn.cursor() as cur, pytest.raises(errors.UniqueViolation):
        cur.execute(
            "INSERT INTO universe.source_snapshots (check_id, source, url, status, fetched_at) "
            "VALUES (%s, 'wikipedia_ledger', 'u', 'ok', now())",
            (run.check_id,),
        )
    conn.rollback()
