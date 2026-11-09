"""Integration tests of the membership database against a throwaway database on a real server.

Never part of the fast, full or CI tiers: the module is marked ``integration`` (and ``db``), and it
skips itself unless ``GGT_TEST_DB_URL`` names a server. Each test gets its own database
``ggt_test_<uuid>`` created from ``template1``, runs the real ``sql/universe_schema.sql`` in it,
and drops only that database afterwards (plus the two NOLOGIN roles, only if this test run was the
one that created them). Nothing here ever touches the database the URL points at, except to issue
CREATE DATABASE / DROP DATABASE for its own scratch name.

Run (the main database is only the door to the server, hence the explicit consent flag)::

    GGT_TEST_DB_URL=... GGT_TEST_DB_ALLOW_SHARED=1 \\
        scripts/run_tests.sh -m integration tests/data/test_universe_db_integration.py
"""

from __future__ import annotations

import gzip
import threading
from datetime import date, datetime, timedelta, timezone

import psycopg2
import pytest
from psycopg2 import errors

from ggTrader.data import universe_db
from ggTrader.data.core.index_constituents import SP500_HISTORY_CSV
from ggTrader.data.universe_algebra import (
    HistoryValidationError,
    content_sha256,
    diff_intervals,
    parse_history_csv,
    raw_sha256,
)
from ggTrader.data.universe_db import (
    LoadRefused,
    LoadRequest,
    apply_ddl,
    connect,
    load_vintage,
    read_vintage,
    verify_vintage,
)
from tests.data.scratch_db import ROLES

pytestmark = [pytest.mark.db, pytest.mark.integration]

V1_AT = datetime(2026, 6, 9, tzinfo=timezone.utc)
V2_AT = datetime(2026, 11, 7, 18, 26, 50, tzinfo=timezone.utc)
D = date.fromisoformat


def _v1_bytes() -> bytes:
    if not SP500_HISTORY_CSV.exists():
        pytest.skip("vintage-1 file not present")
    return gzip.decompress(SP500_HISTORY_CSV.read_bytes())


def _request(raw: bytes, **kwargs) -> LoadRequest:
    defaults = dict(
        raw=raw,
        source="fixture-import",
        source_url="file:test",
        retrieved_at=V1_AT,
        commit=True,
        loader_git_sha="test",
    )
    return LoadRequest(**{**defaults, **kwargs})


def _csv_bytes(history: dict[date, tuple[str, ...]]) -> bytes:
    lines = ["date,tickers"] + [
        f'{d.isoformat()},"{",".join(t)}"' for d, t in sorted(history.items())
    ]
    return ("\n".join(lines) + "\n").encode()


def _v2_bytes(*, rewrites: int = 7, tail: bool = True) -> bytes:
    """Vintage 1 with `rewrites` joins re-dated one snapshot later, plus a new tail snapshot."""
    history = dict(parse_history_csv(_v1_bytes()))
    dates = sorted(history)
    picked = 0
    for index in range(len(dates) - 2, 0, -1):
        newcomers = set(history[dates[index]]) - set(history[dates[index - 1]])
        stays = set(history[dates[index]]) & set(history[dates[index + 1]])
        for symbol in sorted(newcomers & stays):
            history[dates[index]] = tuple(t for t in history[dates[index]] if t != symbol)
            picked += 1
            if picked >= rewrites:
                break
        if picked >= rewrites:
            break
    assert picked == rewrites
    if tail:
        last = history[dates[-1]]
        history[D("2026-08-18")] = tuple(
            sorted((set(last) - {last[0], last[1]}) | {"NEWAAA", "NEWBBB"})
        )
    return _csv_bytes(history)


def _count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM universe.{table}")
        return cur.fetchone()[0]


# ---- schema ----------------------------------------------------------------------------


def test_ddl_is_idempotent_and_stays_inside_the_universe_schema(db_url: str) -> None:
    conn = connect(db_url)
    try:
        apply_ddl(conn)  # second run on top of the fixture's first
        apply_ddl(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relkind IN ('r','v','m','S','i')"
            )
            assert cur.fetchall() == []  # nothing leaked into public
            # template1 may carry other extensions' functions in public; ours must not appear.
            cur.execute(
                "SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                "WHERE n.nspname = 'public' AND (p.proname LIKE 'gbt\\_%' "
                "OR p.proname IN ('membership_guard', 'frozen_guard', 'snapshot'))"
            )
            assert cur.fetchall() == []
            cur.execute(
                "SELECT n.nspname FROM pg_extension e "
                "JOIN pg_namespace n ON n.oid = e.extnamespace "
                "WHERE e.extname = 'btree_gist'"
            )
            assert cur.fetchone()[0] == "universe"
            cur.execute(
                "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'universe' AND c.relkind = 'r' ORDER BY 1"
            )
            assert [r[0] for r in cur.fetchall()] == [
                "check_blobs",
                "checks",
                "load_blobs",
                "load_evidence",
                "loads",
                "membership",
                "snapshot_dates",
                "source_snapshots",
                "symbol_changes",
            ]
            cur.execute(
                "SELECT count(*) FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'universe' AND NOT t.tgisinternal"
            )
            assert cur.fetchone()[0] == 18  # (update/delete + truncate) x 9 tables
            cur.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'membership_no_live_overlap'"
            )
            assert "EXCLUDE USING gist" in cur.fetchone()[0]
    finally:
        conn.close()


def test_roles_hold_only_the_column_privilege_the_loader_needs(conn) -> None:
    with conn.cursor() as cur:

        def has(role: str, column: str | None, priv: str) -> bool:
            if column:
                cur.execute(
                    "SELECT has_column_privilege(%s, 'universe.membership', %s, %s)",
                    (role, column, priv),
                )
            else:
                cur.execute(
                    "SELECT has_table_privilege(%s, 'universe.membership', %s)", (role, priv)
                )
            return cur.fetchone()[0]

        assert has("universe_loader", "superseded_by_load_id", "UPDATE")
        assert not has("universe_loader", "symbol_raw", "UPDATE")
        assert not has("universe_loader", None, "DELETE")
        assert has("universe_loader", None, "INSERT")
        assert has("universe_reader", None, "SELECT")
        assert not has("universe_reader", None, "INSERT")
        cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = ANY(%s)", (list(ROLES),))
        assert [r[0] for r in cur.fetchall()] == [False, False]


# ---- the load --------------------------------------------------------------------------


def test_dry_run_reports_and_writes_nothing(conn) -> None:
    report = load_vintage(conn, _request(_v1_bytes(), commit=False))
    assert not report.committed and report.load_id is None and report.rows_added == 1255
    assert _count(conn, "loads") == _count(conn, "membership") == _count(conn, "checks") == 0


def test_vintage_one_loads_and_verifies_on_every_snapshot_date(conn) -> None:
    raw = _v1_bytes()
    report = load_vintage(conn, _request(raw, parent="none", note="test"))
    assert report.committed and report.load_id == 1 and report.parent_load_id is None
    assert (report.n_snapshot_dates, report.n_intervals, report.n_symbols) == (2712, 1255, 1202)
    assert report.content_sha256 == content_sha256(parse_history_csv(raw))
    assert (_count(conn, "snapshot_dates"), _count(conn, "membership")) == (2712, 1255)
    assert (_count(conn, "load_blobs"), _count(conn, "checks")) == (1, 1)

    outcome = verify_vintage(conn, 1, SP500_HISTORY_CSV)
    assert outcome.ok, [c for c in outcome.checks if not c.ok]
    assert any("2712/2712 snapshot dates equal" in c.detail for c in outcome.checks)

    with conn.cursor() as cur:
        cur.execute("SELECT symbol_yf FROM universe.membership WHERE symbol_raw = 'BRK.B' LIMIT 1")
        assert cur.fetchone()[0] == "BRK-B"
        cur.execute("SELECT raw_sha256, loaded_by <> '' FROM universe.loads WHERE load_id = 1")
        assert cur.fetchone() == (raw_sha256(raw), True)


def test_stored_bytes_export_byte_exact(conn) -> None:
    raw = _v1_bytes()
    load_vintage(conn, _request(raw))
    with conn.cursor() as cur:
        cur.execute("SELECT raw_gzip FROM universe.load_blobs WHERE load_id = 1")
        assert gzip.decompress(bytes(cur.fetchone()[0])) == raw


def test_reloading_the_same_bytes_is_a_no_op_that_records_a_check(conn) -> None:
    raw = _v1_bytes()
    load_vintage(conn, _request(raw))
    again = load_vintage(conn, _request(raw))
    assert again.no_op and again.committed and again.load_id is None
    assert _count(conn, "loads") == 1
    with conn.cursor() as cur:
        cur.execute("SELECT outcome, matched_load_id FROM universe.checks ORDER BY check_id")
        assert cur.fetchall() == [("loaded", 1), ("unchanged", 1)]


def test_validation_and_pins_refuse_before_writing(conn) -> None:
    raw = _v1_bytes()
    with pytest.raises(LoadRefused, match="expect-sha256"):
        load_vintage(conn, _request(raw, expect_sha256="0" * 64))
    with pytest.raises(LoadRefused, match="pinned"):
        load_vintage(conn, _request(raw, parent=7))
    with pytest.raises(HistoryValidationError):
        load_vintage(conn, _request(raw.replace(b"date,tickers", b"day,tickers", 1)))
    with pytest.raises(LoadRefused, match="timezone"):
        load_vintage(conn, _request(raw, retrieved_at=datetime(2026, 6, 9)))
    assert _count(conn, "loads") == 0


def test_a_failed_read_back_rolls_the_whole_load_back(conn, monkeypatch) -> None:
    def broken(*_args, **_kwargs) -> None:
        raise LoadRefused(["simulated read-back failure"])

    monkeypatch.setattr(universe_db, "_verify_readback", broken)
    with pytest.raises(LoadRefused, match="simulated"):
        load_vintage(conn, _request(_v1_bytes()))
    assert all(
        _count(conn, t) == 0
        for t in ("loads", "load_blobs", "snapshot_dates", "membership", "checks")
    )


# ---- vintage 2: supersession, lineage, the four queries ---------------------------------


def test_a_second_vintage_needs_force_and_keeps_vintage_one_exactly(conn) -> None:
    v1 = _v1_bytes()
    v2 = _v2_bytes()
    load_vintage(conn, _request(v1))

    with pytest.raises(LoadRefused, match="historical rewrites"):
        load_vintage(conn, _request(v2, retrieved_at=V2_AT, commit=False))
    with pytest.raises(LoadRefused, match="--force needs --reason"):
        load_vintage(conn, _request(v2, retrieved_at=V2_AT, force=True))
    dry = load_vintage(
        conn, _request(v2, retrieved_at=V2_AT, force=True, force_reason="t", commit=False)
    )
    assert dry.forced and len(dry.historical_rewrites) == 7 and dry.renames == []
    assert set(dry.tail_changes) >= {"NEWAAA", "NEWBBB"}
    assert (
        dry.dates_only_in_new == [D("2026-08-18")]
        and sum(len(v["removed"]) for v in dry.dates_changed.values()) == 7
    )

    report = load_vintage(
        conn, _request(v2, retrieved_at=V2_AT, force=True, force_reason="test rewrite", parent=1)
    )
    assert report.committed and report.load_id == 2 and report.parent_load_id == 1

    with conn.cursor() as cur:
        v1_back = read_vintage(cur, 1)
        v2_back = read_vintage(cur, 2)
    assert v1_back.history() == parse_history_csv(v1)
    assert v2_back.history() == parse_history_csv(v2)
    assert content_sha256(v1_back.history()) != content_sha256(v2_back.history())

    only_1, only_2 = diff_intervals(v1_back.intervals, v2_back.intervals)
    assert only_1 and only_2
    assert report.rows_superseded == len(only_1) and report.rows_added == len(only_2)

    # Old vintage still verifies against the old file after the new one is loaded.
    assert verify_vintage(conn, 1, SP500_HISTORY_CSV).ok
    with conn.cursor() as cur:
        # Q1: members at the latest vintage on a date only v2 covers.
        cur.execute(
            "SELECT symbol_raw FROM universe.membership WHERE index_name = 'sp500' "
            "AND superseded_by_load_id IS NULL AND valid_from <= %s::date "
            "AND (valid_to IS NULL OR valid_to > %s::date) ORDER BY symbol_raw",
            (D("2026-08-18"), D("2026-08-18")),
        )
        assert {r[0] for r in cur.fetchall()} >= {"NEWAAA", "NEWBBB"}
        # Q3b: the date present only in v2.
        cur.execute(
            "SELECT snapshot_date FROM universe.snapshot_dates WHERE load_id IN (1, 2) "
            "GROUP BY snapshot_date HAVING count(*) = 1"
        )
        assert [r[0] for r in cur.fetchall()] == [D("2026-08-18")]
        # Q4: the latest load carries the newest coverage.
        cur.execute(
            "SELECT load_id, coverage_end FROM universe.loads ORDER BY load_id DESC LIMIT 1"
        )
        assert cur.fetchone() == (2, D("2026-08-18"))
        cur.execute(
            "SELECT forced, force_reason, rows_superseded FROM universe.loads WHERE load_id = 2"
        )
        assert cur.fetchone() == (True, "test rewrite", len(only_1))


# ---- append-only and the exclusion constraint -------------------------------------------


def _expect_refused(conn, sql: str, params: tuple = ()) -> None:
    with pytest.raises(psycopg2.errors.RaiseException):
        with conn.cursor() as cur:
            cur.execute(sql, params)
    conn.rollback()


def test_triggers_refuse_update_delete_and_truncate(conn) -> None:
    load_vintage(conn, _request(_v1_bytes()))
    load_vintage(conn, _request(_v2_bytes(), retrieved_at=V2_AT, force=True, force_reason="t"))
    for table in (
        "loads",
        "load_blobs",
        "snapshot_dates",
        "membership",
        "checks",
        "check_blobs",
        "symbol_changes",
        "load_evidence",
        "source_snapshots",
    ):
        _expect_refused(conn, f"TRUNCATE universe.{table} CASCADE")
    _expect_refused(conn, "DELETE FROM universe.loads WHERE load_id = 1")
    _expect_refused(conn, "UPDATE universe.loads SET note = 'edited' WHERE load_id = 1")
    _expect_refused(conn, "UPDATE universe.snapshot_dates SET n_members = 1")
    _expect_refused(conn, "UPDATE universe.checks SET outcome = 'refused'")
    _expect_refused(conn, "DELETE FROM universe.membership")
    _expect_refused(conn, "UPDATE universe.membership SET symbol_raw = 'ZZZ' WHERE interval_id = 1")
    # A row already superseded cannot be superseded again, nor un-superseded.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT interval_id FROM universe.membership WHERE superseded_by_load_id = 2 LIMIT 1"
        )
        done = cur.fetchone()[0]
    _expect_refused(
        conn,
        "UPDATE universe.membership SET superseded_by_load_id = NULL WHERE interval_id = %s",
        (done,),
    )
    _expect_refused(
        conn,
        "UPDATE universe.membership SET superseded_by_load_id = 2 WHERE interval_id = %s",
        (done,),
    )
    assert _count(conn, "loads") == 2


def test_superseding_a_live_row_once_is_the_only_update_allowed(conn) -> None:
    load_vintage(conn, _request(_v1_bytes()))
    load_vintage(conn, _request(_v2_bytes(), retrieved_at=V2_AT, force=True, force_reason="t"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT interval_id FROM universe.membership "
            "WHERE superseded_by_load_id IS NULL LIMIT 1"
        )
        live = cur.fetchone()[0]
        cur.execute(
            "UPDATE universe.membership SET superseded_by_load_id = 2 WHERE interval_id = %s",
            (live,),
        )
        assert cur.rowcount == 1
    conn.rollback()


def test_exclusion_constraint_blocks_overlapping_live_intervals(conn) -> None:
    load_vintage(conn, _request(_v1_bytes()))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol_raw, valid_from, valid_to FROM universe.membership "
            "WHERE valid_to IS NOT NULL ORDER BY interval_id LIMIT 1"
        )
        symbol, start, end = cur.fetchone()
        insert = (
            "INSERT INTO universe.membership (index_name, symbol_raw, symbol_yf, valid_from, "
            "valid_to, "
            "load_id) VALUES ('sp500', %s, %s, %s::date, %s::date, 1)"
        )
        with pytest.raises(errors.ExclusionViolation):
            cur.execute(insert, (symbol, symbol, start + timedelta(days=1), None))
        conn.rollback()
        with conn.cursor() as cur2:
            # Adjacent, not overlapping, is allowed ([a,b) then [b,c)).
            cur2.execute(insert, (symbol, symbol, end, end + timedelta(days=3)))
    conn.rollback()


def test_the_schema_function_matches_the_python_reconstruction(conn) -> None:
    load_vintage(conn, _request(_v1_bytes()))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM universe.snapshot('sp500', 1) WHERE snapshot_date = %s::date",
            (D("2026-06-02"),),
        )
        assert cur.fetchone()[0] == len(parse_history_csv(_v1_bytes())[D("2026-06-02")])


# ---- concurrency and the read-only session -----------------------------------------------


def test_two_loads_pinned_to_the_same_parent_end_with_exactly_one_winner(db_url: str) -> None:
    setup = connect(db_url)
    try:
        load_vintage(setup, _request(_v1_bytes()))
    finally:
        setup.close()
    variants = [_v2_bytes(rewrites=6), _v2_bytes(rewrites=7)]
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def worker(raw: bytes) -> None:
        conn = connect(db_url)
        try:
            barrier.wait()
            load_vintage(
                conn, _request(raw, retrieved_at=V2_AT, force=True, force_reason="race", parent=1)
            )
            outcomes.append("loaded")
        except LoadRefused:
            outcomes.append("refused")
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(raw,)) for raw in variants]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert sorted(outcomes) == ["loaded", "refused"]
    check = connect(db_url)
    try:
        assert _count(check, "loads") == 2
        with check.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM universe.membership WHERE superseded_by_load_id IS NULL"
            )
            live = cur.fetchone()[0]
            winner = read_vintage(cur, 2)
        assert live == len(winner.rows)
    finally:
        check.close()


def test_a_read_only_session_cannot_write(db_url: str) -> None:
    conn = connect(db_url, read_only=True)
    try:
        with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO universe.checks (index_name, outcome) "
                    "VALUES ('sp500', 'unchanged')"
                )
    finally:
        conn.close()


def test_verify_reports_a_mismatch_against_a_different_file(conn, tmp_path) -> None:
    load_vintage(conn, _request(_v1_bytes()))
    other = tmp_path / "other.csv"
    other.write_bytes(_v2_bytes())
    outcome = verify_vintage(conn, 1, other)
    assert not outcome.ok
    failed = {c.name for c in outcome.checks if not c.ok}
    assert "raw bytes hash" in failed and "snapshot dates" in failed


def test_verify_without_a_file_checks_each_load_against_its_own_archived_bytes(conn) -> None:
    load_vintage(conn, _request(_v1_bytes()))
    load_vintage(
        conn,
        _request(
            _v2_bytes(),
            source="fja05680/sp500",
            retrieved_at=V2_AT,
            force=True,
            force_reason="test",
        ),
    )
    for load_id in (1, 2):
        outcome = verify_vintage(conn, load_id)
        assert outcome.ok, [(c.name, c.detail) for c in outcome.checks if not c.ok]
        assert "its own archived bytes" in outcome.checks[0].detail
    # The old behaviour compared load 2 with the vintage-1 file and reported a false FAILED:
    # an explicit reference file is still honoured, and still disagrees for the wrong file.
    assert verify_vintage(conn, 1, SP500_HISTORY_CSV).ok
    assert not verify_vintage(conn, 2, SP500_HISTORY_CSV).ok


def test_verify_without_a_file_fails_when_the_archived_bytes_are_missing(conn) -> None:
    load_vintage(conn, _request(_v1_bytes()))
    conn.rollback()
    assert verify_vintage(conn, 1).ok
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE universe.load_blobs DISABLE TRIGGER USER")
        cur.execute("DELETE FROM universe.load_blobs WHERE load_id = 1")
    conn.commit()  # verify_vintage starts with a rollback
    outcome = verify_vintage(conn, 1)
    assert not outcome.ok and outcome.checks[-1].name == "archived bytes"


# ---- upstream check (slice M4) -----------------------------------------------------------


def _fetched(raw: bytes):
    from ggTrader.data.membership_upstream import UpstreamFile

    return UpstreamFile(
        raw, "https://example.invalid/x", "a" * 40, '"e"', datetime.now(timezone.utc)
    )


def test_an_upstream_check_appends_one_row_and_extends_nothing_and_changes_nothing_else(
    conn, db_url
):
    from ggTrader.data.membership import PgMembershipStore
    from ggTrader.data.membership_upstream import PgCheckStore, check_upstream

    raw = _v1_bytes()
    load_vintage(conn, _request(raw, parent="none"))
    before = {t: _count(conn, t) for t in ("loads", "membership", "snapshot_dates")}
    store = PgCheckStore(db_url)
    verified0 = PgMembershipStore(db_url).read(None, "sp500").verified_at

    differs = check_upstream(store, fetch=lambda: _fetched(_v2_bytes()))
    assert differs.result.outcome == "refused" and differs.check_id == 2
    assert PgMembershipStore(db_url).read(None, "sp500").verified_at == verified0  # not reset

    equal = check_upstream(store, fetch=lambda: _fetched(raw))
    assert equal.result.outcome == "unchanged" and equal.check_id == 3
    # Appendix C3: an upstream check no longer extends anything; only a holdings match does.
    stored = PgMembershipStore(db_url).read(None, "sp500")
    assert stored.verified_at == verified0 and stored.matched_at is None

    assert _count(conn, "checks") == 3  # the load's own row plus the two checks
    assert {t: _count(conn, t) for t in before} == before  # loads and membership untouched
    with conn.cursor() as cur:
        cur.execute("SELECT outcome, detail->>'result' FROM universe.checks ORDER BY check_id")
        assert cur.fetchall() == [("loaded", None), ("refused", "differs"), ("unchanged", "equal")]
        with pytest.raises(errors.RaiseException):
            cur.execute("UPDATE universe.checks SET outcome = 'unchanged' WHERE check_id = 3")
    conn.rollback()
