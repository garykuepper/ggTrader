"""Integration tests of the curated (events-file) load on a throwaway database.

Never part of the fast, full or CI tiers (marked ``integration`` and ``db``; skipped unless
``GGT_TEST_DB_URL`` names a server). Each test works in its own ``ggt_test_<uuid>`` database.
Run (the main database is only the door to the server, hence the explicit consent flag)::

    GGT_TEST_DB_URL=... GGT_TEST_DB_ALLOW_SHARED=1 \\
        scripts/run_tests.sh -m integration tests/data/test_membership_events_integration.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ggTrader.cli import cmd_membership
from ggTrader.cli.cmd_db import register_db_parser
from ggTrader.data import universe_db
from ggTrader.data.membership_events import EVENT_COLUMNS, prepare_events_load
from ggTrader.data.universe_algebra import content_sha256, parse_history_csv, raw_sha256
from ggTrader.data.universe_db import (
    LOADER_URL_ENV,
    LoadRefused,
    load_vintage,
    read_evidence,
    read_symbol_changes,
    read_vintage,
    verify_vintage,
)
from tests.data.test_universe_db_integration import (
    V2_AT,
    _count,
    _expect_refused,
    _request,
    _v1_bytes,
    _v2_bytes,
)

pytestmark = [pytest.mark.db, pytest.mark.integration]

D = date.fromisoformat
TODAY = D("2026-11-08")
PAGE = b"<html>press release</html>"
PAGE_SHA = hashlib.sha256(PAGE).hexdigest()
URL = "https://example.invalid/release"
LOADED_AT = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def _row(day: str, action: str, ticker: str, new: str = "") -> list[str]:
    return [day, action, ticker, day, URL, PAGE_SHA, "test", new]


def _write_events(folder: Path, rows: list[list[str]], name: str = "events.csv") -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EVENT_COLUMNS)
    writer.writerows(rows)
    path = folder / name
    path.write_bytes(buffer.getvalue().encode())
    (folder / "evidence").mkdir(exist_ok=True)
    (folder / "evidence" / "page.html").write_bytes(PAGE)
    return path


def _events_rows() -> list[list[str]]:
    """Against the test vintage 2 (tail: NEWAAA, NEWBBB joined on 2026-08-18)."""
    return [
        _row("2026-09-21", "add", "EVTA"),
        _row("2026-09-21", "remove", "NEWAAA"),
        _row("2026-10-01", "add", "EVTB"),
        _row("2026-10-06", "remove", "NEWBBB"),
        _row("2026-10-06", "add", "EVTC"),
        _row("2026-10-06", "rename", "NEWBBB", new="EVTC"),
    ]


def _two_vintages(conn) -> None:
    load_vintage(conn, _request(_v1_bytes(), parent="none"))
    load_vintage(conn, _request(_v2_bytes(), retrieved_at=V2_AT, force=True, force_reason="t"))


def _prepare(conn, tmp_path: Path, rows=None, **kwargs):
    events = _write_events(tmp_path, rows or _events_rows())
    return prepare_events_load(
        conn,
        events_path=events,
        evidence_dir=tmp_path / "evidence",
        today=TODAY,
        retrieved_at=LOADED_AT,
        loader_git_sha="test",
        **kwargs,
    )


def _all_rows(conn) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    with conn.cursor() as cur:
        for table, order in (("loads", "load_id"), ("snapshot_dates", "load_id, snapshot_date")):
            cur.execute(f"SELECT * FROM universe.{table} ORDER BY {order}")
            out[table] = cur.fetchall()
        cur.execute(
            "SELECT interval_id, index_name, symbol_raw, symbol_yf, valid_from, valid_to, load_id "
            "FROM universe.membership ORDER BY interval_id"
        )
        out["membership"] = cur.fetchall()
    return out


def test_a_dry_run_writes_nothing_and_needs_no_force(conn, tmp_path) -> None:
    _two_vintages(conn)
    before = {t: _count(conn, t) for t in ("loads", "membership", "load_evidence", "checks")}
    prepared = _prepare(conn, tmp_path, parent=2)
    assert prepared.problems == [] and prepared.request.commit is False
    report = load_vintage(conn, prepared.request)
    assert not report.committed and report.refusal_reasons == []
    assert report.historical_rewrites == [] and report.tail_changes
    assert report.n_evidence == 2 and report.n_symbol_changes == 1
    assert {t: _count(conn, t) for t in before} == before
    assert _count(conn, "symbol_changes") == 0


def test_a_curated_vintage_loads_with_its_evidence_and_verifies_on_its_own_bytes(
    conn, tmp_path
) -> None:
    _two_vintages(conn)
    before = _all_rows(conn)
    prepared = _prepare(conn, tmp_path, parent=2, commit=True)
    report = load_vintage(conn, prepared.request)
    assert report.committed and report.load_id == 3 and report.parent_load_id == 2

    with conn.cursor() as cur:
        load = read_vintage(cur, 3).load
        assert (load["source"], load["upstream_commit"], load["parent_load_id"]) == (
            "curated",
            None,
            2,
        )
        assert load["coverage_end"] == D("2026-10-06") and load["loader_git_sha"] == "test"
        assert "events:events.csv" in load["source_url"] and URL in load["source_url"]
        history = read_vintage(cur, 3).history()
        evidence = read_evidence(cur, 3)
        changes = read_symbol_changes(cur, 3)
        cur.execute("SELECT raw_gzip FROM universe.load_blobs WHERE load_id = 3")
        stored_raw = gzip.decompress(bytes(cur.fetchone()[0]))
    assert stored_raw == prepared.request.raw and raw_sha256(stored_raw) == load["raw_sha256"]
    assert content_sha256(parse_history_csv(stored_raw)) == content_sha256(history)
    assert [d.isoformat() for d in sorted(history)][-3:] == [
        "2026-09-21",
        "2026-10-01",
        "2026-10-06",
    ]
    assert sorted(evidence) == ["events.csv", "page.html"]
    assert gzip.decompress(evidence["page.html"]["content_gz"]) == PAGE
    assert (
        evidence["page.html"]["sha256"] == PAGE_SHA and evidence["page.html"]["source_url"] == URL
    )
    assert evidence["events.csv"]["kind"] == "events_file"
    assert [(c.old_symbol, c.new_symbol, c.effective_date) for c in changes] == [
        ("NEWBBB", "EVTC", D("2026-10-06"))
    ]

    for load_id in (1, 2, 3):  # no options: each load against its own archived bytes
        outcome = verify_vintage(conn, load_id)
        assert outcome.ok, [(c.name, c.detail) for c in outcome.checks if not c.ok]
    assert any(c.name == "stored evidence" for c in verify_vintage(conn, 3).checks)

    # Earlier rows are untouched: same ids and content, only superseded markers may appear.
    after = _all_rows(conn)
    for table in ("loads", "snapshot_dates"):
        assert after[table][: len(before[table])] == before[table]
    assert after["membership"][: len(before["membership"])] == before["membership"]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT outcome, matched_load_id FROM universe.checks ORDER BY check_id DESC LIMIT 1"
        )
        assert cur.fetchone() == ("loaded", 3)


def test_events_cannot_be_applied_twice(conn, tmp_path) -> None:
    _two_vintages(conn)
    load_vintage(conn, _prepare(conn, tmp_path, parent=2, commit=True).request)
    again = _prepare(conn, tmp_path, commit=True)
    assert again.problems
    assert any("not after the parent" in p for p in again.problems)
    assert _count(conn, "loads") == 3


def test_a_stale_parent_pin_is_refused(conn, tmp_path) -> None:
    _two_vintages(conn)
    with pytest.raises(LoadRefused, match="pinned to 1"):
        _prepare(conn, tmp_path, parent=1)
    prepared = _prepare(conn, tmp_path, parent=2, commit=True)
    later = _prepare(conn, tmp_path, rows=[_row("2026-11-04", "add", "LATE")], commit=True)
    load_vintage(conn, later.request)  # a different load lands first
    # the pin of an already prepared request is checked again under the loader's lock
    with pytest.raises(LoadRefused, match="pinned to 2"):
        load_vintage(conn, prepared.request)


def test_a_curated_vintage_needs_a_parent_and_its_events_file(conn, tmp_path) -> None:
    with pytest.raises(LoadRefused, match="no parent"):
        _prepare(conn, tmp_path)
    _two_vintages(conn)
    prepared = _prepare(conn, tmp_path, parent=2, commit=True)
    bare = replace(prepared.request, evidence=())
    with pytest.raises(LoadRefused, match="events file"):
        load_vintage(conn, bare)
    plain = _request(_v2_bytes(), evidence=prepared.request.evidence)
    with pytest.raises(LoadRefused, match="curated vintages only"):
        load_vintage(conn, plain)
    assert _count(conn, "loads") == 2


def test_a_failure_while_storing_evidence_rolls_the_whole_load_back(
    conn, tmp_path, monkeypatch
) -> None:
    _two_vintages(conn)
    sizes = {t: _count(conn, t) for t in ("loads", "membership", "snapshot_dates", "load_blobs")}

    def broken(*args, **kwargs) -> None:
        raise LoadRefused(["boom"])

    monkeypatch.setattr(universe_db, "_verify_evidence_readback", broken)
    with pytest.raises(LoadRefused, match="boom"):
        load_vintage(conn, _prepare(conn, tmp_path, parent=2, commit=True).request)
    assert {t: _count(conn, t) for t in sizes} == sizes
    assert _count(conn, "load_evidence") == _count(conn, "symbol_changes") == 0


def test_evidence_and_symbol_changes_are_append_only(conn, tmp_path) -> None:
    _two_vintages(conn)
    load_vintage(conn, _prepare(conn, tmp_path, parent=2, commit=True).request)
    for table in ("symbol_changes", "load_evidence"):
        _expect_refused(conn, f"DELETE FROM universe.{table}")
        _expect_refused(conn, f"TRUNCATE universe.{table}")
    _expect_refused(conn, "UPDATE universe.symbol_changes SET new_symbol = 'X'")
    _expect_refused(conn, "UPDATE universe.load_evidence SET name = 'x'")


def test_the_cli_prints_the_table_and_writes_only_with_commit(
    db_url, conn, tmp_path, monkeypatch, capsys
) -> None:
    _two_vintages(conn)
    monkeypatch.setenv(LOADER_URL_ENV, db_url)
    events = _write_events(tmp_path, _events_rows())
    monkeypatch.setattr(cmd_membership, "git_sha", lambda: "cli-test")

    def args(*extra: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="ggt")
        register_db_parser(parser.add_subparsers(dest="command"))
        return parser.parse_args(
            ["db", "load-sp500", "--from-events", str(events), "--evidence-dir",
             str(tmp_path / "evidence"), "--parent", "2", *extra]
        )  # fmt: skip

    # Dates after 2026-11-08 would be refused; the test events are all earlier, so no clock patch.
    assert cmd_membership.run_load_sp500(args()) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "2026-10-06" in out and "symbol change NEWBBB -> EVTC" in out
    assert _count(conn, "loads") == 2
    assert cmd_membership.run_load_sp500(args("--commit")) == 0
    assert "COMMITTED" in capsys.readouterr().out and _count(conn, "loads") == 3
    assert cmd_membership.run_load_sp500(args("--commit")) == 2  # same events again: refused
    assert _count(conn, "loads") == 3
