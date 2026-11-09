"""Fast, offline tests of ``ggt db membership weekly`` (slice M6).

Everything is replaced: the network by ``World`` (fabricated public sources), the database by
``FakeWeeklyStore`` and Telegram by ``RecordingNotifier`` or the repo's ``DryRunNotifier``. An
autouse fixture makes any real network or Telegram call raise, so a mistake here fails loudly
instead of reaching a server.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import re
import socket
from datetime import date, timedelta
from pathlib import Path

import pytest
import requests

from ggTrader.cli import cmd_membership
from ggTrader.data import membership_weekly as mw
from ggTrader.data.membership_events import EventsFileError, parse_events_csv, plan_events
from ggTrader.data.membership_holdings import HoldingsFetchError
from ggTrader.data.membership_proposal import FILE_COLUMNS
from ggTrader.data.universe_algebra import history_members_asof
from ggTrader.paper.notifier import DryRunNotifier, TelegramNotifier
from tests.data.membership_weekly_fakes import (
    NOW,
    TODAY,
    FakeWeeklyStore,
    NoNetwork,
    RecordingNotifier,
    World,
    base_ours,
    changes_csv,
    components_csv,
    current_json,
    ledger_json,
    no_real_network,
    press_ref,
    reply,
)
from tests.data.test_membership_holdings import _Connection, history
from tests.data.test_membership_upstream import FakeResponse, vintage_data
from tests.data.xlsx_builder import holdings_xlsx

SRC = Path(mw.__file__).parent


@pytest.fixture(autouse=True)
def _nothing_can_reach_the_network_or_telegram(monkeypatch):
    """Any real HTTP, socket or Telegram call made by a test raises ``NoNetwork``."""
    monkeypatch.setattr(requests, "get", no_real_network)
    monkeypatch.setattr(requests, "post", no_real_network)
    monkeypatch.setattr(requests.sessions.Session, "request", no_real_network)
    monkeypatch.setattr(socket.socket, "connect", no_real_network)
    monkeypatch.setattr(TelegramNotifier, "send", no_real_network)


def run(world: World, store=None, notifier=None, tmp_path: Path | None = None, now=NOW):
    store = store or FakeWeeklyStore(world.ours)
    notifier = notifier or RecordingNotifier()
    result = mw.run_weekly(
        store,
        notifier,
        get=world,
        now=lambda: now,
        proposal_dir=(tmp_path or Path("/nonexistent-never-written")) / "proposals",
    )
    return result, store, notifier


def codes(result) -> set[str]:
    return {f.code for f in result.findings}


# --------------------------------------------------------------------------- a clean run


def test_a_clean_run_records_a_match_archives_every_source_and_sends_a_heartbeat(tmp_path):
    world = World(base_ours())
    result, store, notifier = run(world, tmp_path=tmp_path)
    assert result.status == "ok" and result.exit_code == 0 and result.findings == []
    # four sources, two metadata calls, the SPY workbook, the press page
    assert len(world.calls) == 8
    assert len(store.runs) == 1
    check, rows = store.runs[0]
    assert (check.outcome, check.matched_load_id) == ("matched", 1)
    assert (
        check.detail["weekly"]["status"] == "ok" and check.detail["known_through"] == "2026-11-05"
    )
    assert {r.source for r in rows} == {
        "wikipedia_ledger",
        "wikipedia_current",
        "chinobing_components",
        "chinobing_changes",
        "spy_holdings",
        "sp_press_listing",
    }
    by_name = {r.source: r for r in rows}
    assert by_name["wikipedia_ledger"].raw == world.ledger
    assert by_name["chinobing_components"].raw == world.components
    assert by_name["wikipedia_ledger"].raw_sha256 == hashlib.sha256(world.ledger).hexdigest()
    assert by_name["spy_holdings"].raw is None and check.raw == world.spy  # SPY bytes: check_blobs
    assert by_name["chinobing_components"].last_change == date(2026, 11, 8)  # its newest row
    assert by_name["wikipedia_ledger"].last_change == date(2026, 11, 3)
    assert by_name["spy_holdings"].last_change == date(2026, 11, 5)
    assert not (tmp_path / "proposals").exists()
    assert len(notifier.messages) == 1 and len(notifier.messages[0]) < 700
    assert "OK" in notifier.messages[0] and "ACTION" not in notifier.messages[0]
    assert result.check_id == 1 and result.proposal is None


def test_the_run_uses_a_descriptive_user_agent_and_an_independent_get_per_source(tmp_path):
    world = World(base_ours())
    run(world, tmp_path=tmp_path)
    agents = {headers.get("User-Agent", "") for _, _, headers in world.calls}
    assert any("ggTrader-membership-check" in a for a in agents)


# --------------------------------------------------------------------------- the mismatch


def fabricated_mismatch(tmp_path: Path, tweak=None):
    """Our list vs a SPY file with one clean swap, a date disagreement, a rename and an
    announcement-dated ledger row; the sources give candidate dates for each."""
    ours = base_ours()
    spy = [t for t in ours if t not in {"X0009", "X0010", "X0020", "X0030"}] + [
        "ZZNEW",
        "ZZAGR",
        "NEWCO",
        "ZZANN",
    ]
    world = World(ours)
    world.spy = holdings_xlsx(spy, "As of 06-Nov-2026")
    world.ledger = ledger_json(
        [
            (
                "November 5, 2026",
                "ZZANN",
                "Ann Co",
                "X0030",
                "Old Ann",
                "r",
                press_ref("November 5, 2026"),
            ),
            (
                "November 4, 2026",
                "ZZNEW",
                "New Co",
                "X0009",
                "Old Nine",
                "r",
                press_ref("October 30, 2026"),
            ),
            (
                "November 2, 2026",
                "ZZAGR",
                "Agree Co",
                "X0010",
                "Old Ten",
                "r",
                press_ref("October 28, 2026"),
            ),
        ]
    )
    world.changes = changes_csv(
        [
            (date(2026, 11, 2), ["ZZAGR"], ["X0010"]),
            (date(2026, 11, 3), ["NEWCO"], ["X0020"]),
            (date(2026, 11, 5), ["ZZNEW"], ["X0009"]),
            (date(2026, 11, 6), ["ZZANN"], ["X0030"]),
        ]
    )
    if tweak is not None:
        tweak(world)
    result, store, notifier = run(world, tmp_path=tmp_path)
    return world, spy, result, store, notifier


def read_proposal(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def test_a_mismatch_writes_a_proposal_with_dates_sources_and_confirmation_flags(tmp_path):
    _, _, result, store, notifier = fabricated_mismatch(tmp_path)
    assert result.status == "refuse" and result.exit_code == 2
    assert codes(result) >= {"spy_mismatch", "ledger_announcement_date"}
    assert result.spy.outcome == "mismatch" and result.spy.matched_load_id is None
    path = result.proposal_path
    assert path is not None and path.parent == tmp_path / "proposals"
    assert path.name.startswith("proposed_events_20261109T140000Z") and path.suffix == ".csv"
    assert path.read_text().splitlines()[0] == ",".join(FILE_COLUMNS)
    rows = read_proposal(path)
    got = [
        (
            r["effective_date"],
            r["action"],
            r["ticker"],
            r["new_ticker"],
            r["date_source"],
            r["needs_confirmation"],
        )
        for r in rows
    ]
    assert got == [
        # agreed by both sources, not future, not announcement-dated: the only "N"
        ("2026-11-02", "remove", "X0010", "", "ledger+chinobing", "N"),
        ("2026-11-02", "add", "ZZAGR", "", "ledger+chinobing", "N"),
        # no ledger row (it omits ticker changes); chinobing shows both names on one day
        ("2026-11-03", "remove", "X0020", "", "chinobing", "Y"),
        ("2026-11-03", "add", "NEWCO", "", "chinobing", "Y"),
        ("2026-11-03", "rename", "X0020", "NEWCO", "chinobing", "Y"),
        # the ledger says 11-04, chinobing observed 11-05: they disagree by a day
        ("2026-11-04", "remove", "X0009", "", "ledger+chinobing", "Y"),
        ("2026-11-04", "add", "ZZNEW", "", "ledger+chinobing", "Y"),
        # the ledger row carries its announcement date (11-05), so chinobing's 11-06 is used
        ("2026-11-06", "remove", "X0030", "", "chinobing", "Y"),
        ("2026-11-06", "add", "ZZANN", "", "chinobing", "Y"),
    ]
    assert all(r["evidence_url"] == "" and r["evidence_sha256"] == "" for r in rows)
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["ZZNEW"]["announcement_date"] == "2026-10-30"
    assert by_ticker["ZZAGR"]["announcement_date"] == "2026-10-28"
    assert by_ticker["X0020"]["announcement_date"] == ""
    assert "disagree" in by_ticker["ZZNEW"]["note"] and "11-05" in by_ticker["ZZNEW"]["note"]
    assert "announcement" in by_ticker["ZZANN"]["note"]
    assert "ticker change" in by_ticker["NEWCO"]["note"] or "ticker change" in rows[4]["note"]
    # the bytes are also kept in the database as a snapshot row
    check, snapshots = store.runs[0]
    proposal_row = next(r for r in snapshots if r.source == "proposal")
    assert proposal_row.raw == path.read_bytes() and proposal_row.url == f"file:{path.name}"
    assert check.detail["weekly"]["proposal"]["rows"] == 9
    assert check.detail["weekly"]["proposal"]["file"] == path.name
    # one message: the proposal, what to check and the exact commands
    assert len(notifier.messages) == 1
    message = notifier.messages[0]
    rel = f"data/exports/membership_proposals/{path.name}"
    assert "ACTION NEEDED" in message and "nothing was loaded" in message
    assert "press release" in message and "needs_confirmation" in message
    # the message names the command that rebuilds the file from the database, not a path that
    # may vanish with the container (the fake store's first check id is 1)
    assert "<code>ggt db membership proposal --id 1 --out proposed_events.csv</code>" in message
    assert "<code>ggt db load-sp500 --from-events proposed_events.csv" in message
    assert "--commit</code>" in message and rel not in message and str(path) not in message
    assert "ZZNEW" in message and "CONFIRM" in message


def test_the_proposal_cannot_be_loaded_until_a_human_has_confirmed_it(tmp_path):
    _, spy, result, _, _ = fabricated_mismatch(tmp_path)
    raw = result.proposal_path.read_bytes()
    with pytest.raises(EventsFileError) as caught:
        parse_events_csv(raw)
    text = " ".join(caught.value.problems)
    assert "evidence_url is empty" in text and "needs_confirmation" in text
    # the human confirms every date, fills the evidence and the missing announcement date ...
    edited = []
    for row in csv.DictReader(io.StringIO(raw.decode())):
        row["evidence_url"] = "https://press.spglobal.com/example"
        row["needs_confirmation"] = "N"
        row["announcement_date"] = row["announcement_date"] or "2026-10-20"
        edited.append(row)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FILE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(edited)
    events = parse_events_csv(out.getvalue().encode())
    assert len(events) == 9
    # ... and the loader's own planner then reproduces exactly the SPY list
    parent = vintage_data(history(base_ours())).history()
    plan = plan_events(parent, events, today=TODAY)
    assert plan.problems == []
    final = {t.replace(".", "-") for t in history_members_asof(plan.history, date(2026, 11, 6))}
    assert final == {t.replace(".", "-") for t in spy}
    assert [s.old_symbol for s in plan.symbol_changes] == ["X0020"]


def test_a_mismatch_still_records_nothing_but_a_mismatch_check(tmp_path):
    _, _, result, store, _ = fabricated_mismatch(tmp_path)
    check, _ = store.runs[0]
    assert check.outcome == "mismatch" and check.matched_load_id is None
    assert sorted(check.detail["only_in_list"]) == ["X0009", "X0010", "X0020", "X0030"]
    assert sorted(check.detail["only_in_holdings"]) == ["NEWCO", "ZZAGR", "ZZANN", "ZZNEW"]


# --------------------------------------------------------------------------- refusal conditions


def test_a_ledger_row_dated_in_the_future_is_refused_and_proposed(tmp_path):
    world = World(base_ours())
    world.ledger = ledger_json(
        [
            (
                "November 20, 2026",
                "ZZFUT",
                "Future Co",
                "X0040",
                "Old",
                "r",
                press_ref("November 6, 2026"),
            )
        ]
    )
    result, store, notifier = run(world, tmp_path=tmp_path)
    assert result.status == "refuse" and "ledger_future" in codes(result)
    assert store.runs[0][0].outcome == "matched"  # the SPY match is still true and recorded
    rows = read_proposal(result.proposal_path)
    assert [
        (r["action"], r["ticker"], r["effective_date"], r["needs_confirmation"]) for r in rows
    ] == [
        ("remove", "X0040", "2026-11-20", "Y"),
        ("add", "ZZFUT", "2026-11-20", "Y"),
    ]
    assert "ACTION NEEDED" in notifier.messages[0] and "future" in notifier.messages[0]


def test_a_ledger_row_on_its_announcement_date_is_refused(tmp_path):
    world = World(base_ours())
    world.ledger = ledger_json(
        [
            (
                "November 6, 2026",
                "ZZANN",
                "Ann Co",
                "X0040",
                "Old",
                "r",
                press_ref("November 6, 2026"),
            )
        ]
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert "ledger_announcement_date" in codes(result) and result.status == "refuse"
    assert "announcement date" in result.message


def test_rows_already_attested_by_an_earlier_match_are_not_refused(tmp_path):
    # the real page's OKTA row carries its announcement date (Oct 30) but our list is past it
    world = World(base_ours())
    world.ledger = ledger_json(
        [("October 30, 2026", "OKTA", "Okta", "KVUE", "Kenvue", "r", press_ref("October 30, 2026"))]
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert result.status == "ok" and result.findings == []


def test_a_removal_with_no_matching_addition_that_leaves_the_range_is_refused(tmp_path):
    ours = base_ours(476)  # exactly 480 names
    world = World(ours)
    spy = [t for t in ours if t != "X0007"] + ["ZZBACK"]
    world.spy = holdings_xlsx(spy, "As of 06-Nov-2026")
    world.ledger = ledger_json(
        [
            ("November 4, 2026", "ZZBACK", "Back Co", "", "", "r", ""),
            ("November 2, 2026", "", "", "X0007", "Gone Co", "r", ""),
        ]
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert "count_out_of_range" in codes(result)
    problem = next(f.text for f in result.findings if f.code == "count_out_of_range")
    assert "X0007" in problem and "no matching addition" in problem and "479" in problem
    rows = read_proposal(result.proposal_path)
    removal = next(r for r in rows if r["action"] == "remove")
    assert "no matching addition" in removal["note"]


def test_an_unpaired_removal_that_stays_in_range_is_noted_but_not_a_count_refusal(tmp_path):
    ours = base_ours()  # 499 names: one name out for two days is fine (the 504 days of Oct 2026)
    world = World(ours)
    spy = [t for t in ours if t != "X0007"] + ["ZZBACK"]
    world.spy = holdings_xlsx(spy, "As of 06-Nov-2026")
    world.ledger = ledger_json(
        [
            ("November 4, 2026", "ZZBACK", "Back Co", "", "", "r", ""),
            ("November 2, 2026", "", "", "X0007", "Gone Co", "r", ""),
        ]
    )
    result, _, _ = run(world, tmp_path=tmp_path)
    assert "count_out_of_range" not in codes(result) and "spy_mismatch" in codes(result)


@pytest.mark.parametrize(
    ("key", "failure", "reason"),
    [
        ("ledger", requests.ConnectionError("no route"), "unreachable"),
        ("current", reply(b"nope", 503), "http_status"),
        ("components", FakeResponse(200, b"x" * 10, {"content-length": "99"}), "truncated"),
    ],
)
def test_a_missing_source_is_alerted_and_the_others_are_still_archived(
    tmp_path, key, failure, reason
):
    world = World(base_ours())
    world.errors[key] = failure
    result, store, notifier = run(world, tmp_path=tmp_path)
    assert result.status == "refuse" and "source_missing" in codes(result)
    check, rows = store.runs[0]
    assert check.outcome == "matched"  # SPY did not depend on it
    by_name = {r.source: r for r in rows}
    broken = {
        "ledger": "wikipedia_ledger",
        "current": "wikipedia_current",
        "components": "chinobing_components",
    }[key]
    assert by_name[broken].status == "failed" and by_name[broken].raw is None
    assert by_name[broken].detail["reason"] == reason
    # the other three sources, the press page and SPY
    assert sum(1 for r in rows if r.status == "ok") == 5
    assert broken in notifier.messages[0]


def test_a_source_not_updated_for_more_than_14_days_is_alerted_but_14_is_fine(tmp_path):
    world = World(base_ours())
    world.ledger_edit = "2026-10-26T10:00:00Z"  # exactly 14 days before 2026-11-09
    result, _, _ = run(world, tmp_path=tmp_path)
    assert result.status == "ok"
    world.ledger_edit = "2026-10-25T23:59:00Z"  # 15 days
    result, _, _ = run(world, tmp_path=tmp_path)
    assert "source_stale" in codes(result) and "15 days" in " ".join(
        f.text for f in result.findings
    )
    world.ledger_edit = "2026-11-03T10:00:00Z"
    world.commit_date = "2026-10-01T00:00:00Z"
    world.components = components_csv(base_ours(), date(2026, 11, 8))
    result, _, _ = run(world, tmp_path=tmp_path)
    assert [f.text.split(":")[0] for f in result.findings if f.code == "source_stale"] == [
        "chinobing_changes"
    ]  # the components file is judged by its own newest row, not by the commit date


def test_an_unreadable_last_change_date_is_alerted_not_ignored(tmp_path):
    world = World(base_ours())
    world.errors["revisions"] = reply(b"rate limited", 429)
    result, _, _ = run(world, tmp_path=tmp_path)
    unknown = [f for f in result.findings if f.code == "source_age_unknown"]
    assert {f.text.split(":")[0] for f in unknown} == {"wikipedia_ledger", "wikipedia_current"}


def test_a_changed_page_structure_is_alerted_and_the_raw_bytes_are_still_kept(tmp_path):
    from tests.data.membership_weekly_fakes import REAL_HEADER, ledger_wikitext, parse_json

    world = World(base_ours())
    world.ledger = parse_json(
        ledger_wikitext([], header=REAL_HEADER.replace("| Reason", "| Why")), revid=7
    )
    result, store, _ = run(world, tmp_path=tmp_path)
    assert result.status == "refuse" and "source_structure" in codes(result)
    assert "columns_changed" in result.message
    row = {r.source: r for r in store.runs[0][1]}["wikipedia_ledger"]
    assert row.raw == world.ledger and row.detail["structure_error"] == "columns_changed"
    assert store.runs[0][0].outcome == "matched"


@pytest.mark.parametrize(
    ("spy", "code_word"),
    [
        (HoldingsFetchError("http_status", "the host answered HTTP 403"), "http_status"),
        (holdings_xlsx(base_ours(), "As of 20-Oct-2026"), "stale_source"),  # 20 days old
        (holdings_xlsx(base_ours(), "As of 15-Nov-2026"), "as_of_in_future"),
        (b"<html>blocked</html>", "not_xlsx"),
    ],
)
def test_an_unusable_spy_file_is_alerted_and_never_counts_as_a_match(tmp_path, spy, code_word):
    world = World(base_ours())
    if isinstance(spy, Exception):
        world.errors["spy"] = reply(b"denied", 403)
    else:
        world.spy = spy
    result, store, _ = run(world, tmp_path=tmp_path)
    check, rows = store.runs[0]
    assert check.outcome == "fetch_failed" and check.matched_load_id is None
    assert "spy_failed" in codes(result) and code_word in result.message
    assert result.status == "refuse" and result.exit_code == 2
    assert {r.source: r for r in rows}["spy_holdings"].status == "failed"


def test_a_stale_as_of_date_is_the_spy_staleness_guard_not_a_looser_one(tmp_path):
    world = World(base_ours())
    world.spy = holdings_xlsx(base_ours(), "As of 02-Nov-2026")  # 7 days: accepted
    assert run(world, tmp_path=tmp_path)[0].spy.outcome == "matched"
    world.spy = holdings_xlsx(base_ours(), "As of 01-Nov-2026")  # 8 days
    assert run(world, tmp_path=tmp_path)[0].spy.outcome == "fetch_failed"


# --------------------------------------------------------------------------- warnings only


def test_other_sources_disagreeing_with_our_list_are_warnings_only(tmp_path):
    ours = base_ours()
    world = World(ours)
    world.current = current_json([t.replace(".", "-") for t in ours] + ["ZZEXT"])
    world.components = components_csv([t for t in ours if t != "X0003"], date(2026, 11, 8))
    result, store, notifier = run(world, tmp_path=tmp_path)
    assert result.status == "warn" and result.exit_code == 0
    assert codes(result) == {"current_list_differs", "chinobing_differs"}
    assert all(f.severity == "warn" for f in result.findings)
    assert store.runs[0][0].outcome == "matched"  # the SPY match still extends the clock
    assert "ZZEXT" in notifier.messages[0] and "X0003" in notifier.messages[0]
    assert "ACTION NEEDED" not in notifier.messages[0]
    assert result.proposal is None and not (tmp_path / "proposals").exists()


# --------------------------------------------------------------------------- failures of the job


def test_an_unreadable_database_stops_the_run_with_one_alert_and_no_secret(tmp_path):
    world = World(base_ours())
    store = FakeWeeklyStore(world.ours, fail_latest=True)
    result, _, notifier = run(world, store, tmp_path=tmp_path)
    assert result.status == "error" and result.exit_code == 1 and store.runs == []
    assert world.calls == []  # nothing was fetched without a list to compare with
    assert len(notifier.messages) == 1 and "JOB ERROR" in notifier.messages[0]
    assert "secret" not in notifier.messages[0] and "host/db" in notifier.messages[0]


def test_a_failed_record_is_reported_and_exits_nonzero(tmp_path):
    world = World(base_ours())
    store = FakeWeeklyStore(world.ours, fail_record=True)
    result, _, notifier = run(world, store, tmp_path=tmp_path)
    assert result.status == "error" and result.exit_code == 1 and result.check_id is None
    assert "record_failed" in codes(result) and "NOT recorded" in notifier.messages[0]
    assert "secret" not in notifier.messages[0]


def test_a_notifier_that_fails_does_not_hide_the_recorded_result(tmp_path):
    world = World(base_ours())

    class Dead:
        def send(self, message: str) -> bool:
            raise ConnectionError("telegram unreachable")

    result, store, _ = run(world, notifier=Dead(), tmp_path=tmp_path)
    assert result.delivered is False and result.check_id == 1 and len(store.runs) == 1
    refused, _, _ = run(world, notifier=RecordingNotifier(delivers=False), tmp_path=tmp_path)
    assert refused.delivered is False and refused.exit_code == 0


def test_the_previous_match_moves_the_floor_so_older_ledger_rows_are_not_candidates(tmp_path):
    world = World(base_ours())
    world.ledger = ledger_json(
        [
            (
                "November 6, 2026",
                "ZZANN",
                "Ann Co",
                "X0040",
                "Old",
                "r",
                press_ref("November 6, 2026"),
            )
        ]
    )
    # a match at 2026-11-08 already attested everything up to that day (floor is 11-08)
    store = FakeWeeklyStore(world.ours, known_through=date(2026, 11, 8))
    world.spy = holdings_xlsx(world.ours, "As of 05-Nov-2026")
    result, _, _ = run(world, store, tmp_path=tmp_path)
    # SPY matched as of 11-05, so the floor is max(prior 11-08, as-of 11-05) = 11-08
    assert result.floor == date(2026, 11, 8) and result.status == "ok"


# --------------------------------------------------------------------------- never loads


def test_the_job_has_no_way_to_write_loads_or_membership():
    forbidden = re.compile(
        r"(INSERT\s+INTO|UPDATE|DELETE\s+FROM|TRUNCATE)\s+(TABLE\s+)?universe\.(loads|membership|"
        r"snapshot_dates|load_blobs|load_evidence|symbol_changes)",
        re.I,
    )
    for name in ("membership_weekly.py", "membership_sources.py", "membership_proposal.py"):
        assert forbidden.search((SRC / name).read_text()) is None, name
    assert "load_vintage" not in (SRC / "membership_weekly.py").read_text()
    protocol = [m for m in dir(mw.WeeklyStore) if not m.startswith("_")]
    assert protocol == ["known_through", "latest", "record_run"]  # the job's whole database API


def test_the_real_store_only_inserts_checks_blobs_and_source_snapshots(monkeypatch):
    conn = _Connection()
    monkeypatch.setattr(mw, "connect", lambda url=None, **kw: conn)
    store = mw.PgWeeklyStore("ignored")
    world = World(base_ours())
    fake = FakeWeeklyStore(world.ours)
    run(world, fake)
    check, rows = fake.runs[0]
    assert store.record_run("sp500", check, rows) == 42
    statements = [sql for sql, _ in conn.log]
    assert statements[0].startswith("INSERT INTO universe.checks ")
    assert statements[1].startswith("INSERT INTO universe.check_blobs ")
    assert all(s.startswith("INSERT INTO universe.source_snapshots ") for s in statements[2:])
    assert len(statements) == 2 + len(rows) and conn.committed and conn.closed
    snapshot = next(p for s, p in conn.log[2:] if p[1] == "wikipedia_ledger")
    assert gzip.decompress(snapshot[8]) == world.ledger  # exact bytes, gzipped
    spy_row = next(p for s, p in conn.log[2:] if p[1] == "spy_holdings")
    assert spy_row[8] is None  # the SPY workbook lives in check_blobs only


def test_known_through_reads_only(monkeypatch):
    conn = _Connection()
    seen = {}

    def fake_connect(url=None, **kw):
        seen.update(kw)
        return conn

    monkeypatch.setattr(mw, "connect", fake_connect)
    mw.PgWeeklyStore("ignored").known_through(3, "sp500")
    assert seen == {"read_only": True} and conn.log[0][0].lstrip().startswith("SELECT")


# --------------------------------------------------------------------------- no real Telegram


def test_the_dry_notifier_never_reaches_the_inner_notifier(tmp_path):
    class Boom:
        def send(self, message: str) -> bool:
            raise AssertionError("the inner notifier must not be called in a dry run")

    world = World(base_ours())
    result, _, _ = run(world, notifier=DryRunNotifier(Boom()), tmp_path=tmp_path)
    assert result.delivered is False and result.check_id == 1
    result, _, _ = run(World(base_ours()), notifier=DryRunNotifier(None), tmp_path=tmp_path)
    assert result.delivered is False


def test_a_real_telegram_notifier_cannot_send_inside_a_test(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:fake")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    notifier = TelegramNotifier()
    with pytest.raises(NoNetwork):  # the autouse guard replaced send itself
        notifier.send("hello")
    with pytest.raises(NoNetwork):  # and so would the raw HTTP call it makes
        requests.post("https://api.telegram.org/bot0:x/sendMessage", json={})


def _args(**kw) -> argparse.Namespace:
    base = {"membership_command": "weekly", "notify": False, "proposal_dir": None, "json": None}
    return argparse.Namespace(**(base | kw))


def _patch_cli(monkeypatch, world: World, tmp_path: Path, notifiers: list):
    store = FakeWeeklyStore(world.ours)
    monkeypatch.setattr(cmd_membership, "PgWeeklyStore", lambda: store)
    original = mw.run_weekly

    def fake_run(s, n, **kw):
        notifiers.append(n)
        return original(s, n, get=world, now=lambda: NOW, proposal_dir=tmp_path / "p")

    monkeypatch.setattr(cmd_membership, "run_weekly", fake_run)
    return store


def test_the_command_is_dry_by_default_and_builds_no_telegram_notifier(
    monkeypatch, tmp_path, capsys
):
    class NoTelegram:
        def __init__(self) -> None:
            raise AssertionError("TelegramNotifier was built without --notify")

    monkeypatch.setattr(cmd_membership, "TelegramNotifier", NoTelegram)
    notifiers: list = []
    _patch_cli(monkeypatch, World(base_ours()), tmp_path, notifiers)
    assert cmd_membership.run_membership(_args()) == 0
    out = capsys.readouterr().out
    assert isinstance(notifiers[0], DryRunNotifier)
    assert "weekly membership check: OK" in out and "dry run: nothing was sent" in out
    assert "recorded as universe.checks row 1; nothing was loaded" in out
    assert "wikipedia_ledger" in out and "spy_holdings" in out


def test_notify_uses_the_telegram_notifier_and_a_failed_send_exits_nonzero(
    monkeypatch, tmp_path, capsys
):
    sent: list[str] = []

    class Fake:
        ok = True

        def send(self, message: str) -> bool:
            sent.append(message)
            return self.ok

    fake = Fake()
    monkeypatch.setattr(cmd_membership, "TelegramNotifier", lambda: fake)
    notifiers: list = []
    _patch_cli(monkeypatch, World(base_ours()), tmp_path, notifiers)
    assert cmd_membership.run_membership(_args(notify=True)) == 0
    assert notifiers[0] is fake and len(sent) == 1 and "Telegram: sent" in capsys.readouterr().out
    fake.ok = False
    _patch_cli(monkeypatch, World(base_ours()), tmp_path, notifiers)
    assert cmd_membership.run_membership(_args(notify=True)) == 1
    assert "NOT SENT" in capsys.readouterr().out


def test_the_command_exit_codes_for_a_proposal_and_a_job_failure(monkeypatch, tmp_path, capsys):
    notifiers: list = []
    world = World(base_ours())
    world.spy = holdings_xlsx(
        [t for t in base_ours() if t != "X0001"] + ["ZZNEW"], "As of 06-Nov-2026"
    )
    _patch_cli(monkeypatch, world, tmp_path, notifiers)
    assert cmd_membership.run_membership(_args()) == 2
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "proposal:" in out and "ZZNEW" in out

    def boom(*a, **k):
        raise RuntimeError("postgres://user:secret@host/db exploded")

    monkeypatch.setattr(cmd_membership, "run_weekly", boom)
    assert cmd_membership.run_membership(_args()) == 1
    err = capsys.readouterr().err
    assert "weekly check failed" in err and "secret" not in err


def test_the_weekly_subcommand_is_registered_and_dry_by_default():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="cmd")
    cmd_membership.register_membership_parsers(subs)
    args = parser.parse_args(["membership", "weekly"])
    assert args.membership_command == "weekly" and args.notify is False
    assert parser.parse_args(["membership", "weekly", "--notify"]).notify is True


def test_a_test_run_cannot_read_the_real_env_file():
    # tests/conftest keeps load_dotenv a no-op, so no token can leak into TelegramNotifier
    from ggTrader.utils import config

    assert config.load_dotenv("/definitely/not/a/file") is False


def test_proposal_names_never_collide_and_are_never_overwritten(tmp_path):
    world = World(base_ours())
    world.ledger = ledger_json(
        [
            (
                "November 20, 2026",
                "ZZFUT",
                "Future Co",
                "X0040",
                "Old",
                "r",
                press_ref("November 6, 2026"),
            )
        ]
    )
    first, _, _ = run(world, tmp_path=tmp_path)
    again, _, _ = run(world, tmp_path=tmp_path, now=NOW + timedelta(seconds=1))
    assert first.proposal_path != again.proposal_path
    assert {p.name for p in (tmp_path / "proposals").iterdir()} == {
        first.proposal_path.name,
        again.proposal_path.name,
    }
    clash = tmp_path / "proposals" / first.proposal_path.name
    before = clash.read_bytes()
    third, _, _ = run(world, tmp_path=tmp_path)  # same second as the first
    assert "proposal_not_written" in codes(third) and clash.read_bytes() == before


# --------------------------------------------------------------------------- the stored proposal


def test_without_a_recorded_check_id_the_message_falls_back_to_the_file_path(tmp_path):
    ours = base_ours()
    world = World(ours)
    world.spy = holdings_xlsx([t for t in ours if t != "X0010"] + ["ZZAGR"], "As of 06-Nov-2026")
    world.changes = changes_csv([(date(2026, 11, 2), ["ZZAGR"], ["X0010"])])
    store = FakeWeeklyStore(ours, fail_record=True)
    result, _, notifier = run(world, store=store, tmp_path=tmp_path)
    assert result.check_id is None and result.proposal_path is not None
    message = notifier.messages[0]
    assert "membership proposal --id" not in message
    assert result.proposal_path.name in message


def test_the_stored_proposal_comes_back_byte_for_byte_and_is_checked_against_its_hash():
    data = b"effective_date,action,ticker\n2026-11-02,add,AAA\n"
    digest = hashlib.sha256(data).hexdigest()
    assert mw.decode_stored_proposal(7, gzip.compress(data), digest) == data
    with pytest.raises(ValueError, match="sha256"):
        mw.decode_stored_proposal(7, gzip.compress(data), "0" * 64)
    with pytest.raises(mw.ProposalNotFound, match="no bytes"):
        mw.decode_stored_proposal(7, None, digest)


def _proposal_args(tmp_path: Path, out: str = "p.csv", check: int = 7):
    return argparse.Namespace(id=check, out=tmp_path / out)


def test_the_proposal_command_writes_the_stored_bytes_and_will_not_overwrite(
    tmp_path, monkeypatch, capsys
):
    data = b"effective_date,action,ticker\n2026-11-02,add,AAA\n"
    monkeypatch.setattr(cmd_membership, "read_stored_proposal", lambda check_id: data)
    args = _proposal_args(tmp_path, "sub/p.csv")
    assert cmd_membership.run_proposal(args) == 0
    assert args.out.read_bytes() == data
    assert str(args.out) in capsys.readouterr().out
    args.out.write_bytes(b"edited by a human")
    assert cmd_membership.run_proposal(args) == 1  # refuses to overwrite the edited file
    assert args.out.read_bytes() == b"edited by a human"


def test_the_proposal_command_reports_a_missing_proposal_and_hides_connection_text(
    tmp_path, monkeypatch, capsys
):
    def missing(check_id):
        raise mw.ProposalNotFound(f"check {check_id} stored no proposal")

    monkeypatch.setattr(cmd_membership, "read_stored_proposal", missing)
    assert cmd_membership.run_proposal(_proposal_args(tmp_path)) == 1
    assert "stored no proposal" in capsys.readouterr().err

    def broken(check_id):
        raise RuntimeError("postgresql://user:secret@host/db refused")

    monkeypatch.setattr(cmd_membership, "read_stored_proposal", broken)
    assert cmd_membership.run_proposal(_proposal_args(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "secret" not in err and "RuntimeError" in err
    assert not (tmp_path / "p.csv").exists()


def test_the_proposal_command_is_registered_with_help():
    import io as _io
    from contextlib import redirect_stdout

    parser = argparse.ArgumentParser()
    from ggTrader.cli.cmd_db import register_db_parser

    register_db_parser(parser.add_subparsers(dest="command"))
    buffer = _io.StringIO()
    with pytest.raises(SystemExit) as caught, redirect_stdout(buffer):
        parser.parse_args(["db", "membership", "proposal", "--help"])
    assert caught.value.code == 0 and "--id" in buffer.getvalue() and "--out" in buffer.getvalue()
