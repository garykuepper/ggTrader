"""Pins the weekly membership job's findings and message text, path by path (refactor II, D7b).

Each scenario runs ``run_weekly`` against the offline fakes (no database, no network, no Telegram)
and records everything a person or the stored history would see: the full finding list (severity,
code, text), the Telegram message, the stored ``weekly`` detail and the snapshot rows. The record
is compared with ``golden/membership_weekly_pins.json``, written once from the code *before* the
job was split into steps; a refactor of ``membership_weekly.py`` must leave it byte-identical.
Never regenerate the golden to make a test pass.
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import requests

from ggTrader.data import membership_weekly as mw
from ggTrader.data.membership_holdings import HoldingsResult
from ggTrader.data.membership_weekly import WeeklyRun
from ggTrader.paper.notifier import TelegramNotifier
from tests.data.membership_weekly_fakes import (
    NOW,
    FakeWeeklyStore,
    RecordingNotifier,
    World,
    base_ours,
    components_csv,
    current_json,
    ledger_json,
    ledger_wikitext,
    no_real_network,
    parse_json,
    press_listing_html,
    press_ref,
    press_release_html,
    reply,
)
from tests.data.test_membership_upstream import FakeResponse
from tests.data.test_membership_weekly import fabricated_mismatch, run
from tests.data.xlsx_builder import holdings_xlsx

GOLDEN = Path(__file__).parent / "golden" / "membership_weekly_pins.json"
OKTA_SLUG = "okta-set-to-join-sp-500-others-to-join-sp-midcap-400-302925405"
OKTA_CARD = ("Oct 30, 2026, 17:51 ET", OKTA_SLUG, "Okta Set to Join S&amp;P 500; Others")
TMP = "<TMP>"

Outcome = tuple[WeeklyRun, FakeWeeklyStore, RecordingNotifier]


@pytest.fixture(autouse=True)
def _nothing_can_reach_the_network_or_telegram(monkeypatch):
    """Any real HTTP, socket or Telegram call made by a test raises ``NoNetwork``."""
    monkeypatch.setattr(requests, "get", no_real_network)
    monkeypatch.setattr(requests, "post", no_real_network)
    monkeypatch.setattr(requests.sessions.Session, "request", no_real_network)
    monkeypatch.setattr(socket.socket, "connect", no_real_network)
    monkeypatch.setattr(TelegramNotifier, "send", no_real_network)


def _announce(world: World, rows: list[tuple[str, ...]]) -> None:
    world.press_listing = press_listing_html([OKTA_CARD])
    world.press_releases = {"302925405": press_release_html(rows)}


def _ledger(world: World, *rows: tuple[str, ...]) -> None:
    world.ledger = ledger_json(list(rows))


# --------------------------------------------------------------------------- scenarios


def clean(tmp: Path) -> Outcome:
    return run(World(base_ours()), tmp_path=tmp)


def database_unreadable(tmp: Path) -> Outcome:
    world = World(base_ours())
    return run(world, FakeWeeklyStore(world.ours, fail_latest=True), tmp_path=tmp)


def no_vintage(tmp: Path) -> Outcome:
    world = World(base_ours())
    store = FakeWeeklyStore(world.ours)
    store._vintage = None
    return run(world, store, tmp_path=tmp)


def record_failed(tmp: Path) -> Outcome:
    world = World(base_ours())
    return run(world, FakeWeeklyStore(world.ours, fail_record=True), tmp_path=tmp)


def spy_http_failed(tmp: Path) -> Outcome:
    world = World(base_ours())
    world.errors["spy"] = reply(b"denied", 403)
    return run(world, tmp_path=tmp)


def spy_stale(tmp: Path) -> Outcome:
    world = World(base_ours())
    world.spy = holdings_xlsx(base_ours(), "As of 20-Oct-2026")
    return run(world, tmp_path=tmp)


def spy_mismatch_proposal(tmp: Path) -> Outcome:
    return fabricated_mismatch(tmp)[2:]


def spy_mismatch_with_press_flood(tmp: Path) -> Outcome:
    rows = [("Aug 3, 2026", "S&P 500", "Addition", f"C{i}", f"ZQ{i:03d}") for i in range(60)]
    return fabricated_mismatch(tmp, tweak=lambda w: _announce(w, rows))[2:]


def proposal_not_written(tmp: Path) -> Outcome:
    world = World(base_ours())
    _ledger(
        world,
        ("November 6, 2026", "ZZANN", "Ann Co", "X0040", "Old", "r", press_ref("November 6, 2026")),
    )
    run(world, tmp_path=tmp)  # the first run writes the file; the second may not overwrite it
    return run(world, tmp_path=tmp)


def ledger_row_in_the_future(tmp: Path) -> Outcome:
    world = World(base_ours())
    _ledger(world, ("November 20, 2026", "ZZFUT", "Future Co", "X0040", "Old", "r", ""))
    return run(world, tmp_path=tmp)


def ledger_row_on_announcement_date(tmp: Path) -> Outcome:
    world = World(base_ours())
    _ledger(
        world,
        (
            "November 6, 2026",
            "ZZANN",
            "Ann Co",
            "X0040",
            "Old",
            "r",
            press_ref("November 6, 2026"),
        ),
    )
    return run(world, tmp_path=tmp)


def count_out_of_range(tmp: Path) -> Outcome:
    ours = base_ours(476)
    world = World(ours)
    spy = [t for t in ours if t != "X0007"] + ["ZZBACK"]
    world.spy = holdings_xlsx(spy, "As of 06-Nov-2026")
    _ledger(
        world,
        ("November 4, 2026", "ZZBACK", "Back Co", "", "", "r", ""),
        ("November 2, 2026", "", "", "X0007", "Gone Co", "r", ""),
    )
    return run(world, tmp_path=tmp)


def _source_down(key: str, failure: Exception | FakeResponse):
    def scenario(tmp: Path) -> Outcome:
        world = World(base_ours())
        world.errors[key] = failure
        return run(world, tmp_path=tmp)

    return scenario


def source_structure_changed(tmp: Path) -> Outcome:
    from tests.data.membership_weekly_fakes import REAL_HEADER

    world = World(base_ours())
    world.ledger = parse_json(
        ledger_wikitext([], header=REAL_HEADER.replace("| Reason", "| Why")), revid=7
    )
    return run(world, tmp_path=tmp)


def source_stale(tmp: Path) -> Outcome:
    world = World(base_ours())
    world.ledger_edit = "2026-10-25T23:59:00Z"
    world.commit_date = "2026-10-01T00:00:00Z"
    return run(world, tmp_path=tmp)


def source_age_unknown(tmp: Path) -> Outcome:
    world = World(base_ours())
    world.errors["revisions"] = reply(b"rate limited", 429)
    return run(world, tmp_path=tmp)


def ledger_with_unreadable_old_rows(tmp: Path) -> Outcome:
    world = World(base_ours())
    world.ledger = parse_json(ledger_wikitext([]).replace("ZA100", "ZA 100!"))
    return run(world, tmp_path=tmp)


def other_lists_differ(tmp: Path) -> Outcome:
    ours = base_ours()
    world = World(ours)
    world.current = current_json([t.replace(".", "-") for t in ours] + ["ZZEXT"])
    world.components = components_csv([t for t in ours if t != "X0003"], date(2026, 11, 8))
    return run(world, tmp_path=tmp)


def press_change_not_in_list(tmp: Path) -> Outcome:
    world = World(base_ours())
    _announce(
        world,
        [
            ("Oct 20, 2026", "S&P 500", "Addition", "Ghost", "GHOST"),
            ("Oct 20, 2026", "S&P 500", "Deletion", "Kept", "X0002"),
            ("Dec 1, 2026", "S&P 500", "Addition", "Later", "LATER"),
        ],
    )
    return run(world, tmp_path=tmp)


def press_listing_down(tmp: Path) -> Outcome:
    world = World(base_ours())
    _announce(world, [("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001")])
    world.errors["press_listing"] = reply(b"down", 503)
    return run(world, tmp_path=tmp)


def press_page_down(tmp: Path) -> Outcome:
    world = World(base_ours())
    _announce(world, [("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001")])
    world.press_releases = {}
    return run(world, tmp_path=tmp)


def press_page_changed(tmp: Path) -> Outcome:
    world = World(base_ours())
    _announce(world, [("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001")])
    world.press_releases["302925405"] = b"<html><body class='x'>redesign</body></html>"
    return run(world, tmp_path=tmp)


def press_listing_changed(tmp: Path) -> Outcome:
    world = World(base_ours())
    _announce(world, [("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001")])
    world.press_listing = b"<html>redesign</html>"
    return run(world, tmp_path=tmp)


def press_quiet_and_archived(tmp: Path) -> Outcome:
    world = World(base_ours())
    _announce(
        world,
        [
            ("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001"),
            ("Oct 20, 2026", "S&P 500", "Deletion", "Gone", "GONE"),
        ],
    )
    return run(world, tmp_path=tmp)


def everything_down(tmp: Path) -> Outcome:
    """Every fetched source and the press pages unavailable at once: the repeated 'not fetched'."""
    world = World(base_ours())
    _announce(world, [("Oct 20, 2026", "S&P 500", "Addition", "X", "X0001")])
    world.errors["ledger"] = requests.ConnectionError("no route")
    world.errors["current"] = reply(b"nope", 503)
    world.errors["components"] = FakeResponse(200, b"x" * 10, {"content-length": "99"})
    world.errors["changes"] = reply(b"gone", 404)
    world.errors["press_listing"] = reply(b"down", 503)
    return run(world, tmp_path=tmp)


SCENARIOS: dict[str, Callable[[Path], Outcome]] = {
    "clean": clean,
    "database_unreadable": database_unreadable,
    "no_vintage": no_vintage,
    "record_failed": record_failed,
    "spy_http_failed": spy_http_failed,
    "spy_stale": spy_stale,
    "spy_mismatch_proposal": spy_mismatch_proposal,
    "spy_mismatch_with_press_flood": spy_mismatch_with_press_flood,
    "proposal_not_written": proposal_not_written,
    "ledger_row_in_the_future": ledger_row_in_the_future,
    "ledger_row_on_announcement_date": ledger_row_on_announcement_date,
    "count_out_of_range": count_out_of_range,
    "source_missing_ledger": _source_down("ledger", requests.ConnectionError("no route")),
    "source_missing_current": _source_down("current", reply(b"nope", 503)),
    "source_missing_components": _source_down(
        "components", FakeResponse(200, b"x" * 10, {"content-length": "99"})
    ),
    "source_missing_changes": _source_down("changes", reply(b"gone", 404)),
    "source_structure_changed": source_structure_changed,
    "source_stale": source_stale,
    "source_age_unknown": source_age_unknown,
    "ledger_with_unreadable_old_rows": ledger_with_unreadable_old_rows,
    "other_lists_differ": other_lists_differ,
    "press_change_not_in_list": press_change_not_in_list,
    "press_listing_down": press_listing_down,
    "press_page_down": press_page_down,
    "press_page_changed": press_page_changed,
    "press_listing_changed": press_listing_changed,
    "press_quiet_and_archived": press_quiet_and_archived,
    "everything_down": everything_down,
}


# --------------------------------------------------------------------------- the record


def _check(store: FakeWeeklyStore) -> HoldingsResult | None:
    return store.runs[0][0] if store.runs else None


def record_of(outcome: Outcome, tmp: Path) -> dict[str, Any]:
    """Everything the run decided, in plain JSON types (the temp directory made constant)."""
    result, store, notifier = outcome
    check = _check(store)
    record: dict[str, Any] = {
        "status": result.status,
        "exit_code": result.exit_code,
        "findings": [[f.severity, f.code, f.text] for f in result.findings],
        "message": result.message,
        "messages_sent": notifier.messages,
        "delivered": result.delivered,
        "load_id": result.load_id,
        "check_id": result.check_id,
        "record_error": result.record_error,
        "floor": result.floor.isoformat() if result.floor else None,
        "spy_outcome": None if result.spy is None else result.spy.outcome,
        "proposal": None
        if result.proposal is None
        else {
            "rows": len(result.proposal.rows),
            "needs_confirmation": result.proposal.n_needing_confirmation,
            "count_problems": list(result.proposal.count_problems),
            "file": None if result.proposal_path is None else result.proposal_path.name,
            "sha256": result.proposal_sha256,
            "csv_sha256": None
            if result.proposal_path is None
            else hashlib.sha256(result.proposal_path.read_bytes()).hexdigest(),
        },
        "n_press_changes": len(result.press),
        "source_names": list(result.sources),
        "sources": {
            name: {
                "url": row.url,
                "status": row.status,
                "fetched_at": row.fetched_at.isoformat(),
                "last_change": row.last_change.isoformat() if row.last_change else None,
                "raw_sha256": row.raw_sha256,
                "size_bytes": row.size_bytes,
                "detail": row.detail,
            }
            for name, row in result.sources.items()
        },
        "stored": None
        if check is None
        else {
            "outcome": check.outcome,
            "matched_load_id": check.matched_load_id,
            "weekly": check.detail.get("weekly"),
        },
        "recorded_source_names": [] if not store.runs else [r.source for r in store.runs[0][1]],
    }
    text = json.dumps(record, sort_keys=True, default=str, ensure_ascii=False)
    return _without_zip_clock(json.loads(text.replace(str(tmp), TMP)))


def _without_zip_clock(node: Any) -> Any:
    """The fake SPY workbook is a zip stamped with the time it was built, so its hash varies."""
    if isinstance(node, dict):
        fixed = {k: _without_zip_clock(v) for k, v in node.items()}
        spy = fixed.get("spy_holdings")
        if isinstance(spy, dict) and "raw_sha256" in spy:
            spy["raw_sha256"] = "<zip-clock>"
        return fixed
    if isinstance(node, list):
        return [_without_zip_clock(v) for v in node]
    return node


def all_records(base: Path) -> dict[str, Any]:
    out = {}
    for name, scenario in SCENARIOS.items():
        tmp = base / name
        tmp.mkdir()
        out[name] = record_of(scenario(tmp), tmp)
    return out


# --------------------------------------------------------------------------- the tests


def test_the_golden_covers_exactly_the_scenarios():
    assert sorted(json.loads(GOLDEN.read_text())) == sorted(SCENARIOS)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_each_path_through_the_weekly_job_keeps_its_findings_and_message(name, tmp_path):
    expected = json.loads(GOLDEN.read_text())[name]
    assert record_of(SCENARIOS[name](tmp_path), tmp_path) == expected


def test_the_pinned_paths_include_the_repeated_not_fetched_and_structure_findings():
    """A guard on the golden itself: the near-identical findings are really exercised."""
    golden = json.loads(GOLDEN.read_text())
    found = {
        (code, text.split(":")[0])
        for record in golden.values()
        for _, code, text in record["findings"]
        if code in {"source_missing", "press_unavailable", "press_structure", "source_structure"}
    }
    assert {c for c, _ in found} == {
        "source_missing",
        "press_unavailable",
        "press_structure",
        "source_structure",
    }
    names = {n for _, n in found}
    assert {"sp_press_listing", "sp_press_302925405", "wikipedia_ledger"} <= names


# ------------------------------------------- evaluation order and call-time lookup (fix round 1)


class _TruthRaises:
    """A stand-in for a directory whose truthiness is never to be asked (it raises)."""

    def __bool__(self) -> bool:
        raise RuntimeError("the proposal directory was evaluated")


def test_without_a_vintage_the_proposal_directory_is_never_evaluated(tmp_path):
    """The directory default is resolved only when a vintage is compared, after the findings."""
    world = World(base_ours())
    store = FakeWeeklyStore(world.ours)
    store._vintage = None
    notifier = RecordingNotifier()
    result = mw.run_weekly(store, notifier, get=world, now=lambda: NOW, proposal_dir=_TruthRaises())
    assert "no_vintage" in {f.code for f in result.findings}
    assert result.status == "refuse" and len(store.runs) == 1
    assert len(notifier.messages) == 1


@pytest.mark.parametrize("parser", ["parse_ledger", "parse_current_list"])
def test_a_parser_is_looked_up_by_module_name_when_the_source_is_parsed(
    parser, monkeypatch, tmp_path
):
    """A parser replaced after import is the one used; its failure reads as ``internal``."""

    def boom(raw: bytes) -> Any:
        raise RuntimeError("parser bug")

    monkeypatch.setattr(mw, parser, boom)
    result, _, _ = run(World(base_ours()), tmp_path=tmp_path)
    structure = [f for f in result.findings if f.code == "source_structure"]
    assert len(structure) == 1
    assert "(internal): RuntimeError: parser bug" in structure[0].text


def test_the_fetches_run_in_the_order_spy_then_sources_then_press(monkeypatch, tmp_path):
    log: list[str] = []

    def logged(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log.append(name)
            return original(*args, **kwargs)

        return wrapper

    for attr, name in (
        ("fetch_holdings", "spy"),
        ("fetch_sources", "sources"),
        ("fetch_press", "press"),
    ):
        monkeypatch.setattr(mw, attr, logged(name, getattr(mw, attr)))
    run(World(base_ours()), tmp_path=tmp_path)
    assert log == ["spy", "sources", "press"]
