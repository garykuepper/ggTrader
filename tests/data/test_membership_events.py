"""Unit tests of the events-file path (curated vintages): parsing, applying, counts, evidence.

Fast tier, offline: pure functions only (``ggTrader.data.membership_events``), no database. The
real writes are tested on a scratch database in ``test_membership_events_integration.py``.
"""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import date
from pathlib import Path

import pytest

from ggTrader.data.membership_events import (
    EVENT_COLUMNS,
    EventsFileError,
    collect_evidence,
    parse_events_csv,
    plan_events,
    render_history_csv,
)
from ggTrader.data.universe_algebra import History, content_sha256, parse_history_csv

D = date.fromisoformat
TODAY = D("2026-11-08")
EVIDENCE = b"<html>press release</html>"
SHA = hashlib.sha256(EVIDENCE).hexdigest()
URL = "https://example.invalid/page"


def _parent() -> History:
    """503 members on the parent's last day, plus an older snapshot."""
    base = [f"S{n:03d}" for n in range(493)] + [
        "BLDR", "TAP", "TTD", "CTVA", "WBD", "PSKY", "KVUE", "EQR", "AAA", "BBB",
    ]  # fmt: skip
    older = tuple(sorted(base[:-1] + ["XXX"]))
    return {D("2026-08-05"): older, D("2026-08-18"): tuple(sorted(base))}


def _csv(*rows: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EVENT_COLUMNS)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode()


def _row(day: str, action: str, ticker: str, new: str = "", sha: str = SHA) -> tuple[str, ...]:
    return (day, action, ticker, day, URL, sha, "n", new)


#: The four real gap dates (adds, removes) and the member counts the design requires.
GAP = [
    ("2026-09-21", ("BE", "ILMN", "P"), ("TTD", "BLDR", "TAP"), 503),
    ("2026-10-01", ("VYLR",), (), 504),
    ("2026-10-06", ("TWLO", "SKYD"), ("CTVA", "WBD", "PSKY"), 503),
    ("2026-11-04", ("OKTA",), ("KVUE",), 503),
]


def _gap_rows() -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = [_row("2026-10-06", "rename", "PSKY", new="SKYD")]
    for day, adds, removes, _ in GAP:
        rows += [_row(day, "add", t) for t in adds] + [_row(day, "remove", t) for t in removes]
    return rows


def _plan(rows: list[tuple[str, ...]], parent: History | None = None):
    return plan_events(parent or _parent(), parse_events_csv(_csv(*rows)), today=TODAY)


def test_the_four_gap_dates_give_the_required_member_counts() -> None:
    plan = _plan(_gap_rows())
    assert plan.problems == []
    assert plan.parent_members == 503 and plan.parent_last_day == D("2026-08-18")
    assert [(c.day.isoformat(), c.members) for c in plan.days] == [
        (day, count) for day, _, _, count in GAP
    ]
    assert [c.net for c in plan.days] == [0, 1, -1, 0]


def test_the_504_day_is_real_and_keeps_its_own_snapshot_row() -> None:
    plan = _plan(_gap_rows())
    history = plan.history
    assert sorted(history) == [D("2026-08-05"), D("2026-08-18"), *(D(g[0]) for g in GAP)]
    assert len(history[D("2026-10-01")]) == 504  # Vylor joins before Corteva leaves
    assert "VYLR" in history[D("2026-10-01")] and "CTVA" in history[D("2026-10-01")]
    assert "CTVA" not in history[D("2026-10-06")] and "VYLR" in history[D("2026-10-06")]
    # Carry forward: parent rows are untouched and the 09-21 row differs only by its events.
    assert history[D("2026-08-18")] == _parent()[D("2026-08-18")]
    assert set(history[D("2026-09-21")]) - set(history[D("2026-08-18")]) == {"BE", "ILMN", "P"}


def test_the_generated_csv_is_the_same_format_as_the_other_vintages() -> None:
    plan = _plan(_gap_rows())
    raw = render_history_csv(plan.history)
    assert raw.startswith(b"date,tickers\n") and raw.endswith(b'"\n')
    assert parse_history_csv(raw) == plan.history  # the loader's own parser reads it back
    assert render_history_csv(parse_history_csv(raw)) == raw  # and rendering is stable
    parent_raw = render_history_csv(_parent())
    assert raw.startswith(parent_raw)  # the parent's rows come first, byte for byte
    assert content_sha256(plan.history) != content_sha256(_parent())


def test_rename_rows_become_symbol_changes_and_move_no_member_by_themselves() -> None:
    plan = _plan(_gap_rows())
    assert [(c.old_symbol, c.new_symbol, c.effective_date) for c in plan.symbol_changes] == [
        ("PSKY", "SKYD", D("2026-10-06"))
    ]
    assert SHA in plan.symbol_changes[0].evidence
    only_rename = _plan([_row("2026-10-06", "rename", "PSKY", new="SKYD")])
    assert any("nothing" in p for p in only_rename.problems)


# (name, rows, text that must appear in a problem)
REFUSED = [
    ("removal without a matching member", [_row("2026-09-21", "remove", "NOPE")], "no matching"),
    ("duplicate addition", [_row("2026-09-21", "add", "AAA")], "already a member"),
    (
        "the same name added and removed on one date",
        [_row("2026-09-21", "add", "ZZZ"), _row("2026-09-21", "remove", "ZZZ")],
        "both added and removed",
    ),
    (
        "count above the bound",
        [_row("2026-09-21", "add", f"N{n}") for n in range(18)],
        "outside 480..520",
    ),
    (
        "count below the bound",
        [_row("2026-09-21", "remove", t) for t in ("BLDR", "TAP")]
        + [_row("2026-10-01", "remove", f"S{n:03d}") for n in range(25)],
        "outside 480..520",
    ),
    ("event on a parent date", [_row("2026-08-18", "add", "ZZZ")], "not after the parent"),
    ("event before the parent end", [_row("2026-08-10", "add", "ZZZ")], "not after the parent"),
    ("event in the future", [_row("2026-12-01", "add", "ZZZ")], "after today"),
    (
        "rename whose new name is not a member",
        [_row("2026-10-06", "remove", "PSKY"), _row("2026-10-06", "rename", "PSKY", new="SKYD")],
        "does not hold SKYD",
    ),
    ("rename of a non-member", [_row("2026-10-06", "rename", "QQQQ", new="SKYD")], "not a member"),
]


@pytest.mark.parametrize(("name", "rows", "needle"), REFUSED, ids=[r[0] for r in REFUSED])
def test_invalid_events_are_refused_with_a_reason(name: str, rows, needle: str) -> None:
    problems = _plan(rows).problems
    assert any(needle in p for p in problems), (name, problems)


def test_a_wrong_count_names_the_date_and_the_count() -> None:
    plan = _plan([_row("2026-09-21", "add", f"N{n}") for n in range(18)])
    assert any("2026-09-21: 521 members" in p for p in plan.problems)


def test_events_that_change_nothing_are_refused() -> None:
    with pytest.raises(EventsFileError, match="no events"):
        parse_events_csv(_csv())
    renames_only = _plan([_row("2026-10-06", "rename", "PSKY", new="SKYD")])
    assert any("add or remove nothing" in p for p in renames_only.problems)


# (name, csv text, text in the error)
MALFORMED = [
    ("wrong header", b"date,action\n", "header must be"),
    ("bad action", _csv(_row("2026-09-21", "swap", "AAA")), "action 'swap'"),
    ("bad date", _csv(("21/09/2026", "add", "ZZZ", "2026-09-04", "u", SHA, "", "")), "ISO date"),
    ("no evidence url", _csv(("2026-09-21", "add", "ZZZ", "2026-09-04", "", SHA, "", "")), "url"),
    ("bad sha", _csv(("2026-09-21", "add", "ZZZ", "2026-09-04", "u", "abc", "", "")), "64"),
    (
        "announced after effective",
        _csv(("2026-09-21", "add", "ZZZ", "2026-09-22", "u", SHA, "", "")),
        "after it took effect",
    ),
    (
        "duplicate row",
        _csv(_row("2026-09-21", "add", "ZZZ"), _row("2026-09-21", "add", "ZZZ")),
        "duplicate",
    ),
    ("rename without a new ticker", _csv(_row("2026-10-06", "rename", "PSKY")), "new_ticker"),
    ("new_ticker on an add", _csv(_row("2026-09-21", "add", "ZZZ", new="Y")), "only for rename"),
    ("padded ticker", _csv(_row("2026-09-21", "add", " ZZZ")), "padded"),
]


@pytest.mark.parametrize(("name", "raw", "needle"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_a_malformed_events_file_is_refused(name: str, raw: bytes, needle: str) -> None:
    with pytest.raises(EventsFileError) as caught:
        parse_events_csv(raw)
    assert needle in str(caught.value), name


def test_applying_the_same_events_twice_is_refused_not_a_silent_no_op() -> None:
    rows = [_row("2026-09-21", "add", "ZZZ"), _row("2026-09-21", "remove", "AAA")]
    first = _plan(rows)
    assert first.problems == []
    assert content_sha256(first.history) != content_sha256(_parent())
    again = plan_events(first.history, parse_events_csv(_csv(*rows)), today=TODAY)
    assert any("not after the parent" in p for p in again.problems)


def test_evidence_is_found_by_hash_and_must_exist(tmp_path: Path) -> None:
    events = parse_events_csv(_csv(*_gap_rows()))
    files, problems, warnings = collect_evidence(b"events", "e.csv", events, None)
    assert [f.kind for f in files] == ["events_file"] and problems and not warnings
    (tmp_path / "page.html").write_bytes(EVIDENCE)
    (tmp_path / "extra.html").write_bytes(b"not cited")
    files, problems, warnings = collect_evidence(b"events", "e.csv", events, tmp_path)
    assert problems == []
    assert [(f.name, f.kind) for f in files] == [
        ("e.csv", "events_file"),
        ("page.html", "evidence"),
    ]
    assert files[1].source_url is not None and files[1].raw == EVIDENCE
    assert any("extra.html" in w for w in warnings)


def test_a_row_without_a_hash_is_a_warning_not_a_refusal(tmp_path: Path) -> None:
    events = parse_events_csv(_csv(_row("2026-09-21", "add", "ZZZ", sha="")))
    files, problems, warnings = collect_evidence(b"events", "e.csv", events, tmp_path)
    assert problems == [] and len(files) == 1 and "secondary" in warnings[0]


# ---- CLI wiring (no database is opened) ---------------------------------------------------------


def _load_args(*argv: str):
    import argparse

    from ggTrader.cli.cmd_db import register_db_parser

    parser = argparse.ArgumentParser(prog="ggt")
    register_db_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["db", "load-sp500", *argv])


def test_load_sp500_takes_a_file_or_an_events_file_not_both_and_one_of_them() -> None:
    assert _load_args("--from-events", "e.csv").from_events == Path("e.csv")
    assert (
        _load_args("--from-file", "f.csv", "--retrieved-at", "2026-06-09T00:00:00Z").from_events
        is None
    )
    with pytest.raises(SystemExit):
        _load_args("--from-file", "f.csv", "--from-events", "e.csv")
    with pytest.raises(SystemExit):
        _load_args()


def test_events_mode_refuses_upstream_options_and_force_before_opening_the_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ggTrader.cli import cmd_membership

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the database was opened")

    monkeypatch.setattr(cmd_membership, "connect", forbidden)
    for extra in (["--source", "fixture-import"], ["--upstream-commit", "a" * 40], ["--force"]):
        assert cmd_membership.run_load_sp500(_load_args("--from-events", "e.csv", *extra)) == 2
    assert "REFUSED" in capsys.readouterr().err
    needs_time = _load_args("--from-file", "f.csv")
    assert cmd_membership.run_load_sp500(needs_time) == 2  # --retrieved-at is required here
