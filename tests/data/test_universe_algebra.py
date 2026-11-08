"""DB-free tests of the membership interval algebra, content hash, diff and refusal rules."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import random
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from ggTrader.cli.cmd_db import register_db_parser
from ggTrader.data.core.index_constituents import SP500_HISTORY_CSV, sp500_members_asof
from ggTrader.data.universe_algebra import (
    MAX_HISTORICAL_REWRITES,
    MAX_TAIL_CHANGES,
    HistoryValidationError,
    Interval,
    as_date,
    classify_changes,
    collapse,
    content_sha256,
    diff_histories,
    diff_intervals,
    history_members_asof,
    members_on,
    parse_history_csv,
    plan_supersession,
    raw_sha256,
    reconstruct,
    refusal_reasons,
)

D = date.fromisoformat
REAL_FILE = pytest.mark.skipif(not SP500_HISTORY_CSV.exists(), reason="vintage-1 file not present")


def _csv(history: dict[str, list[str]]) -> bytes:
    lines = ["date,tickers"] + [f'{d},"{",".join(t)}"' for d, t in history.items()]
    return ("\n".join(lines) + "\n").encode()


def _pool(n: int) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _random_history(seed: int, n_dates: int = 40, size: int = 485) -> dict[date, tuple[str, ...]]:
    """A history with churn, leave-and-rejoin and quiet spells (members stay within 480..520)."""
    rng = random.Random(seed)
    pool = _pool(560)
    members = set(rng.sample(pool, size))
    day = date(2000, 1, 3)
    out: dict[date, tuple[str, ...]] = {}
    for _ in range(n_dates):
        day += timedelta(days=rng.randint(1, 40))
        if rng.random() < 0.7:  # quiet spells repeat the previous snapshot
            for _ in range(rng.randint(1, 6)):
                members.discard(rng.choice(sorted(members)))
            outside = sorted(set(pool) - members)
            members.update(rng.sample(outside, min(len(outside), rng.randint(1, 6))))
            while len(members) < 482:
                members.add(rng.choice(sorted(set(pool) - members)))
        out[day] = tuple(sorted(members))
    return out


def _as_bytes(history: dict[date, tuple[str, ...]]) -> bytes:
    return _csv({d.isoformat(): list(t) for d, t in history.items()})


# ---- parsing and validation ------------------------------------------------------------


def test_parse_round_trips_a_valid_file() -> None:
    history = _random_history(1)
    assert parse_history_csv(_as_bytes(history)) == history


def _valid_two_rows() -> list[str]:
    full = ",".join(_pool(485))
    return ["date,tickers", f'2000-01-03,"{full}"', f'2000-02-03,"{full}"']


@pytest.mark.parametrize(
    "row, message",
    [
        ('2000-02-03,"' + ",".join(_pool(485) + ["T000"]) + '"', "duplicate"),
        ('2000-02-03,"' + ",".join(_pool(484) + [" T999"]) + '"', "padded"),
        ('2000-02-03,"' + ",".join(_pool(484) + [""]) + '"', "empty"),
        ('20xx-02-03,"' + ",".join(_pool(485)) + '"', "bad ISO date"),
    ],
)
def test_parse_refuses_malformed_rows(row: str, message: str) -> None:
    lines = _valid_two_rows()[:2] + [row]
    with pytest.raises(HistoryValidationError, match=message):
        parse_history_csv(("\n".join(lines) + "\n").encode())


def test_parse_refuses_a_wrong_header() -> None:
    lines = _valid_two_rows()
    lines[0] = "day,tickers"
    with pytest.raises(HistoryValidationError, match="header"):
        parse_history_csv(("\n".join(lines) + "\n").encode())


def test_parse_refuses_dates_that_do_not_increase() -> None:
    rows = {"2000-02-03": _pool(485), "2000-01-03": _pool(485)}
    with pytest.raises(HistoryValidationError, match="does not increase"):
        parse_history_csv(_csv(rows))


@pytest.mark.parametrize("size", [479, 521])
def test_parse_refuses_member_counts_outside_the_range(size: int) -> None:
    with pytest.raises(HistoryValidationError, match="outside"):
        parse_history_csv(_csv({"2000-01-03": _pool(size)}))


# ---- collapse and reconstruct ----------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_snapshots_to_intervals_to_snapshots_is_identity(seed: int) -> None:
    history = _random_history(seed)
    intervals = collapse(history)
    assert reconstruct(intervals, history) == history
    # half-open and minimal: no zero-length interval, no two intervals of a symbol touching
    by_symbol: dict[str, list[Interval]] = {}
    for interval in intervals:
        assert interval.valid_to is None or interval.valid_to > interval.valid_from
        by_symbol.setdefault(interval.symbol_raw, []).append(interval)
    for runs in by_symbol.values():
        for earlier, later in zip(runs, runs[1:]):
            assert earlier.valid_to is not None and earlier.valid_to < later.valid_from


def test_a_symbol_that_leaves_and_rejoins_has_two_intervals() -> None:
    base = tuple(_pool(485))
    without = tuple(t for t in base if t != "T010")
    history = {D("2000-01-03"): base, D("2000-02-01"): without, D("2000-03-01"): base}
    runs = [i for i in collapse(history) if i.symbol_raw == "T010"]
    assert runs == [
        Interval("T010", D("2000-01-03"), D("2000-02-01")),
        Interval("T010", D("2000-03-01"), None),
    ]


@pytest.mark.parametrize("seed", range(4))
def test_every_calendar_day_equals_the_latest_snapshot_carry_forward(seed: int) -> None:
    history = _random_history(seed)
    intervals = collapse(history)
    days = [min(history) - timedelta(days=3) + timedelta(days=n) for n in range(0, 1500, 7)]
    calendar = reconstruct(intervals, days)
    for day in days:
        assert calendar[day] == history_members_asof(history, day)
        assert calendar[day] == members_on(intervals, day)


@REAL_FILE
def test_vintage_one_round_trips_and_matches_the_project_reader() -> None:
    raw = gzip.decompress(SP500_HISTORY_CSV.read_bytes())
    history = parse_history_csv(raw)
    assert len(history) == 2712
    intervals = collapse(history)
    assert reconstruct(intervals, history) == history
    assert len(intervals) < 2000  # ~1.3k intervals, not 1.4M (date, symbol) rows
    probe = [D("1996-01-02"), D("2008-09-15"), D("2020-12-18"), D("2020-12-21"), D("2026-06-02")]
    calendar = reconstruct(intervals, probe + [D("2026-07-02")])
    for day in probe:
        assert list(calendar[day]) == sp500_members_asof(pd.Timestamp(day))
    assert calendar[D("2026-07-02")] == calendar[D("2026-06-02")]  # carry-forward past coverage


# ---- content hash ----------------------------------------------------------------------


def test_content_hash_is_the_documented_canonical_form() -> None:
    history = {D("2000-01-03"): ("B", "A"), D("2000-02-01"): ("A", "C")}
    expected = hashlib.sha256(b"2000-01-03,A,B\n2000-02-01,A,C\n").hexdigest()
    assert content_sha256(history) == expected


def test_content_hash_ignores_input_order_and_quoting_but_not_content() -> None:
    history = _random_history(3)
    shuffled = {d: tuple(reversed(t)) for d, t in reversed(list(history.items()))}
    assert content_sha256(shuffled) == content_sha256(history)
    assert content_sha256(history) == content_sha256(parse_history_csv(_as_bytes(history)))
    day = min(history)
    changed = dict(history)
    changed[day] = (*history[day][:-1], "ZZZ")
    assert content_sha256(changed) != content_sha256(history)


def test_raw_hash_sees_bytes_the_content_hash_does_not() -> None:
    history = _random_history(4)
    raw = _as_bytes(history)
    other = raw.replace(b"\n", b"\r\n")
    assert raw_sha256(raw) != raw_sha256(other)
    assert content_sha256(parse_history_csv(raw)) == content_sha256(parse_history_csv(other))


@REAL_FILE
def test_vintage_one_hashes_are_pinned() -> None:
    raw = gzip.decompress(SP500_HISTORY_CSV.read_bytes())
    assert raw_sha256(raw) == "bc9c2aaed27e70406a247cd3f75513e786b0f173c70c3e05c1e243bf25498787"
    assert content_sha256(parse_history_csv(raw)) == (
        "256e022dc1580a82aaef38d303a772611469bc6a705374704a79333eefcef71c"
    )


# ---- diff and rewrite detection --------------------------------------------------------


def _grown(history: dict[date, tuple[str, ...]], day: date, drop=(), add=()) -> dict:
    out = dict(history)
    last = out[max(out)]
    out[day] = tuple(sorted((set(last) - set(drop)) | set(add)))
    return out


def test_diff_histories_reports_dates_on_one_side_and_changed_dates() -> None:
    a = _random_history(5)
    b = dict(a)
    first = min(a)
    b[first] = (*a[first][:-1], "NEW")
    b[date(2030, 1, 1)] = a[max(a)]
    del b[sorted(a)[1]]
    diff = diff_histories(a, b)
    assert diff.only_in_b == [date(2030, 1, 1)]
    assert diff.only_in_a == [sorted(a)[1]]
    assert list(diff.changed) == [first]


def test_a_quiet_extension_is_a_tail_change_and_nothing_else() -> None:
    parent = _random_history(6)
    last = max(parent)
    new = _grown(parent, last + timedelta(days=30), drop=[parent[last][0]], add=["NEWCO"])
    c = classify_changes(collapse(parent), collapse(new), last)
    assert c.historical == () and c.renames == ()
    assert set(c.tail) == {parent[last][0], "NEWCO"}
    assert (
        refusal_reasons(
            parent_coverage=(min(parent), last),
            new_coverage=(min(new), max(new)),
            classification=c,
        )
        == []
    )


def test_redating_an_old_event_is_a_historical_rewrite() -> None:
    parent = _random_history(7)
    dates = sorted(parent)
    joined = next(iter(set(parent[dates[10]]) - set(parent[dates[9]])), None)
    assert joined is not None
    new = dict(parent)
    new[dates[10]] = tuple(t for t in parent[dates[10]] if t != joined)  # joins one snapshot later
    if len(new[dates[10]]) < 480:
        pytest.skip("random history too small")
    c = classify_changes(collapse(parent), collapse(new), dates[-1])
    assert joined in c.historical


def test_a_rename_is_neither_historical_nor_tail() -> None:
    parent = _random_history(8)
    dates = sorted(parent)
    old = parent[dates[-1]][0]
    new = {
        d: tuple(sorted("RENAMED" if t == old else t for t in tick)) for d, tick in parent.items()
    }
    c = classify_changes(collapse(parent), collapse(new), dates[-1])
    assert c.renames == ((old, "RENAMED"),)
    assert c.historical == () and c.tail == ()


def test_refusal_rules_fire_on_shrinking_and_on_large_rewrites() -> None:
    parent = _random_history(9)
    dates = sorted(parent)
    cov = (dates[0], dates[-1])
    shrunk = refusal_reasons(
        parent_coverage=cov,
        new_coverage=(dates[2], dates[-3]),
        classification=classify_changes([], [], dates[-1]),
    )
    assert any("coverage_start" in r for r in shrunk) and any("coverage_end" in r for r in shrunk)

    names = [f"X{i}" for i in range(MAX_HISTORICAL_REWRITES + 1)]
    old = [Interval(n, dates[0], dates[5]) for n in names]
    redated = [Interval(n, dates[0], dates[6]) for n in names]
    hist = classify_changes(old, redated, dates[-1])
    assert len(hist.historical) == MAX_HISTORICAL_REWRITES + 1
    assert any(
        "historical" in r
        for r in refusal_reasons(parent_coverage=cov, new_coverage=cov, classification=hist)
    )

    tail_names = [f"Y{i}" for i in range(MAX_TAIL_CHANGES + 1)]
    after = dates[-1] + timedelta(days=10)
    tail = classify_changes([], [Interval(n, after, None) for n in tail_names], dates[-1])
    assert any(
        "tail" in r
        for r in refusal_reasons(
            parent_coverage=cov, new_coverage=(dates[0], after), classification=tail
        )
    )


# ---- supersession: visible set at v equals the vintage's own intervals ------------------


def test_visibility_at_each_vintage_reproduces_that_vintage() -> None:
    rng = random.Random(11)
    store: list[dict] = []  # rows: interval_id, interval, load_id, superseded_by
    vintages: list[set[Interval]] = []
    history = _random_history(12)
    dates = sorted(history)
    for load_id in range(1, 7):
        history = dict(history)
        day = max(history) + timedelta(days=rng.randint(1, 30))
        last = history[max(history)]
        history[day] = tuple(
            sorted((set(last) - set(rng.sample(last, 3))) | {f"N{load_id}{k}" for k in range(3)})
        )
        if rng.random() < 0.5:  # a correction of old history, not just a new tail
            i = rng.randrange(1, len(dates) - 1)
            history[dates[i]] = tuple(sorted(set(history[dates[i]]) ^ {"FIX"}))
        intervals = collapse(history)
        visible = [
            (r["id"], r["iv"])
            for r in store
            if r["load"] <= load_id - 1 and (r["sup"] is None or r["sup"] > load_id - 1)
        ]
        to_sup, to_ins = plan_supersession(visible, intervals)
        for r in store:
            if r["id"] in to_sup:
                r["sup"] = load_id
        store.extend(
            {"id": len(store) + 1, "iv": iv, "load": load_id, "sup": None} for iv in to_ins
        )
        vintages.append(set(intervals))
        for v, expected in enumerate(vintages, start=1):
            seen = {r["iv"] for r in store if r["load"] <= v and (r["sup"] is None or r["sup"] > v)}
            assert seen == expected, f"vintage {v} after load {load_id}"
    only_1, only_6 = diff_intervals(vintages[0], vintages[-1])
    assert only_1 or only_6


def test_an_identical_interval_is_not_rewritten() -> None:
    history = _random_history(13)
    intervals = collapse(history)
    visible = list(enumerate(intervals, start=1))
    assert plan_supersession(visible, intervals) == ([], [])


# ---- date handling ---------------------------------------------------------------------


def test_sql_dates_refuse_timestamps_so_a_timezone_cannot_shift_the_day() -> None:
    assert as_date(date(2026, 6, 2)) == date(2026, 6, 2)
    for bad in (
        pd.Timestamp("2026-06-02", tz="UTC"),
        datetime(2026, 6, 2, tzinfo=timezone.utc),
        "2026-06-02",
    ):
        with pytest.raises(TypeError, match="datetime.date"):
            as_date(bad)


# ---- CLI wiring ------------------------------------------------------------------------


def _db_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ggt")
    register_db_parser(parser.add_subparsers(dest="command"))
    return parser


@pytest.mark.parametrize(
    "argv",
    [
        ["db", "load-sp500", "--help"],
        ["db", "membership", "--help"],
        ["db", "membership", "verify", "--help"],
    ],
)
def test_new_commands_answer_help(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _db_parser().parse_args(argv)
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_existing_db_subcommands_still_parse() -> None:
    parser = _db_parser()
    for name in ("diag", "clean", "truncate"):
        assert parser.parse_args(["db", name]).db_command == name
    assert parser.parse_args(["db", "compression", "--status"]).db_command == "compression"
    assert parser.parse_args(["db", "export", "-j", "2"]).db_command == "export"
    args = parser.parse_args(
        ["db", "load-sp500", "--from-file", "x.csv", "--retrieved-at", "2026-06-09T00:00:00Z"]
    )
    assert args.commit is False and args.retrieved_at.tzinfo is not None
    with pytest.raises(SystemExit):  # a retrieval time without a zone is refused
        parser.parse_args(["db", "load-sp500", "--from-file", "x", "--retrieved-at", "2026-06-09"])
