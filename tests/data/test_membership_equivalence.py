"""The new door returns exactly what the old file functions return, for vintage 1.

Compared against ``ggTrader.data.core.index_constituents`` and the pure helpers in
``ggTrader.lab.universe``: every one of the 2,712 snapshot dates, every calendar day from
1996-01-02 to 2026-07-02 (weekends included), 1,500 random timestamps (naive, UTC, Pacific and
Tokyo, past both ends of the file), and 1,300 random windows. Two backends are checked: the frozen
fixture, and the database source fed by an in-memory store holding the same vintage (the real
database is compared in ``test_membership_db_integration.py``). The two must also agree with
each other. Fast tier: no database, no network.
"""

from __future__ import annotations

import gzip
import random
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ggTrader.data.core.index_constituents import (
    SP500_HISTORY_CSV,
    all_members_between,
    load_sp500_history,
    normalize_yf_ticker,
    sp500_members_asof,
    universe_all_between,
    universe_members_asof,
)
from ggTrader.data.membership import (
    DbMembershipSource,
    FixtureMembershipSource,
    MembershipSource,
)
from ggTrader.data.universe_algebra import parse_history_csv
from ggTrader.lab.universe import check_constituent_age, membership_mask, pit_mask
from tests.data.membership_fakes import FakeMembershipStore

pytestmark = pytest.mark.skipif(
    not SP500_HISTORY_CSV.exists(), reason="vintage-1 constituent file not present"
)

V1 = "sp500_v1_2026-06-02"
#: The old reader: the frozen file parsed straight from its path (no source selection).
OLD_TABLE = load_sp500_history(SP500_HISTORY_CSV)
FIRST = date(1996, 1, 2)
LAST_SNAPSHOT = date(2026, 6, 2)
CALENDAR_END = date(2026, 7, 2)
N_SNAPSHOT_DATES = 2712
N_RANDOM_STAMPS = 1500
N_RANDOM_WINDOWS = 600
TIMEZONES = (None, "UTC", "US/Pacific", "Asia/Tokyo")


@pytest.fixture(scope="module", params=["fixture", "db-in-memory"])
def source(request: pytest.FixtureRequest) -> MembershipSource:
    """Each backend in turn, both holding vintage 1."""
    return _backend(request.param)


def _backend(kind: str) -> MembershipSource:
    if kind == "fixture":
        return FixtureMembershipSource(V1)
    store = FakeMembershipStore()
    store.add(parse_history_csv(gzip.decompress(SP500_HISTORY_CSV.read_bytes())))
    return DbMembershipSource("latest", store=store)


@pytest.fixture(scope="module")
def old_history() -> pd.DataFrame:
    return OLD_TABLE


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """('ok', value) or ('error', ValueError class name): compare results and failures alike."""
    try:
        return "ok", call()
    except ValueError as exc:
        return "error", type(exc).__name__


@pytest.fixture(scope="module")
def calendar_days() -> list[date]:
    return [FIRST + timedelta(days=n) for n in range((CALENDAR_END - FIRST).days + 1)]


@pytest.fixture(scope="module")
def old_calendar(calendar_days: list[date]) -> dict[date, list[str]]:
    """The old reader's answer for every calendar day, computed once for both backends."""
    return {d: sp500_members_asof(pd.Timestamp(d), OLD_TABLE) for d in calendar_days}


def _localize(stamp: pd.Timestamp, zone: str) -> pd.Timestamp:
    """Attach a zone, resolving a daylight-saving gap or overlap instead of raising."""
    return stamp.tz_localize(zone, ambiguous=True, nonexistent="shift_forward")


def _random_stamps() -> list[pd.Timestamp]:
    rng = random.Random(20261108)
    out = []
    for _ in range(N_RANDOM_STAMPS):
        day = date(1995, 12, 1) + timedelta(
            days=rng.randint(0, (date(2029, 1, 1) - date(1995, 12, 1)).days)
        )
        stamp = pd.Timestamp(day) + pd.Timedelta(
            hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
        )
        zone = rng.choice(TIMEZONES)
        out.append(stamp if zone is None else _localize(stamp, zone))
    return out


# ---- the history table -----------------------------------------------------------------------


def test_history_frame_equals_load_sp500_history(
    source: MembershipSource, old_history: pd.DataFrame
) -> None:
    pd.testing.assert_frame_equal(source.history(), old_history, check_exact=True)
    assert source.history().index.equals(old_history.index)
    assert len(source.history()) == N_SNAPSHOT_DATES


def test_coverage_matches_the_old_table(
    source: MembershipSource, old_history: pd.DataFrame
) -> None:
    assert source.coverage_start == old_history.index[0].date() == FIRST
    assert source.coverage_end == old_history.index[-1].date() == LAST_SNAPSHOT
    assert source.snapshot_dates() == [t.date() for t in old_history.index]


# ---- members as of a day ---------------------------------------------------------------------


def test_every_snapshot_date_matches_the_old_reader(
    source: MembershipSource, old_history: pd.DataFrame
) -> None:
    checked = 0
    for stamp in old_history.index:
        old = sp500_members_asof(stamp, OLD_TABLE)
        assert source.members_asof(stamp.date()) == old
        assert source.members_asof(stamp) == old
        assert universe_members_asof("sp500", stamp, OLD_TABLE) == old
        day, tickers = source.snapshot_asof(stamp.date())
        assert (day, tickers) == (stamp.date(), old)
        checked += 1
    assert checked == N_SNAPSHOT_DATES


def test_every_calendar_day_matches_the_old_reader(
    source: MembershipSource, calendar_days: list[date], old_calendar: dict[date, list[str]]
) -> None:
    weekends = sum(d.weekday() >= 5 for d in calendar_days)
    assert len(calendar_days) > 11_000 and weekends > 3_000  # the days tested include weekends
    mismatches = [d for d in calendar_days if source.members_asof(d) != old_calendar[d]]
    assert mismatches == []


def test_the_date_edges_match_including_the_errors(source: MembershipSource) -> None:
    edges = [
        FIRST - timedelta(days=1),  # before the file: both raise
        FIRST,
        FIRST + timedelta(days=1),
        LAST_SNAPSHOT - timedelta(days=1),
        LAST_SNAPSHOT,
        LAST_SNAPSHOT + timedelta(days=1),  # carried forward
        LAST_SNAPSHOT + timedelta(days=400),
        date(2040, 1, 1),
    ]
    for day in edges:
        assert _outcome(lambda: source.members_asof(day)) == _outcome(
            lambda: sp500_members_asof(pd.Timestamp(day), OLD_TABLE)
        ), day
    assert _outcome(lambda: source.members_asof(FIRST - timedelta(days=1)))[0] == "error"


def test_random_timestamps_in_any_zone_match_the_old_reader(source: MembershipSource) -> None:
    stamps = _random_stamps()
    assert len(stamps) >= 1000
    kinds = {_outcome(lambda: sp500_members_asof(s, OLD_TABLE))[0] for s in stamps}
    assert kinds == {"ok", "error"}  # the sample reaches both before the first snapshot and past it
    for stamp in stamps:
        assert _outcome(lambda: source.members_asof(stamp)) == _outcome(
            lambda: sp500_members_asof(stamp, OLD_TABLE)
        ), stamp


# ---- the union over a window -----------------------------------------------------------------


def _windows() -> list[tuple[date, date]]:
    rng = random.Random(1108)
    out = [
        (FIRST, FIRST),
        (FIRST, LAST_SNAPSHOT),
        (LAST_SNAPSHOT, LAST_SNAPSHOT),
        (LAST_SNAPSHOT, LAST_SNAPSHOT + timedelta(days=90)),
        (LAST_SNAPSHOT + timedelta(days=1), LAST_SNAPSHOT + timedelta(days=400)),
        (FIRST - timedelta(days=1), FIRST + timedelta(days=30)),  # starts before the file
        (date(2021, 1, 31), date(2026, 4, 30)),  # the pinned 17-fold window
    ]
    for _ in range(N_RANDOM_WINDOWS):
        start = date(1995, 12, 1) + timedelta(days=rng.randint(0, 11_500))
        out.append((start, start + timedelta(days=rng.randint(0, 4_000))))
    for _ in range(100):  # end before start
        start = date(1996, 1, 2) + timedelta(days=rng.randint(100, 11_000))
        out.append((start, start - timedelta(days=rng.randint(1, 90))))
    return out


@pytest.fixture(scope="module")
def old_windows() -> list[tuple[date, date, tuple[str, Any], tuple[str, Any] | None]]:
    """The old functions' answer per window, computed once for both backends.

    ``universe_all_between`` (a thin wrapper over the same function) is asked for every fifth.
    """
    out = []
    fixture = FixtureMembershipSource(V1)
    for n, (start, end) in enumerate(_windows()):
        ts, te = pd.Timestamp(start), pd.Timestamp(end)
        wrapped = (
            _outcome(lambda: universe_all_between("sp500", ts, te, membership=fixture))
            if n % 5 == 0
            else None
        )
        out.append((start, end, _outcome(lambda: all_members_between(ts, te, OLD_TABLE)), wrapped))
    return out


def test_windows_match_all_members_between_and_universe_all_between(
    source: MembershipSource, old_windows: list[tuple[date, date, Any, Any]]
) -> None:
    assert len(old_windows) >= 700
    assert {"ok", "error"} <= {old[0] for _, _, old, _ in old_windows}
    for start, end, old, wrapped in old_windows:
        assert _outcome(lambda: source.all_members_between(start, end)) == old, (start, end)
        assert (
            _outcome(lambda: source.all_members_between(pd.Timestamp(start), pd.Timestamp(end)))
            == old
        )
        if wrapped is not None:
            assert wrapped == old


# ---- the pure helpers the lab feeds from the table --------------------------------------------


def test_membership_mask_and_pit_mask_equal_on_the_new_table(
    source: MembershipSource, old_history: pd.DataFrame
) -> None:
    index = pd.bdate_range("2019-03-01", "2026-06-02")
    members = {
        normalize_yf_ticker(t)
        for tickers in old_history.loc["2019-01-01":, "tickers"]
        for t in tickers
    }
    symbols = sorted(members)
    new_table = source.history()

    pd.testing.assert_frame_equal(
        membership_mask(index, symbols, new_table), membership_mask(index, symbols, old_history)
    )
    pd.testing.assert_frame_equal(
        membership_mask(index, symbols, new_table), membership_mask(index, symbols, None)
    )

    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        rng.uniform(10, 100, (len(index), len(symbols))), index=index, columns=symbols
    )
    close.iloc[:120, ::7] = np.nan  # a third of the names start late
    kwargs: dict[str, Any] = {"min_history_bars": 60, "max_stocks": 40, "on_stale": "ignore"}
    pd.testing.assert_frame_equal(
        pit_mask(close, new_table, **kwargs), pit_mask(close, old_history, **kwargs)
    )
    pd.testing.assert_frame_equal(
        pit_mask(close, new_table, **kwargs), pit_mask(close, None, **kwargs)
    )


def test_the_lab_age_record_is_the_same_for_the_new_table(
    source: MembershipSource, old_history: pd.DataFrame
) -> None:
    for asof in ("2026-06-02", "2026-11-08", "2020-01-01"):
        assert check_constituent_age(
            asof, source.history(), on_stale="ignore"
        ) == check_constituent_age(asof, old_history, on_stale="ignore")


# ---- the two backends agree with each other ---------------------------------------------------


def test_the_database_source_and_the_fixture_source_agree() -> None:
    fixture, database = _backend("fixture"), _backend("db-in-memory")
    pd.testing.assert_frame_equal(fixture.history(), database.history(), check_exact=True)
    a, b = fixture.vintage(), database.vintage()
    assert (a.content_sha256, a.coverage_start, a.coverage_end, a.index_name) == (
        b.content_sha256,
        b.coverage_start,
        b.coverage_end,
        b.index_name,
    )
    assert (a.kind, b.kind) == ("fixture", "db")
    assert fixture.snapshot_dates() == database.snapshot_dates()
    day = FIRST
    while day <= CALENDAR_END:
        assert fixture.members_asof(day) == database.members_asof(day)
        day += timedelta(days=1)
    for start, end in _windows()[:300]:
        assert _outcome(lambda: fixture.all_members_between(start, end)) == _outcome(
            lambda: database.all_members_between(start, end)
        )
