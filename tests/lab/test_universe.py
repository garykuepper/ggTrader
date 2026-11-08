"""Vectorised PIT universe: parity with ``eligible_at`` plus synthetic edge cases."""

from __future__ import annotations

import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.data import snapshot as snap
from ggTrader.data.core import index_constituents as ic
from ggTrader.lab import data as lab_data
from ggTrader.lab import universe as uni
from ggTrader.lab.strategy import LabConfig

REAL_PARQUET = Path(snap.DEFAULT_SNAPSHOT_DIR) / "ohlcv_20261106.parquet"
needs_real = pytest.mark.skipif(
    not REAL_PARQUET.exists(),
    reason="frozen tape data/snapshots/ohlcv_20261106.parquet is absent (git-ignored; rebuild "
    "with scripts/make_tape_snapshot.py)",
)


def _history(rows: dict[str, list[str]]) -> pd.DataFrame:
    """A constituent table: snapshot date -> tickers (index-style symbols)."""
    return pd.DataFrame(
        {"tickers": [sorted(v) for v in rows.values()]},
        index=pd.DatetimeIndex(pd.to_datetime(list(rows)), tz="UTC", name="date"),
    )


def _bars(days: int, symbols: list[str], tz: str | None = "UTC") -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=days, tz=tz)
    return pd.DataFrame(1.0, index=idx, columns=symbols)


# ------------------------------------------------------------------ membership


def test_membership_enter_and_leave_same_day_effective() -> None:
    hist = _history({"2023-12-01": ["AAA", "BBB"], "2024-01-03": ["AAA", "CCC"]})
    close = _bars(5, ["AAA", "BBB", "CCC"])  # Jan 1..5 2024
    m = uni.membership_mask(close.index, list(close.columns), hist)
    assert m["AAA"].all()
    assert m["BBB"].tolist() == [True, True, False, False, False]  # left on the change date
    assert m["CCC"].tolist() == [False, False, True, True, True]  # joined on the change date


def test_membership_weekend_change_takes_effect_next_bar() -> None:
    hist = _history({"2023-12-01": ["AAA"], "2024-01-06": ["AAA", "BBB"]})  # Saturday
    idx = pd.bdate_range("2024-01-04", periods=4, tz="UTC")  # Thu, Fri, Mon, Tue
    m = uni.membership_mask(idx, ["AAA", "BBB"], hist)
    assert m["BBB"].tolist() == [False, False, True, True]


def test_membership_normalises_history_tickers_but_not_columns() -> None:
    hist = _history({"2023-12-01": ["BRK.B", "AAA"]})
    idx = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    m = uni.membership_mask(idx, ["BRK-B", "BRK.B", "AAA"], hist)
    assert m["BRK-B"].all() and m["AAA"].all()
    assert not m["BRK.B"].any()  # eligible_at never matched a raw-dotted column either


def test_membership_naive_index_is_read_as_utc() -> None:
    hist = _history({"2023-12-01": ["AAA"], "2024-01-03": ["AAA", "BBB"]})
    idx = pd.bdate_range("2024-01-02", periods=3)  # naive
    m = uni.membership_mask(idx, ["AAA", "BBB"], hist)
    assert m["BBB"].tolist() == [False, True, True]
    assert m.index.equals(idx)


def test_membership_before_first_snapshot_raises() -> None:
    hist = _history({"2024-01-03": ["AAA"]})
    with pytest.raises(ValueError, match="predates"):
        uni.membership_mask(pd.bdate_range("2024-01-01", periods=3, tz="UTC"), ["AAA"], hist)


def test_membership_duplicate_columns_rejected() -> None:
    hist = _history({"2023-12-01": ["AAA"]})
    with pytest.raises(ValueError, match="duplicate"):
        uni.membership_mask(pd.bdate_range("2024-01-02", periods=2, tz="UTC"), ["AAA", "AAA"], hist)


def test_static_membership_constant() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    m = uni.static_membership_mask(idx, ["AAA", "BBB", "BRK-B"], ["AAA", "BRK.B"])
    assert m.to_numpy().tolist() == [[True, False, True]] * 3


# ------------------------------------------------------------------ pit_mask rules


def test_min_history_counts_non_nan_closes_only() -> None:
    hist = _history({"2023-12-01": ["AAA"]})
    close = _bars(8, ["AAA"])
    close.iloc[[1, 2], 0] = np.nan  # a NaN gap must not count toward history
    m = uni.pit_mask(close, hist, min_history_bars=4, on_stale="ignore")
    # valid counts: 1,1,1,2,3,4,5,6 -> eligible from bar index 5
    assert m["AAA"].tolist() == [False] * 5 + [True] * 3


def test_dead_ticker_stays_eligible_until_removed() -> None:
    """eligible_at only asks for enough past history, so a delisted name remains in the mask."""
    hist = _history({"2023-12-01": ["AAA", "DEAD"]})
    close = _bars(6, ["AAA", "DEAD"])
    close.loc[close.index[3] :, "DEAD"] = np.nan
    m = uni.pit_mask(close, hist, min_history_bars=2, on_stale="ignore")
    assert m["DEAD"].tolist() == [False, True, True, True, True, True]


def test_max_stocks_cap_is_alphabetical_and_keeps_column_order() -> None:
    hist = _history({"2023-12-01": ["AAA", "BBB", "CCC", "DDD"]})
    close = _bars(4, ["DDD", "AAA", "CCC", "BBB"])  # deliberately unsorted columns
    close.loc[close.index[:2], "AAA"] = np.nan  # AAA not eligible at first (min_history 3)
    m = uni.pit_mask(close, hist, min_history_bars=3, max_stocks=2, on_stale="ignore")
    assert list(m.columns) == ["DDD", "AAA", "CCC", "BBB"]
    # bar 2: valid counts AAA=1 -> not eligible; BBB, CCC, DDD eligible -> cap keeps BBB, CCC
    assert m.iloc[2].to_dict() == {"DDD": False, "AAA": False, "CCC": True, "BBB": True}
    # bar 3: AAA now has 2 (<3) -> still out
    assert m.iloc[3].to_dict() == {"DDD": False, "AAA": False, "CCC": True, "BBB": True}


def test_cap_applies_after_history_and_membership() -> None:
    hist = _history({"2023-12-01": ["AAA", "BBB", "CCC"], "2024-01-03": ["BBB", "CCC"]})
    close = _bars(4, ["AAA", "BBB", "CCC"])
    m = uni.pit_mask(close, hist, min_history_bars=1, max_stocks=1, on_stale="ignore")
    assert m["AAA"].tolist() == [True, True, False, False]
    assert m["BBB"].tolist() == [False, False, True, True]  # next name once AAA leaves
    assert not m["CCC"].any()


def test_max_stocks_zero_and_negative() -> None:
    hist = _history({"2023-12-01": ["AAA"]})
    close = _bars(3, ["AAA"])
    assert (
        not uni.pit_mask(close, hist, min_history_bars=1, max_stocks=0, on_stale="ignore")
        .any()
        .any()
    )
    with pytest.raises(ValueError, match="max_stocks"):
        uni.pit_mask(close, hist, max_stocks=-1, on_stale="ignore")


def test_pit_mask_rejects_unsorted_or_duplicate_index() -> None:
    hist = _history({"2023-12-01": ["AAA"]})
    close = _bars(3, ["AAA"])
    with pytest.raises(ValueError, match="increasing"):
        uni.pit_mask(close.iloc[::-1], hist, on_stale="ignore")
    with pytest.raises(ValueError, match="increasing"):
        uni.pit_mask(pd.concat([close, close.iloc[[0]]]), hist, on_stale="ignore")


def test_pit_mask_does_not_mutate_input_and_is_bool() -> None:
    hist = _history({"2023-12-01": ["AAA", "BBB"]})
    close = _bars(5, ["AAA", "BBB"])
    before = close.copy()
    m = uni.pit_mask(close, hist, min_history_bars=2, on_stale="ignore")
    pd.testing.assert_frame_equal(close, before)
    assert all(dt is np.dtype(bool) or dt == np.dtype(bool) for dt in m.dtypes)
    assert m.index.equals(close.index)


def test_pit_mask_static_universe_skips_age_check() -> None:
    close = _bars(4, ["AAA", "BBB"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = uni.pit_mask(close, min_history_bars=2, static_members=["BBB"])
    assert m["BBB"].tolist() == [False, True, True, True]
    assert not m["AAA"].any()


def test_pit_mask_empty_frame() -> None:
    hist = _history({"2023-12-01": ["AAA"]})
    m = uni.pit_mask(pd.DataFrame({"AAA": []}, index=pd.DatetimeIndex([], tz="UTC")), hist)
    assert m.shape == (0, 1)


def test_weekend_stamps_in_index_are_handled_like_any_date() -> None:
    hist = _history({"2023-12-01": ["AAA"], "2024-01-06": ["AAA", "BBB"]})
    idx = pd.date_range("2024-01-05", periods=4, tz="UTC")  # Fri, Sat, Sun, Mon
    close = pd.DataFrame(1.0, index=idx, columns=["AAA", "BBB"])
    m = uni.pit_mask(close, hist, min_history_bars=1, on_stale="ignore")
    assert m["BBB"].tolist() == [False, True, True, True]


# ------------------------------------------------------------------ synthetic parity vs eligible_at


def _synthetic_world(seed: int, n_days: int = 90, n_sym: int = 30):
    """Random bars with late starts, early ends and gaps, plus a churning constituent table."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-06-01", periods=n_days, tz="UTC")
    names = [f"S{c}{i:02d}" for i, c in zip(range(n_sym), "ZYXWVUTSRQPONMLKJIHGFEDCBA" * 2)]
    close = pd.DataFrame(1.0, index=idx, columns=names)
    for name in names:
        start = int(rng.integers(0, n_days // 2))
        end = n_days if rng.random() < 0.7 else int(rng.integers(start + 5, n_days))
        close.iloc[:start, close.columns.get_loc(name)] = np.nan
        close.iloc[end:, close.columns.get_loc(name)] = np.nan
        for g in rng.integers(0, n_days, size=3):
            if rng.random() < 0.5:
                close.iloc[g, close.columns.get_loc(name)] = np.nan
    snap_dates = sorted(set(rng.choice(idx[1:-1], size=8, replace=False)) | {idx[0]})
    rows = {}
    for d in snap_dates:
        k = int(rng.integers(n_sym // 2, n_sym))
        members = rng.choice(names, size=k, replace=False).tolist()
        rows[pd.Timestamp(d).tz_convert("UTC").strftime("%Y-%m-%d")] = [
            m.replace("-", ".") for m in members
        ]
    return close, _history(rows)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize(
    "kw",
    [
        {"min_history_bars": 30},
        {"min_history_bars": 5},
        {"min_history_bars": 5, "max_stocks": 7},
        {"min_history_bars": 0},
        {"min_history_bars": 20, "max_stocks": 3},
    ],
)
@pytest.mark.slow
def test_synthetic_parity_with_eligible_at(seed, kw, monkeypatch) -> None:
    close, hist = _synthetic_world(seed)
    monkeypatch.setattr(
        lab_data,
        "universe_members_asof",
        lambda universe, asof, membership=None: ic.sp500_members_asof(asof, hist),
    )
    ohlcv = pd.concat({s: close[[s]].rename(columns={s: "close"}) for s in close.columns}, axis=1)
    mask = uni.pit_mask(close, hist, on_stale="ignore", **kw)
    cfg = LabConfig(min_history_bars=kw["min_history_bars"], max_stocks=kw.get("max_stocks"))
    mismatches = []
    for d in close.index:
        expected, _ = lab_data.eligible_at(d, ohlcv.loc[:d], cfg)
        got = set(mask.columns[mask.loc[d].to_numpy()])
        if got != set(expected):
            mismatches.append((str(d.date()), sorted(got ^ set(expected))[:4]))
    assert mismatches == []


# ------------------------------------------------------------------ age check


def test_age_check_warns_raises_or_ignores() -> None:
    hist = _history({"2024-01-02": ["AAA"]})
    with pytest.warns(uni.StaleConstituentsWarning, match="stale"):
        rec = uni.check_constituent_age("2024-03-01", hist, max_age_days=45)
    assert rec["stale"] and rec["age_days"] == 59 and rec["last_snapshot"] == "2024-01-02"
    with pytest.raises(uni.StaleConstituentsError):
        uni.check_constituent_age("2024-03-01", hist, max_age_days=45, on_stale="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = uni.check_constituent_age("2024-03-01", hist, max_age_days=45, on_stale="ignore")
        fresh = uni.check_constituent_age("2024-02-01", hist, max_age_days=45)
    assert quiet["stale"] and not fresh["stale"]
    with pytest.raises(ValueError, match="on_stale"):
        uni.check_constituent_age("2024-02-01", hist, on_stale="loud")


def _source():
    from ggTrader.data.membership import FixtureMembershipSource

    return FixtureMembershipSource("sp500_v1_2026-06-02")


def test_age_check_with_a_source_counts_days_since_verification_not_since_change() -> None:
    """Spec 3.4: stale means our copy was last proved current too long ago (v1: 2026-06-09)."""
    source = _source()
    fresh = uni.check_constituent_age("2026-06-20", membership=source, on_stale="raise")
    assert (fresh["age_days"], fresh["age_since_verified_days"]) == (18, 11)  # change 06-02
    assert fresh["last_snapshot"] == "2026-06-02" and not fresh["stale"]
    assert fresh["membership_sha256"] == source.vintage().content_sha256
    with pytest.raises(uni.StaleConstituentsError, match="last verification"):
        uni.check_constituent_age("2026-08-01", membership=source, on_stale="raise")
    # a replay before the list was verified is never stale, however old the last change
    replay = uni.check_constituent_age("2025-01-01", membership=source, on_stale="raise")
    assert not replay["stale"] and replay["age_since_verified_days"] < 0


def test_age_check_without_a_table_or_source_needs_the_environment(monkeypatch) -> None:
    from ggTrader.data import membership
    from ggTrader.data.membership import MembershipSourceNotSelected

    membership.clear_source_cache()
    monkeypatch.delenv("GGT_MEMBERSHIP")
    with pytest.raises(MembershipSourceNotSelected):
        uni.check_constituent_age("2026-06-20")
    with pytest.raises(MembershipSourceNotSelected):
        uni.membership_mask(pd.bdate_range("2024-01-01", periods=2, tz="UTC"), ["AAA"])
    with pytest.raises(MembershipSourceNotSelected):
        uni.pit_mask(_bars(3, ["AAA"]), min_history_bars=1)


def test_a_source_and_its_table_give_the_same_masks() -> None:
    source = _source()
    idx = pd.bdate_range("2023-10-02", periods=60, tz="UTC")
    symbols = ["AAPL", "BRK-B", "UBER", "TSLA", "GONE"]
    assert uni.membership_mask(idx, symbols, membership=source).equals(
        uni.membership_mask(idx, symbols, source.history())
    )
    close = pd.DataFrame(1.0, index=idx, columns=symbols)
    assert uni.pit_mask(close, membership=source, min_history_bars=5, on_stale="ignore").equals(
        uni.pit_mask(close, source.history(), min_history_bars=5, on_stale="ignore")
    )


def test_pit_mask_warns_when_constituents_are_stale() -> None:
    hist = _history({"2023-01-02": ["AAA"]})
    close = _bars(3, ["AAA"])
    with pytest.warns(uni.StaleConstituentsWarning):
        uni.pit_mask(close, hist, min_history_bars=1)
    with pytest.raises(uni.StaleConstituentsError):
        uni.pit_mask(close, hist, min_history_bars=1, on_stale="raise")


# ------------------------------------------------------------------ removal exits


def test_removal_exits_first_non_member_bar() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6, tz="UTC")
    member = pd.DataFrame(
        {
            "STAY": [True] * 6,
            "LEAVE": [True, True, True, False, False, False],
            "BACK": [True, False, False, True, False, True],
            "LATE": [False, False, True, True, True, True],
        },
        index=idx,
    )
    ex = uni.removal_exits(member)
    assert ex["STAY"].sum() == 0
    assert ex["LEAVE"].tolist() == [False, False, False, True, False, False]
    assert ex["BACK"].tolist() == [False, True, False, False, True, False]
    assert ex["LATE"].sum() == 0  # joining is not a removal


# ------------------------------------------------------------------ terminal exits


def _tail_world() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=8, tz="UTC")
    close = pd.DataFrame(
        {
            "LIVE": np.arange(1.0, 9.0),
            "DEAD": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan],
            "GAP": [1.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "EMPTY": np.nan,
            "STALE1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan],
        },
        index=idx,
    )
    return close


def test_terminal_exit_on_last_valid_bar_only_for_dead_series() -> None:
    close = _tail_world()
    res = uni.terminal_exits(close)
    ex = res.exits
    assert ex["DEAD"].tolist() == [False, False, False, True, False, False, False, False]
    assert ex["LIVE"].sum() == 0 and ex["GAP"].sum() == 0 and ex["EMPTY"].sum() == 0
    assert ex["STALE1"].tolist() == [False] * 6 + [True, False]  # default: 1 trailing NaN = dead
    assert res.entries is None


def test_terminal_exit_min_trailing_threshold_ignores_ragged_end() -> None:
    res = uni.terminal_exits(_tail_world(), min_trailing_nan_bars=2)
    assert res.exits["STALE1"].sum() == 0
    assert res.exits["DEAD"].sum() == 1
    assert set(res.meta["dead_symbols"]) == {"DEAD"}


def test_terminal_exit_suppresses_same_bar_entry() -> None:
    close = _tail_world()
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    entries.loc[close.index[3], "DEAD"] = True  # entry on the dead series' last bar
    entries.loc[close.index[1], "DEAD"] = True  # an earlier entry must survive
    entries.loc[close.index[3], "LIVE"] = True
    res = uni.terminal_exits(close, entries)
    assert res.entries is not None
    assert res.entries["DEAD"].tolist() == [False, True] + [False] * 6
    assert res.entries.loc[close.index[3], "LIVE"]
    assert res.meta["same_bar_entries_suppressed"] == 1
    assert entries.loc[close.index[3], "DEAD"]  # input untouched


def test_terminal_exit_meta_labels_the_result_optimistic() -> None:
    meta = uni.terminal_exits(_tail_world()).meta
    assert meta["optimistic"] is True
    assert "overstates" in meta["note"]
    assert meta["dead_symbols"]["DEAD"] == "2024-01-04"
    assert meta["n_dead"] == 2


def test_terminal_exit_meta_labels_both_uses_of_knowledge_of_the_series_end() -> None:
    """Review MINOR-4: the exit price is optimistic AND knowing bar t is the last bar is
    lookahead (it also lets the suppression of a same-bar entry avoid buying a dying name)."""
    close, entries = _tail_world(), None
    close_with_entry = close.copy()
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    entries.loc[close.index[3], "DEAD"] = True  # the dead name's last bar
    meta = uni.terminal_exits(close_with_entry, entries).meta
    biases = meta["biases"]
    assert set(biases) == {"exit_price", "same_bar_entry_suppression"}
    assert biases["exit_price"]["direction"] == "optimistic"
    assert "last traded price" in biases["exit_price"]["what"]
    suppression = biases["same_bar_entry_suppression"]
    assert suppression["direction"] == "favourable"
    assert suppression["uses_knowledge_of_series_end"] is True
    assert suppression["entries_suppressed"] == meta["same_bar_entries_suppressed"] == 1
    assert "lookahead" in suppression["what"]
    assert meta["lookahead"] is True
    assert meta["uses_knowledge_of_series_end"] == ["exit_on_last_bar", "entry_suppression"]
    assert "series ends" in meta["lookahead_note"]


def test_terminal_exit_forced_exit_itself_also_uses_the_series_end() -> None:
    meta = uni.terminal_exits(_tail_world()).meta
    assert meta["biases"]["same_bar_entry_suppression"]["entries_suppressed"] == 0
    assert meta["biases"]["exit_price"]["uses_knowledge_of_series_end"] is True


def test_terminal_exit_all_valid_frame_has_no_exits() -> None:
    close = _tail_world()[["LIVE", "GAP"]]
    res = uni.terminal_exits(close)
    assert not res.exits.to_numpy().any() and res.meta["n_dead"] == 0


def test_same_bar_entry_plus_exit_is_the_vectorbt_trap() -> None:
    """Without suppression the entry and the forced exit coincide on one bar (vbt ignores both)."""
    close = _tail_world()
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    entries.loc[close.index[3], "DEAD"] = True
    raw_exit = uni.terminal_exits(close).exits
    assert (entries & raw_exit).to_numpy().any()
    res = uni.terminal_exits(close, entries)
    assert not (res.entries & res.exits).to_numpy().any()


# ------------------------------------------------------------------ coverage report


def test_coverage_by_year() -> None:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"]), tz="UTC"
    )
    member = pd.DataFrame({"AAA": True, "GHOST": True}, index=idx)
    close = pd.DataFrame({"AAA": [1.0, 1.0, np.nan, 1.0]}, index=idx)  # GHOST has no bars
    out = uni.coverage_by_year(member, close)
    assert out["2023"] == {"member_days": 4, "coverage_pct": 50.0}
    assert out["2024"] == {"member_days": 4, "coverage_pct": 25.0}


# ------------------------------------------------------------------ real tape parity


def _real_close() -> pd.DataFrame:
    from ggTrader.data.bars import load_bars
    from ggTrader.data.membership import FixtureMembershipSource

    return load_bars(
        universe="sp500",
        universe_span=("2021-01-31", "2026-04-30"),
        start="2019-03-01",
        end="2026-10-30",
        extra_symbols=["SPY", "TLT", "GLD", "PDBC", "BIL"],
        membership=FixtureMembershipSource("sp500_v1_2026-06-02"),
    ).close


def sample_dates(
    close: pd.DataFrame, n_random: int, seed: int, n_changes: int = 17
) -> list[pd.Timestamp]:
    """Random bars plus the bar before, on and after an even spread of constituent changes."""
    hist = ic.load_sp500_history()
    in_tape = [d for d in hist.index if close.index[0] <= d <= close.index[-1]]
    changes = in_tape[:: max(1, len(in_tape) // n_changes)]
    edge = {
        close.index[min(max(close.index.searchsorted(d) + k, 0), len(close) - 1)]
        for d in changes
        for k in (-1, 0, 1)
    }
    rng = np.random.default_rng(seed)
    rand = set(pd.DatetimeIndex(rng.choice(close.index[1:], size=n_random, replace=False)))
    return sorted(edge | rand)


REAL_CONFIGS = {
    "min_history_400": {"min_history_bars": 400},
    "min_history_60": {"min_history_bars": 60},
    "min_history_60_cap_50": {"min_history_bars": 60, "max_stocks": 50},
    "min_history_60_cap_off": {"min_history_bars": 60, "max_stocks": None},
}


@needs_real
@pytest.mark.slow
def test_real_tape_parity_with_eligible_at_on_70_plus_dates_x_4_configs() -> None:
    close = _real_close()
    ohlcv = pd.concat({s: close[[s]].rename(columns={s: "close"}) for s in close.columns}, axis=1)
    hist = ic.load_sp500_history()
    # 20 random bars + the neighbourhood of ~19 constituent changes (57 dates, 77 in total). It was
    # 150 random bars (202 dates, 139 s of the suite): scripts/p3_pit_parity.py runs every change
    # date and the synthetic parity above covers the edge cases, so the random sample only
    # spot-checks. eligible_at costs ~0.19 s per call.
    all_dates = sample_dates(close, n_random=20, seed=20261106)
    assert len(all_dates) >= 70
    masks = {
        name: uni.pit_mask(close, hist, on_stale="ignore", **kw)
        for name, kw in REAL_CONFIGS.items()
    }
    mismatch: dict[str, list] = {name: [] for name in REAL_CONFIGS}
    for d in all_dates:
        past = ohlcv.loc[:d]
        for name, kw in REAL_CONFIGS.items():
            cfg = LabConfig(
                min_history_bars=kw["min_history_bars"], max_stocks=kw.get("max_stocks")
            )
            expected, _ = lab_data.eligible_at(d, past, cfg)
            got = set(masks[name].columns[masks[name].loc[d].to_numpy()])
            diff = got ^ set(expected)
            if diff:
                mismatch[name].append((str(d.date()), sorted(diff)[:5]))
    assert mismatch == {name: [] for name in REAL_CONFIGS}
    # Guard against a vacuous pass: the caps and thresholds actually bite.
    assert 30 < masks["min_history_60_cap_50"].sum(axis=1).max() <= 50
    assert masks["min_history_400"].to_numpy().sum() < masks["min_history_60"].to_numpy().sum()


@needs_real
def test_real_tape_mask_build_time_and_memory() -> None:
    """S6 measured 0.05 s and 1.2 MB; allow generous headroom, record exact numbers in evidence."""
    close = _real_close()
    hist = ic.load_sp500_history(ic.SP500_HISTORY_CSV)
    tracemalloc.start()
    t0 = time.perf_counter()
    mask = uni.pit_mask(close, hist, min_history_bars=60, on_stale="ignore")
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert mask.shape == close.shape
    assert elapsed < 2.0
    assert mask.to_numpy().nbytes < 3e6  # bool, 1929 x 605
    assert peak < 60e6  # working arrays (cumsum is int64) stay far below the 5 GB budget
