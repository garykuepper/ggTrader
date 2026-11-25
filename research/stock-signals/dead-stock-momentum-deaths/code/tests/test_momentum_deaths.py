"""Tests for momentum_deaths: pure counts, boundaries, bands, bootstrap, profile, jumps, D."""

from __future__ import annotations

import dm_dead_stock_first_test as fp
import dm_replay
import momentum_deaths as md
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.study


def _make_panel(
    start_cal: str,
    end_cal: str,
    n_stocks: int,
    *,
    seed: int = 7,
    tz: str | None = "UTC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic business-day closes and all-True eligibility for ``n_stocks`` names."""
    days = pd.bdate_range(start_cal, end_cal, tz=tz)
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.015, size=(len(days), n_stocks))
    prices = 50.0 * np.exp(np.cumsum(steps, axis=0))
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(prices, index=days, columns=cols)
    eligible = pd.DataFrame(True, index=days, columns=cols)
    return close, eligible


def _empty_spans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=object),
            "last_date": pd.to_datetime([]),
            "has_successor": pd.Series(dtype=bool),
            "total_loss": pd.Series(dtype=bool),
        }
    )


def test_top_50_equals_plan_trades_with_deaths_successors_and_file_end_rows() -> None:
    close, eligible = _make_panel("2018-01-01", "2020-06-30", 65, seed=11)
    # Give two stocks identical closes to verify stable tie-breaking order.
    close["S001"] = close["S000"].copy()
    # S005 dies mid-window; S006 has a successor; S007 ends at CRSP_END (file-end row).
    close.loc["2019-06-18":, "S005"] = np.nan
    close.loc["2019-08-15":, "S006"] = np.nan
    eligible.loc["2019-09-01":"2019-11-30", ["S010", "S011"]] = False

    span_rows = pd.DataFrame(
        {
            "symbol": ["S005", "S006", "S007"],
            "last_date": pd.to_datetime(["2019-06-17", "2019-08-14", "2020-06-30"]),
            "has_successor": [False, True, False],
            "total_loss": [False, False, False],
        }
    )
    start, end = "2019-01-02", "2020-06-30"
    spans, ends, successors, death_dates, n_censored = md.prepare_deaths(
        close, span_rows, end, crsp_end="2020-06-30"
    )
    assert n_censored == 1
    assert "S007" not in ends.index
    assert successors == frozenset({"S006"})
    assert set(death_dates.index) == {"S005"}

    sets = md.eligible_sets(
        close,
        eligible,
        spans,
        lookback=fp.LOOKBACK,
        skip=fp.SKIP,
        start=start,
        end=end,
    )
    rebuilt = md.rule_ranking_plan(sets, np.array(close.columns), top_n=fp.TOP_N)
    reference = dm_replay.plan_trades(
        close,
        eligible,
        spans,
        top_n=fp.TOP_N,
        lookback=fp.LOOKBACK,
        skip=fp.SKIP,
        dead_loss=1.0,
        start=start,
        end=end,
    )
    assert rebuilt == reference
    assert len(rebuilt) > 10


def test_death_in_holding_month_boundaries_successor_and_file_end() -> None:
    # Build 3 decision months: Jan 2020 -> exec Feb 3, Feb 2020 -> exec Mar 2,
    # Mar 2020 -> exec Apr 1; window ends Apr 30, 2020.
    close, eligible = _make_panel("2018-06-01", "2020-04-30", 12, seed=21, tz=None)
    start, end = "2020-01-02", "2020-04-30"

    # Verify the exact calendar dates of our 3 decision/execution pairs.
    # Month 0: dec 2020-01-31, exec 2020-02-03, hold_end 2020-03-01 (last trading day 2020-02-28)
    # Month 1: dec 2020-02-28, exec 2020-03-02, hold_end 2020-03-31
    # Month 2: dec 2020-03-31, exec 2020-04-01, hold_end 2020-04-30
    deaths_spec = [
        ("S000", "2020-01-31", False),  # on decision 0 -> before exec 0 -> NOT in month 0
        ("S001", "2020-02-03", False),  # on exec 0 -> IN month 0
        ("S002", "2020-02-28", False),  # last trading day before exec 1 -> IN month 0
        ("S003", "2020-03-02", False),  # on exec 1 itself -> NOT month 0, IN month 1
        ("S004", "2020-02-14", True),  # successor inside month 0 -> NOT counted
        ("S005", "2020-04-30", False),  # file-end row (equals crsp_end) -> NOT counted
        ("S006", "2020-04-29", False),  # before file end, inside last month -> IN month 2
    ]
    for sym, last_day, _ in deaths_spec:
        after = close.index > pd.Timestamp(last_day)
        close.loc[after, sym] = np.nan

    span_rows = pd.DataFrame(
        {
            "symbol": [s for s, _, _ in deaths_spec],
            "last_date": pd.to_datetime([d for _, d, _ in deaths_spec]),
            "has_successor": [succ for _, _, succ in deaths_spec],
            "total_loss": [False] * len(deaths_spec),
        }
    )

    spans, _ends, _succ, death_dates, n_censored = md.prepare_deaths(
        close, span_rows, end, crsp_end="2020-04-30"
    )
    assert n_censored == 1
    sets = md.eligible_sets(
        close, eligible, spans, lookback=fp.LOOKBACK, skip=fp.SKIP, start=start, end=end
    )
    months = md.holding_months(sets, close.index, end)
    assert len(months) == 3
    assert months[0].execution_date == pd.Timestamp("2020-02-03")
    assert months[0].hold_end == pd.Timestamp("2020-03-01")
    assert months[1].execution_date == pd.Timestamp("2020-03-02")
    assert months[1].hold_end == pd.Timestamp("2020-03-31")
    assert months[2].execution_date == pd.Timestamp("2020-04-01")
    assert months[2].hold_end == pd.Timestamp("2020-04-30")

    symbols = np.array(close.columns)
    m0 = {str(symbols[c]) for c in md.deaths_in_month(months[0], symbols, death_dates)}
    m1 = {str(symbols[c]) for c in md.deaths_in_month(months[1], symbols, death_dates)}
    m2 = {str(symbols[c]) for c in md.deaths_in_month(months[2], symbols, death_dates)}

    assert m0 == {"S001", "S002"}
    assert m1 == {"S003"}
    assert m2 == {"S006"}


def test_four_rank_bands_counts_and_totals() -> None:
    # 520 stocks over 2 decision months with deterministic distinct linear trends so rank is exact.
    days = pd.bdate_range("2018-10-01", "2020-03-31", tz=None)
    n_stocks = 520
    t = np.arange(len(days), dtype=float)[:, None]
    # Stock i has daily drift (n_stocks - i) * 1e-4 so S000 is rank 1 through S519 at rank 520.
    drifts = (n_stocks - np.arange(n_stocks, dtype=float))[None, :] * 1e-4
    prices = 100.0 * np.exp(t * drifts)
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(prices, index=days, columns=cols)
    eligible = pd.DataFrame(True, index=days, columns=cols)

    # Plant deaths in month 0 (decision 2020-01-31, holding Feb 2020):
    # 2 in band 1-50 (S000, S049), 1 in band 51-150 (S050),
    # 3 in band 151-500 (S150, S300, S499), 1 in band 501+ (S500).
    dying = ["S000", "S049", "S050", "S150", "S300", "S499", "S500"]
    close.loc["2020-02-15":, dying] = np.nan
    span_rows = pd.DataFrame(
        {
            "symbol": dying,
            "last_date": pd.to_datetime(["2020-02-14"] * len(dying)),
            "has_successor": [False] * len(dying),
            "total_loss": [False] * len(dying),
        }
    )

    out = md.compute_measures(
        close,
        eligible,
        span_rows,
        start="2020-01-02",
        end="2020-03-31",
        n_resamples=200,
    )
    # Month 0 has 520 usable; Month 1 (Feb 28 -> Mar 2) has 513 usable (7 died in Feb).
    assert out["A1_totals"]["n_months"] == 2
    assert out["A1_totals"]["stock_months"] == 520 + 513
    assert out["A1_totals"]["deaths"] == 7
    assert out["A1_totals"]["death_rate"] == pytest.approx(7 / (520 + 513))

    bands = out["A2_by_band"]
    assert bands["1_to_50"]["stock_months"] == 50 + 50
    assert bands["1_to_50"]["deaths"] == 2
    assert bands["1_to_50"]["death_rate"] == pytest.approx(2 / 100)

    assert bands["51_to_150"]["stock_months"] == 100 + 100
    assert bands["51_to_150"]["deaths"] == 1
    assert bands["51_to_150"]["death_rate"] == pytest.approx(1 / 200)

    assert bands["151_to_500"]["stock_months"] == 350 + 350
    assert bands["151_to_500"]["deaths"] == 3
    assert bands["151_to_500"]["death_rate"] == pytest.approx(3 / 700)

    assert bands["501_plus"]["stock_months"] == 20 + 13
    assert bands["501_plus"]["deaths"] == 1
    assert bands["501_plus"]["death_rate"] == pytest.approx(1 / 33)

    assert sum(b["stock_months"] for b in bands.values()) == out["A1_totals"]["stock_months"]
    assert sum(b["deaths"] for b in bands.values()) == out["A1_totals"]["deaths"]
    # Cross-measure reconciliation across A1, A2, A4, A5, B, and C.
    assert out["A4_expected_vs_actual"]["actual_top_50_deaths"] == bands["1_to_50"]["deaths"]
    assert out["B_dying_profile"]["in_top_50"]["n"] == bands["1_to_50"]["deaths"]
    assert (
        out["B_dying_profile"]["in_top_50"]["n"] + out["B_dying_profile"]["usable_not_top_50"]["n"]
        == out["A1_totals"]["deaths"]
    )
    assert out["C_one_day_jumps"]["top_50_died"]["n"] == bands["1_to_50"]["deaths"]
    assert (
        out["C_one_day_jumps"]["top_50_died"]["n"]
        + out["C_one_day_jumps"]["top_50_did_not_die"]["n"]
        == bands["1_to_50"]["stock_months"]
    )
    assert out["C_one_day_jumps"]["all_usable"]["n"] == out["A1_totals"]["stock_months"]
    assert (
        sum(yr["usable_stock_months"] for yr in out["A5_by_year"].values())
        == out["A1_totals"]["stock_months"]
    )
    assert (
        sum(yr["top_50_deaths"] for yr in out["A5_by_year"].values())
        == out["A4_expected_vs_actual"]["actual_top_50_deaths"]
    )
    assert sum(
        yr["expected_random_50_deaths"] for yr in out["A5_by_year"].values()
    ) == pytest.approx(out["A4_expected_vs_actual"]["expected_random_50_deaths"])


def test_a3_rate_ratio_and_bootstrap_interval_known_answer() -> None:
    # Constant-ratio months: every month top-50 has 2/50 = 4% deaths, all usable has 4/200 = 2%.
    # Rate ratio is identically 2.0 in every single resample -> ci_low == ci_high == 2.0.
    res = md.bootstrap_rate_ratio(
        top_n_counts=np.array([50, 50, 50, 50]),
        top_deaths=np.array([2, 2, 2, 2]),
        all_counts=np.array([200, 200, 200, 200]),
        all_deaths=np.array([4, 4, 4, 4]),
        n_resamples=2000,
        seed=20261125,
    )
    assert res["rate_ratio"] == pytest.approx(2.0)
    assert res["top_50_rate"] == pytest.approx(0.04)
    assert res["all_usable_rate"] == pytest.approx(0.02)
    assert res["ci_low"] == pytest.approx(2.0)
    assert res["ci_high"] == pytest.approx(2.0)
    assert res["n_resamples"] == 2000
    assert res["n_valid_resamples"] == 2000
    assert res["seed"] == 20261125

    # Varying months: check exact match against independent numpy draw with seed 20261125.
    top_n = np.array([50, 50, 50, 50])
    top_d = np.array([3, 1, 2, 0])
    all_n = np.array([200, 200, 200, 200])
    all_d = np.array([4, 2, 4, 2])
    got = md.bootstrap_rate_ratio(top_n, top_d, all_n, all_d, n_resamples=2000, seed=20261125)
    # Pooled point estimate: (6/200) / (12/800) = 0.03 / 0.015 = 2.0
    assert got["rate_ratio"] == pytest.approx(2.0)
    rng = np.random.default_rng(20261125)
    idx = rng.integers(0, 4, size=(2000, 4))
    ratios = (top_d[idx].sum(axis=1) / top_n[idx].sum(axis=1)) / (
        all_d[idx].sum(axis=1) / all_n[idx].sum(axis=1)
    )
    want_lo, want_hi = np.percentile(ratios, [2.5, 97.5])
    assert got["ci_low"] == pytest.approx(want_lo)
    assert got["ci_high"] == pytest.approx(want_hi)
    assert got["ci_low"] < got["rate_ratio"] < got["ci_high"]


def test_a4_expected_count_and_a5_by_year_hand_made() -> None:
    # Construct 3 months across a year boundary:
    # Month 0: dec 2019-11-29, exec 2019-12-02 (year 2019), 60 usable, 6 deaths (2 in top 50)
    #   expected = min(50, 60) * 6 / 60 = 5.0
    # Month 1: dec 2019-12-31, exec 2020-01-02 (year 2020), 30 usable (< 50), 3 deaths (all top)
    #   expected = min(50, 30) * 3 / 30 = 3.0
    # Month 2: dec 2020-01-31, exec 2020-02-03 (year 2020), 50 usable, 1 death (0 in top 50)
    #   expected = min(50, 50) * 1 / 50 = 1.0  (Wait: if 50 usable, all 50 are in top 50, so
    #   make month 2 have 51 usable with rank 51 dying so 0 in top 50!)
    #   expected = min(50, 51) * 1 / 51 = 50 / 51.
    days = pd.bdate_range("2018-09-01", "2020-02-28", tz=None)
    n_stocks = 60
    t = np.arange(len(days), dtype=float)[:, None]
    drifts = (n_stocks - np.arange(n_stocks, dtype=float))[None, :] * 1e-4
    prices = 50.0 * np.exp(t * drifts)
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(prices, index=days, columns=cols)
    eligible = pd.DataFrame(True, index=days, columns=cols)

    # Month 0 (dec 2019-11-29 -> exec 2019-12-02): all 60 eligible.
    # 6 die in Dec 2019: S000, S001 (top 50) and S056, S057, S058, S059 (ranks 57-60).
    dec_dead = ["S000", "S001", "S056", "S057", "S058", "S059"]
    close.loc["2019-12-16":, dec_dead] = np.nan

    # Month 1 (dec 2019-12-31 -> exec 2020-01-02): only 30 eligible (S002..S031).
    # 3 die in Jan 2020: S002, S003, S004 (all in top 50 since usable=30 <= 50).
    eligible.loc["2019-12-31", cols] = False
    eligible.loc["2019-12-31", [f"S{i:03d}" for i in range(2, 32)]] = True
    jan_dead = ["S002", "S003", "S004"]
    close.loc["2020-01-16":, jan_dead] = np.nan

    # Month 2 (dec 2020-01-31 -> exec 2020-02-03): all remaining 51 alive stocks eligible!
    # 1 dies in Feb 2020: S055 (the lowest-ranked alive stock -> rank 51, outside top 50).
    feb_dead = ["S055"]
    close.loc["2020-02-15":, feb_dead] = np.nan

    span_rows = pd.DataFrame(
        {
            "symbol": dec_dead + jan_dead + feb_dead,
            "last_date": pd.to_datetime(
                ["2019-12-13"] * len(dec_dead)
                + ["2020-01-15"] * len(jan_dead)
                + ["2020-02-14"] * len(feb_dead)
            ),
            "has_successor": [False] * 10,
            "total_loss": [False] * 10,
        }
    )

    out = md.compute_measures(
        close,
        eligible,
        span_rows,
        start="2019-11-01",
        end="2020-02-28",
        random_baseline_median=7.5,
        n_resamples=100,
    )
    a4 = out["A4_expected_vs_actual"]
    expected_total = 50 * (6 / 60) + 30 * (3 / 30) + 50 * (1 / 51)
    assert a4["expected_random_50_deaths"] == pytest.approx(expected_total)
    assert a4["actual_top_50_deaths"] == 2 + 3 + 0
    assert a4["excess_deaths"] == pytest.approx(5 - expected_total)
    assert a4["random_baseline_fresh_draw_median"] == 7.5

    a5 = out["A5_by_year"]
    assert list(a5) == ["2019", "2020"]
    assert a5["2019"]["usable_stock_months"] == 60
    assert a5["2019"]["top_50_deaths"] == 2
    assert a5["2019"]["expected_random_50_deaths"] == pytest.approx(5.0)
    assert a5["2020"]["usable_stock_months"] == 30 + 51
    assert a5["2020"]["top_50_deaths"] == 3
    assert a5["2020"]["expected_random_50_deaths"] == pytest.approx(3.0 + 50 / 51)


def test_b_dying_profile_buckets_rank_share_and_63d_return() -> None:
    days = pd.bdate_range("2018-10-01", "2020-02-28", tz=None)
    n_stocks = 60
    # Give every stock flat price 100, then craft exact momentum scores and 63d returns.
    close = pd.DataFrame(100.0, index=days, columns=[f"S{i:03d}" for i in range(n_stocks)])
    eligible = pd.DataFrame(True, index=days, columns=close.columns)

    dec_pos = int(close.index.get_loc(pd.Timestamp("2020-01-31")))
    # Set price at dec_pos - 252 so that close[dec_pos - 21] / close[dec_pos - 252] - 1
    # puts S000..S049 at higher scores than S050..S059.
    for i in range(n_stocks):
        # Score = (n_stocks - i) * 0.01  =>  past = 100 / (1 + score)
        score = (n_stocks - i) * 0.01
        close.iloc[: dec_pos - 251, i] = 100.0 / (1.0 + score)
        # Keep dec_pos - 63 at 80.0 for top-50 stocks (63d return = 100/80 - 1 = 0.25)
        # and at 125.0 for non-top-50 stocks (63d return = 100/125 - 1 = -0.20).
        if i < 50:
            close.iat[dec_pos - 63, i] = 80.0
        else:
            close.iat[dec_pos - 63, i] = 125.0

    # 4 dying stocks in top 50, testing exact bucket edges:
    # S000 (rank 1): last close 49.0 / 100.0 = 0.49 -> lt_0.5
    # S001 (rank 2): last close 50.0 / 100.0 = 0.50 -> 0.5_to_0.9 (edge belongs to upper)
    # S002 (rank 3): last close 90.0 / 100.0 = 0.90 -> 0.9_to_1.1 (edge belongs to upper)
    # S003 (rank 4): last close 110.0 / 100.0 = 1.10 -> ge_1.1 (edge belongs to upper)
    # 2 dying stocks outside top 50:
    # S050 (rank 51): last close 120.0 / 100.0 = 1.20 -> ge_1.1
    # S059 (rank 60): decision close NaN -> no_history (still usable via ffill score!)
    death_day = pd.Timestamp("2020-02-14")
    close.loc[death_day, "S000"] = 49.0
    close.loc[death_day, "S001"] = 50.0
    close.loc[death_day, "S002"] = 90.0
    close.loc[death_day, "S003"] = 110.0
    close.loc[death_day, "S050"] = 120.0
    close.loc[pd.Timestamp("2020-01-31"), "S059"] = np.nan
    close.loc[death_day, "S059"] = 95.0

    dying = ["S000", "S001", "S002", "S003", "S050", "S059"]
    close.loc["2020-02-17":, dying] = np.nan
    span_rows = pd.DataFrame(
        {
            "symbol": dying,
            "last_date": pd.to_datetime(["2020-02-14"] * len(dying)),
            "has_successor": [False] * len(dying),
            "total_loss": [False] * len(dying),
        }
    )

    out = md.compute_measures(
        close, eligible, span_rows, start="2020-01-02", end="2020-02-28", n_resamples=50
    )
    b_top = out["B_dying_profile"]["in_top_50"]
    b_rest = out["B_dying_profile"]["usable_not_top_50"]

    assert b_top["n"] == 4
    assert b_rest["n"] == 2

    # Check exact bucket shares in top 50 (one in each of the four buckets, 0 in no_history).
    for bucket in ("lt_0.5", "0.5_to_0.9", "0.9_to_1.1", "ge_1.1"):
        assert b_top["final_month_move_buckets"][bucket]["count"] == 1
        assert b_top["final_month_move_buckets"][bucket]["share"] == pytest.approx(0.25)
    assert b_top["final_month_move_buckets"]["no_history"]["count"] == 0

    # Outside top 50: one ge_1.1, one no_history.
    assert b_rest["final_month_move_buckets"]["ge_1.1"]["count"] == 1
    assert b_rest["final_month_move_buckets"]["no_history"]["count"] == 1
    assert b_rest["final_month_move_buckets"]["ge_1.1"]["share"] == pytest.approx(0.5)

    # Rank share: top 50 has ranks 1, 2, 3, 4 out of 60 -> median 2.5 / 60.
    assert b_top["rank_share"]["p50"] == pytest.approx(2.5 / 60.0)
    # Outside top 50 has ranks 51, 60 out of 60 -> median 55.5 / 60.
    assert b_rest["rank_share"]["p50"] == pytest.approx(55.5 / 60.0)

    # 63-trading-day return: top 50 all +0.25; outside top 50 S050 has -0.20 (S059 is NaN).
    assert b_top["return_63d"]["n"] == 4
    assert b_top["return_63d"]["p50"] == pytest.approx(0.25)
    assert b_rest["return_63d"]["n"] == 1
    assert b_rest["return_63d"]["p50"] == pytest.approx(-0.20)


def test_c_one_day_jump_window_boundaries_and_nan_gaps() -> None:
    days = pd.bdate_range("2018-06-01", "2020-02-28", tz=None)
    # 55 stocks so top 50 is S000..S049, rest is S050..S054.
    n_stocks = 55
    close = pd.DataFrame(100.0, index=days, columns=[f"S{i:03d}" for i in range(n_stocks)])
    eligible = pd.DataFrame(True, index=days, columns=close.columns)

    dec_pos = int(close.index.get_loc(pd.Timestamp("2020-01-31")))
    # Give S000..S049 slightly higher price from dec_pos - 251 onward so they are top 50,
    # using tiny daily steps (never near 20%).
    for i in range(n_stocks):
        close.iloc[: dec_pos - 255, i] = 90.0 + i * 0.1

    # Helper to plant a single-day jump at lag `lag` before dec_pos:
    # close[dec_pos - lag] = factor * close[dec_pos - lag - 1]
    # and keep prices flat after so only that one day jumps (and scores stay clean).
    def plant_jump(sym: str, lag: int, factor: float) -> None:
        j = dec_pos - lag
        base = float(close.iat[j - 1, close.columns.get_loc(sym)])
        close.iloc[j:, close.columns.get_loc(sym)] = base * factor

    # S000 (top 50, dies): exact 20% jump at EXACTLY 252 trading days before -> FLAGGED
    plant_jump("S000", 252, 1.20)
    # S001 (top 50, dies): 25% jump at EXACTLY 21 trading days before -> FLAGGED
    plant_jump("S001", 21, 1.25)
    # S002 (top 50, dies): 30% jump at 253 trading days before (just outside!) -> NOT flagged
    plant_jump("S002", 253, 1.30)
    # S003 (top 50, dies): 30% jump at 20 trading days before (just outside!) -> NOT flagged
    plant_jump("S003", 20, 1.30)
    # S004 (top 50, survives): 19.9% gain inside window -> NOT flagged
    plant_jump("S004", 100, 1.199)
    # S005 (top 50, survives): NaN gap where price doubles across gap -> NOT flagged
    col5 = close.columns.get_loc("S005")
    close.iat[dec_pos - 100, col5] = np.nan
    close.iloc[dec_pos - 99 :, col5] = 200.0
    # S006 (top 50, survives): 22% jump at 100 days before -> FLAGGED
    plant_jump("S006", 100, 1.22)
    # S054 (outside top 50, survives): 25% jump at 50 days before -> FLAGGED
    plant_jump("S054", 50, 1.25)

    # Ensure S000..S006 stay in top 50 and S054 stays outside top 50 despite planted prices:
    # Check using exact scores by keeping S050..S054 below S000..S049 in momentum:
    for i in range(50, n_stocks):
        # Give S050..S054 a higher starting price at dec_pos - 252 so their 12-1 score is lower.
        close.iloc[: dec_pos - 251, i] = 300.0

    dying = ["S000", "S001", "S002", "S003"]
    close.loc["2020-02-17":, dying] = np.nan
    span_rows = pd.DataFrame(
        {
            "symbol": dying,
            "last_date": pd.to_datetime(["2020-02-14"] * len(dying)),
            "has_successor": [False] * len(dying),
            "total_loss": [False] * len(dying),
        }
    )

    out = md.compute_measures(
        close, eligible, span_rows, start="2020-01-02", end="2020-02-28", n_resamples=50
    )
    c = out["C_one_day_jumps"]
    # top_50_died: 4 stocks (S000, S001 flagged; S002, S003 not flagged) -> 2 / 4 = 0.5
    assert c["top_50_died"]["n"] == 4
    assert c["top_50_died"]["n_flagged"] == 2
    assert c["top_50_died"]["share_flagged"] == pytest.approx(0.5)
    # top_50_did_not_die: 46 stocks, only S006 flagged -> 1 / 46
    assert c["top_50_did_not_die"]["n"] == 46
    assert c["top_50_did_not_die"]["n_flagged"] == 1
    assert c["top_50_did_not_die"]["share_flagged"] == pytest.approx(1 / 46)
    # all_usable: 55 stocks, S000, S001, S006, S054 flagged -> 4 / 55
    assert c["all_usable"]["n"] == 55
    assert c["all_usable"]["n_flagged"] == 4
    assert c["all_usable"]["share_flagged"] == pytest.approx(4 / 55)


def test_d_check_against_held_at_death_flag_boundaries() -> None:
    exact = md.check_against_held_at_death(31, 31)
    assert exact["difference"] == 0
    assert exact["ratio"] == pytest.approx(1.0)
    assert exact["exceeds_10pct_flag"] is False

    # Exactly 10% difference (3 out of 30) is NOT over 10%.
    at_10 = md.check_against_held_at_death(33, 30)
    assert at_10["difference"] == 3
    assert at_10["ratio"] == pytest.approx(1.1)
    assert at_10["exceeds_10pct_flag"] is False

    # Over 10% positive (4 out of 30) -> flagged.
    over_pos = md.check_against_held_at_death(34, 30)
    assert over_pos["difference"] == 4
    assert over_pos["exceeds_10pct_flag"] is True

    # Over 10% negative (26 out of 30) -> flagged.
    over_neg = md.check_against_held_at_death(26, 30)
    assert over_neg["difference"] == -4
    assert over_neg["exceeds_10pct_flag"] is True

    # Zero held_at_death edge cases.
    zero_ok = md.check_against_held_at_death(0, 0)
    assert zero_ok["ratio"] is None
    assert zero_ok["exceeds_10pct_flag"] is False

    zero_bad = md.check_against_held_at_death(1, 0)
    assert zero_bad["ratio"] is None
    assert zero_bad["exceeds_10pct_flag"] is True


def _cum(flag_bars: list[int], n: int) -> np.ndarray:
    flags = np.zeros(n, dtype=np.int32)
    flags[flag_bars] = 1
    out = np.zeros(n + 1, dtype=np.int32)
    out[1:] = np.cumsum(flags)
    return out


def test_scoring_window_jump_endpoints_without_holes() -> None:
    valid = np.arange(400)
    dec = 399  # span runs from valid bar 399-252 = 147 to 399-21 = 378, both included
    for bar, expected in [(146, False), (147, True), (378, True), (379, False)]:
        assert (
            md.scoring_window_has_jump(valid, _cum([bar], 400), dec, far=252, near=21) is expected
        )


def test_scoring_window_follows_the_stocks_own_valid_closes_when_it_has_holes() -> None:
    # Ten missing closes just before the decision: the stock's last valid close is bar 389, so
    # its span is valid bars 137 to 368 (not the market days 147 to 378).
    valid = np.array([b for b in range(400) if not 390 <= b <= 399])
    dec = 399
    # 260 market days before the decision: inside the true span, outside the market-day span.
    assert md.scoring_window_has_jump(valid, _cum([139], 400), dec, far=252, near=21) is True
    # 22 market days before: outside the true span (only 12 valid closes back).
    assert md.scoring_window_has_jump(valid, _cum([377], 400), dec, far=252, near=21) is False


def test_scoring_window_without_enough_history_has_no_span() -> None:
    valid = np.arange(200)
    assert md.scoring_window_has_jump(valid, _cum([100], 300), 199, far=252, near=21) is False


def test_usable_set_excludes_short_history_bad_scores_ineligible_and_ended_names() -> None:
    days = pd.bdate_range("2018-01-01", "2020-03-31", tz=None)
    cols = [f"S{i:03d}" for i in range(60)] + [
        "NEW",  # eligible but only 100 days of closes
        "ZERO",  # a zero price 252 closes back: no finite score
        "OUT",  # not eligible on the decision day
        "GONE",  # series ended before the decision day
        "LAST",  # series ends exactly on the decision day: still usable
    ]
    rng = np.random.default_rng(3)
    prices = 50.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (len(days), len(cols))), axis=0))
    close = pd.DataFrame(prices, index=days, columns=cols)
    decision = pd.Timestamp("2020-01-31")
    close.loc[: days[-101], "NEW"] = np.nan
    pos = int(close.index.get_loc(decision))
    # Zero price at the bar the score reads as the older close (valid closes 252 back).
    close.iloc[pos - 252, close.columns.get_loc("ZERO")] = 0.0
    close.loc[decision + pd.Timedelta(days=1) :, "LAST"] = np.nan
    close.loc[pd.Timestamp("2020-01-15") :, "GONE"] = np.nan
    eligible = pd.DataFrame(True, index=days, columns=cols)
    eligible.loc[decision, "OUT"] = False
    span_rows = pd.DataFrame(
        {
            "symbol": ["GONE", "LAST"],
            "last_date": pd.to_datetime(["2020-01-14", "2020-01-31"]),
            "has_successor": [False, False],
            "total_loss": [False, False],
        }
    )
    spans, *_ = md.prepare_deaths(close, span_rows, "2020-03-31", crsp_end="2024-12-31")
    sets = md.eligible_sets(close, eligible, spans, start="2020-01-02", end="2020-03-31")
    first = next(s for s in sets if s.decision_date == decision)
    names = {cols[j] for j in first.usable}
    assert names == {f"S{i:03d}" for i in range(60)} | {"LAST"}
    assert len(first.ranked) == len(first.usable) == 61
