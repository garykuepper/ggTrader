"""L1: window statistics against independent formulas, the finite guard and the DSR.

Sortino and CAGR are vectorbt's own forms (owner decision 2026-11-06) and are checked here
against independent textbook formulas. Sharpe, volatility and drawdown keep the lab's
``curve_stats`` conventions (252 bars, ddof=1, window from the second bar), which the tests in
``test_window_stats`` hold the engine to.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import stats
from tests.oracles.legacy_dsr import dsr_check

from .panels import ref_max_drawdown

RTOL = 1e-9
TRADING_DAYS = 252


def _returns(seed: int = 0, n: int = 300, cols: int = 3, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(drift, 0.01, (n, cols)),
        index=pd.bdate_range("2021-01-04", periods=n),
        columns=[f"c{i}" for i in range(cols)],
    )


# --- independent references -----------------------------------------------------------


def ref_sharpe(r: np.ndarray) -> float:
    return float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def ref_vol(r: np.ndarray) -> float:
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def ref_sortino_textbook(r: np.ndarray) -> float:
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * math.sqrt(TRADING_DAYS))


def ref_cagr_bar_annualised(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r) ** (TRADING_DAYS / len(r)) - 1.0)


def test_sharpe_matches_reference_at_252_and_ddof_1(engine: str) -> None:
    r = _returns()
    out = stats.sharpe(r)
    for c in r:
        assert out[c] == pytest.approx(ref_sharpe(r[c].to_numpy()), rel=RTOL)


def test_sharpe_is_not_the_365_day_number() -> None:
    r = _returns()
    wrong = r["c1"].mean() / r["c1"].std(ddof=1) * math.sqrt(365)
    assert stats.sharpe(r)["c1"] != pytest.approx(wrong, rel=0.05)


def test_volatility_matches_reference(engine: str) -> None:
    r = _returns()
    out = stats.volatility(r)
    for c in r:
        assert out[c] == pytest.approx(ref_vol(r[c].to_numpy()), rel=RTOL)


def test_max_drawdown_matches_reference(engine: str) -> None:
    r = _returns(seed=4)
    out = stats.max_drawdown(r)
    for c in r:
        assert out[c] == pytest.approx(ref_max_drawdown(r[c].to_numpy()), rel=RTOL)
        assert out[c] <= 0.0


def test_sortino_is_vectorbts_textbook_form(engine: str) -> None:
    r = _returns(seed=2)
    out = stats.sortino(r)
    for c in r:
        assert out[c] == pytest.approx(ref_sortino_textbook(r[c].to_numpy()), rel=RTOL)


def test_cagr_is_bar_annualised(engine: str) -> None:
    r = _returns(seed=3)
    out = stats.cagr(r)
    for c in r:
        assert out[c] == pytest.approx(ref_cagr_bar_annualised(r[c].to_numpy()), rel=RTOL)


def test_series_input_gives_scalars() -> None:
    r = _returns(cols=1)["c0"]
    assert isinstance(stats.sharpe(r), float)
    assert isinstance(stats.max_drawdown(r), float)


# --- finite guard ---------------------------------------------------------------------


def _flat_and_real() -> pd.DataFrame:
    r = _returns(cols=2)
    r["flat"] = 0.0
    r["const"] = 0.001  # vectorbt gives 4.5e16 here, not inf
    return r


def test_flat_curve_gives_nan_never_inf(engine: str) -> None:
    out = stats.sharpe(_flat_and_real())
    assert np.isnan(out["flat"]) and np.isnan(out["const"])
    assert np.isfinite(out[["c0", "c1"]]).all()


def test_no_downside_sortino_is_nan_not_inf() -> None:
    r = pd.DataFrame({"up": np.abs(np.random.default_rng(1).normal(0.001, 0.0005, 100))})
    assert np.isnan(stats.sortino(r)["up"])


def test_a_flat_column_in_a_48_column_grid_never_wins() -> None:
    rng = np.random.default_rng(7)
    grid = pd.DataFrame(
        rng.normal(0.0001, 0.01, (250, 48)), index=pd.bdate_range("2021-01-04", periods=250)
    )
    grid[17] = 0.0
    scores = stats.sharpe(grid)
    assert np.isnan(scores[17])
    assert scores.idxmax() != 17  # idxmax skips NaN; an inf here would have won
    assert np.isfinite(scores.max())


def test_zero_trades_window_is_nan() -> None:
    r = _returns(cols=2)
    out = stats.sharpe(r, n_trades=pd.Series({"c0": 0, "c1": 5}))
    assert np.isnan(out["c0"]) and np.isfinite(out["c1"])


def test_too_few_observations_is_nan() -> None:
    assert np.isnan(stats.sharpe(pd.Series([0.01])))


def test_window_stats_slice_by_bar_position() -> None:
    """Bars 50..149 of the equity: 99 returns, the first starting at bar 51 (legacy window)."""
    r = _returns()
    table = stats.window_stats(r, 50, 150)
    assert list(table.index) == ["sharpe", "volatility", "max_drawdown", "sortino", "cagr", "n_obs"]
    assert table.loc["n_obs", "c0"] == 99
    assert table.loc["sharpe", "c0"] == pytest.approx(
        ref_sharpe(r["c0"].to_numpy()[51:150]), rel=RTOL
    )


# --- trade counts per window (OD12) ---------------------------------------------------


def test_trade_counts_use_the_entry_index_inside_the_window() -> None:
    trades = pd.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b"],
            "entry_idx": [5, 50, 99, 100, 120],
            "exit_idx": [60, 70, 130, 110, 140],
        }
    )
    out = stats.trade_counts(trades, 50, 100, groups=["a", "b", "c"])
    assert out.to_dict() == {"a": 2, "b": 0, "c": 0}  # window is [50, 100): entry 99 in, 100 out


def test_trade_counts_whole_history_equals_the_group_totals() -> None:
    trades = pd.DataFrame({"group": ["a", "a", "b"], "entry_idx": [1, 2, 3], "exit_idx": [5, 6, 7]})
    assert stats.trade_counts(trades, 0, 10).to_dict() == {"a": 2, "b": 1}


def test_trade_counts_on_an_empty_frame() -> None:
    empty = pd.DataFrame({"group": pd.Series(dtype=object), "entry_idx": pd.Series(dtype=int)})
    assert stats.trade_counts(empty, 0, 10, groups=["a"]).to_dict() == {"a": 0}


# --- deflated Sharpe ratio ------------------------------------------------------------


def _legacy_dsr(returns: pd.Series, n_trials: int) -> float:
    """What lab/wfo.py does today: pandas skew/kurtosis into gates.dsr_check."""
    n_obs = len(returns)
    skew, kurt = float(returns.skew()), float(returns.kurtosis())
    sr_annual = float(returns.mean() / returns.std() * math.sqrt(TRADING_DAYS))
    return dsr_check(sr_annual, n_obs, n_trials, skew, kurt).dsr_value


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n_trials", [48, 80])
def test_configured_dsr_reproduces_the_legacy_statistic(seed: int, n_trials: int) -> None:
    rng = np.random.default_rng(seed)
    r = pd.Series(
        rng.standard_t(5, 300) * 0.01 + 0.0006, index=pd.bdate_range("2021-01-04", periods=300)
    )
    assert stats.dsr(r, n_trials) == pytest.approx(_legacy_dsr(r, n_trials), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_dsr_with_one_trial_is_the_psr_against_zero_like_the_legacy_check(seed: int) -> None:
    """One combination: nothing was selected among alternatives, so nothing is deflated.

    vectorbt's own formula has an expected-maximum term of -inf for one trial; the engine pins
    it to 0 (the legacy ``dsr_check`` value for ``n_trials=1``), so the gate is the probabilistic
    Sharpe ratio against a zero benchmark, computed by vectorbt.
    """
    rng = np.random.default_rng(seed)
    r = pd.Series(
        rng.standard_t(5, 200 + 25 * seed) * 0.01 + 0.0005,
        index=pd.bdate_range("2021-01-04", periods=200 + 25 * seed),
    )
    assert stats.dsr(r, 1) == pytest.approx(_legacy_dsr(r, 1), abs=1e-12)
    assert stats.dsr(r, 1) >= stats.dsr(r, 2)  # deflating by more trials never raises the value


def test_dsr_rejects_fewer_than_one_trial() -> None:
    r = _returns(cols=1).iloc[:, 0]
    with pytest.raises(ValueError, match="n_trials"):
        stats.dsr(r, 0)


def test_dsr_var_sharpe_is_252_over_t_minus_1_not_1_over_t_minus_1() -> None:
    """S5's 1/(T-1) loosens the 0.80 gate (0.99 against 0.62); spec Appendix A."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 252), index=pd.bdate_range("2021-01-04", periods=252))
    expected = _legacy_dsr(r, 48)
    wrong = r.vbt.returns(freq="1D", year_freq="252 days").deflated_sharpe_ratio(
        nb_trials=48, var_sharpe=1 / (len(r) - 1), bias=False
    )
    assert stats.dsr(r, 48) == pytest.approx(expected, rel=1e-9)
    assert abs(float(wrong) - expected) > 0.05


def test_dsr_uses_unbiased_moments_like_pandas() -> None:
    """bias=True (scipy's biased skew/kurtosis) differs from pandas by ~1e-6; legacy is pandas."""
    rng = np.random.default_rng(3)
    r = pd.Series(
        rng.standard_t(5, 300) * 0.01 + 0.0006, index=pd.bdate_range("2021-01-04", periods=300)
    )
    biased = r.vbt.returns(freq="1D", year_freq="252 days").deflated_sharpe_ratio(
        nb_trials=48, var_sharpe=TRADING_DAYS / (len(r) - 1), bias=True
    )
    assert abs(float(biased) - _legacy_dsr(r, 48)) > 1e-8
    assert abs(stats.dsr(r, 48) - _legacy_dsr(r, 48)) < 1e-12


def test_dsr_frame_returns_one_value_per_column() -> None:
    r = _returns(cols=3)
    out = stats.dsr(r, 48)
    assert list(out.index) == ["c0", "c1", "c2"]
    for c in r:
        assert out[c] == pytest.approx(_legacy_dsr(r[c], 48), rel=1e-9, abs=1e-15)


def test_dsr_needs_at_least_two_trials_and_observations() -> None:
    with pytest.raises(ValueError):
        stats.dsr(_returns(cols=1)["c0"], 0)
    assert np.isnan(stats.dsr(pd.Series([0.01]), 10))


# --- second, independent protections (review: six safety properties rested on one test) -----


def _sim_with_late_entries(first_entry_bar: int):
    """A real simulation (the engine, not hand-made returns) with entries only from a bar on."""
    from ggTrader.engine import portfolio

    rng = np.random.default_rng(11)
    index = pd.bdate_range("2021-01-04", periods=400)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (400, 3)), axis=0)),
        index=index,
        columns=list("abc"),
    )
    entries = pd.DataFrame(False, index=index, columns=close.columns)
    entries.iloc[first_entry_bar::40] = True
    exits = entries.shift(15, fill_value=False)
    cfg = {"START_CASH": 10_000.0, "FEES": 0.0, "SLIPPAGE": 0.0005, "FREQ": "1D"}
    return portfolio.run_signals(entries, exits, close, cfg, groups=["g"] * 3)


def test_zero_trade_guard_end_to_end_on_a_simulated_window() -> None:
    """A window before the first entry is flat cash: NaN. One entry inside it: a Sharpe."""
    sim = _sim_with_late_entries(first_entry_bar=300)
    flat_window = (50, 250)  # no entry yet, equity is exactly the starting cash
    live_window = (290, 399)  # entries at 300, 340, 380
    for window, expect_finite in ((flat_window, False), (live_window, True)):
        counts = stats.trade_counts(sim.trades, *window, groups=list(sim.value))
        returns = sim.returns.iloc[window[0] + 1 : window[1]]
        score = stats.sharpe(returns, n_trades=counts)
        assert bool(np.isfinite(score["g"])) is expect_finite, (window, counts.to_dict())
    assert stats.trade_counts(sim.trades, *flat_window, groups=["g"])["g"] == 0


def test_zero_trade_guard_on_the_scalar_path_and_with_a_missing_count() -> None:
    returns = _returns(cols=1)["c0"]
    assert np.isnan(stats.sharpe(returns, n_trades=pd.Series({"c0": 0})))
    assert np.isfinite(stats.sharpe(returns, n_trades=pd.Series({"c0": 1})))
    # a column absent from the counts has no trades; it is not "unguarded"
    frame = _returns(cols=2)
    out = stats.sharpe(frame, n_trades=pd.Series({"c0": 3}))
    assert np.isfinite(out["c0"]) and np.isnan(out["c1"])


def test_zero_trade_guard_accepts_a_plain_integer_count_for_a_series() -> None:
    """The scalar path is a separate branch from the frame path; cover it with ints too."""
    returns = _returns(cols=1)["c0"]
    assert np.isnan(stats.sharpe(returns, n_trades=0))
    assert np.isfinite(stats.sharpe(returns, n_trades=2))
    assert np.isfinite(stats.sharpe(returns))  # no count supplied: no guard


def test_trade_count_window_is_half_open_on_a_real_simulation() -> None:
    """Entries on bars start, end-1 and end of an engine run: [start, end) keeps the first two."""
    from ggTrader.engine import portfolio

    index = pd.bdate_range("2021-01-04", periods=200)
    close = pd.DataFrame({"a": np.linspace(50.0, 60.0, 200)}, index=index)
    entries = pd.DataFrame(False, index=index, columns=["a"])
    exits = entries.copy()
    entries.iloc[[60, 99, 100], 0] = True
    exits.iloc[[70, 105, 110], 0] = True
    cfg = {"START_CASH": 10_000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    sim = portfolio.run_signals(entries, exits, close, cfg)
    assert sim.trades["entry_idx"].tolist() == [60, 99]  # the bar-100 entry is a re-entry blocked
    counts = {
        w: stats.trade_counts(sim.trades, *w, groups=["all"])["all"]
        for w in [(60, 99), (60, 100), (61, 100), (99, 100), (100, 150), (0, 200)]
    }
    assert counts == {
        (60, 99): 1,
        (60, 100): 2,
        (61, 100): 1,
        (99, 100): 1,
        (100, 150): 0,
        (0, 200): 2,
    }


def test_trade_counts_equal_a_brute_force_count_for_random_windows() -> None:
    rng = np.random.default_rng(5)
    entry_idx = rng.integers(0, 500, 300)
    trades = pd.DataFrame(
        {
            "group": rng.choice(["a", "b", "c"], 300),
            "entry_idx": entry_idx,
            "exit_idx": entry_idx + 3,
        }
    )
    for _ in range(50):
        start = int(rng.integers(0, 450))
        end = start + int(rng.integers(1, 60))
        got = stats.trade_counts(trades, start, end, groups=["a", "b", "c"])
        for g in "abc":
            mask = (
                (trades["group"] == g)
                & (trades["entry_idx"] >= start)
                & (trades["entry_idx"] < end)
            )
            assert got[g] == int(mask.sum())


# --- degeneracy guards of the composite (P5b, review MAJOR-1) ---------------------------


def _few_losers(n: int = 250, losers: int = 3, seed: int = 3) -> pd.Series:
    """A near-flat winner: tiny gains every day but ``losers`` small losing days."""
    rng = np.random.default_rng(seed)
    r = np.abs(rng.normal(0.0004, 0.0001, n))
    r[rng.choice(n, losers, replace=False)] = -0.0002
    return pd.Series(r, index=pd.bdate_range("2021-01-04", periods=n))


def test_min_losing_days_default_is_twenty_and_documented() -> None:
    from ggTrader.engine import config

    assert config.MIN_LOSING_DAYS == 20
    assert "MIN_LOSING_DAYS" in (config.__doc__ or "")


def test_sortino_needs_enough_losing_days_to_estimate_downside_deviation() -> None:
    few, many = _few_losers(losers=3), _few_losers(losers=40, seed=4)
    assert np.isfinite(stats.sortino(few))  # vectorbt's own number: huge, but finite
    assert stats.sortino(few) > 10  # the artefact the guard exists for
    assert np.isnan(stats.sortino(few, min_losing_days=20))
    assert stats.sortino(many, min_losing_days=20) == pytest.approx(stats.sortino(many), rel=1e-12)


def test_the_losing_day_threshold_is_inclusive_and_counts_only_negative_returns() -> None:
    r = _few_losers(losers=20)
    assert int((r < 0).sum()) == 20
    assert np.isfinite(stats.sortino(r, min_losing_days=20))  # exactly 20 counts
    assert np.isnan(stats.sortino(r, min_losing_days=21))
    zero_days = r.copy()
    zero_days.iloc[:10] = 0.0  # a zero return is not a losing day
    assert int((zero_days < 0).sum()) == int((r.iloc[10:] < 0).sum())


def test_the_guard_never_changes_a_sortino_that_passes_it() -> None:
    """A guard, not a new definition: vectorbt's formula is untouched above the threshold."""
    frame = _returns(cols=4, n=300)
    plain = stats.sortino(frame)
    guarded = stats.sortino(frame, min_losing_days=20)
    pd.testing.assert_series_equal(plain, guarded)


def test_window_stats_applies_both_guards_to_sharpe_and_sortino_only() -> None:
    n = 250
    frame = pd.DataFrame({"few": _few_losers(n=n), "many": _returns(cols=1, n=n)["c0"].to_numpy()})
    counts = pd.Series({"few": 4, "many": 0})  # `many` has no entry inside the window
    table = stats.window_stats(frame, 0, n, n_trades=counts, min_losing_days=20)
    assert np.isnan(table.loc["sortino", "few"]) and np.isfinite(table.loc["sharpe", "few"])
    assert np.isnan(table.loc["sharpe", "many"]) and np.isnan(table.loc["sortino", "many"])
    for stat in ("volatility", "max_drawdown", "cagr"):  # not trade-dependent statistics
        assert np.isfinite(table.loc[stat, "many"])
    unguarded = stats.window_stats(frame, 0, n, min_losing_days=0)
    assert np.isfinite(unguarded.loc["sortino", "few"])


def test_window_stats_uses_the_configured_threshold_by_default() -> None:
    from ggTrader.engine import config

    r = _few_losers(losers=config.MIN_LOSING_DAYS - 1).to_frame("a")
    assert np.isnan(stats.window_stats(r, 0, len(r)).loc["sortino", "a"])
    assert np.isfinite(stats.window_stats(r, 0, len(r), min_losing_days=0).loc["sortino", "a"])


def test_the_guards_are_stamped_into_the_definitions() -> None:
    from ggTrader.engine import provenance

    labels = provenance.definitions()
    assert "20" in labels["sortino_guard"] and "entry" in labels["zero_trade_guard"]


def test_window_stats_of_a_series_takes_its_trade_count_as_a_scalar_or_one_element_series() -> None:
    r = _returns(cols=1)["c0"]
    assert np.isnan(stats.window_stats(r, 0, len(r), n_trades=0).loc["sharpe"].iloc[0])
    assert np.isnan(stats.window_stats(r, 0, len(r), n_trades=pd.Series([0])).loc["sharpe"].iloc[0])
    assert np.isfinite(
        stats.window_stats(r, 0, len(r), n_trades=pd.Series([3])).loc["sharpe"].iloc[0]
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "returns",
    [
        pd.Series([0.01]),
        pd.Series([np.nan, np.nan, 0.01]),
        pd.Series([np.nan, np.nan, np.nan]),
        pd.DataFrame({"a": [np.nan, np.nan, 0.1], "b": [0.1, 0.2, 0.3]}),
    ],
    ids=["one_obs", "one_finite", "all_nan", "frame_one_short_column"],
)
def test_std_of_fewer_than_two_observations_is_nan_without_a_warning(returns) -> None:
    """A window with under two observations gives NaN quietly (no 'Degrees of freedom <= 0')."""
    out = stats._std(returns)
    if isinstance(returns, pd.Series):
        assert np.isnan(out)
    else:
        assert np.isnan(out["a"])
        assert out["b"] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1), rel=0, abs=0)


def test_std_is_bit_identical_to_numpy_nanstd_when_defined() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(0, 0.01, (50, 4)))
    frame.iloc[:10, 1] = np.nan
    expected = np.nanstd(frame.to_numpy(), axis=0, ddof=1)
    assert np.array_equal(stats._std(frame).to_numpy(), expected)
