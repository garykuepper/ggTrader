"""Tests for the audit's pre-registered excess-return statistics."""

import math

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.significance import (
    CAGR_BAR_PP,
    NW_LAGS,
    T_BAR,
    ExcessStats,
    excess_stats,
    newey_west_se,
    passes_pretax_bar,
    years_for_t,
)


def test_newey_west_lag0_is_plain_se():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert newey_west_se(x, 0) == pytest.approx(np.std(x, ddof=0) / np.sqrt(4))


def test_newey_west_widens_under_positive_autocorrelation():
    x = np.sin(np.arange(200) / 10.0)
    assert newey_west_se(x, 5) > newey_west_se(x, 0)


def test_identical_curves_have_zero_excess():
    eq = pd.Series(np.linspace(100, 120, 300), index=pd.bdate_range("2024-01-01", periods=300))
    stats = excess_stats(eq, eq)
    assert stats.t_stat == 0.0
    assert stats.cagr_diff_pp == pytest.approx(0.0, abs=1e-9)
    assert stats.information_ratio == 0.0


def test_constant_outperformance_is_detected():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=1260)
    rb = rng.normal(0.0004, 0.01, len(idx))
    rs = rb + 0.0001 + rng.normal(0.0, 0.0005, len(idx))
    bench = pd.Series(100 * np.cumprod(1 + rb), index=idx)
    strat = pd.Series(100 * np.cumprod(1 + rs), index=idx)
    stats = excess_stats(strat, bench)
    assert stats.ann_excess_pct == pytest.approx(2.52, abs=1.0)
    assert stats.t_stat > 3.0


def test_years_for_t():
    assert years_for_t(0.5) == pytest.approx(16.0)
    assert years_for_t(0.0) == math.inf
    assert years_for_t(-0.5) == math.inf


def test_alignment_precedes_returns_and_uses_shared_cagr_helper():
    idx = pd.bdate_range("2024-01-01", periods=6)
    strat = pd.Series([100, 102, np.nan, 103, 105, 104], index=idx)
    bench = pd.Series([100, 101, 102, 103, 104], index=idx[1:])
    dates = idx[[1, 3, 4, 5]]
    stats = excess_stats(strat, bench)
    excess = strat.loc[dates].pct_change().dropna() - bench.loc[dates].pct_change().dropna()
    assert stats.n_obs == 3
    assert stats.ann_excess_pct == pytest.approx(excess.mean() * 252 * 100)
    assert stats.cagr_diff_pp == pytest.approx(
        curve_stats(strat.loc[dates])["cagr_pct"] - curve_stats(bench.loc[dates])["cagr_pct"]
    )


def test_pretax_bar_requires_both_thresholds_inclusive():
    assert (T_BAR, NW_LAGS, CAGR_BAR_PP) == (3.0, 5, 1.0)
    assert passes_pretax_bar(ExcessStats(252, 3.0, 2.0, 1.0, 2.0, 1.0))
    assert not passes_pretax_bar(ExcessStats(252, 2.99, 2.0, 1.0, 2.0, 1.0))
    assert not passes_pretax_bar(ExcessStats(252, 3.0, 2.0, 1.0, 2.0, 0.99))


@pytest.mark.parametrize(
    "x,lags", [(np.array([]), 0), (np.array([np.nan]), 0), (np.array([1.0, 2.0]), -1)]
)
def test_newey_west_rejects_invalid_inputs(x, lags):
    with pytest.raises(ValueError):
        newey_west_se(x, lags)
