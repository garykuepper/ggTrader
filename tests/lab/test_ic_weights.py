"""Tests for the causal IC weight schedule."""

import pandas as pd

from ggTrader.lab.strategies.ic_weights import (
    daily_cross_sectional_ic,
    forward_returns,
)


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_forward_returns_shifts_up_by_horizon():
    idx = _idx(5)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=idx)
    fwd = forward_returns(close, horizon=1)
    # fwd[t] = close[t+1]/close[t]-1
    assert abs(fwd["A"].iloc[0] - (11.0 / 10.0 - 1.0)) < 1e-9
    assert pd.isna(fwd["A"].iloc[-1])  # no t+1 for the last bar


def test_daily_ic_perfect_positive_rank():
    """raw ranking that matches forward-return ranking gives IC ~ +1."""
    idx = _idx(1)
    raw = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03]}, index=idx)
    ic = daily_cross_sectional_ic(raw, fwd, min_names=3)
    assert abs(ic.iloc[0] - 1.0) < 1e-9


def test_daily_ic_perfect_negative_rank():
    idx = _idx(1)
    raw = pd.DataFrame({"A": [3.0], "B": [2.0], "C": [1.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03]}, index=idx)
    ic = daily_cross_sectional_ic(raw, fwd, min_names=3)
    assert abs(ic.iloc[0] - (-1.0)) < 1e-9


def test_daily_ic_nan_below_min_names():
    idx = _idx(1)
    raw = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01], "B": [0.02]}, index=idx)
    ic = daily_cross_sectional_ic(raw, fwd, min_names=3)
    assert pd.isna(ic.iloc[0])
