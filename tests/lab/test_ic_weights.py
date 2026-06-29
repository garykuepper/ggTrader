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


# ---------------------------------------------------------------------------
# Task 3: ic_weight_schedule tests
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from ggTrader.lab.strategies.ic_weights import ic_weight_schedule  # noqa: E402


def _ramp_close(n=900, n_syms=12, seed=3):
    np.random.seed(seed)
    idx = _idx(n)
    cols = {f"S{i}": 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))) for i in range(n_syms)}
    return pd.DataFrame(cols, index=idx)


def test_weights_sum_to_one_each_row():
    close = _ramp_close()
    raw = {"a": close.pct_change(), "b": -close.pct_change()}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    row_sums = w.sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), 1.0, atol=1e-9)


def test_warmup_is_equal_weight():
    close = _ramp_close()
    raw = {"a": close.pct_change(), "b": -close.pct_change()}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    # First row (no full trailing window yet) is equal-weight.
    assert np.allclose(w.iloc[0].to_numpy(), 0.5, atol=1e-9)


def test_all_nonpositive_ic_falls_back_to_equal():
    """Two voters that both anti-predict -> clip(0) zeros both -> equal weights."""
    close = _ramp_close()
    fwd = forward_returns(close, 3)
    anti = -fwd  # perfectly anti-correlated raw -> negative IC
    raw = {"a": anti, "b": anti}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    assert np.allclose(w.iloc[-1].to_numpy(), 0.5, atol=1e-9)


def test_predictive_voter_gets_more_weight():
    """A voter whose raw == forward return should out-weight a noise voter."""
    close = _ramp_close()
    fwd = forward_returns(close, 3)
    np.random.seed(99)
    noise = pd.DataFrame(
        np.random.normal(size=close.shape), index=close.index, columns=close.columns
    )
    raw = {"good": fwd.fillna(0.0), "noise": noise}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    assert w["good"].iloc[-1] > w["noise"].iloc[-1]


def test_truncation_invariance_leak_guard():
    """Weights up to date d are identical whether or not post-d rows exist."""
    close = _ramp_close()
    raw = {"a": close.pct_change(), "b": -close.pct_change()}
    d = close.index[600]
    full = ic_weight_schedule(raw, close, lookback_months=6)
    raw_trunc = {k: v.loc[:d] for k, v in raw.items()}
    trunc = ic_weight_schedule(raw_trunc, close.loc[:d], lookback_months=6)
    aligned = full.loc[:d]
    assert np.allclose(aligned.to_numpy(), trunc.to_numpy(), atol=1e-9, equal_nan=True)


def test_nan_ic_voter_pruned_rows_still_sum_to_one():
    """A voter whose raw is all-NaN over the window gets weight 0; rows sum to 1."""
    close = _ramp_close()
    # Use perfect-predictor raw (forward returns) so the "good" voter always has
    # positive IC regardless of the random seed — this guarantees the degenerate
    # equal-weight fallback never fires and the nanvoter is unambiguously 0.
    good = forward_returns(close, 3).fillna(0.0)
    nanvoter = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    raw = {"good": good, "nanvoter": nanvoter}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    assert np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    # after warmup, the all-NaN voter carries ~zero weight
    assert w["nanvoter"].iloc[-1] < 1e-9
