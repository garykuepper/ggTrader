"""Tests for the portfolio-blend helper and orchestrator."""

import numpy as np
import pandas as pd

from ggTrader.lab.blend import blend_curves


def _idx(n, start="2021-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _equity_from_returns(rets: pd.Series, start=100000.0) -> pd.Series:
    return (1.0 + rets).cumprod() * start


def test_blend_curves_aligns_on_intersection_and_blends():
    idx = _idx(400)
    np.random.seed(0)
    a = _equity_from_returns(pd.Series(np.random.normal(0.0004, 0.01, 400), index=idx))
    # b starts 50 bars later -> intersection trims the blend to the common span
    b = _equity_from_returns(pd.Series(np.random.normal(0.0004, 0.01, 350), index=idx[50:]))
    blend_eq, returns_df, diag = blend_curves({"A@sp500": a, "B@nasdaq100": b})
    assert list(returns_df.columns) == ["A@sp500", "B@nasdaq100"]
    assert returns_df.index.min() >= idx[50]  # trimmed to the later start
    assert blend_eq.notna().all()
    assert (blend_eq > 0).all()


def test_blend_curves_equal_vol_gives_balanced_weights():
    """Two sleeves with the same vol get ~50/50 inverse-vol weight (diag)."""
    idx = _idx(400)
    rng = np.random.default_rng(1)
    a = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.012, 400), index=idx))
    b = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.012, 400), index=idx))
    _, _, diag = blend_curves({"A@sp500": a, "B@nasdaq100": b}, window=60)
    last = diag.iloc[-1]
    assert abs(last["w_A@sp500"] - last["w_B@nasdaq100"]) < 0.15  # near-balanced
