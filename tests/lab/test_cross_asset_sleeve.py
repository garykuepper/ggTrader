"""Tests for the cross_asset_sleeve strategy (SPY + TLT/GLD/PDBC sleeve, static or trend)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.cross_asset_sleeve import (
    CROSS_ASSET_SLEEVE_UNIVERSE,
    CrossAssetSleeveStrategy,
)
from ggTrader.lab.strategy import LabConfig


def _ohlcv(n: int = 300) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    paths = {
        "SPY": np.linspace(100, 150, n),
        "TLT": np.linspace(100, 70, n),  # downtrend -> off in trend arm
        "GLD": np.linspace(100, 130, n),
        "PDBC": np.linspace(100, 110, n),
        "BIL": np.linspace(100, 104, n),
    }
    cols = pd.MultiIndex.from_product([CROSS_ASSET_SLEEVE_UNIVERSE, ["close"]])
    return pd.DataFrame({(s, "close"): paths[s] for s, _ in cols}, index=idx)


def _weights(plan):
    return {s["symbol"]: s["weight"] for s in plan}


def test_static_arm_holds_every_leg():
    data = _ohlcv()
    w = _weights(
        CrossAssetSleeveStrategy(LabConfig(), trend=False).select(data.index[-1], data, [])
    )
    assert w["SPY"] == 0.8 and w["BIL"] == 0.0
    assert all(abs(w[leg] - 0.2 / 3) < 1e-12 for leg in ("TLT", "GLD", "PDBC"))


def test_trend_arm_moves_downtrending_leg_to_bil_and_sums_to_one():
    data = _ohlcv()
    w = _weights(CrossAssetSleeveStrategy(LabConfig(lookback=252)).select(data.index[-1], data, []))
    assert w["TLT"] == 0.0 and abs(w["BIL"] - 0.2 / 3) < 1e-12
    assert w["GLD"] > 0 and w["PDBC"] > 0
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_trend_arm_ignores_data_after_asof():
    data = _ohlcv()
    asof = data.index[260]
    later = data.copy()
    later.loc[later.index > asof, ("GLD", "close")] = 1.0  # future crash must not matter
    strat = CrossAssetSleeveStrategy(LabConfig(lookback=252))
    assert _weights(strat.select(asof, data, [])) == _weights(strat.select(asof, later, []))


def test_targets_apply_on_next_bar():
    data = _ohlcv()
    strat = CrossAssetSleeveStrategy(LabConfig(), trend=False)
    asof = data.index[100]
    t = strat.to_targets({asof: strat.select(asof, data, [])}, data)
    assert t.loc[asof].isna().all()
    assert t.loc[data.index[101], "SPY"] == 0.8
