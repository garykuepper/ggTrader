"""The oracle itself: known values, and agreement with the engine on the shared conventions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.wfo.report import curve_summary
from tests.oracles.curve_stats import curve_stats


def test_curve_stats_known_values() -> None:
    idx = pd.date_range("2021-01-01", periods=253, freq="B", tz="UTC")
    curve = pd.Series(10000.0 * 1.0005 ** np.arange(253), index=idx)  # steady up
    s = curve_stats(curve)
    assert s["total_return_pct"] > 0
    assert s["max_drawdown_pct"] == 0.0  # monotonic up -> no drawdown
    assert s["sharpe"] > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engine_summary_matches_the_oracle_on_sharpe_vol_and_drawdown(seed: int) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=400, tz="UTC")
    curve = pd.Series(10000 * np.cumprod(1 + rng.normal(0.0004, 0.012, 400)), index=idx)
    mine, ref = curve_summary(curve), curve_stats(curve)
    for key in ("sharpe", "max_drawdown_pct", "total_return_pct", "ann_vol_pct"):
        assert mine[key] == pytest.approx(ref[key], rel=1e-10)
