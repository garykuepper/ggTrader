import numpy as np
import pandas as pd

from ggTrader.lab.metrics import benchmark, curve_stats


def test_curve_stats_known_values():
    idx = pd.date_range("2021-01-01", periods=253, freq="B", tz="UTC")
    curve = pd.Series(10000.0 * 1.0005 ** np.arange(253), index=idx)  # steady up
    s = curve_stats(curve)
    assert s["total_return_pct"] > 0
    assert s["max_drawdown_pct"] == 0.0  # monotonic up -> no drawdown
    assert s["sharpe"] > 0


def test_benchmark_shape():
    idx = pd.date_range("2021-01-01", periods=253, freq="B", tz="UTC")
    equity = pd.Series(10000.0 * 1.0006 ** np.arange(253), index=idx)
    spy = pd.Series(400.0 * 1.0004 ** np.arange(253), index=idx)
    rep = benchmark(equity, spy, 10000.0)
    assert set(rep) == {"strategy", "spy", "monthly_hit_rate_vs_spy", "n_months"}
    assert "sharpe" in rep["strategy"] and "sharpe" in rep["spy"]
