"""Tests for cross-sectional idiosyncratic-volatility indicator and strategy."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.idio_vol import idiosyncratic_variance


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _returns(symbols, n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    market = rng.normal(0.0005, 0.01, n)
    data = {}
    for i, s in enumerate(symbols):
        # Each symbol = beta*market + idiosyncratic noise of varying scale.
        idio_scale = 0.005 * (i + 1)
        data[s] = 0.8 * market + rng.normal(0, idio_scale, n)
    return pd.DataFrame(data, index=idx), pd.Series(market, index=idx)


class TestIdiosyncraticVariance:
    def test_output_shape(self):
        returns, market = _returns(["A", "B", "C"], n=200)
        resid_var = idiosyncratic_variance(returns, market, window=20)
        assert resid_var.shape == returns.shape
        assert list(resid_var.columns) == ["A", "B", "C"]

    def test_warmup_is_nan(self):
        returns, market = _returns(["A"], n=100)
        resid_var = idiosyncratic_variance(returns, market, window=20)
        assert resid_var["A"].iloc[:19].isna().all()
        assert resid_var["A"].iloc[38:].notna().all()

    def test_higher_idio_noise_gives_higher_residual_variance(self):
        """Symbol C has 3x the idiosyncratic noise scale of A by construction."""
        returns, market = _returns(["A", "B", "C"], n=300, seed=7)
        resid_var = idiosyncratic_variance(returns, market, window=60)
        last = resid_var.iloc[-1]
        assert last["C"] > last["A"]

    def test_zero_market_variance_does_not_raise(self):
        idx = _idx(50)
        returns = pd.DataFrame({"A": np.full(50, 0.001)}, index=idx)
        market = pd.Series(np.zeros(50), index=idx)  # constant market -> Var=0
        resid_var = idiosyncratic_variance(returns, market, window=10)
        assert resid_var.shape == (50, 1)
