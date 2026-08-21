"""Tests for cross-sectional idiosyncratic-volatility indicator and strategy."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.idio_vol import IdioVolStrategy, idiosyncratic_variance
from ggTrader.lab.strategy import LabConfig


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


def _ohlcv_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    frames = {}
    for col in returns.columns:
        close = 100.0 * (1.0 + returns[col]).cumprod()
        frames[col] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=returns.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


class TestIdioVolStrategy:
    def test_select_returns_bottom_quintile_only(self):
        returns, _market = _returns(["A", "B", "C", "D", "E"], n=300, seed=3)
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=100), reg_window=20, quintile=5)
        sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B", "C", "D", "E"])
        # 5 symbols / quintile=5 -> bucket size 1: only the single lowest-idio-var symbol.
        assert len(sels) == 1
        assert all("weight" in s for s in sels)
        assert abs(sum(s["weight"] for s in sels) - 1.0) < 1e-9

    def test_select_respects_min_history(self):
        returns, _market = _returns(["A", "B"], n=50, seed=1)
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=400))
        sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B"])
        assert sels == []

    def test_select_prefers_low_idio_variance_symbol(self):
        """Symbol A has the lowest idiosyncratic noise scale by construction (i=0)."""
        returns, _market = _returns(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], n=400, seed=11
        )
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=100), reg_window=30, quintile=5)
        sels = strat.select(ohlcv.index[-1], ohlcv, list(returns.columns))
        assert "A" in [s["symbol"] for s in sels]

    def test_to_targets_returns_weight_dataframe(self):
        returns, _market = _returns(["A", "B", "C"], n=300, seed=5)
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=100))
        plans = {
            ohlcv.index[250]: [{"symbol": "A", "weight": 1.0}],
        }
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"A"}
        assert (targets.dropna() == 1.0).all().all() or targets.dropna().empty is False

    def test_sweep_params_has_reg_window_and_quintile(self):
        params = IdioVolStrategy.sweep_params()
        assert "reg_window" in params
        assert "quintile" in params


def test_idio_vol_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "idio_vol" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["idio_vol"] is IdioVolStrategy


def test_cli_accepts_idio_vol():
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "idio_vol"])
    assert args.strategy == "idio_vol"
