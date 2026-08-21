"""Tests for the MAX-effect (lottery-demand) indicator and strategy."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.max_effect import MaxEffectStrategy, trailing_max_return
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


class TestTrailingMaxReturn:
    def test_output_shape(self):
        idx = _idx(60)
        returns = pd.DataFrame({"A": np.full(60, 0.001), "B": np.full(60, 0.002)}, index=idx)
        max_ret = trailing_max_return(returns, window=20)
        assert max_ret.shape == returns.shape
        assert list(max_ret.columns) == ["A", "B"]

    def test_warmup_is_nan(self):
        idx = _idx(60)
        returns = pd.DataFrame({"A": np.full(60, 0.001)}, index=idx)
        max_ret = trailing_max_return(returns, window=20)
        assert max_ret["A"].iloc[:19].isna().all()
        assert max_ret["A"].iloc[19:].notna().all()

    def test_picks_the_single_largest_return_in_window(self):
        idx = _idx(30)
        vals = np.full(30, 0.001)
        vals[10] = 0.15  # one big spike inside the trailing window at the end
        returns = pd.Series(vals, index=idx)
        max_ret = trailing_max_return(pd.DataFrame({"A": returns}), window=20)
        # Last bar's trailing 20-day window (bars 10..29) includes the spike at bar 10.
        assert max_ret["A"].iloc[-1] == pytest.approx(0.15)

    def test_spike_outside_window_is_excluded(self):
        idx = _idx(40)
        vals = np.full(40, 0.001)
        vals[5] = 0.20  # spike well before the trailing window ends
        returns = pd.Series(vals, index=idx)
        max_ret = trailing_max_return(pd.DataFrame({"A": returns}), window=10)
        # Last bar's trailing 10-day window (bars 30..39) does not include bar 5.
        assert max_ret["A"].iloc[-1] == pytest.approx(0.001)


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


def _returns_with_distinct_spikes(symbols, n=60, seed=1):
    """Each symbol gets a baseline drift plus its own one-off spike of
    increasing size (symbols later in the list get bigger spikes), so MAX
    ranking is deterministic and distinguishable across symbols."""
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    data = {}
    for i, s in enumerate(symbols):
        vals = rng.normal(0.0002, 0.001, n)
        vals[-5] = 0.02 * (i + 1)  # spike inside the trailing window, scaled by index
        data[s] = vals
    return pd.DataFrame(data, index=idx)


class TestMaxEffectStrategy:
    def test_select_returns_bottom_quintile_only(self):
        returns = _returns_with_distinct_spikes(["A", "B", "C", "D", "E"], n=60, seed=2)
        ohlcv = _ohlcv_from_returns(returns)
        strat = MaxEffectStrategy(LabConfig(min_history_bars=30), window=20, quintile=5)
        sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B", "C", "D", "E"])
        # 5 symbols / quintile=5 -> bucket size 1: only the single lowest-MAX symbol.
        assert len(sels) == 1
        assert all("weight" in s for s in sels)
        assert abs(sum(s["weight"] for s in sels) - 1.0) < 1e-9

    def test_select_prefers_low_max_symbol(self):
        """Symbol A has the smallest spike by construction (i=0) -> lowest MAX."""
        returns = _returns_with_distinct_spikes(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], n=60, seed=11
        )
        ohlcv = _ohlcv_from_returns(returns)
        strat = MaxEffectStrategy(LabConfig(min_history_bars=30), window=20, quintile=5)
        sels = strat.select(ohlcv.index[-1], ohlcv, list(returns.columns))
        assert "A" in [s["symbol"] for s in sels]
        assert "J" not in [s["symbol"] for s in sels]

    def test_select_respects_min_history(self):
        returns = _returns_with_distinct_spikes(["A", "B"], n=25, seed=1)
        ohlcv = _ohlcv_from_returns(returns)
        strat = MaxEffectStrategy(LabConfig(min_history_bars=400))
        sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B"])
        assert sels == []

    def test_to_targets_returns_weight_dataframe(self):
        returns = _returns_with_distinct_spikes(["A", "B", "C"], n=60, seed=5)
        ohlcv = _ohlcv_from_returns(returns)
        strat = MaxEffectStrategy(LabConfig(min_history_bars=30))
        plans = {ohlcv.index[50]: [{"symbol": "A", "weight": 1.0}]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"A"}

    def test_sweep_params_has_window_and_quintile(self):
        params = MaxEffectStrategy.sweep_params()
        assert "window" in params
        assert "quintile" in params


def test_max_effect_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "max_effect" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["max_effect"] is MaxEffectStrategy
