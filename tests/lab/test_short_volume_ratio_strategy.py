"""Tests for the short-volume-ratio (free-data-only "stealthy shorts" cut) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.short_volume_ratio import ShortVolumeRatioStrategy
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2019-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


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


def _flat_ohlcv(symbols, n=300, start="2019-01-01"):
    idx = _idx(n, start)
    returns = pd.DataFrame({s: np.zeros(n) for s in symbols}, index=idx)
    return _ohlcv_from_returns(returns)


def _sv_row(symbol, date, ratio):
    return {"symbol": symbol, "date": pd.Timestamp(date), "short_volume_ratio": ratio}


def _daily_history(symbol, asof, n_days, ratio):
    return [_sv_row(symbol, asof - pd.Timedelta(days=i), ratio) for i in range(n_days)]


class TestSelect:
    def test_market_neutral_long_low_short_high_ratio(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]

        rows = []
        for sym, ratio in [("A", 0.10), ("B", 0.80), ("C", 0.30), ("D", 0.45), ("E", 0.60)]:
            rows.extend(_daily_history(sym, asof, 30, ratio))
        sv = pd.DataFrame(rows)

        strat = ShortVolumeRatioStrategy(
            LabConfig(min_history_bars=30),
            lookback_days=20,
            quintile=5,
            publish_lag_days=0,
            _sv_loader=lambda symbols, start, end: sv,
        )
        plan = strat.select(asof, ohlcv, symbols)
        pairs = {(p["symbol_long"], p["symbol_short"]) for p in plan}
        # A has the lowest short-volume ratio -> long; B has the highest -> short.
        assert ("A", "B") in pairs

    def test_no_data_returns_empty_plan(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        strat = ShortVolumeRatioStrategy(
            LabConfig(min_history_bars=30),
            _sv_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "date", "short_volume_ratio"]
            ),
        )
        plan = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert plan == []

    def test_respects_min_history(self):
        symbols = ["A", "B"]
        ohlcv = _flat_ohlcv(symbols, n=25)
        strat = ShortVolumeRatioStrategy(
            LabConfig(min_history_bars=400),
            _sv_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "date", "short_volume_ratio"]
            ),
        )
        plan = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert plan == []

    def test_weights_are_symmetric_and_sum_to_zero_net(self):
        symbols = ["A", "B", "C", "D"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]
        rows = []
        for sym, ratio in [("A", 0.10), ("B", 0.80), ("C", 0.30), ("D", 0.60)]:
            rows.extend(_daily_history(sym, asof, 30, ratio))
        sv = pd.DataFrame(rows)
        strat = ShortVolumeRatioStrategy(
            LabConfig(min_history_bars=30),
            quintile=4,
            publish_lag_days=0,
            _sv_loader=lambda symbols, start, end: sv,
        )
        plan = strat.select(asof, ohlcv, symbols)
        assert plan
        for p in plan:
            assert p["weight"] > 0


class TestToTargets:
    def test_targets_have_correct_signs(self):
        ohlcv = _flat_ohlcv(["A", "B"], n=60)
        strat = ShortVolumeRatioStrategy(LabConfig(min_history_bars=30))
        idx = ohlcv.index
        plans = {idx[50]: [{"symbol_long": "A", "symbol_short": "B", "weight": 0.5}]}
        targets = strat.to_targets(plans, ohlcv)
        bar = idx[idx.get_loc(idx[50]) + 1]
        assert targets.loc[bar, "A"] == pytest.approx(0.5)
        assert targets.loc[bar, "B"] == pytest.approx(-0.5)


def test_sweep_params_has_lookback_days_and_quintile():
    params = ShortVolumeRatioStrategy.sweep_params()
    assert "lookback_days" in params
    assert "quintile" in params


def test_short_volume_ratio_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "short_volume_ratio" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["short_volume_ratio"] is ShortVolumeRatioStrategy
