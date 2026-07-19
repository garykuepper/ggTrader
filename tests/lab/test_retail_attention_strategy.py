"""Tests for the retail-attention (Google Trends search-spike) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ggTrader.lab.strategies.retail_attention import RetailAttentionStrategy

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


def _flat_ohlcv(symbols, n=400, start="2019-01-01"):
    idx = _idx(n, start)
    returns = pd.DataFrame({s: np.zeros(n) for s in symbols}, index=idx)
    return _ohlcv_from_returns(returns)


def _interest_row(symbol, date, value):
    return {"symbol": symbol, "date": pd.Timestamp(date), "search_interest": value}


class TestSelect:
    def test_longs_the_biggest_search_spike_symbol(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        month_starts = pd.date_range(asof - pd.Timedelta(days=180), asof, freq="MS")

        rows = []
        for sym, spike in [("A", 1.0), ("B", 5.0), ("C", 1.1), ("D", 0.9), ("E", 1.2)]:
            for d in month_starts[:-1]:
                rows.append(_interest_row(sym, d, 20))
            rows.append(_interest_row(sym, month_starts[-1], 20 * spike))
        interest = pd.DataFrame(rows)

        strat = RetailAttentionStrategy(
            LabConfig(min_history_bars=30),
            lookback_months=3,
            quintile=5,
            publish_lag_days=0,
            _interest_loader=lambda symbols, start, end: interest,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert len(sels) == 1
        assert sels[0]["symbol"] == "B"

    def test_no_interest_data_returns_empty_plan(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        strat = RetailAttentionStrategy(
            LabConfig(min_history_bars=30),
            _interest_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "date", "search_interest"]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_respects_min_history(self):
        symbols = ["A", "B"]
        ohlcv = _flat_ohlcv(symbols, n=25)
        strat = RetailAttentionStrategy(
            LabConfig(min_history_bars=400),
            _interest_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "date", "search_interest"]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_publish_lag_excludes_too_recent_reading(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        rows = [_interest_row(sym, asof - pd.Timedelta(days=1), 50) for sym in symbols]
        interest = pd.DataFrame(rows)
        strat = RetailAttentionStrategy(
            LabConfig(min_history_bars=30),
            lookback_months=3,
            publish_lag_days=7,
            _interest_loader=lambda symbols, start, end: interest,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_weight_is_one_over_bucket_size(self):
        symbols = ["A", "B", "C", "D"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        month_starts = pd.date_range(asof - pd.Timedelta(days=120), asof, freq="MS")
        rows = []
        for sym, spike in [("A", 1.0), ("B", 3.0), ("C", 1.2), ("D", 0.8)]:
            for d in month_starts[:-1]:
                rows.append(_interest_row(sym, d, 20))
            rows.append(_interest_row(sym, month_starts[-1], 20 * spike))
        interest = pd.DataFrame(rows)
        strat = RetailAttentionStrategy(
            LabConfig(min_history_bars=30),
            lookback_months=2,
            quintile=4,
            publish_lag_days=0,
            _interest_loader=lambda symbols, start, end: interest,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels
        for s in sels:
            assert s["weight"] == pytest.approx(1.0 / len(sels))


class TestToTargets:
    def test_returns_weight_dataframe(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=60)
        strat = RetailAttentionStrategy(LabConfig(min_history_bars=30))
        plans = {ohlcv.index[50]: [{"symbol": "A", "weight": 1.0}]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"A"}


def test_sweep_params_has_lookback_months_and_quintile():
    params = RetailAttentionStrategy.sweep_params()
    assert "lookback_months" in params
    assert "quintile" in params


def test_retail_attention_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "retail_attention" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["retail_attention"] is RetailAttentionStrategy
