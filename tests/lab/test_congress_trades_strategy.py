"""Tests for the Congressional (House STOCK Act) trade-mirroring strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.congress_trades import CongressTradeMirrorStrategy
from ggTrader.lab.strategy import LabConfig


def _tx(symbol, date, tx_type="P"):
    return {
        "symbol": symbol,
        "transaction_type": tx_type,
        "transaction_date": pd.Timestamp(date),
        "filing_date": pd.Timestamp(date) + pd.Timedelta(days=3),
    }


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


class TestSelect:
    def test_holds_a_symbol_with_a_recent_purchase(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame([_tx("B", asof - pd.Timedelta(days=20))])
        strat = CongressTradeMirrorStrategy(
            LabConfig(min_history_bars=30),
            hold_days=252,
            report_lag_days=0,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert len(sels) == 1
        assert sels[0]["symbol"] == "B"

    def test_ignores_sale_transactions(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame([_tx("B", asof - pd.Timedelta(days=20), tx_type="S")])
        strat = CongressTradeMirrorStrategy(
            LabConfig(min_history_bars=30),
            report_lag_days=0,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_no_transactions_returns_empty_plan(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        strat = CongressTradeMirrorStrategy(
            LabConfig(min_history_bars=30),
            _tx_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "transaction_type", "transaction_date", "filing_date"]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_purchase_older_than_hold_days_is_excluded(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame([_tx("B", asof - pd.Timedelta(days=300))])
        strat = CongressTradeMirrorStrategy(
            LabConfig(min_history_bars=30),
            hold_days=126,
            report_lag_days=0,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_report_lag_excludes_too_recently_filed_purchase(self):
        """STOCK Act allows up to 45 days between the trade and its public
        disclosure -- the filing_date (not the trade date) is what gates
        real-world knowability."""
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame([_tx("B", asof - pd.Timedelta(days=1))])  # filed practically today
        strat = CongressTradeMirrorStrategy(
            LabConfig(min_history_bars=30),
            report_lag_days=5,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_respects_min_history(self):
        symbols = ["A", "B"]
        ohlcv = _flat_ohlcv(symbols, n=25)
        strat = CongressTradeMirrorStrategy(
            LabConfig(min_history_bars=400),
            _tx_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "transaction_type", "transaction_date", "filing_date"]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_multiple_active_purchases_are_equal_weighted(self):
        symbols = ["A", "B", "C", "D"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame(
            [_tx("B", asof - pd.Timedelta(days=20)), _tx("D", asof - pd.Timedelta(days=15))]
        )
        strat = CongressTradeMirrorStrategy(
            LabConfig(min_history_bars=30),
            hold_days=252,
            report_lag_days=0,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert {s["symbol"] for s in sels} == {"B", "D"}
        for s in sels:
            assert s["weight"] == pytest.approx(0.5)


class TestToTargets:
    def test_returns_weight_dataframe(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=60)
        strat = CongressTradeMirrorStrategy(LabConfig(min_history_bars=30))
        plans = {ohlcv.index[50]: [{"symbol": "B", "weight": 1.0}]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"B"}


def test_sweep_params_has_hold_days():
    params = CongressTradeMirrorStrategy.sweep_params()
    assert "hold_days" in params


def test_congress_trades_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "congress_trades" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["congress_trades"] is CongressTradeMirrorStrategy
