"""Tests for the post-earnings-announcement-drift (PEAD) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ggTrader.lab.strategies.pead import PeadStrategy

from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
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


def _flat_ohlcv(symbols, n=300):
    idx = _idx(n)
    returns = pd.DataFrame({s: np.zeros(n) for s in symbols}, index=idx)
    return _ohlcv_from_returns(returns)


def _surprise_row(symbol, earnings_date, surprise_pct):
    return {
        "symbol": symbol,
        "earnings_date": pd.Timestamp(earnings_date),
        "surprise_pct": surprise_pct,
    }


class TestSelect:
    def test_longs_the_biggest_positive_surprise_symbol(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]

        surprises = pd.DataFrame(
            [
                _surprise_row("A", asof - pd.Timedelta(days=20), 25.0),  # biggest beat -> long
                _surprise_row("B", asof - pd.Timedelta(days=20), -10.0),
                _surprise_row("C", asof - pd.Timedelta(days=20), 0.0),
                _surprise_row("D", asof - pd.Timedelta(days=20), 2.0),
                _surprise_row("E", asof - pd.Timedelta(days=20), 5.0),
            ]
        )

        strat = PeadStrategy(
            LabConfig(min_history_bars=100),
            max_age_days=90,
            quintile=5,
            report_lag_days=0,
            _surprise_loader=lambda symbols, start, end: surprises,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert len(sels) == 1
        assert sels[0]["symbol"] == "A"

    def test_no_surprise_data_returns_empty_plan(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        strat = PeadStrategy(
            LabConfig(min_history_bars=100),
            _surprise_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "earnings_date", "surprise_pct"]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_respects_min_history(self):
        symbols = ["A", "B"]
        ohlcv = _flat_ohlcv(symbols, n=50)
        strat = PeadStrategy(
            LabConfig(min_history_bars=400),
            _surprise_loader=lambda symbols, start, end: pd.DataFrame(
                [_surprise_row("A", "2020-01-15", 5.0), _surprise_row("B", "2020-01-15", 1.0)]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_stale_surprise_outside_drift_window_is_excluded(self):
        """A surprise older than max_age_days has drifted its course --
        must not still be driving a position."""
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]

        surprises = pd.DataFrame(
            [_surprise_row(s, asof - pd.Timedelta(days=200), 20.0) for s in symbols]
        )
        strat = PeadStrategy(
            LabConfig(min_history_bars=100),
            max_age_days=90,
            report_lag_days=0,
            _surprise_loader=lambda symbols, start, end: surprises,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_report_lag_excludes_too_recent_earnings(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]

        surprises = pd.DataFrame(
            [_surprise_row(s, asof, 20.0) for s in symbols]  # reported literally today
        )
        strat = PeadStrategy(
            LabConfig(min_history_bars=100),
            report_lag_days=1,
            _surprise_loader=lambda symbols, start, end: surprises,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_weight_is_one_over_bucket_size(self):
        symbols = ["A", "B", "C", "D"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]
        surprises = pd.DataFrame(
            [
                _surprise_row("A", asof - pd.Timedelta(days=10), 20.0),
                _surprise_row("B", asof - pd.Timedelta(days=10), 15.0),
                _surprise_row("C", asof - pd.Timedelta(days=10), 5.0),
                _surprise_row("D", asof - pd.Timedelta(days=10), 1.0),
            ]
        )
        strat = PeadStrategy(
            LabConfig(min_history_bars=100),
            quintile=4,
            report_lag_days=0,
            _surprise_loader=lambda symbols, start, end: surprises,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels
        for s in sels:
            assert s["weight"] == pytest.approx(1.0 / len(sels))


class TestToTargets:
    def test_returns_weight_dataframe(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=60)
        strat = PeadStrategy(LabConfig(min_history_bars=30))
        plans = {ohlcv.index[50]: [{"symbol": "A", "weight": 1.0}]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"A"}


def test_sweep_params_has_max_age_days_and_quintile():
    params = PeadStrategy.sweep_params()
    assert "max_age_days" in params
    assert "quintile" in params


def test_pead_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "pead" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["pead"] is PeadStrategy
