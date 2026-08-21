"""Tests for the S&P 500 index-deletion-overshoot fade strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.index_deletion import IndexDeletionFadeStrategy, deletion_events
from ggTrader.lab.strategy import LabConfig


def _history(rows: list[tuple[str, list[str]]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d, _ in rows], name="date")
    return pd.DataFrame({"tickers": [t for _, t in rows]}, index=idx)


class TestDeletionEvents:
    def test_detects_a_single_deletion(self):
        history = _history(
            [
                ("2020-01-01", ["A", "B", "C"]),
                ("2020-02-01", ["A", "C"]),  # B deleted
            ]
        )
        events = deletion_events(history)
        assert len(events) == 1
        assert events.iloc[0]["symbol"] == "B"
        assert events.iloc[0]["deletion_date"] == pd.Timestamp("2020-02-01", tz="UTC")

    def test_detects_multiple_deletions_in_one_transition(self):
        history = _history(
            [
                ("2020-01-01", ["A", "B", "C", "D"]),
                ("2020-02-01", ["A"]),  # B, C, D all deleted
            ]
        )
        events = deletion_events(history)
        assert set(events["symbol"]) == {"B", "C", "D"}

    def test_an_addition_alone_is_not_a_deletion(self):
        history = _history(
            [
                ("2020-01-01", ["A", "B"]),
                ("2020-02-01", ["A", "B", "C"]),  # C added, nothing removed
            ]
        )
        events = deletion_events(history)
        assert events.empty

    def test_symbol_readded_later_produces_a_second_event_if_deleted_again(self):
        history = _history(
            [
                ("2020-01-01", ["A", "B"]),
                ("2020-02-01", ["A"]),  # B deleted
                ("2020-03-01", ["A", "B"]),  # B re-added
                ("2020-04-01", ["A"]),  # B deleted again
            ]
        )
        events = deletion_events(history)
        b_events = events[events["symbol"] == "B"]
        assert len(b_events) == 2
        assert set(b_events["deletion_date"]) == {
            pd.Timestamp("2020-02-01", tz="UTC"),
            pd.Timestamp("2020-04-01", tz="UTC"),
        }

    def test_normalizes_index_style_tickers_to_yfinance_style(self):
        history = _history(
            [
                ("2020-01-01", ["BRK.B", "A"]),
                ("2020-02-01", ["A"]),  # BRK.B deleted
            ]
        )
        events = deletion_events(history)
        assert events.iloc[0]["symbol"] == "BRK-B"

    def test_single_row_history_has_no_events(self):
        history = _history([("2020-01-01", ["A", "B"])])
        events = deletion_events(history)
        assert events.empty


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


class TestSelect:
    def test_holds_a_recently_deleted_symbol(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        events = pd.DataFrame([{"symbol": "B", "deletion_date": asof - pd.Timedelta(days=10)}])
        strat = IndexDeletionFadeStrategy(
            LabConfig(min_history_bars=30),
            hold_days=63,
            _events_loader=lambda: events,
        )
        # "eligible" reflects CURRENT membership -- B is deleted, so it's
        # deliberately absent here; the strategy must still find it via the
        # deletion-events side channel, the same pattern pairs_stat_arb.py
        # uses for its sector map.
        sels = strat.select(asof, ohlcv, ["A", "C"])
        assert len(sels) == 1
        assert sels[0]["symbol"] == "B"

    def test_deletion_outside_hold_window_is_excluded(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        events = pd.DataFrame([{"symbol": "B", "deletion_date": asof - pd.Timedelta(days=200)}])
        strat = IndexDeletionFadeStrategy(
            LabConfig(min_history_bars=30),
            hold_days=63,
            _events_loader=lambda: events,
        )
        sels = strat.select(asof, ohlcv, ["A", "C"])
        assert sels == []

    def test_future_deletion_relative_to_asof_is_excluded(self):
        """No look-ahead: a deletion dated after asof must not be visible yet."""
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-100]
        events = pd.DataFrame(
            [{"symbol": "B", "deletion_date": ohlcv.index[-1]}]  # far in "the future"
        )
        strat = IndexDeletionFadeStrategy(
            LabConfig(min_history_bars=30),
            hold_days=63,
            _events_loader=lambda: events,
        )
        sels = strat.select(asof, ohlcv, ["A", "C"])
        assert sels == []

    def test_symbol_with_no_price_data_at_asof_is_skipped(self):
        """An acquired/fully-delisted symbol whose price series has already
        gone to NaN by asof (yfinance stopped covering it) must not be
        selected -- there's nothing tradeable to hold."""
        symbols = ["A", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        # Give "B" a price series that goes NaN well before asof.
        idx = ohlcv.index
        b_close = pd.Series(100.0, index=idx)
        b_close.iloc[-50:] = np.nan
        b_frame = pd.DataFrame(
            {
                "open": b_close,
                "high": b_close,
                "low": b_close,
                "close": b_close,
                "volume": np.full(len(idx), 1e6),
            }
        )
        ohlcv = pd.concat([ohlcv, pd.concat({"B": b_frame}, axis=1)], axis=1)
        ohlcv.columns = ohlcv.columns.set_names(["symbol", "field"])

        asof = idx[-1]
        events = pd.DataFrame([{"symbol": "B", "deletion_date": asof - pd.Timedelta(days=10)}])
        strat = IndexDeletionFadeStrategy(
            LabConfig(min_history_bars=30),
            hold_days=63,
            _events_loader=lambda: events,
        )
        sels = strat.select(asof, ohlcv, ["A", "C"])
        assert sels == []

    def test_multiple_active_deletions_are_equal_weighted(self):
        symbols = ["A", "B", "C", "D"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        events = pd.DataFrame(
            [
                {"symbol": "B", "deletion_date": asof - pd.Timedelta(days=10)},
                {"symbol": "D", "deletion_date": asof - pd.Timedelta(days=20)},
            ]
        )
        strat = IndexDeletionFadeStrategy(
            LabConfig(min_history_bars=30),
            hold_days=63,
            _events_loader=lambda: events,
        )
        sels = strat.select(asof, ohlcv, ["A", "C"])
        assert {s["symbol"] for s in sels} == {"B", "D"}
        for s in sels:
            assert s["weight"] == pytest.approx(0.5)


class TestToTargets:
    def test_returns_weight_dataframe(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=60)
        strat = IndexDeletionFadeStrategy(LabConfig(min_history_bars=30))
        plans = {ohlcv.index[50]: [{"symbol": "B", "weight": 1.0}]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"B"}


def test_sweep_params_has_hold_days():
    params = IndexDeletionFadeStrategy.sweep_params()
    assert "hold_days" in params


def test_index_deletion_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "index_deletion_fade" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["index_deletion_fade"] is IndexDeletionFadeStrategy
