"""Tests for the live blend's trailing-curve and rebalance-weight overlay."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _mock_ohlcv(symbols: list[str], n_days: int = 150) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-10", periods=n_days, tz="UTC")
    np.random.seed(7)
    frames = {}
    for sym in symbols:
        price = 100.0 * np.exp(np.random.randn(n_days).cumsum() * 0.015)
        frames[sym] = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1e6,
            },
            index=dates,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


class TestComputeSleeveCurve:
    @patch("ggTrader.paper.overlay.universe_members_asof")
    @patch("ggTrader.paper.overlay.fetch_stock_ohlcv")
    def test_returns_equity_series_over_window(self, mock_fetch, mock_members):
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)

        from ggTrader.paper.overlay import compute_sleeve_curve

        asof = pd.Timestamp("2026-07-10", tz="UTC")
        curve = compute_sleeve_curve("sp500", asof, window_days=90)

        assert isinstance(curve, pd.Series)
        assert len(curve) > 0
        assert curve.index.max() <= asof

    @patch("ggTrader.paper.overlay.universe_members_asof")
    @patch("ggTrader.paper.overlay.fetch_stock_ohlcv")
    def test_uses_same_ensemble_construction_as_signal_runner(self, mock_fetch, mock_members):
        """Invariant 1: overlay must build EnsembleSignal identically to
        signal_runner.py -- no separate params, or the vol estimate silently
        describes a different strategy than the one actually live-trading."""
        import inspect

        from ggTrader.paper import overlay, signal_runner

        overlay_src = inspect.getsource(overlay.compute_sleeve_curve)
        runner_src = inspect.getsource(signal_runner.generate_signals)

        assert "EnsembleSignal(cfg)" in overlay_src
        assert "EnsembleSignal(cfg)" in runner_src
        assert "LabConfig(min_history_bars=60)" in overlay_src
        assert "LabConfig(min_history_bars=60)" in runner_src


class TestComputeWeightsAndScale:
    def test_weights_sum_to_one_and_scale_capped(self):
        from ggTrader.paper.overlay import compute_weights_and_scale

        dates = pd.bdate_range("2026-01-01", periods=120, tz="UTC")
        rng = np.random.default_rng(3)
        curves = {
            "sp500": pd.Series(10000 * np.exp(rng.normal(0, 0.01, 120).cumsum()), index=dates),
            "midcap400": pd.Series(10000 * np.exp(rng.normal(0, 0.02, 120).cumsum()), index=dates),
            "nasdaq100": pd.Series(10000 * np.exp(rng.normal(0, 0.015, 120).cumsum()), index=dates),
        }

        weights, scale = compute_weights_and_scale(curves, max_leverage=1.0)

        assert set(weights) == {"sp500", "midcap400", "nasdaq100"}
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert 0.0 <= scale <= 1.0


class TestShouldRebalance:
    def test_none_last_rebalance_triggers(self):
        from ggTrader.paper.overlay import should_rebalance

        assert should_rebalance(None, pd.Timestamp("2026-07-13", tz="UTC")) is True

    def test_same_month_does_not_trigger(self):
        from ggTrader.paper.overlay import should_rebalance

        assert should_rebalance("2026-07-01", pd.Timestamp("2026-07-13", tz="UTC")) is False

    def test_new_month_triggers(self):
        from ggTrader.paper.overlay import should_rebalance

        assert should_rebalance("2026-06-15", pd.Timestamp("2026-07-01", tz="UTC")) is True
