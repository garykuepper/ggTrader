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
