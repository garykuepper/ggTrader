"""Tests for the live blend's trailing-curve and rebalance-weight overlay."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _mock_ohlcv(symbols: list[str], n_days: int = 150, end: str = "2026-07-10") -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=n_days, tz="UTC")
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


def _mock_ohlcv_truncated(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Like _mock_ohlcv but actually respects a [start, end] range, unlike
    the fixed-range helper above -- needed to catch warmup-buffer bugs that
    a mock ignoring its requested date range would silently mask."""
    dates = pd.bdate_range(start=start, end=end, tz="UTC")
    np.random.seed(7)
    frames = {}
    for sym in symbols:
        price = 100.0 * np.exp(np.random.randn(len(dates)).cumsum() * 0.015)
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

    @patch("ggTrader.paper.overlay.universe_members_asof")
    @patch("ggTrader.paper.overlay.fetch_stock_ohlcv")
    def test_warmup_buffer_leaves_full_window_of_trading_days(self, mock_fetch, mock_members):
        """min_history_bars=60 is TRADING days; the fetch-start buffer must be
        padded in calendar days generously enough that consuming 60 trading
        days of indicator warmup still leaves close to a full window_days of
        signal-eligible curve left over. compute_weights_and_scale computes
        returns via curve.pct_change().dropna(), which drops one row -- a
        curve trimmed to exactly the vol window (default 60 trading days)
        yields only 59 returns, permanently one short of that function's
        `len(r) >= window` guard, and silently collapses to the
        equal-weight/scale=0.0 fallback without raising. This asserts a
        margin past that boundary (>= 61, not just >= 60) -- exactly the
        off-by-one failure this test guards against."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols

        def _fetch(syms, start, end):
            return _mock_ohlcv_truncated(syms, start=start, end=end)

        mock_fetch.side_effect = _fetch

        from ggTrader.paper.overlay import compute_sleeve_curve

        asof = pd.Timestamp("2026-07-10", tz="UTC")
        curve = compute_sleeve_curve("sp500", asof, window_days=90)

        assert len(curve) >= 61


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
