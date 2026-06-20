"""Tests for the daily signal runner."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _mock_ohlcv(symbols: list[str], n_days: int = 120) -> pd.DataFrame:
    """Create a fake MultiIndex OHLCV DataFrame."""
    dates = pd.bdate_range(end="2026-06-19", periods=n_days, tz="UTC")
    np.random.seed(42)
    frames = {}
    for sym in symbols:
        price = 100.0 * np.exp(np.random.randn(n_days).cumsum() * 0.02)
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


class TestGenerateSignals:
    @patch("ggTrader.paper.signal_runner.sp500_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_returns_buys_and_sells(self, mock_fetch, mock_members):
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120)

        assert "buys" in result
        assert "sells" in result
        assert "as_of" in result
        assert "universe_size" in result
        assert isinstance(result["buys"], list)
        assert isinstance(result["sells"], list)
        assert result["universe_size"] == 3

    @patch("ggTrader.paper.signal_runner.sp500_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_buys_and_sells_are_disjoint(self, mock_fetch, mock_members):
        symbols = [f"SYM{i}" for i in range(20)]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols, n_days=120)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120)

        buys = set(result["buys"])
        sells = set(result["sells"])
        assert buys.isdisjoint(sells), "A symbol cannot be both a buy and sell"

    @patch("ggTrader.paper.signal_runner.sp500_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_empty_data_returns_no_signals(self, mock_fetch, mock_members):
        mock_members.return_value = ["AAPL"]
        empty = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                [
                    ("AAPL", "open"),
                    ("AAPL", "high"),
                    ("AAPL", "low"),
                    ("AAPL", "close"),
                    ("AAPL", "volume"),
                ],
                names=["symbol", "field"],
            )
        )
        mock_fetch.return_value = empty

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120)

        assert result["buys"] == []
        assert result["sells"] == []
        assert result["universe_size"] == 0

    @patch("ggTrader.paper.signal_runner.sp500_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_as_of_is_last_bar_date(self, mock_fetch, mock_members):
        symbols = ["AAPL"]
        mock_members.return_value = symbols
        ohlcv = _mock_ohlcv(symbols)
        mock_fetch.return_value = ohlcv

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120)

        expected_date = str(ohlcv.index[-1].date())
        assert result["as_of"] == expected_date
