"""Tests for the daily signal runner."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


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
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
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

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
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

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
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

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
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

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_universe_param_passed_through(self, mock_fetch, mock_members):
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(universe="midcap400", lookback_days=120)

        mock_members.assert_called_once()
        assert mock_members.call_args[0][0] == "midcap400"
        assert result["universe_size"] == 3


class TestGenerateBlendedSignals:
    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_first_run_rebalances_and_returns_all_sleeves(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = None  # no prior rebalance
        mock_curve.return_value = pd.Series([1.0, 1.01, 1.02])
        mock_weights.return_value = ({"sp500": 0.4, "midcap400": 0.3, "nasdaq100": 0.3}, 0.9)

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert set(result["sleeves"]) == {"sp500", "midcap400", "nasdaq100"}
        assert result["weights"] == {"sp500": 0.4, "midcap400": 0.3, "nasdaq100": 0.3}
        assert result["scale"] == 0.9
        assert result["rebalanced_today"] is True
        assert result["fallback_used"] is False
        mock_save.assert_called_once()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_mid_month_reuses_stored_weights(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        today_str = str(pd.Timestamp.now(tz="UTC").normalize().date())
        mock_get_state.return_value = {
            "rebalance_date": today_str,
            "weights": {"sp500": 0.5, "midcap400": 0.25, "nasdaq100": 0.25},
            "scale": 0.8,
        }

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert result["weights"] == {"sp500": 0.5, "midcap400": 0.25, "nasdaq100": 0.25}
        assert result["scale"] == 0.8
        assert result["rebalanced_today"] is False
        mock_curve.assert_not_called()
        mock_save.assert_not_called()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_rebalance_fetch_failure_falls_back_to_stored_weights(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = {
            "rebalance_date": "2026-05-01",  # stale -- would normally trigger a rebalance
            "weights": {"sp500": 0.6, "midcap400": 0.2, "nasdaq100": 0.2},
            "scale": 0.7,
        }
        mock_curve.side_effect = RuntimeError("OHLCV fetch failed")

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert result["weights"] == {"sp500": 0.6, "midcap400": 0.2, "nasdaq100": 0.2}
        assert result["scale"] == 0.7
        assert result["fallback_used"] is True
        mock_save.assert_not_called()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_first_run_rebalance_failure_raises(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = None  # no prior rebalance -- nothing to fall back to
        mock_curve.side_effect = RuntimeError("OHLCV fetch failed")

        from ggTrader.paper.signal_runner import generate_blended_signals

        with pytest.raises(RuntimeError, match="OHLCV fetch failed"):
            generate_blended_signals()

        mock_save.assert_not_called()
