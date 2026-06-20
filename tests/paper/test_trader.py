"""Tests for the paper trader orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_trader(positions=None, portfolio_value=100000.0, cash=50000.0):
    from ggTrader.paper.trader import PaperTrader

    broker = MagicMock()
    broker.get_account.return_value = {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "buying_power": cash * 2,
    }
    broker.get_positions.return_value = positions or {}
    broker.submit_buy.return_value = "buy-order-1"
    broker.submit_sell.return_value = "sell-order-1"

    notifier = MagicMock()
    notifier.trade_alert.return_value = True
    notifier.daily_summary.return_value = True

    return PaperTrader(broker, notifier, position_size=0.02), broker, notifier


class TestSellExits:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_sells_positions_with_exit_signal(self, mock_signals):
        mock_signals.return_value = {
            "buys": [],
            "sells": ["AAPL"],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, notifier = _make_trader(
            positions={
                "AAPL": {
                    "qty": 10.0,
                    "market_value": 1500.0,
                    "avg_entry": 145.0,
                    "unrealized_pl": 50.0,
                }
            }
        )
        result = trader.run()
        broker.submit_sell.assert_called_once_with("AAPL", 10.0)
        assert "AAPL" in result["sells"]

    @patch("ggTrader.paper.trader.generate_signals")
    def test_skips_sell_if_not_holding(self, mock_signals):
        mock_signals.return_value = {
            "buys": [],
            "sells": ["AAPL"],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, _ = _make_trader(positions={})
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert result["sells"] == []


class TestBuyEntries:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_buys_new_positions(self, mock_signals):
        mock_signals.return_value = {
            "buys": ["MSFT"],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_called_once_with("MSFT", 2000.0)  # 0.02 * 100000
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_signals")
    def test_skips_buy_if_already_holding(self, mock_signals):
        mock_signals.return_value = {
            "buys": ["AAPL"],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, _ = _make_trader(
            positions={
                "AAPL": {
                    "qty": 5.0,
                    "market_value": 750.0,
                    "avg_entry": 145.0,
                    "unrealized_pl": 25.0,
                }
            }
        )
        result = trader.run()
        broker.submit_buy.assert_not_called()
        assert result["buys"] == []


class TestNotifications:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_sends_trade_alerts(self, mock_signals):
        mock_signals.return_value = {
            "buys": ["MSFT"],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, notifier = _make_trader()
        trader.run()
        notifier.trade_alert.assert_called_once()
        call_args = notifier.trade_alert.call_args[0]
        assert call_args[0] == "BUY"
        assert call_args[1] == "MSFT"

    @patch("ggTrader.paper.trader.generate_signals")
    def test_sends_daily_summary(self, mock_signals):
        mock_signals.return_value = {
            "buys": [],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, _, notifier = _make_trader()
        trader.run()
        notifier.daily_summary.assert_called_once()


class TestErrorHandling:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_order_error_captured_not_raised(self, mock_signals):
        mock_signals.return_value = {
            "buys": ["MSFT"],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, _ = _make_trader()
        broker.submit_buy.side_effect = Exception("API error")
        result = trader.run()
        assert len(result["errors"]) == 1
        assert "MSFT" in result["errors"][0]
