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

    return PaperTrader(broker, notifier), broker, notifier


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestSellExits:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_sells_positions_with_exit_signal(self, mock_signals, *_):
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
    def test_skips_sell_if_not_holding(self, mock_signals, *_):
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


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestBuyEntries:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_buys_new_positions(self, mock_signals, *_):
        mock_signals.return_value = {
            "buys": ["MSFT"],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_called_once_with("MSFT", 3300.0)  # 0.033 * 100000
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_signals")
    def test_skips_buy_if_already_holding(self, mock_signals, *_):
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


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestNotifications:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_sends_trade_alerts(self, mock_signals, *_):
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
    def test_sends_daily_summary(self, mock_signals, *_):
        mock_signals.return_value = {
            "buys": [],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, _, notifier = _make_trader()
        trader.run()
        notifier.daily_summary.assert_called_once()


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestErrorHandling:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_order_error_captured_not_raised(self, mock_signals, *_):
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

    @patch("ggTrader.paper.trader.generate_signals")
    def test_db_schema_failure_does_not_block_trading(self, mock_signals, mock_schema, *_):
        mock_schema.side_effect = Exception("DB down")
        mock_signals.return_value = {
            "buys": ["MSFT"],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, _ = _make_trader()
        result = trader.run()
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_signals")
    def test_db_snapshot_failure_does_not_crash(self, mock_signals, mock_schema, *_a):
        mock_signals.return_value = {
            "buys": [],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, _, _ = _make_trader()
        # mock_log_snapshot is the 2nd positional after mock_schema
        # The patch order is: init_paper_schema, log_trade, log_snapshot, get_latest_snapshot
        # But *_a captures: mock_schema=init, then *_a = (log_trade, log_snapshot, get_latest)
        # Actually the decorator order is bottom-up for positional args after mock_signals
        # Let's just patch directly
        with patch("ggTrader.paper.trader.log_snapshot", side_effect=Exception("DB write fail")):
            result = trader.run()
        assert result["errors"] == []

    @patch("ggTrader.paper.trader.generate_signals")
    def test_signal_failure_sends_notification(self, mock_signals, *_):
        mock_signals.side_effect = ValueError("yfinance returned no data")
        trader, _, notifier = _make_trader()
        import pytest

        with pytest.raises(ValueError, match="yfinance"):
            trader.run()
        notifier.send.assert_called_once()
        assert "signal generation error" in notifier.send.call_args[0][0]


@patch("ggTrader.paper.trader.get_latest_snapshot")
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestDailyPnl:
    @patch("ggTrader.paper.trader.generate_signals")
    def test_uses_previous_snapshot_for_pnl(self, mock_signals, _schema, _trade, _snap, mock_prev):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = {
            "buys": [],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        trader.run()
        pnl_arg = notifier.daily_summary.call_args[0][1]
        assert pnl_arg == 1000.0

    @patch("ggTrader.paper.trader.generate_signals")
    def test_falls_back_to_pre_trade_value(self, mock_signals, _schema, _trade, _snap, mock_prev):
        mock_prev.return_value = None
        mock_signals.return_value = {
            "buys": [],
            "sells": [],
            "as_of": "2026-06-19",
            "universe_size": 100,
        }
        trader, _, notifier = _make_trader(portfolio_value=100000.0)
        trader.run()
        pnl_arg = notifier.daily_summary.call_args[0][1]
        assert pnl_arg == 0.0
