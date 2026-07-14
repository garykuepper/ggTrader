"""Tests for the paper trader orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _stub_pending_order_db():
    """Keep the pending-order reconciliation off the real DB by default.
    Individual tests override these patches to exercise reconciliation."""
    with patch("ggTrader.paper.trader.get_pending_orders", return_value=[]):
        with patch("ggTrader.paper.trader.log_pending_order"):
            with patch("ggTrader.paper.trader.clear_pending_order"):
                yield


def _blend(buys, sells, as_of, universe="sp500"):
    """Wrap a flat buys/sells list into generate_blended_signals()'s shape,
    with full weight+scale on one sleeve -- reproduces today's flat-3%
    single-universe behavior exactly (see Task 6's collapse-to-flat test)."""
    all_universes = ("sp500", "midcap400", "nasdaq100")
    sleeves = {
        u: {
            "buys": buys if u == universe else [],
            "sells": sells if u == universe else [],
            "as_of": as_of,
            "universe_size": 100 if u == universe else 0,
            "gate": {},
        }
        for u in all_universes
    }
    return {
        "sleeves": sleeves,
        "weights": {u: (1.0 if u == universe else 0.0) for u in all_universes},
        "scale": 1.0,
        "rebalanced_today": False,
        "fallback_used": False,
    }


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
    broker.get_clock.return_value = {"is_open": True}
    broker.get_order.side_effect = lambda oid: {
        "id": oid,
        "symbol": "MSFT" if "buy" in oid else "AAPL",
        "side": "buy" if "buy" in oid else "sell",
        "qty": 10.0,
        "notional": 3300.0,
        "filled_qty": 10.0,
        "filled_avg_price": 150.0,
        "status": "filled",
    }

    notifier = MagicMock()
    notifier.trade_alert.return_value = True
    notifier.daily_summary.return_value = True

    return PaperTrader(broker, notifier, dry_run=False), broker, notifier


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestSellExits:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_sells_positions_with_exit_signal(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=[], sells=["AAPL"], as_of="2026-06-19")
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

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_skips_sell_if_not_holding(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=[], sells=["AAPL"], as_of="2026-06-19")
        trader, broker, _ = _make_trader(positions={})
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert result["sells"] == []


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestBuyEntries:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_buys_new_positions(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_called_once_with("MSFT", 3300.0)  # 0.033 * 100000
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_skips_buy_if_already_holding(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["AAPL"], sells=[], as_of="2026-06-19")
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
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_sends_trade_alerts(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, notifier = _make_trader()
        trader.run()
        notifier.trade_alert.assert_called_once()
        call_args = notifier.trade_alert.call_args[0]
        assert call_args[0] == "BUY"
        assert call_args[1] == "MSFT"

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_sends_daily_summary(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, _, notifier = _make_trader()
        trader.run()
        notifier.daily_summary.assert_called_once()


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestErrorHandling:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_order_error_captured_not_raised(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, _ = _make_trader()
        broker.submit_buy.side_effect = Exception("API error")
        result = trader.run()
        assert len(result["errors"]) == 1
        assert "MSFT" in result["errors"][0]

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_db_schema_failure_does_not_block_trading(self, mock_signals, mock_schema, *_):
        mock_schema.side_effect = Exception("DB down")
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, _ = _make_trader()
        result = trader.run()
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_db_snapshot_failure_does_not_crash(self, mock_signals, mock_schema, *_a):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, _, _ = _make_trader()
        # mock_log_snapshot is the 2nd positional after mock_schema
        # The patch order is: init_paper_schema, log_trade, log_snapshot, get_latest_snapshot
        # But *_a captures: mock_schema=init, then *_a = (log_trade, log_snapshot, get_latest)
        # Actually the decorator order is bottom-up for positional args after mock_signals
        # Let's just patch directly
        with patch("ggTrader.paper.trader.log_snapshot", side_effect=Exception("DB write fail")):
            result = trader.run()
        assert result["errors"] == []

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_signal_failure_sends_notification(self, mock_signals, *_):
        mock_signals.side_effect = ValueError("yfinance returned no data")
        trader, _, notifier = _make_trader()
        import pytest

        with pytest.raises(ValueError, match="yfinance"):
            trader.run()
        notifier.send.assert_called_once()
        assert "signal generation error" in notifier.send.call_args[0][0]


def _clock_after(open_calls):
    """time.time() stub: returns 0.0 for the first `open_calls` invocations
    (so the poll loop opens), then a large value (so it times out instantly)."""
    state = {"n": 0}

    def _t():
        state["n"] += 1
        return 0.0 if state["n"] <= open_calls else 1000.0

    return _t


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestFillLogging:
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_logs_actual_fill_value_not_notional(self, mock_signals, mock_log_trade, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        trader.run()
        # Order submitted for $3300 notional but filled 10 sh @ $150 = $1500.
        mock_log_trade.assert_called_once()
        logged_amount = mock_log_trade.call_args[0][3]
        assert logged_amount == 1500.0

    @patch("ggTrader.paper.trader.time.sleep")
    @patch("ggTrader.paper.trader.time.time", side_effect=_clock_after(2))
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_unfilled_order_alerts_but_not_logged(
        self, mock_signals, mock_log_trade, _time, _sleep, *_
    ):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        broker.get_order.side_effect = lambda oid: {
            "id": oid,
            "symbol": "MSFT",
            "side": "buy",
            "qty": None,
            "notional": 3300.0,
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "status": "accepted",
        }
        with patch("ggTrader.paper.trader.log_pending_order") as mock_pending:
            trader.run()
        # No phantom ledger entry for an order that never filled...
        mock_log_trade.assert_not_called()
        # ...but it is persisted for next-run reconciliation...
        mock_pending.assert_called_once()
        assert mock_pending.call_args[0][4] == "buy-order-1"  # order_id
        # ...and the user is still alerted the order was placed.
        notifier.trade_alert.assert_called_once()
        assert notifier.trade_alert.call_args.kwargs["status"] == "accepted"

    @patch("ggTrader.paper.trader.time.sleep")
    @patch("ggTrader.paper.trader.time.time", side_effect=_clock_after(2))
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_partial_fill_persisted_not_booked(
        self, mock_signals, mock_log_trade, _time, _sleep, *_
    ):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        broker.get_order.side_effect = lambda oid: {
            "id": oid,
            "symbol": "MSFT",
            "side": "buy",
            "qty": None,
            "notional": 3300.0,
            "filled_qty": 4.0,
            "filled_avg_price": 150.0,
            "status": "partially_filled",
        }
        with patch("ggTrader.paper.trader.log_pending_order") as mock_pending:
            trader.run()
        # A still-working partial fill is NOT booked yet (it may complete);
        # it is persisted so the next run settles the final fill.
        mock_log_trade.assert_not_called()
        mock_pending.assert_called_once()


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestReconciliation:
    @patch("ggTrader.paper.trader.clear_pending_order")
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_filled_pending_order_booked_and_cleared(
        self, mock_signals, mock_get_pending, mock_log_trade, mock_clear, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-22")
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-19",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
            }
        ]
        trader, broker, notifier = _make_trader()
        broker.get_order.side_effect = lambda oid: {
            "id": oid,
            "symbol": "MSFT",
            "side": "buy",
            "qty": 22.0,
            "notional": 3300.0,
            "filled_qty": 22.0,
            "filled_avg_price": 150.0,
            "status": "filled",
        }
        trader.run()
        # Booked at the real fill value, under the original order date.
        mock_log_trade.assert_called_once_with("2026-06-19", "BUY", "MSFT", 3300.0, "buy-order-99")
        mock_clear.assert_called_once_with("buy-order-99")

    @patch("ggTrader.paper.trader.clear_pending_order")
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_canceled_pending_order_dropped_without_booking(
        self, mock_signals, mock_get_pending, mock_log_trade, mock_clear, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-22")
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-19",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
            }
        ]
        trader, broker, _ = _make_trader()
        broker.get_order.side_effect = lambda oid: {
            "id": oid,
            "symbol": "MSFT",
            "side": "buy",
            "qty": None,
            "notional": 3300.0,
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "status": "canceled",
        }
        trader.run()
        mock_log_trade.assert_not_called()
        mock_clear.assert_called_once_with("buy-order-99")

    @patch("ggTrader.paper.trader.clear_pending_order")
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_still_working_pending_order_left_alone(
        self, mock_signals, mock_get_pending, mock_log_trade, mock_clear, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-22")
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-19",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
            }
        ]
        trader, broker, _ = _make_trader()
        broker.get_order.side_effect = lambda oid: {
            "id": oid,
            "symbol": "MSFT",
            "side": "buy",
            "qty": None,
            "notional": 3300.0,
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "status": "accepted",
        }
        trader.run()
        mock_log_trade.assert_not_called()
        mock_clear.assert_not_called()


@patch("ggTrader.paper.trader.get_latest_snapshot")
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestDailyPnl:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_uses_previous_snapshot_for_pnl(self, mock_signals, _schema, _trade, _snap, mock_prev):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        trader.run()
        pnl_arg = notifier.daily_summary.call_args[0][1]
        assert pnl_arg == 1000.0

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_falls_back_to_pre_trade_value(self, mock_signals, _schema, _trade, _snap, mock_prev):
        mock_prev.return_value = None
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, _, notifier = _make_trader(portfolio_value=100000.0)
        trader.run()
        pnl_arg = notifier.daily_summary.call_args[0][1]
        assert pnl_arg == 0.0


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestDryRun:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_dry_run_does_not_submit_orders(self, mock_signals, *_):
        """In dry_run mode (the new default), buys/sells are computed and
        reported but no real order is submitted."""
        from ggTrader.paper.trader import PaperTrader

        mock_signals.return_value = _blend(buys=["AAPL"], sells=[], as_of="2026-07-13")

        broker = MagicMock()
        broker.get_account.return_value = {
            "cash": 10000.0,
            "portfolio_value": 10000.0,
            "buying_power": 10000.0,
        }
        broker.get_positions.return_value = {}
        notifier = MagicMock()
        notifier.trade_alert.return_value = True
        notifier.daily_summary.return_value = True

        trader = PaperTrader(broker, notifier)  # dry_run=True default

        result = trader.run()

        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert result["buys"] == ["AAPL"]
        notifier.send.assert_any_call(
            f"<b>🔍 DRY RUN buy:</b> AAPL (${round(10000.0 * 1.0 * 1.0 * 0.033, 0):.0f}, sleeve=sp500)"
        )
