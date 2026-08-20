"""Tests for the paper trader orchestration."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# Fixed "today" every test's signals["as_of"] defaults to, so the Item-4
# freshness gate (as_of must equal today in ET) doesn't need to be
# individually stubbed per test. Tests that deliberately exercise staleness
# override as_of to something else.
_TEST_TODAY = "2026-06-19"


@pytest.fixture(autouse=True)
def _stub_today_et():
    with patch("ggTrader.paper.trader._today_et", return_value=date.fromisoformat(_TEST_TODAY)):
        yield


@pytest.fixture(autouse=True)
def _stub_pending_order_db():
    """Keep the pending-order reconciliation off the real DB by default.
    Individual tests override these patches to exercise reconciliation."""
    with patch("ggTrader.paper.trader.get_pending_orders", return_value=[]):
        with patch("ggTrader.paper.trader.log_pending_order"):
            with patch("ggTrader.paper.trader.clear_pending_order"):
                with patch("ggTrader.paper.trader.mark_pending_order_stale"):
                    yield


@pytest.fixture(autouse=True)
def _stub_risk_state_db():
    """Keep persisted-peak reads/writes off the real DB by default.
    Individual tests override to exercise cross-run drawdown halts."""
    with patch("ggTrader.paper.trader.get_peak_value", return_value=None):
        with patch("ggTrader.paper.trader.save_peak_value"):
            yield


@pytest.fixture(autouse=True)
def _stub_dividend_db():
    """Keep dividend accrual reads/writes off the real DB by default.
    Individual tests override these patches to exercise accrual."""
    with patch("ggTrader.paper.trader.get_total_dividend_accrual", return_value=0.0):
        with patch("ggTrader.paper.trader.get_snapshot_history", return_value=[]):
            with patch("ggTrader.paper.trader.get_accrued_dividend_keys", return_value=set()):
                with patch("ggTrader.paper.trader.log_dividend_accrual"):
                    yield


def _blend(buys, sells, as_of=_TEST_TODAY, universe="sp500"):
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
    broker.get_split_corrections.return_value = {}
    broker.get_dividend_corrections.return_value = {"corp_dividends": {}, "credited_keys": set()}
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

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_sell_submits_brokers_uncorrected_qty_not_split_adjusted(self, mock_signals, *_):
        # Broker holds 20.80410098 shares (never doubled for MNST's 2-for-1
        # split) -- that is what can actually be sold. A sell for the
        # split-adjusted 41.6 shares would be rejected by the broker.
        mock_signals.return_value = _blend(buys=[], sells=["MNST"], as_of="2026-06-19")
        mnst_qty = 20.80410098
        trader, broker, _ = _make_trader(
            positions={
                "MNST": {
                    "qty": mnst_qty,
                    "market_value": mnst_qty * 47.77,
                    "unrealized_pl": mnst_qty * 47.77 - 1887.11,
                    "cost_basis": 1887.11,
                }
            }
        )
        broker.get_split_corrections.return_value = {"MNST": 2.0}
        trader.run()
        broker.submit_sell.assert_called_once_with("MNST", mnst_qty)


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

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_concentration_check_uses_split_corrected_market_value(self, mock_signals, *_):
        # MSFT is a new buy candidate; MNST is an existing, split-flagged
        # holding. The concentration check's `positions` argument must carry
        # MNST's corrected (doubled) market_value -- true economic exposure
        # -- not the broker's uncorrected figure.
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        mnst_qty = 20.80410098
        mnst_broker_market_value = mnst_qty * 47.77
        trader, broker, _ = _make_trader(
            positions={
                "MNST": {
                    "qty": mnst_qty,
                    "market_value": mnst_broker_market_value,
                    "unrealized_pl": mnst_broker_market_value - 1887.11,
                    "cost_basis": 1887.11,
                }
            }
        )
        broker.get_split_corrections.return_value = {"MNST": 2.0}
        trader._risk.check_concentration = MagicMock(return_value=False)
        trader.run()
        call_args = trader._risk.check_concentration.call_args
        positions_arg = call_args[0][1]
        assert positions_arg["MNST"]["market_value"] == pytest.approx(
            mnst_broker_market_value * 2.0, abs=0.01
        )


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
        mock_signals.return_value = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
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
        mock_signals.return_value = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
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
        mock_signals.return_value = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
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

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_unapplied_split_corrects_the_daily_summary(
        self, mock_signals, _schema, _trade, _snap, mock_prev
    ):
        # Real MNST numbers from the 2026-08-19 incident: broker qty
        # (20.80410098) never doubled for the 2-for-1 split, so the broker's
        # own unrealized_pl (-893.30) is phantom -- the true corrected
        # position is a +100.51 gain. The broker-native corrector (mocked
        # here via AlpacaBroker.get_split_corrections) reports the factor;
        # the trader must apply it to positions/P&L before reporting, not
        # hide the position's P&L.
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        mnst_qty = 20.80410098
        mnst_cost_basis = 1887.11
        mnst_broker_market_value = mnst_qty * 47.77  # 993.81..., broker's bogus (pre-split) value
        mnst_broker_pl = mnst_broker_market_value - mnst_cost_basis  # -893.30...
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={
                "MNST": {
                    "qty": mnst_qty,
                    "market_value": mnst_broker_market_value,
                    "unrealized_pl": mnst_broker_pl,
                    "unrealized_plpc": mnst_broker_pl / mnst_cost_basis,
                    "cost_basis": mnst_cost_basis,
                    "current_price": 47.77,
                },
                "VZ": {
                    "qty": 77.05,
                    "market_value": 3743.85,
                    "unrealized_pl": 365.22,
                    "cost_basis": 3378.63,
                },
            },
        )
        broker.get_split_corrections.return_value = {"MNST": 2.0}
        trader.run()

        kwargs = notifier.daily_summary.call_args[1]
        mnst_true_pl = mnst_broker_market_value * 2.0 - mnst_cost_basis  # +100.51...
        assert kwargs["unrealized_pnl"] == pytest.approx(mnst_true_pl + 365.22, abs=0.01)
        assert kwargs["split_corrections"] == {"MNST": 2.0}
        assert kwargs["booked_equity"] == 100000.0
        # portfolio_value (first positional arg) is the split-adjusted total.
        corrected_value = notifier.daily_summary.call_args[0][0]
        assert corrected_value == pytest.approx(
            100000.0 + (mnst_broker_market_value * 2.0 - mnst_broker_market_value), abs=0.01
        )
        # The corrected position's own P&L is shown as a real number.
        reported_positions = notifier.daily_summary.call_args[0][2]
        assert reported_positions["MNST"]["unrealized_pl"] == pytest.approx(mnst_true_pl, abs=0.01)

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_no_split_corrections_reports_broker_values_unchanged(
        self, mock_signals, _schema, _trade, _snap, mock_prev
    ):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={"VZ": {"qty": 77.05, "market_value": 3743.85, "unrealized_pl": 365.22}},
        )
        broker.get_split_corrections.return_value = {}
        trader.run()
        kwargs = notifier.daily_summary.call_args[1]
        assert kwargs["split_corrections"] == {}
        assert kwargs["booked_equity"] is None
        assert kwargs["unrealized_pnl"] == 365.22


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestDividendAccrual:
    """Alpaca's PAPER account credits zero dividends (verified 2026-08-18:
    /v2/account/activities/DIV returns [] while the corporate-actions feed
    lists real cash dividends on held symbols). These tests exercise the
    trader-level wiring of the accrual correction: dividend_check.py's pure
    functions are tested independently in test_dividend_check.py.
    """

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_held_dividend_is_accrued_and_added_to_reported_equity(
        self, mock_signals, _schema, _trade, _snap, mock_prev
    ):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={"VZ": {"qty": 77.05, "market_value": 3743.85, "unrealized_pl": 365.22}},
        )
        broker.get_dividend_corrections.return_value = {
            "corp_dividends": {"VZ": [(date(2026, 6, 1), 0.6725)]},
            "credited_keys": set(),
        }
        with patch(
            "ggTrader.paper.trader.get_snapshot_history",
            return_value=[(date(2026, 5, 20), {"VZ": {"qty": 77.05}})],
        ):
            with patch(
                "ggTrader.paper.trader.get_total_dividend_accrual",
                side_effect=[0.0, 77.05 * 0.6725],
            ):
                with patch("ggTrader.paper.trader.log_dividend_accrual") as mock_log:
                    trader.run()

        mock_log.assert_called_once_with(
            "VZ", date(2026, 6, 1), 0.6725, 77.05, pytest.approx(51.82, abs=0.01)
        )
        kwargs = notifier.daily_summary.call_args[1]
        assert kwargs["dividend_total"] == pytest.approx(51.82, abs=0.01)
        assert len(kwargs["new_dividend_accruals"]) == 1
        # Reported equity (first positional arg) includes the accrual.
        corrected_value = notifier.daily_summary.call_args[0][0]
        assert corrected_value == pytest.approx(100000.0 + 51.82, abs=0.01)

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_dividend_already_credited_by_broker_is_not_accrued(
        self, mock_signals, _schema, _trade, _snap, mock_prev
    ):
        # The case that would double-count if Alpaca's paper environment
        # ever starts crediting dividends correctly.
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={"VZ": {"qty": 77.05, "market_value": 3743.85, "unrealized_pl": 365.22}},
        )
        broker.get_dividend_corrections.return_value = {
            "corp_dividends": {"VZ": [(date(2026, 6, 1), 0.6725)]},
            "credited_keys": {("VZ", date(2026, 6, 1))},
        }
        with patch(
            "ggTrader.paper.trader.get_snapshot_history",
            return_value=[(date(2026, 5, 20), {"VZ": {"qty": 77.05}})],
        ):
            with patch("ggTrader.paper.trader.log_dividend_accrual") as mock_log:
                trader.run()

        mock_log.assert_not_called()
        kwargs = notifier.daily_summary.call_args[1]
        assert kwargs["dividend_total"] == 0.0
        assert kwargs["new_dividend_accruals"] == []

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_symbol_bought_after_ex_date_is_not_accrued(
        self, mock_signals, _schema, _trade, _snap, mock_prev
    ):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={"VZ": {"qty": 77.05, "market_value": 3743.85, "unrealized_pl": 365.22}},
        )
        broker.get_dividend_corrections.return_value = {
            "corp_dividends": {"VZ": [(date(2026, 6, 1), 0.6725)]},
            "credited_keys": set(),
        }
        with patch(
            "ggTrader.paper.trader.get_snapshot_history",
            # Only bought (first appears in a snapshot) after the ex-date.
            return_value=[(date(2026, 6, 15), {"VZ": {"qty": 77.05}})],
        ):
            with patch("ggTrader.paper.trader.log_dividend_accrual") as mock_log:
                trader.run()

        mock_log.assert_not_called()
        kwargs = notifier.daily_summary.call_args[1]
        assert kwargs["dividend_total"] == 0.0

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_dividend_api_failure_fails_soft(self, mock_signals, _schema, _trade, _snap, mock_prev):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={"VZ": {"qty": 77.05, "market_value": 3743.85, "unrealized_pl": 365.22}},
        )
        broker.get_dividend_corrections.side_effect = Exception("API down")

        # Must not raise -- the broker call itself failing soft is the
        # broker's job, but _accrue_dividends must also tolerate the
        # underlying persistence calls failing.
        with patch("ggTrader.paper.trader.get_snapshot_history", side_effect=Exception("DB down")):
            result = trader.run()

        assert result["errors"] == []
        kwargs = notifier.daily_summary.call_args[1]
        assert kwargs["dividend_total"] == 0.0
        assert kwargs["new_dividend_accruals"] == []

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_no_dividend_accrual_leaves_equity_unchanged(
        self, mock_signals, _schema, _trade, _snap, mock_prev
    ):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        broker.get_dividend_corrections.return_value = {
            "corp_dividends": {},
            "credited_keys": set(),
        }
        trader.run()
        corrected_value = notifier.daily_summary.call_args[0][0]
        assert corrected_value == pytest.approx(100000.0, abs=0.01)


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

        mock_signals.return_value = _blend(buys=["AAPL"], sells=[], as_of=_TEST_TODAY)

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
            f"<b>🔍 DRY RUN buy:</b> AAPL "
            f"(${round(10000.0 * 1.0 * 1.0 * 0.033, 0):.0f}, sleeve=sp500)"
        )


class TestRunPaperTrading:
    """The margin pre-flight check must not block dry-run: dry-run submits no
    real orders regardless of the account's margin capability, so gating it
    on account type defeats the smoke-test/burn-in mode it exists to enable.
    The check should only fire before real order submission (dry_run=False)."""

    @patch("ggTrader.paper.trader.TelegramNotifier")
    @patch("ggTrader.paper.trader.AlpacaBroker")
    @patch("ggTrader.paper.trader.PaperTrader")
    def test_margin_account_allowed_in_dry_run(self, mock_trader_cls, mock_broker_cls, _notif):
        from ggTrader.paper.trader import run_paper_trading

        mock_broker = mock_broker_cls.return_value
        mock_broker.get_account.return_value = {"multiplier": 4.0}
        mock_trader_cls.return_value.run.return_value = {"buys": [], "sells": [], "errors": []}

        run_paper_trading(dry_run=True)

        mock_trader_cls.assert_called_once()
        assert mock_trader_cls.call_args.kwargs["dry_run"] is True

    @patch("ggTrader.paper.trader.TelegramNotifier")
    @patch("ggTrader.paper.trader.AlpacaBroker")
    @patch("ggTrader.paper.trader.PaperTrader")
    def test_margin_account_rejected_when_going_live(
        self, mock_trader_cls, mock_broker_cls, _notif
    ):
        from ggTrader.paper.trader import run_paper_trading

        mock_broker = mock_broker_cls.return_value
        mock_broker.get_account.return_value = {"multiplier": 4.0}

        with pytest.raises(RuntimeError, match="margin-enabled"):
            run_paper_trading(dry_run=False)

        mock_trader_cls.assert_not_called()

    @patch("ggTrader.paper.trader.TelegramNotifier")
    @patch("ggTrader.paper.trader.AlpacaBroker")
    @patch("ggTrader.paper.trader.PaperTrader")
    def test_unlevered_account_allowed_when_going_live(
        self, mock_trader_cls, mock_broker_cls, _notif
    ):
        from ggTrader.paper.trader import run_paper_trading

        mock_broker = mock_broker_cls.return_value
        mock_broker.get_account.return_value = {"multiplier": 1.0}
        mock_trader_cls.return_value.run.return_value = {"buys": [], "sells": [], "errors": []}

        run_paper_trading(dry_run=False)

        assert mock_trader_cls.call_args.kwargs["dry_run"] is False


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestPendingOrderDedup:
    """A symbol with an already-open (non-terminal) pending order must not
    get a second buy/sell submitted against it this run."""

    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_skips_buy_when_symbol_has_pending_order(self, mock_signals, mock_get_pending, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-18",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
                "created_at": datetime.now(timezone.utc),
                "flagged_stale": False,
            }
        ]
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
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
        result = trader.run()
        broker.submit_buy.assert_not_called()
        assert "MSFT" not in result["buys"]

    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_skips_sell_when_symbol_has_pending_order(self, mock_signals, mock_get_pending, *_):
        mock_signals.return_value = _blend(buys=[], sells=["AAPL"])
        mock_get_pending.return_value = [
            {
                "order_id": "sell-order-99",
                "run_date": "2026-06-18",
                "side": "SELL",
                "symbol": "AAPL",
                "notional": 1500.0,
                "created_at": datetime.now(timezone.utc),
                "flagged_stale": False,
            }
        ]
        trader, broker, _ = _make_trader(
            portfolio_value=100000.0,
            positions={
                "AAPL": {
                    "qty": 10.0,
                    "market_value": 1500.0,
                    "avg_entry": 145.0,
                    "unrealized_pl": 50.0,
                }
            },
        )
        broker.get_order.side_effect = lambda oid: {
            "id": oid,
            "symbol": "AAPL",
            "side": "sell",
            "qty": None,
            "notional": 1500.0,
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "status": "accepted",
        }
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert "AAPL" not in result["sells"]

    @patch("ggTrader.paper.trader.clear_pending_order")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_terminal_pending_order_does_not_block_new_order(
        self, mock_signals, mock_get_pending, mock_clear, *_
    ):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-18",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
                "created_at": datetime.now(timezone.utc),
                "flagged_stale": False,
            }
        ]
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        # Reconciliation resolves the old order (filled) -- it is terminal,
        # so MSFT is no longer "pending" and a fresh signal may proceed.
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
        result = trader.run()
        broker.submit_buy.assert_called_once_with("MSFT", 3300.0)
        assert "MSFT" in result["buys"]


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestMarketGates:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_market_closed_skips_run_entirely(self, mock_signals, *_):
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        broker.get_clock.return_value = {"is_open": False}
        result = trader.run()
        mock_signals.assert_not_called()
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        notifier.daily_summary.assert_not_called()
        assert result == {"buys": [], "sells": [], "errors": []}
        assert any("Market closed" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_market_open_get_clock_failure_fails_open(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        broker.get_clock.side_effect = Exception("API down")
        result = trader.run()
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_stale_as_of_skips_trading(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2020-01-01")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert result == {"buys": [], "sells": [], "errors": []}
        assert any("Stale signal data" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_fresh_as_of_trades_normally(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=_TEST_TODAY)
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_prior_session_as_of_trades_normally(self, mock_signals, *_):
        """The normal case in production: signals come from the most recent
        COMPLETED daily bar, so as_of is the prior trading session, never
        today. Requiring as_of == today froze every live run (caught only
        after deploy, because the other tests all pinned as_of to today).
        4 days covers a Friday bar still being latest on a post-holiday
        Tuesday run.
        """
        for age_days in (1, 3, 4):
            as_of = (date.fromisoformat(_TEST_TODAY) - timedelta(days=age_days)).isoformat()
            mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=as_of)
            trader, broker, notifier = _make_trader(portfolio_value=100000.0)
            result = trader.run()
            assert "MSFT" in result["buys"], f"blocked at age {age_days}d"
            assert not any(
                "Stale signal data" in c.args[0] for c in notifier.send.call_args_list
            ), f"stale alert at age {age_days}d"

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_as_of_beyond_max_age_blocks_trading(self, mock_signals, *_):
        as_of = (date.fromisoformat(_TEST_TODAY) - timedelta(days=5)).isoformat()
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=as_of)
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_not_called()
        assert result == {"buys": [], "sells": [], "errors": []}
        assert any("Stale signal data" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_unparseable_as_of_blocks_trading(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="not-a-date")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_not_called()
        assert result == {"buys": [], "sells": [], "errors": []}
        assert any("unparseable" in c.args[0] for c in notifier.send.call_args_list)


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestPersistedDrawdown:
    """The max-drawdown halt is dead code unless the peak survives across
    process restarts (each cron run constructs a fresh PaperTrader)."""

    @patch("ggTrader.paper.trader.get_peak_value")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_halt_fires_using_persisted_peak_across_two_runs(self, mock_signals, mock_get_peak, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        mock_get_peak.return_value = 120000.0  # peak persisted from a prior run
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)  # -16.7% drawdown
        result = trader.run()
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert result["errors"] and "drawdown" in result["errors"][0].lower()
        assert any("HALTED" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.save_peak_value")
    @patch("ggTrader.paper.trader.get_peak_value")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_peak_persisted_after_new_high(self, mock_signals, mock_get_peak, mock_save_peak, *_):
        mock_signals.return_value = _blend(buys=[], sells=[])
        mock_get_peak.return_value = 90000.0
        trader, _, _ = _make_trader(portfolio_value=100000.0)
        trader.run()
        mock_save_peak.assert_called_once_with(100000.0)

    @patch("ggTrader.paper.trader.save_peak_value")
    @patch("ggTrader.paper.trader.get_peak_value")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_peak_not_regressed_when_persisted_peak_higher(
        self, mock_signals, mock_get_peak, mock_save_peak, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[])
        mock_get_peak.return_value = 120000.0
        # 110000 is an 8.3% drawdown from 120000 -- below the 15% halt
        # threshold, so trading continues, but the peak must stay at 120000.
        trader, _, _ = _make_trader(portfolio_value=110000.0)
        trader.run()
        mock_save_peak.assert_called_once_with(120000.0)


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestStalePendingOrders:
    @patch("ggTrader.paper.trader.mark_pending_order_stale")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_stale_pending_order_flagged_once(
        self, mock_signals, mock_get_pending, mock_mark_stale, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[])
        old_created = datetime.now(timezone.utc) - timedelta(days=6)
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-01",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
                "created_at": old_created,
                "flagged_stale": False,
            }
        ]
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
        trader.run()
        mock_mark_stale.assert_called_once_with("buy-order-99")
        assert any("Stale pending order" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.mark_pending_order_stale")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_already_flagged_stale_order_not_re_alerted(
        self, mock_signals, mock_get_pending, mock_mark_stale, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[])
        old_created = datetime.now(timezone.utc) - timedelta(days=6)
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-01",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
                "created_at": old_created,
                "flagged_stale": True,
            }
        ]
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
        trader.run()
        mock_mark_stale.assert_not_called()
        assert not any("Stale pending order" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.mark_pending_order_stale")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_fresh_pending_order_not_flagged(
        self, mock_signals, mock_get_pending, mock_mark_stale, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[])
        recent_created = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_get_pending.return_value = [
            {
                "order_id": "buy-order-99",
                "run_date": "2026-06-19",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 3300.0,
                "created_at": recent_created,
                "flagged_stale": False,
            }
        ]
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
        trader.run()
        mock_mark_stale.assert_not_called()
        assert not any("Stale pending order" in c.args[0] for c in notifier.send.call_args_list)


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestOrderErrorAlerting:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_order_failure_sends_distinct_alert(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        broker.submit_buy.side_effect = Exception("insufficient buying power")
        trader.run()
        assert any(
            "failed" in c.args[0].lower() and "MSFT" in c.args[0]
            for c in notifier.send.call_args_list
        )

    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_errors_passed_through_to_daily_summary(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        broker.submit_buy.side_effect = Exception("insufficient buying power")
        trader.run()
        kwargs = notifier.daily_summary.call_args[1]
        assert len(kwargs["errors"]) == 1
        assert "MSFT" in kwargs["errors"][0]
