"""Tests for the paper trader orchestration."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper.gates import SignalGateError

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
def _stub_now_pt():
    """Pin the wall clock to 12:46 PT so the late-run gate (cutoff 12:55 PT) never depends on
    when the suite runs. Tests of the gate itself patch `_now_pt` again."""
    from ggTrader.paper.gates import PT

    with patch(
        "ggTrader.paper.trader._now_pt",
        return_value=datetime(2026, 6, 19, 12, 46, tzinfo=PT),
    ):
        yield


@pytest.fixture(autouse=True)
def _stub_write_path_db():
    """Keep the schema/snapshot/trade write path off the real DB.

    These five were the only persist functions `trader.py` imports that no
    autouse fixture stubbed. Locally that went unnoticed: `utils/config.py`
    `_load_env()` reads the `.env` FILE from the project root, so a developer
    machine always resolves a connection string and the tests look hermetic.
    On a clean checkout with no `.env` -- i.e. CI -- 42 of them died in
    `get_db_connection_string()` before reaching any assertion.

    Tests that need to assert on these still patch them individually; a
    test-level `@patch` applies inside this fixture and wins.
    """
    with patch("ggTrader.paper.trader.init_paper_schema"):
        with patch("ggTrader.paper.trader.log_trade"):
            with patch("ggTrader.paper.trader.log_snapshot"):
                with patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None):
                    with patch("ggTrader.paper.trader.get_earliest_snapshot", return_value=None):
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


@pytest.fixture(autouse=True)
def _stub_split_trade_db():
    """Keep the split-check's paper_trades lookup off the real DB by
    default. `get_snapshot_history` (also read by `_compute_split_corrections`)
    is already stubbed to `[]` by `_stub_dividend_db` above, so with both
    stubbed the default is "no snapshot evidence" -- individual split tests
    override one or both to exercise the applied/unapplied/traded paths.

    `get_open_split_states` must be stubbed too: unstubbed, every run()-level
    test queried the production DB, and the connection pool's `time.time()`
    calls consumed `_clock_after`'s ticks so `_poll_orders` spun forever."""
    with patch("ggTrader.paper.trader.get_trade_history_dates", return_value=[]):
        with patch("ggTrader.paper.trader.get_open_split_states", return_value={}):
            yield


@pytest.fixture(autouse=True)
def _stub_benchmark_tape():
    """Keep the benchmark/ETF tape keepalive off yfinance and the real DB by
    default. It is a side job of `run()`, so every run()-level test would
    otherwise download five symbols and upsert them. The two tests in
    `TestBenchmarkTapeKeepalive` patch it explicitly; a test-level patch
    applies inside this fixture and wins."""
    with patch("ggTrader.paper.trader.refresh_benchmark_tape", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _stub_live_membership_guard():
    """Let the live-run tests below keep the fixture list from the conftest environment.

    The guard that refuses a live run on anything but the latest database vintage has its own
    tests in `test_live_membership_guard.py`, which do not use this stub."""
    with patch("ggTrader.paper.trader.require_live_membership_spec"):
        yield


@pytest.fixture(autouse=True)
def _stub_last_buy_dates():
    """Keep the missed-exit catch-up's paper_trades lookup off the real DB.
    Default `{}` means "no entry dates known", which disables the catch-up."""
    with patch("ggTrader.paper.trader.get_last_buy_dates", return_value={}):
        yield


def _blend(buys, sells, as_of=_TEST_TODAY, universe="sp500", last_exit=None):
    """Wrap a flat buys/sells list into `generate_core_signals()`'s actual
    shape: exactly one sleeve, `weights={universe: 1.0}`, `scale=1.0` -- no
    zero-weight sibling sleeves, matching production's standalone-core
    return value verbatim (see `signal_runner.generate_core_signals`, and
    Finding 2 of the 2026-09-11 remediation review: every call site here
    already only ever used the default `universe="sp500"`, so this single
    -sleeve shape covers every existing test with no site-level changes
    needed while also closing the "real shape has zero coverage" gap)."""
    return {
        "sleeves": {
            universe: {
                "buys": buys,
                "sells": sells,
                "as_of": as_of,
                "universe_size": 100,
                "last_exit": last_exit or {},
            }
        },
        "weights": {universe: 1.0},
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
    broker.get_untradable.return_value = set()
    broker.get_split_corrections.return_value = {}
    broker.get_split_evidence.return_value = {"corp_splits": {}, "activity_applied": set()}
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_skips_sell_if_not_holding(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=[], sells=["AAPL"], as_of="2026-06-19")
        trader, broker, _ = _make_trader(positions={})
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert result["sells"] == []

    @patch("ggTrader.paper.trader.generate_core_signals")
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
        broker.get_split_evidence.return_value = {
            "corp_splits": {"MNST": [(date(2026, 8, 11), 2.0)]},
            "activity_applied": set(),
        }
        trader.run()
        broker.submit_sell.assert_called_once_with("MNST", mnst_qty)


_AAPL_POS = {
    "AAPL": {"qty": 10.0, "market_value": 1500.0, "avg_entry": 145.0, "unrealized_pl": 50.0}
}


@patch("ggTrader.paper.trader.generate_core_signals")
class TestMissedExitCatchUp:
    """Live checks exits on one partial bar, so a crossover that lands after
    the 12:45 PT run (or on a skipped day) was lost forever. A held name
    whose most recent exit bar is after its entry must be sold, matching
    the backtest's every-bar exit semantics."""

    def _run(self, mock_signals, last_exit, buy_dates):
        mock_signals.return_value = _blend(buys=[], sells=[], last_exit=last_exit)
        trader, broker, _ = _make_trader(positions=dict(_AAPL_POS))
        with patch("ggTrader.paper.trader.get_last_buy_dates", return_value=buy_dates):
            result = trader.run()
        return result, broker

    def test_sells_when_exit_fired_after_entry(self, mock_signals):
        result, broker = self._run(mock_signals, {"AAPL": "2026-06-10"}, {"AAPL": date(2026, 6, 5)})
        broker.submit_sell.assert_called_once_with("AAPL", 10.0)
        assert "AAPL" in result["sells"]

    def test_keeps_when_exit_predates_entry(self, mock_signals):
        _, broker = self._run(mock_signals, {"AAPL": "2026-06-03"}, {"AAPL": date(2026, 6, 5)})
        broker.submit_sell.assert_not_called()

    def test_keeps_when_exit_on_entry_day(self, mock_signals):
        _, broker = self._run(mock_signals, {"AAPL": "2026-06-05"}, {"AAPL": date(2026, 6, 5)})
        broker.submit_sell.assert_not_called()

    def test_keeps_when_no_buy_record(self, mock_signals):
        _, broker = self._run(mock_signals, {"AAPL": "2026-06-10"}, {})
        broker.submit_sell.assert_not_called()

    def test_buy_date_lookup_failure_degrades_to_last_bar_only(self, mock_signals):
        mock_signals.return_value = _blend(buys=[], sells=[], last_exit={"AAPL": "2026-06-10"})
        trader, broker, _ = _make_trader(positions=dict(_AAPL_POS))
        with patch("ggTrader.paper.trader.get_last_buy_dates", side_effect=RuntimeError("db")):
            trader.run()
        broker.submit_sell.assert_not_called()


@patch("ggTrader.paper.trader.generate_core_signals")
class TestUntradableHoldings:
    """AVB merged into EQR and Alpaca marks it tradable=false; a sell is
    rejected every run and the frozen position ate a slot."""

    def test_untradable_position_is_not_sold_and_frees_its_slot(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=["AVB"])
        positions = {
            "AVB": {"qty": 4.05, "market_value": 745.0, "avg_entry": 184.0, "unrealized_pl": 0.0}
        }
        trader, broker, notifier = _make_trader(positions=positions)
        broker.get_untradable.return_value = {"AVB"}
        with patch.dict(os.environ, {"MAX_POSITIONS": "1"}):
            from ggTrader.paper.risk import config_from_env

            trader._risk.cfg = config_from_env()
            trader.run()
        sold = [c.args[0] for c in broker.submit_sell.call_args_list]
        assert "AVB" not in sold
        broker.submit_buy.assert_called()  # frozen AVB no longer fills the only slot
        assert any("AVB" in str(c) for c in notifier.send.call_args_list)

    def test_untradable_lookup_failure_changes_nothing(self, mock_signals):
        mock_signals.return_value = _blend(buys=[], sells=["AAPL"])
        trader, broker, _ = _make_trader(positions=dict(_AAPL_POS))
        broker.get_untradable.side_effect = RuntimeError("api down")
        trader.run()
        broker.submit_sell.assert_called_once_with("AAPL", 10.0)


class TestMaxPositionsEnv:
    def test_default_is_30(self):
        from ggTrader.paper.trader import PaperTrader

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_POSITIONS", None)
            trader = PaperTrader(MagicMock(), MagicMock())
        assert trader._risk.cfg.max_positions == 30

    def test_env_override(self):
        from ggTrader.paper.trader import PaperTrader

        with patch.dict(os.environ, {"MAX_POSITIONS": "3"}):
            trader = PaperTrader(MagicMock(), MagicMock())
        assert trader._risk.cfg.max_positions == 3

    def test_explicit_risk_cfg_wins(self):
        from ggTrader.paper.risk import RiskConfig
        from ggTrader.paper.trader import PaperTrader

        with patch.dict(os.environ, {"MAX_POSITIONS": "3"}):
            trader = PaperTrader(MagicMock(), MagicMock(), risk_cfg=RiskConfig(max_positions=7))
        assert trader._risk.cfg.max_positions == 7


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestBenchmarkTapeKeepalive:
    @patch("ggTrader.paper.cash_sweep.sweep_enabled", return_value=False)
    @patch("ggTrader.paper.trader.refresh_benchmark_tape", return_value=[])
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_refresh_returning_nothing_does_not_abort_run(
        self, mock_signals, mock_refresh, _sweep_off, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, broker, _notifier = _make_trader(positions={})

        result = trader.run()

        mock_refresh.assert_called_once()
        assert result["errors"] == []
        assert result["buys"] == []

    @patch("ggTrader.paper.cash_sweep.sweep_enabled", return_value=False)
    @patch("ggTrader.paper.trader.refresh_benchmark_tape", return_value=["SPY", "TLT"])
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_refresh_is_called_after_signals(self, mock_signals, mock_refresh, _sweep_off, *_):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, _broker, _notifier = _make_trader(positions={})

        trader.run()

        assert mock_signals.call_count == 1
        mock_refresh.assert_called_once_with()


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestBuyEntries:
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_buys_new_positions(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_called_once_with("MSFT", 3300.0)  # 0.033 * 100000
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_buys_new_positions_from_literal_generate_core_signals_shape(self, mock_signals, *_):
        """Finding 2 (2026-09-11 remediation review): `_blend()` mirrors
        `generate_core_signals()`'s real single-sleeve return, but this test
        feeds `run()` that exact literal dict -- not the helper -- so the
        real production shape (one `sp500` sleeve, `weights={"sp500": 1.0}`,
        `scale=1.0`) has direct, unmistakable end-to-end coverage."""
        mock_signals.return_value = {
            "sleeves": {
                "sp500": {
                    "buys": ["MSFT"],
                    "sells": [],
                    "as_of": "2026-06-19",
                    "universe_size": 100,
                }
            },
            "weights": {"sp500": 1.0},
            "scale": 1.0,
            "rebalanced_today": False,
            "fallback_used": False,
        }
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_called_once_with("MSFT", 3300.0)  # 0.033 * 100000
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
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
        broker.get_split_evidence.return_value = {
            "corp_splits": {"MNST": [(date(2026, 8, 11), 2.0)]},
            "activity_applied": set(),
        }
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
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sends_trade_alerts(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, notifier = _make_trader()
        trader.run()
        notifier.trade_alert.assert_called_once()
        call_args = notifier.trade_alert.call_args[0]
        assert call_args[0] == "BUY"
        assert call_args[1] == "MSFT"

    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_order_error_captured_not_raised(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, _ = _make_trader()
        broker.submit_buy.side_effect = Exception("API error")
        result = trader.run()
        assert len(result["errors"]) == 1
        assert "MSFT" in result["errors"][0]

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_db_schema_failure_does_not_block_trading(self, mock_signals, mock_schema, *_):
        mock_schema.side_effect = Exception("DB down")
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2026-06-19")
        trader, broker, _ = _make_trader()
        result = trader.run()
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
class TestRunDateIsSessionDateNotAsOf:
    """Regression coverage for the paper-tables day-shift (2026-08-22).

    `paper_trades.run_date` / `paper_snapshots.run_date` must record the ET
    calendar date of THIS run, not `signals["as_of"]` (the date of the last
    completed bar the signal was computed from -- which trails by one
    session by design, and which compounded with the now-fixed OHLCV
    day-shift bug to land run_date a session-plus-a-day early: a Sun-Thu
    weekday census with zero Fridays). These tests pin `_today_et` directly
    (bypassing the module's own `_stub_today_et` autouse fixture, which is
    itself just a patch on `_today_et`) so a run "near midnight UTC" is
    exercised without any dependency on the real wall clock.
    """

    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_log_trade_stamps_today_et_not_as_of(self, mock_signals, mock_log_trade, *_):
        # Tuesday 2026-06-23, run fires at 19:45 UTC (12:45 PT) -- same
        # calendar day in UTC/ET/PT, so no wall-clock ambiguity here. The
        # signal's as_of is the PRIOR session (Monday), which is the normal,
        # by-design lag -- not itself a bug -- but must never be what gets
        # persisted as the trade date.
        tuesday = date(2026, 6, 23)
        monday = date(2026, 6, 22)
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=monday.isoformat())
        with patch("ggTrader.paper.trader._today_et", return_value=tuesday):
            trader, _broker, _notifier = _make_trader(portfolio_value=100000.0)
            trader.run()
        mock_log_trade.assert_called_once()
        logged_run_date = mock_log_trade.call_args[0][0]
        assert logged_run_date == tuesday.isoformat()
        assert logged_run_date != monday.isoformat()

    # The order stays "accepted", so _poll_orders waits out its budget: stub the clock and the sleep
    # (as the sibling poll tests do) instead of burning 15 real seconds.
    @patch("ggTrader.paper.trader.time.sleep")
    @patch("ggTrader.paper.trader.time.time", side_effect=_clock_after(2))
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_log_pending_order_stamps_today_et_not_as_of(
        self, mock_signals, _log_trade, _time, _sleep, *_
    ):
        tuesday = date(2026, 6, 23)
        monday = date(2026, 6, 22)
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=monday.isoformat())
        trader, broker, _notifier = _make_trader(portfolio_value=100000.0)
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
        with patch("ggTrader.paper.trader._today_et", return_value=tuesday):
            with patch("ggTrader.paper.trader.log_pending_order") as mock_pending:
                trader.run()
        mock_pending.assert_called_once()
        logged_run_date = mock_pending.call_args[0][0]
        assert logged_run_date == tuesday.isoformat()
        assert logged_run_date != monday.isoformat()

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_log_snapshot_stamps_today_et_not_as_of(
        self, mock_signals, _mock_schema, mock_log_snapshot, *_
    ):
        tuesday = date(2026, 6, 23)
        monday = date(2026, 6, 22)
        mock_signals.return_value = _blend(buys=[], sells=[], as_of=monday.isoformat())
        with patch("ggTrader.paper.trader._today_et", return_value=tuesday):
            trader, _broker, _notifier = _make_trader(portfolio_value=100000.0)
            trader.run()
        mock_log_snapshot.assert_called_once()
        logged_run_date = mock_log_snapshot.call_args[0][0]
        assert logged_run_date == tuesday.isoformat()
        assert logged_run_date != monday.isoformat()


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestReconciliation:
    @patch("ggTrader.paper.trader.clear_pending_order")
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_core_signals")
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
        mock_log_trade.assert_called_once_with(
            "2026-06-19", "BUY", "MSFT", 3300.0, "buy-order-99", None
        )
        mock_clear.assert_called_once_with("buy-order-99")

    @patch("ggTrader.paper.trader.clear_pending_order")
    @patch("ggTrader.paper.trader.log_trade")
    @patch("ggTrader.paper.trader.get_pending_orders")
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_uses_previous_snapshot_for_pnl(self, mock_signals, _schema, _trade, _snap, mock_prev):
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        trader.run()
        pnl_arg = notifier.daily_summary.call_args[0][1]
        assert pnl_arg == 1000.0

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_falls_back_to_pre_trade_value(self, mock_signals, _schema, _trade, _snap, mock_prev):
        mock_prev.return_value = None
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, _, notifier = _make_trader(portfolio_value=100000.0)
        trader.run()
        pnl_arg = notifier.daily_summary.call_args[0][1]
        assert pnl_arg == 0.0

    @patch("ggTrader.paper.trader.generate_core_signals")
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
        broker.get_split_evidence.return_value = {
            "corp_splits": {"MNST": [(date(2026, 8, 11), 2.0)]},
            "activity_applied": set(),
        }
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

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.get_trade_history_dates")
    @patch("ggTrader.paper.trader.get_snapshot_history")
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_snapshot_confirmed_split_suppresses_the_correction(
        self, mock_signals, mock_snap_hist, mock_trades, _schema, _trade, _snap, mock_prev
    ):
        # Primary evidence (paper_snapshots qty doubling across the ex-date,
        # no trade to explain it another way) confirms the broker DID apply
        # MNST's split -- unlike the never-exercised activities-feed guard,
        # this must actually suppress the correction rather than double it.
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        mnst_qty = 41.60820196  # already split-adjusted -- broker applied it
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={
                "MNST": {
                    "qty": mnst_qty,
                    "market_value": mnst_qty * 47.77,
                    "unrealized_pl": mnst_qty * 47.77 - 1887.11,
                    "cost_basis": 1887.11,
                    "current_price": 47.77,
                },
            },
        )
        # ex-date/snapshots deliberately fall before `_TEST_TODAY`
        # (2026-06-19) -- `find_split_applied_symbols` only considers a
        # split whose ex-date has already passed "today".
        broker.get_split_evidence.return_value = {
            "corp_splits": {"MNST": [(date(2026, 6, 1), 2.0)]},
            "activity_applied": set(),
        }
        mock_snap_hist.return_value = [
            (date(2026, 5, 28), {"MNST": {"qty": 20.80410098}}),
            (date(2026, 6, 3), {"MNST": {"qty": mnst_qty}}),
        ]
        mock_trades.return_value = []

        trader.run()

        kwargs = notifier.daily_summary.call_args[1]
        assert kwargs["split_corrections"] == {}
        reported_positions = notifier.daily_summary.call_args[0][2]
        # Untouched -- the broker's own (already-correct) numbers stand.
        assert reported_positions["MNST"]["qty"] == mnst_qty

    @patch("ggTrader.paper.trader.get_trade_history_dates")
    @patch("ggTrader.paper.trader.get_snapshot_history")
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_missing_snapshot_evidence_falls_back_to_correcting(
        self, mock_signals, mock_snap_hist, mock_trades, _schema, _trade, _snap, mock_prev, caplog
    ):
        # No paper_snapshots history at all to compare across the ex-date
        # (e.g. a fresh account, or history that doesn't reach far enough
        # back) -- with no evidence either way, this must fall back to the
        # pre-hardening default of correcting the position, not silently
        # assume the split was applied.
        mock_prev.return_value = 99000.0
        mock_signals.return_value = _blend(buys=[], sells=[])
        mnst_qty = 20.80410098
        mnst_cost_basis = 1887.11
        mnst_broker_market_value = mnst_qty * 47.77
        trader, broker, notifier = _make_trader(
            portfolio_value=100000.0,
            positions={
                "MNST": {
                    "qty": mnst_qty,
                    "market_value": mnst_broker_market_value,
                    "unrealized_pl": mnst_broker_market_value - mnst_cost_basis,
                    "cost_basis": mnst_cost_basis,
                    "current_price": 47.77,
                },
            },
        )
        # ex-date deliberately before `_TEST_TODAY` (2026-06-19), same as
        # the suppression test above, so this actually reaches
        # `find_split_applied_symbols`'s "no snapshot evidence" branch
        # instead of being filtered out as a not-yet-happened split.
        broker.get_split_evidence.return_value = {
            "corp_splits": {"MNST": [(date(2026, 6, 1), 2.0)]},
            "activity_applied": set(),
        }
        mock_snap_hist.return_value = []  # no history to bracket the ex-date
        mock_trades.return_value = []

        with caplog.at_level("WARNING", logger="ggTrader.paper.trader"):
            trader.run()

        kwargs = notifier.daily_summary.call_args[1]
        assert kwargs["split_corrections"] == {"MNST": 2.0}
        assert any(
            "No snapshot evidence" in record.getMessage() and "MNST" in record.getMessage()
            for record in caplog.records
        )

    def test_split_correction_persists_past_lookback_window(
        self, _schema, _trade, _snap, mock_prev
    ):
        """A split whose ex_date is now outside the broker-feed lookback
        window (corp_splits comes back empty) must still be corrected if it
        was previously detected and persisted -- the MNST bug."""
        trader, broker, _ = _make_trader()
        broker.get_split_evidence.return_value = {"corp_splits": {}, "activity_applied": set()}
        positions = {"MNST": {"qty": 20.8041, "cost_basis": 1887.11, "market_value": 986.67}}

        with patch(
            "ggTrader.paper.trader.get_open_split_states",
            return_value={"MNST": {"factor": 2.0, "ex_date": date(2026, 8, 11)}},
        ):
            result = trader._compute_split_corrections(
                positions, date(2026, 9, 8) - timedelta(days=14), date(2026, 9, 8)
            )

        assert result == {"MNST": 2.0}

    def test_persisted_split_still_reverified_once_broker_feed_forgets_it(
        self, _schema, _trade, _snap, mock_prev
    ):
        """A persisted correction outside the broker feed's own lookback
        window (`corp_splits` comes back empty, Finding 1) is still checked
        against snapshot evidence using its *persisted* ex_date/factor, and
        cleared once that evidence confirms the broker applied it -- so a
        belated broker-side apply can no longer double-count forever."""
        trader, broker, _ = _make_trader()
        broker.get_split_evidence.return_value = {"corp_splits": {}, "activity_applied": set()}
        mnst_qty = 41.6082  # already split-adjusted -- broker belatedly applied it
        positions = {
            "MNST": {"qty": mnst_qty, "cost_basis": 1887.11, "market_value": mnst_qty * 47.77}
        }

        with (
            patch(
                "ggTrader.paper.trader.get_open_split_states",
                return_value={"MNST": {"factor": 2.0, "ex_date": date(2026, 6, 1)}},
            ),
            patch(
                "ggTrader.paper.trader.get_snapshot_history",
                return_value=[
                    (date(2026, 5, 28), {"MNST": {"qty": 20.8041}}),
                    (date(2026, 6, 3), {"MNST": {"qty": mnst_qty}}),
                ],
            ),
            patch("ggTrader.paper.trader.get_trade_history_dates", return_value=[]),
            patch("ggTrader.paper.trader.clear_split_correction") as mock_clear,
        ):
            result = trader._compute_split_corrections(
                positions, date(2026, 9, 8) - timedelta(days=14), date(2026, 9, 8)
            )

        mock_clear.assert_called_once_with("MNST")
        assert result == {}

    def test_split_correction_clears_persisted_state_once_applied(
        self, _schema, _trade, _snap, mock_prev
    ):
        """Once snapshot history confirms the broker applied the split (qty
        jumped ~factor), persisted state for that symbol is cleared."""
        trader, broker, _ = _make_trader()
        broker.get_split_evidence.return_value = {
            "corp_splits": {"MNST": [(date(2026, 8, 11), 2.0)]},
            "activity_applied": set(),
        }
        positions = {"MNST": {"qty": 41.6082, "cost_basis": 1887.11, "market_value": 1973.5}}

        with (
            patch(
                "ggTrader.paper.trader.get_open_split_states",
                return_value={"MNST": {"factor": 2.0, "ex_date": date(2026, 8, 11)}},
            ),
            patch("ggTrader.paper.trader.get_snapshot_history", return_value=[]),
            patch("ggTrader.paper.trader.get_trade_history_dates", return_value=[]),
            patch(
                "ggTrader.paper.trader.find_split_applied_symbols",
                return_value=({"MNST"}, set()),
            ) as mock_find,
            patch("ggTrader.paper.trader.clear_split_correction") as mock_clear,
        ):
            result = trader._compute_split_corrections(
                positions, date(2026, 9, 8) - timedelta(days=14), date(2026, 9, 8)
            )

        assert mock_find.called
        mock_clear.assert_called_once_with("MNST")
        assert result == {}

    def test_split_correction_drops_persisted_state_for_closed_position(
        self, _schema, _trade, _snap, mock_prev
    ):
        """A symbol no longer held has nothing left to correct; its persisted
        row is cleaned up."""
        trader, broker, _ = _make_trader()
        broker.get_split_evidence.return_value = {"corp_splits": {}, "activity_applied": set()}

        with (
            patch(
                "ggTrader.paper.trader.get_open_split_states",
                return_value={"MNST": {"factor": 2.0, "ex_date": date(2026, 8, 11)}},
            ),
            patch("ggTrader.paper.trader.clear_split_correction") as mock_clear,
        ):
            result = trader._compute_split_corrections(
                {}, date(2026, 9, 8) - timedelta(days=14), date(2026, 9, 8)
            )

        mock_clear.assert_called_once_with("MNST")
        assert result == {}


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

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_market_open_get_clock_failure_fails_open(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        broker.get_clock.side_effect = Exception("API down")
        result = trader.run()
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_stale_as_of_skips_trading(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of="2020-01-01")
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert result == {"buys": [], "sells": [], "errors": []}
        assert any("Stale signal data" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_fresh_as_of_trades_normally(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=_TEST_TODAY)
        trader, broker, _ = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        assert "MSFT" in result["buys"]

    @patch("ggTrader.paper.trader.generate_core_signals")
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

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_as_of_beyond_max_age_blocks_trading(self, mock_signals, *_):
        as_of = (date.fromisoformat(_TEST_TODAY) - timedelta(days=5)).isoformat()
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=as_of)
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        result = trader.run()
        broker.submit_buy.assert_not_called()
        assert result == {"buys": [], "sells": [], "errors": []}
        assert any("Stale signal data" in c.args[0] for c in notifier.send.call_args_list)

    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_peak_persisted_after_new_high(self, mock_signals, mock_get_peak, mock_save_peak, *_):
        mock_signals.return_value = _blend(buys=[], sells=[])
        mock_get_peak.return_value = 90000.0
        trader, _, _ = _make_trader(portfolio_value=100000.0)
        trader.run()
        mock_save_peak.assert_called_once_with(100000.0)

    @patch("ggTrader.paper.trader.save_peak_value")
    @patch("ggTrader.paper.trader.get_peak_value")
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
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
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_order_failure_sends_distinct_alert(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        broker.submit_buy.side_effect = Exception("insufficient buying power")
        trader.run()
        assert any(
            "failed" in c.args[0].lower() and "MSFT" in c.args[0]
            for c in notifier.send.call_args_list
        )

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_errors_passed_through_to_daily_summary(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, notifier = _make_trader(portfolio_value=100000.0)
        broker.submit_buy.side_effect = Exception("insufficient buying power")
        trader.run()
        kwargs = notifier.daily_summary.call_args[1]
        assert len(kwargs["errors"]) == 1
        assert "MSFT" in kwargs["errors"][0]


def _spy_position(qty=50.0, price=400.0):
    return {
        "qty": qty,
        "market_value": qty * price,
        "current_price": price,
        "avg_entry": price,
        "unrealized_pl": 0.0,
        "unrealized_plpc": 0.0,
        "change_today": 0.0,
        "cost_basis": qty * price,
    }


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestCashSweepFlagOff:
    """The sweep must be a strict no-op when CASH_SWEEP_ENABLED is unset."""

    @patch.dict("os.environ", {}, clear=False)
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_disabled_by_default_no_sweep_sizing_called(self, mock_signals, *_):
        os.environ.pop("CASH_SWEEP_ENABLED", None)
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader(
            positions={"SPY": _spy_position()}, portfolio_value=100_000.0, cash=1_000.0
        )
        with patch("ggTrader.paper.trader.cash_sweep.compute_sweep_buy") as mock_buy:
            with patch(
                "ggTrader.paper.trader.cash_sweep.compute_sweep_sell_for_funding"
            ) as mock_sell:
                trader.run()
        mock_buy.assert_not_called()
        mock_sell.assert_not_called()
        # Only the strategy's own MSFT buy is submitted -- no SPY order.
        broker.submit_buy.assert_called_once_with("MSFT", 3300.0)
        broker.submit_sell.assert_not_called()

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_flag_off_sweep_symbol_counts_like_any_other_position(self, mock_signals, *_):
        """Zero behavior change: with the flag off, a held SPY position
        consumes a slot exactly like any other holding did before this
        feature existed."""
        from ggTrader.paper.risk import RiskConfig, RiskGuard

        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader(
            positions={"SPY": _spy_position()}, portfolio_value=100_000.0
        )
        trader._risk = RiskGuard(RiskConfig(max_positions=1))
        trader.run()
        broker.submit_buy.assert_not_called()


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestCashSweepEnabled:
    @patch.dict("os.environ", {"CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sweep_symbol_excluded_from_slot_cap(self, mock_signals, *_):
        """With the flag on, a held SPY (sweep) position must not count
        against RiskGuard's max_positions -- the strategy buy must still go
        through even with max_positions=1 and SPY already 'held'."""
        from ggTrader.paper.risk import RiskConfig, RiskGuard

        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader(
            positions={"SPY": _spy_position()}, portfolio_value=100_000.0, cash=50_000.0
        )
        trader._risk = RiskGuard(RiskConfig(max_positions=1))
        result = trader.run()
        broker.submit_buy.assert_any_call("MSFT", 3300.0)
        assert "MSFT" in result["buys"]

    @patch.dict("os.environ", {"CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sweep_symbol_not_sellable_by_strategy_exit_signal(self, mock_signals, *_):
        """Even if a strategy sleeve's exit signals happened to name the
        sweep symbol, it must never be sold as a strategy exit -- it's not a
        strategy position."""
        mock_signals.return_value = _blend(buys=[], sells=["SPY"])
        trader, broker, _ = _make_trader(
            positions={"SPY": _spy_position()}, portfolio_value=100_000.0, cash=50_000.0
        )
        result = trader.run()
        # No strategy SELL of the sweep symbol; the pre-buy funding check may
        # still submit its own sweep SELL, so assert on the strategy-exit
        # bookkeeping instead of "submit_sell never called".
        assert "SPY" not in result["sells"]

    @patch.dict("os.environ", {"CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sweep_position_ignored_by_concentration_check(self, mock_signals, *_):
        """The concentration check must not see the sweep position: a large
        held SPY sweep stake (20% of portfolio, far over the concentration
        cap) must not block SPY itself as a strategy buy candidate -- the
        sweep exposure is cash-in-waiting, not a strategy bet."""
        mock_signals.return_value = _blend(buys=["SPY"], sells=[])
        # 50 shares x $400 = $20,000 = 20% of the $100k portfolio; counted,
        # this trips any sane max_concentration_pct and the buy is skipped.
        trader, broker, _ = _make_trader(
            positions={"SPY": _spy_position(qty=50.0, price=400.0)},
            portfolio_value=100_000.0,
            cash=50_000.0,
        )
        result = trader.run()
        broker.submit_buy.assert_any_call("SPY", 3300.0)
        assert "SPY" in result["buys"]

    @patch.dict("os.environ", {"CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sells_sweep_position_before_submitting_strategy_buy(self, mock_signals, *_):
        """Ordering: when cash on hand can't cover the day's strategy buys
        plus the reserve, the sweep sell must be submitted before the
        strategy buy (see cash_sweep.py's module docstring / trader.py's
        wiring comment for the same-run market-sell assumption)."""
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        # cash 1000 is far short of the 3300 buy + 5000 reserve (5% of 100k);
        # the 20000-notional SPY holding must be sold down to cover it.
        trader, broker, _ = _make_trader(
            positions={"SPY": _spy_position(qty=50.0, price=400.0)},
            portfolio_value=100_000.0,
            cash=1_000.0,
        )
        trader.run()

        order_calls = [c for c in broker.method_calls if c[0] in ("submit_sell", "submit_buy")]
        assert [c[0] for c in order_calls] == ["submit_sell", "submit_buy"]
        sell_call, buy_call = order_calls
        assert sell_call.args[0] == "SPY"
        assert buy_call.args[0] == "MSFT"

    @patch.dict("os.environ", {"CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_waits_for_sweep_sell_fill_before_strategy_buy(self, mock_signals, *_):
        """Alpaca credits sell proceeds only on fill. On 2026-09-25 buys sent
        while the funding sell was still working were rejected for
        insufficient buying power (6 of 8), so the sell must be polled to a
        terminal status before any buy goes out."""
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader(
            positions={"SPY": _spy_position(qty=50.0, price=400.0)},
            portfolio_value=100_000.0,
            cash=1_000.0,
        )
        trader.run()

        names = [c[0] for c in broker.method_calls]
        first_poll = next(
            i
            for i, c in enumerate(broker.method_calls)
            if c[0] == "get_order" and c.args[0] == "sell-order-1"
        )
        assert first_poll < names.index("submit_buy")

    @patch.dict("os.environ", {"CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sweep_buy_deploys_leftover_cash_after_strategy_orders(self, mock_signals, *_):
        """After strategy orders, leftover cash above the reserve gets swept
        into the sweep ETF as a BUY."""
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, _ = _make_trader(positions={}, portfolio_value=100_000.0, cash=50_000.0)
        # Post-trade account (fetched fresh for the sweep-buy sizing, and
        # again afterward for the run's own snapshot) reflects the same
        # unchanged cash since there were no strategy orders this run.
        broker.get_account.return_value = {
            "cash": 50_000.0,
            "portfolio_value": 100_000.0,
            "buying_power": 100_000.0,
        }
        trader.run()
        # reserve = 5% * 100000 = 5000; sweep buy = 50000 - 5000 = 45000
        broker.submit_buy.assert_called_once_with("SPY", 45_000.0)

    @patch.dict("os.environ", {"CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sweep_buy_skipped_below_min_clip(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, _ = _make_trader(positions={}, portfolio_value=100_000.0, cash=5_100.0)
        broker.get_account.return_value = {
            "cash": 5_100.0,
            "portfolio_value": 100_000.0,
            "buying_power": 100_000.0,
        }
        trader.run()
        # reserve 5000, target 100 < default min_clip 500 -> no sweep buy
        broker.submit_buy.assert_not_called()


def _losing_position(qty=100.0, cost_basis=10000.0, loss_pct=-0.30):
    """A position whose unrealized loss (from `cost_basis`, split-invariant)
    is exactly `loss_pct` -- e.g. -0.30 for -30%."""
    unrealized_pl = cost_basis * loss_pct
    market_value = cost_basis + unrealized_pl
    price = market_value / qty if qty else 0.0
    return {
        "qty": qty,
        "market_value": market_value,
        "current_price": price,
        "avg_entry": cost_basis / qty if qty else 0.0,
        "unrealized_pl": unrealized_pl,
        "unrealized_plpc": loss_pct,
        "change_today": 0.0,
        "cost_basis": cost_basis,
    }


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestCatastropheStopFlagOff:
    """The stop must be a strict no-op when CATASTROPHE_STOP_ENABLED is unset
    -- zero behavior change for a live-money production path."""

    @patch.dict("os.environ", {}, clear=False)
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_disabled_by_default_no_stop_sell(self, mock_signals, *_):
        os.environ.pop("CATASTROPHE_STOP_ENABLED", None)
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, _ = _make_trader(
            positions={"NXPI": _losing_position(loss_pct=-0.50)}, portfolio_value=100_000.0
        )
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert result["sells"] == []


@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestCatastropheStopEnabled:
    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_fires_at_exactly_threshold_sells_full_position(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, _ = _make_trader(
            positions={"NXPI": _losing_position(qty=100.0, cost_basis=10000.0, loss_pct=-0.25)},
            portfolio_value=100_000.0,
        )
        result = trader.run()
        broker.submit_sell.assert_called_once_with("NXPI", 100.0)
        assert "NXPI" in result["sells"]

    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_does_not_fire_one_point_above_threshold(self, mock_signals, *_):
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, _ = _make_trader(
            positions={"NXPI": _losing_position(qty=100.0, cost_basis=10000.0, loss_pct=-0.24)},
            portfolio_value=100_000.0,
        )
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert result["sells"] == []

    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_stopped_symbol_not_also_sold_by_coincident_strategy_exit(self, mock_signals, *_):
        # NXPI both breaches the catastrophe floor AND has a strategy exit
        # signal this run -- it must be sold exactly once, by the stop.
        mock_signals.return_value = _blend(buys=[], sells=["NXPI"])
        trader, broker, _ = _make_trader(
            positions={"NXPI": _losing_position(qty=100.0, cost_basis=10000.0, loss_pct=-0.40)},
            portfolio_value=100_000.0,
        )
        result = trader.run()
        broker.submit_sell.assert_called_once_with("NXPI", 100.0)
        assert result["sells"].count("NXPI") == 1

    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_stopped_symbol_not_rebought_by_coincident_buy_signal(self, mock_signals, *_):
        # NXPI breaches the catastrophe floor AND appears in this run's buy
        # signals. Removing it from strategy_positions (the double-sell fix)
        # also blinds the buy loop's held-check to it, so without an explicit
        # guard the backstop's sell would be immediately re-opened same-run.
        # The position is deliberately small (1.4% of portfolio) so the
        # concentration cap cannot incidentally block the re-buy — only the
        # stopped_symbols guard prevents it.
        mock_signals.return_value = _blend(buys=["NXPI"], sells=[])
        trader, broker, _ = _make_trader(
            positions={"NXPI": _losing_position(qty=20.0, cost_basis=2000.0, loss_pct=-0.30)},
            portfolio_value=100_000.0,
        )
        result = trader.run()
        broker.submit_sell.assert_called_once_with("NXPI", 20.0)
        broker.submit_buy.assert_not_called()
        assert "NXPI" in result["sells"]
        assert "NXPI" not in result["buys"]

    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true", "CASH_SWEEP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_sweep_symbol_excluded_even_at_a_deep_loss(self, mock_signals, *_):
        # The sweep ETF position (SPY) must never be catastrophe-stopped --
        # it is cash-in-waiting, not a strategy bet, and trader.py builds
        # corrected_positions (what the stop check reads) from
        # strategy_positions, which already excludes it.
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, _ = _make_trader(
            positions={"SPY": _losing_position(qty=50.0, cost_basis=20000.0, loss_pct=-0.90)},
            portfolio_value=100_000.0,
            cash=50_000.0,  # generous cash: no sweep buy/sell action this run either
        )
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert result["sells"] == []

    @patch("ggTrader.paper.trader.log_trade")
    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_reason_tagged_catastrophe_stop(self, mock_signals, mock_log_trade, *_):
        mock_signals.return_value = _blend(buys=[], sells=[])
        trader, broker, _ = _make_trader(
            positions={"NXPI": _losing_position(qty=100.0, cost_basis=10000.0, loss_pct=-0.40)},
            portfolio_value=100_000.0,
        )
        broker.get_order.side_effect = lambda oid: {
            "id": oid,
            "symbol": "NXPI",
            "side": "sell",
            "qty": 100.0,
            "notional": None,
            "filled_qty": 100.0,
            "filled_avg_price": 60.0,
            "status": "filled",
        }
        trader.run()
        mock_log_trade.assert_called_once()
        reason_arg = mock_log_trade.call_args[0][5]
        assert reason_arg == "catastrophe_stop"

    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_slot_math_counts_stop_exactly_once_at_position_cap(self, mock_signals, *_):
        # Account sits exactly at max_positions=30: 29 ordinary holdings plus
        # NXPI, which this run's catastrophe stop closes out. That frees
        # exactly ONE slot. `strategy_positions` already reflects the stop
        # (NXPI removed) -- if `executed_sells` (which also records the stop)
        # is then subtracted a second time in the slots_available formula,
        # the trader thinks TWO slots are free and lets both buy candidates
        # through, breaching the cap. Filler positions carry no cost_basis so
        # `unrealized_pct` returns None for them and they never enter the
        # catastrophe-stop check themselves.
        positions = {
            f"FILLER{i}": {"qty": 1.0, "market_value": 100.0, "cost_basis": 0.0} for i in range(29)
        }
        positions["NXPI"] = _losing_position(qty=100.0, cost_basis=10000.0, loss_pct=-0.30)
        mock_signals.return_value = _blend(buys=["MSFT", "GOOGL"], sells=[])
        trader, broker, _ = _make_trader(positions=positions, portfolio_value=100_000.0)
        trader.run()
        assert broker.submit_buy.call_count == 1

    @patch.dict("os.environ", {"CATASTROPHE_STOP_ENABLED": "true"})
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_split_adjusted_cost_basis_no_phantom_trigger(self, mock_signals, *_):
        # MNST-shaped fixture: broker qty pre-split (never doubled for the
        # 2-for-1 split), current_price/market_value post-split -- the
        # broker's own uncorrected unrealized_pl looks like a huge loss, but
        # the trader corrects it (via split_check.apply_corrections_to_positions,
        # using cost_basis) before the catastrophe check ever sees it, so no
        # phantom stop-out fires.
        mock_signals.return_value = _blend(buys=[], sells=[])
        mnst_qty = 20.80410098
        mnst_cost_basis = 1887.11
        mnst_broker_market_value = mnst_qty * 47.77  # 993.81..., bogus pre-correction value
        trader, broker, _ = _make_trader(
            positions={
                "MNST": {
                    "qty": mnst_qty,
                    "market_value": mnst_broker_market_value,
                    "unrealized_pl": mnst_broker_market_value - mnst_cost_basis,  # -893.30...
                    "unrealized_plpc": (mnst_broker_market_value - mnst_cost_basis)
                    / mnst_cost_basis,
                    "cost_basis": mnst_cost_basis,
                    "current_price": 47.77,
                }
            },
            portfolio_value=100_000.0,
        )
        broker.get_split_evidence.return_value = {
            "corp_splits": {"MNST": [(date(2026, 8, 11), 2.0)]},
            "activity_applied": set(),
        }
        result = trader.run()
        broker.submit_sell.assert_not_called()
        assert result["sells"] == []


# ---------------------------------------------------------------------------------------------
# P7: fail-closed gates, untradable symbols (OD17), the read-only dry run
# ---------------------------------------------------------------------------------------------


class _ApiError(Exception):
    """Stand-in for alpaca-py's APIError (status_code plus the JSON body in the text)."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(body)
        self.status_code = status_code


_AVB_422 = _ApiError(422, '{"code":42210000,"message":"asset AVB is not active"}')


@patch("ggTrader.paper.trader.generate_core_signals")
class TestFailClosed:
    """Spec 8.3 / S7 section 4: any engine, data or gate problem means no orders and an alert."""

    def _positions(self):
        return dict(_AAPL_POS)

    @pytest.mark.parametrize(
        "code",
        [
            "engine_selfcheck",
            "engine_not_rust",
            "empty_frame",
            "duplicate_days",
            "weekend_bars",
            "small_universe",
            "short_history",
            "last_bar_coverage",
            "no_eligible_symbols",
            "absurd_signal_count",
        ],
    )
    def test_every_gate_failure_places_no_order_and_alerts(self, mock_signals, code):
        from ggTrader.paper.gates import SignalGateError

        mock_signals.side_effect = SignalGateError(code, "boom")
        trader, broker, notifier = _make_trader(positions=self._positions())
        with pytest.raises(SignalGateError, match=code):
            trader.run()
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        alert = " ".join(str(c) for c in notifier.send.call_args_list)
        assert "no orders" in alert and code in alert

    def test_unexpected_signal_exception_also_fails_closed(self, mock_signals):
        mock_signals.side_effect = RuntimeError("numba JIT blew up")
        trader, broker, notifier = _make_trader(positions=self._positions())
        with pytest.raises(RuntimeError):
            trader.run()
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert any("signal generation error" in str(c) for c in notifier.send.call_args_list)

    def test_late_run_places_no_orders(self, mock_signals):
        from ggTrader.paper.gates import PT

        mock_signals.return_value = _blend(buys=["MSFT"], sells=["AAPL"], as_of=_TEST_TODAY)
        trader, broker, notifier = _make_trader(positions=self._positions())
        with patch(
            "ggTrader.paper.trader._now_pt",
            return_value=datetime(2026, 6, 19, 12, 56, tzinfo=PT),
        ):
            result = trader.run()
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert result["gate"] == "late_run"
        assert any("Late run" in str(c) for c in notifier.send.call_args_list)

    def test_on_time_run_trades(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=["AAPL"], as_of=_TEST_TODAY)
        trader, broker, _ = _make_trader(positions=self._positions())
        trader.run()
        broker.submit_sell.assert_called_once()
        broker.submit_buy.assert_called_once()

    def test_late_dry_run_only_warns(self, mock_signals):
        from ggTrader.paper.gates import PT
        from ggTrader.paper.trader import PaperTrader

        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=_TEST_TODAY)
        _t, broker, notifier = _make_trader()
        trader = PaperTrader(broker, notifier)  # dry run
        with patch(
            "ggTrader.paper.trader._now_pt",
            return_value=datetime(2026, 6, 19, 15, 0, tzinfo=PT),
        ):
            result = trader.run()
        assert result["buys"] == ["MSFT"]
        assert any("late-run notice" in str(c) for c in notifier.send.call_args_list)

    def test_half_day_close_from_the_broker_clock_moves_the_cutoff(self, mock_signals):
        from ggTrader.paper.gates import PT

        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=_TEST_TODAY)
        trader, broker, _ = _make_trader()
        broker.get_clock.return_value = {
            "is_open": True,
            "next_close": "2026-06-19 10:00:00-07:00",
        }
        with patch(
            "ggTrader.paper.trader._now_pt",
            return_value=datetime(2026, 6, 19, 9, 58, tzinfo=PT),
        ):
            result = trader.run()
        assert result["gate"] == "late_run"
        broker.submit_buy.assert_not_called()

    def test_signal_engine_info_is_logged(self, mock_signals, caplog):
        blend = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
        blend["sleeves"]["sp500"]["engine"] = {"engine": "rust", "vectorbt": "1.2.0"}
        mock_signals.return_value = blend
        trader, _b, _n = _make_trader()
        with caplog.at_level("INFO", logger="ggTrader.paper.trader"):
            trader.run()
        assert any("engine={'engine': 'rust'" in r.getMessage() for r in caplog.records)


@patch("ggTrader.paper.trader.generate_core_signals")
class TestUntradableAvb:
    """OD17: the stuck AVB holding (4.05 shares, inactive ticker) never draws an order and a
    broker "asset is not active" 422 is a skip-and-alert, not a crash or an error."""

    _AVB = {"AVB": {"qty": 4.05, "market_value": 745.0, "avg_entry": 184.0, "unrealized_pl": 0.0}}

    def test_422_on_a_sell_is_skipped_and_alerted_not_an_error(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=["AVB", "AAPL"])
        trader, broker, notifier = _make_trader(positions={**self._AVB, **_AAPL_POS})
        broker.get_untradable.return_value = set()  # status check missed it
        broker.submit_sell.side_effect = lambda sym, qty: (
            (_ for _ in ()).throw(_AVB_422) if sym == "AVB" else "sell-order-1"
        )
        result = trader.run()
        assert result["errors"] == []
        assert "AVB" in result["untradable"]
        assert "AAPL" in result["sells"] and "AVB" not in result["sells"]
        broker.submit_buy.assert_called_once_with("MSFT", 3300.0)
        assert any("Untradable (order skipped)" in str(c) for c in notifier.send.call_args_list)

    def test_422_on_a_buy_is_skipped_and_the_slot_is_not_consumed(self, mock_signals):
        mock_signals.return_value = _blend(buys=["EQR", "MSFT"], sells=[])
        trader, broker, _ = _make_trader()
        broker.submit_buy.side_effect = lambda sym, n: (
            (_ for _ in ()).throw(_AVB_422) if sym == "EQR" else "buy-order-1"
        )
        with patch.dict(os.environ, {"MAX_POSITIONS": "1"}):
            from ggTrader.paper.risk import config_from_env

            trader._risk.cfg = config_from_env()
            result = trader.run()
        assert result["buys"] == ["MSFT"]  # the one slot went to the tradable name
        assert result["errors"] == []

    def test_other_422_is_still_a_real_error(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader()
        broker.submit_buy.side_effect = _ApiError(422, '{"message":"qty must be > 0"}')
        result = trader.run()
        assert result["errors"] and "qty must be > 0" in result["errors"][0]

    def test_untradable_buy_candidate_is_filtered_before_any_order(self, mock_signals):
        mock_signals.return_value = _blend(buys=["AVB", "MSFT"], sells=[])
        trader, broker, notifier = _make_trader()
        broker.get_untradable.return_value = {"AVB"}
        result = trader.run()
        assert result["buys"] == ["MSFT"]
        assert [c.args[0] for c in broker.submit_buy.call_args_list] == ["MSFT"]
        assert any("buy candidates skipped" in str(c) for c in notifier.send.call_args_list)

    def test_configured_set_works_even_if_the_status_lookup_is_down(self, mock_signals):
        mock_signals.return_value = _blend(buys=["AVB", "MSFT"], sells=["AVB"])
        trader, broker, _ = _make_trader(positions=dict(self._AVB))
        broker.get_untradable.side_effect = RuntimeError("api down")
        with patch.dict(os.environ, {"UNTRADABLE_SYMBOLS": "AVB"}):
            result = trader.run()
        broker.submit_sell.assert_not_called()
        assert [c.args[0] for c in broker.submit_buy.call_args_list] == ["MSFT"]
        assert "AVB" in result["untradable"]

    def test_avb_does_not_take_a_slot(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[])
        trader, broker, _ = _make_trader(positions=dict(self._AVB))
        broker.get_untradable.return_value = {"AVB"}
        with patch.dict(os.environ, {"MAX_POSITIONS": "1"}):
            from ggTrader.paper.risk import config_from_env

            trader._risk.cfg = config_from_env()
            trader.run()
        broker.submit_buy.assert_called_once()


_PERSIST_NAMES = (
    "init_paper_schema",
    "log_trade",
    "log_snapshot",
    "save_peak_value",
    "log_pending_order",
    "clear_pending_order",
    "mark_pending_order_stale",
    "save_split_correction",
    "clear_split_correction",
    "log_dividend_accrual",
    "refresh_benchmark_tape",
)


def _patch_persist():
    """Patch every write the trader can make; returns {name: mock} and the context stack."""
    from contextlib import ExitStack

    stack = ExitStack()
    mocks = {
        name: stack.enter_context(patch(f"ggTrader.paper.trader.{name}")) for name in _PERSIST_NAMES
    }
    return stack, mocks


@patch("ggTrader.paper.trader.generate_core_signals")
class TestReadOnlyDryRun:
    """Spec 8.4 item 2 / risk 14: a dry run differs from a live run only in order submission,
    and a read-only dry run writes nothing."""

    def _dry_trader(self, **broker_kw):
        from ggTrader.paper.trader import PaperTrader

        _t, broker, notifier = _make_trader(**broker_kw)
        return PaperTrader(broker, notifier), broker, notifier

    def test_dry_run_is_read_only_by_default(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=["AAPL"], as_of=_TEST_TODAY)
        trader, broker, _ = self._dry_trader(positions=dict(_AAPL_POS))
        stack, mocks = _patch_persist()
        with stack:
            result = trader.run()
        for name, mock in mocks.items():
            assert not mock.called, f"read-only dry run called {name}"
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert result["buys"] == ["MSFT"] and result["sells"] == ["AAPL"]
        assert {"init_paper_schema", "log_snapshot", "refresh_benchmark_tape"} <= set(
            result["skipped_writes"]
        )
        assert "save_peak_value" in result["skipped_writes"]

    def test_signals_are_loaded_without_persisting_the_tape(self, mock_signals):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
        trader, _b, _n = self._dry_trader()
        trader.run()
        mock_signals.assert_called_once()
        assert mock_signals.call_args.kwargs["persist_tape"] is False

    def test_the_trader_hands_its_chosen_membership_source_to_the_signals(self, mock_signals):
        from ggTrader.paper.trader import PaperTrader

        mock_signals.return_value = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
        _t, broker, notifier = _make_trader()
        trader = PaperTrader(
            broker, notifier, dry_run=True, membership="fixture:sp500_v1_2026-06-02"
        )
        trader.run()
        source = mock_signals.call_args.kwargs["membership"]
        assert source.vintage().fixture == "sp500_v1_2026-06-02"  # the frozen file, by name

    def test_a_trader_with_no_membership_choice_fails_loudly_and_trades_nothing(
        self, mock_signals, monkeypatch
    ):
        from ggTrader.data import membership
        from ggTrader.data.membership import MembershipSourceNotSelected

        membership.clear_source_cache()
        monkeypatch.delenv("GGT_MEMBERSHIP")
        trader, broker, notifier = self._dry_trader()
        with pytest.raises(MembershipSourceNotSelected):
            trader.run()
        mock_signals.assert_not_called()  # no signals, hence no orders, and no guessed source
        assert "signal generation error" in notifier.send.call_args.args[0]

    def test_legacy_dry_run_with_writes_still_persists(self, mock_signals):
        from ggTrader.paper.trader import PaperTrader

        mock_signals.return_value = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
        _t, broker, notifier = _make_trader()
        trader = PaperTrader(broker, notifier, dry_run=True, read_only=False)
        stack, mocks = _patch_persist()
        with stack:
            trader.run()
        assert mocks["init_paper_schema"].called and mocks["log_snapshot"].called
        assert mocks["refresh_benchmark_tape"].called
        mock_signals.assert_called_once()
        assert mock_signals.call_args.kwargs["persist_tape"] is True

    def test_live_default_is_unchanged(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT"], sells=[], as_of=_TEST_TODAY)
        trader, broker, _ = _make_trader()
        assert trader._read_only is False
        stack, mocks = _patch_persist()
        with stack:
            result = trader.run()
        assert mocks["init_paper_schema"].called
        assert mocks["log_snapshot"].called
        assert mocks["save_peak_value"].called
        assert mocks["refresh_benchmark_tape"].called
        assert result["skipped_writes"] == []
        mock_signals.assert_called_once()
        assert mock_signals.call_args.kwargs["persist_tape"] is True
        broker.submit_buy.assert_called_once()

    def test_a_live_run_cannot_be_read_only(self, mock_signals):
        from ggTrader.paper.trader import PaperTrader

        with pytest.raises(ValueError, match="read_only requires dry_run"):
            PaperTrader(MagicMock(), MagicMock(), dry_run=False, read_only=True)

    def test_pending_orders_are_neither_polled_nor_booked(self, mock_signals):
        mock_signals.return_value = _blend(buys=["MSFT", "GOOG"], sells=[], as_of=_TEST_TODAY)
        trader, broker, _ = self._dry_trader()
        pending = [
            {
                "order_id": "o1",
                "symbol": "MSFT",
                "side": "BUY",
                "run_date": date(2026, 6, 18),
                "created_at": datetime(2026, 6, 18, tzinfo=timezone.utc),
            }
        ]
        stack, mocks = _patch_persist()
        with stack, patch("ggTrader.paper.trader.get_pending_orders", return_value=pending):
            result = trader.run()
        broker.get_order.assert_not_called()
        assert not mocks["log_trade"].called and not mocks["clear_pending_order"].called
        assert result["buys"] == ["GOOG"]  # MSFT still has an order working

    def test_write_helper_skips_and_records(self, mock_signals):
        trader, _b, _n = self._dry_trader()
        fn = MagicMock(__name__="log_dividend_accrual")
        assert trader._write(fn, "AAPL", 1) is None
        fn.assert_not_called()
        assert trader.skipped_writes == ["log_dividend_accrual"]

    def test_dry_and_live_take_the_same_path_up_to_order_submission(self, mock_signals):
        """Risk 14: identical inputs give identical intended orders in both modes."""
        blend = _blend(buys=["MSFT", "GOOG"], sells=["AAPL"], last_exit={"AAPL": _TEST_TODAY})
        mock_signals.return_value = blend
        dry, dry_broker, _ = self._dry_trader(positions=dict(_AAPL_POS))
        live, live_broker, _ = _make_trader(positions=dict(_AAPL_POS))
        dry_result, live_result = dry.run(), live.run()
        assert dry_result["buys"] == live_result["buys"] == ["MSFT", "GOOG"]
        assert dry_result["sells"] == live_result["sells"] == ["AAPL"]
        assert dry_broker.submit_buy.call_count == dry_broker.submit_sell.call_count == 0
        assert live_broker.submit_buy.call_count == 2 and live_broker.submit_sell.call_count == 1


class TestDryRunNotifier:
    def test_nothing_is_forwarded_by_default(self):
        from ggTrader.paper.notifier import DryRunNotifier

        inner = MagicMock()
        n = DryRunNotifier(inner)
        assert n.send("hello") is False
        assert n.trade_alert("BUY", "AAPL", 100.0, "o1", qty=1.0) is False
        assert n.daily_summary(100000.0, 12.0, {}, total_pnl=1.0) is False
        inner.send.assert_not_called()
        inner.trade_alert.assert_not_called()
        inner.daily_summary.assert_not_called()

    def test_forward_prefixes_free_text_only(self):
        from ggTrader.paper.notifier import DryRunNotifier

        inner = MagicMock()
        inner.send.return_value = True
        n = DryRunNotifier(inner, forward=True)
        assert n.send("<b>x</b>") is True
        inner.send.assert_called_once_with("[DRY RUN] <b>x</b>")
        n.trade_alert("BUY", "AAPL", 100.0, "o1")
        n.daily_summary(1.0, 1.0, {})
        inner.trade_alert.assert_not_called()
        inner.daily_summary.assert_not_called()


class TestRunPaperTradingReadOnly:
    @patch("ggTrader.paper.trader.TelegramNotifier")
    @patch("ggTrader.paper.trader.AlpacaBroker")
    @patch("ggTrader.paper.trader.PaperTrader")
    def test_dry_run_wires_a_read_only_trader_and_a_dry_notifier(
        self, mock_trader_cls, mock_broker_cls, mock_notifier_cls
    ):
        from ggTrader.paper.notifier import DryRunNotifier
        from ggTrader.paper.trader import run_paper_trading

        mock_broker_cls.return_value.get_account.return_value = {"multiplier": 1.0}
        mock_trader_cls.return_value.run.return_value = {"buys": [], "sells": [], "errors": []}
        run_paper_trading(dry_run=True)
        kwargs = mock_trader_cls.call_args.kwargs
        assert kwargs["dry_run"] is True and kwargs["read_only"] is True
        assert isinstance(mock_trader_cls.call_args.args[1], DryRunNotifier)

    @patch("ggTrader.paper.trader.TelegramNotifier")
    @patch("ggTrader.paper.trader.AlpacaBroker")
    @patch("ggTrader.paper.trader.PaperTrader")
    def test_dry_run_writes_flag_restores_the_old_dry_run(
        self, mock_trader_cls, mock_broker_cls, mock_notifier_cls
    ):
        from ggTrader.paper.trader import run_paper_trading

        mock_broker_cls.return_value.get_account.return_value = {"multiplier": 1.0}
        mock_trader_cls.return_value.run.return_value = {"buys": [], "sells": [], "errors": []}
        run_paper_trading(dry_run=True, read_only=False)
        assert mock_trader_cls.call_args.kwargs["read_only"] is False
        assert mock_trader_cls.call_args.args[1] is mock_notifier_cls.return_value

    @patch("ggTrader.paper.trader.TelegramNotifier")
    @patch("ggTrader.paper.trader.AlpacaBroker")
    @patch("ggTrader.paper.trader.PaperTrader")
    def test_live_stays_unwrapped_and_writable(
        self, mock_trader_cls, mock_broker_cls, mock_notifier_cls
    ):
        from ggTrader.paper.trader import run_paper_trading

        mock_broker_cls.return_value.get_account.return_value = {"multiplier": 1.0}
        mock_trader_cls.return_value.run.return_value = {"buys": [], "sells": [], "errors": []}
        run_paper_trading(dry_run=False)
        assert mock_trader_cls.call_args.kwargs["read_only"] is False
        assert mock_trader_cls.call_args.args[1] is mock_notifier_cls.return_value

    def test_live_with_read_only_is_refused_before_any_broker_call(self):
        from ggTrader.paper.trader import run_paper_trading

        with patch("ggTrader.paper.trader.AlpacaBroker") as broker_cls:
            with pytest.raises(ValueError):
                run_paper_trading(dry_run=False, read_only=True)
        broker_cls.assert_not_called()


_FIXTURE = "fixture:sp500_v1_2026-06-02"


class TestLiveMembershipGate:
    """Slice M4: the live trader reads the database list and stops, with no orders, if that list
    is unreadable, empty, unmatched against the SPDR SPY holdings for too long or attested too far
    back (spec 3.4 as revised by Appendix C3)."""

    @pytest.fixture(autouse=True)
    def _fresh_source_cache(self):
        from ggTrader.data import membership

        membership.clear_source_cache()
        yield
        membership.clear_source_cache()

    @staticmethod
    def _store(monkeypatch, *, verified_days_ago=1, history_end=None):
        """Serve a fake database vintage through the real ``PgMembershipStore.read`` seam.

        ``verified_days_ago`` is the age of the last holdings match (``None``: never matched);
        the match attests the list through its own day or the history's end, whichever is first.
        """
        from datetime import timedelta

        from ggTrader.data import membership
        from tests.data.membership_fakes import FakeMembershipStore, synthetic_history

        store = FakeMembershipStore()
        now = datetime.now(timezone.utc)
        history = synthetic_history()
        if history_end is not None:  # shift the history so it ends where the test wants
            offset = history_end - max(history)
            history = {day + offset: members for day, members in history.items()}
        matched_at = None if verified_days_ago is None else now - timedelta(days=verified_days_ago)
        store.add(
            history,
            retrieved_at=now - timedelta(days=400),
            matched_at=matched_at,
            matched_as_of=None if matched_at is None else min(matched_at.date(), max(history)),
        )
        monkeypatch.setattr(
            membership.PgMembershipStore,
            "read",
            lambda self, load_id, index_name="sp500": store.read(load_id, index_name),
        )
        return store

    @staticmethod
    def _live_trader(membership="db"):
        from ggTrader.paper.trader import PaperTrader

        _t, broker, notifier = _make_trader()
        return PaperTrader(broker, notifier, dry_run=False, membership=membership), broker, notifier

    @staticmethod
    def _assert_no_order_path_was_reached(broker, mock_fetch):
        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        broker.get_account.assert_not_called()  # the run stopped before sizing anything
        mock_fetch.assert_not_called()  # and before a single bar was fetched

    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    @patch("ggTrader.paper.signal_runner.engine_selfcheck.selfcheck")
    def test_a_stale_verification_stops_the_run_with_no_orders(
        self, mock_selfcheck, mock_fetch, monkeypatch
    ):
        mock_selfcheck.return_value = {"engine": "rust"}
        self._store(monkeypatch, verified_days_ago=46, history_end=datetime.now().date())
        trader, broker, notifier = self._live_trader()
        with pytest.raises(SignalGateError) as caught:
            trader.run()
        assert caught.value.code == "membership_stale"
        self._assert_no_order_path_was_reached(broker, mock_fetch)
        message = notifier.send.call_args.args[0]
        assert "Signal gate" in message and "ggt db membership check-holdings" in message

    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    @patch("ggTrader.paper.signal_runner.engine_selfcheck.selfcheck")
    def test_a_list_never_matched_against_spy_stops_the_run_with_no_orders(
        self, mock_selfcheck, mock_fetch, monkeypatch
    ):
        mock_selfcheck.return_value = {"engine": "rust"}
        self._store(monkeypatch, verified_days_ago=None, history_end=datetime.now().date())
        trader, broker, notifier = self._live_trader()
        with pytest.raises(SignalGateError) as caught:
            trader.run()
        assert caught.value.code == "membership_stale"
        self._assert_no_order_path_was_reached(broker, mock_fetch)
        assert "never been matched" in notifier.send.call_args.args[0]

    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    @patch("ggTrader.paper.signal_runner.engine_selfcheck.selfcheck")
    def test_a_list_that_lags_by_180_days_stops_the_run(
        self, mock_selfcheck, mock_fetch, monkeypatch
    ):
        from datetime import timedelta

        mock_selfcheck.return_value = {"engine": "rust"}
        end = datetime.now(timezone.utc).date() - timedelta(days=180)
        self._store(monkeypatch, verified_days_ago=0, history_end=end)
        trader, broker, notifier = self._live_trader()
        with pytest.raises(SignalGateError) as caught:
            trader.run()
        assert caught.value.code == "membership_lag"
        self._assert_no_order_path_was_reached(broker, mock_fetch)
        assert "ggt db load-sp500" in notifier.send.call_args.args[0]

    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_an_unreadable_database_stops_the_run_and_never_reads_the_frozen_file(
        self, mock_fetch, monkeypatch
    ):
        from ggTrader.data import membership
        from tests.data.membership_fakes import FailingStore

        failing = FailingStore()
        monkeypatch.setattr(
            membership.PgMembershipStore,
            "read",
            lambda self, load_id, index_name="sp500": failing.read(load_id, index_name),
        )
        reads = []
        monkeypatch.setattr(
            membership,
            "FixtureMembershipSource",
            lambda *a, **k: reads.append(a) or pytest.fail("fell back to the fixture"),
        )
        trader, broker, notifier = self._live_trader()
        with pytest.raises(SignalGateError) as caught:
            trader.run()
        assert caught.value.code == "membership_unavailable"
        self._assert_no_order_path_was_reached(broker, mock_fetch)
        assert reads == []
        assert "never falls back to the frozen file" in notifier.send.call_args.args[0]

    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_an_empty_table_stops_the_run(self, mock_fetch, monkeypatch):
        from ggTrader.data import membership
        from tests.data.membership_fakes import FakeMembershipStore

        empty = FakeMembershipStore()
        monkeypatch.setattr(
            membership.PgMembershipStore,
            "read",
            lambda self, load_id, index_name="sp500": empty.read(load_id, index_name),
        )
        trader, broker, _n = self._live_trader()
        with pytest.raises(SignalGateError, match="membership_unavailable"):
            trader.run()
        self._assert_no_order_path_was_reached(broker, mock_fetch)

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_a_warning_reaches_the_notifier_and_trading_goes_on(self, mock_signals):
        blend = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
        blend["sleeves"]["sp500"]["membership_gate"] = {
            "applied": True,
            "level": "warn",
            "message": "the list's last change date is 2026-06-02, 100 days before 2026-09-10",
        }
        mock_signals.return_value = blend
        trader, broker, notifier = self._live_trader(_FIXTURE)
        trader.run()
        sent = [call.args[0] for call in notifier.send.call_args_list]
        assert any("Membership list" in text and "100 days" in text for text in sent)
        broker.get_account.assert_called()  # a warning never stops the run

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_the_warning_is_logged_not_sent_in_a_silent_dry_run(self, mock_signals):
        """With the real DryRunNotifier the line stays out of Telegram unless --notify."""
        from ggTrader.paper.notifier import DryRunNotifier
        from ggTrader.paper.trader import PaperTrader

        blend = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
        blend["sleeves"]["sp500"]["membership_gate"] = {
            "level": "warn",
            "message": "lag warning",
        }
        mock_signals.return_value = blend
        _t, broker, _n = _make_trader()
        telegram = MagicMock()
        silent = PaperTrader(broker, DryRunNotifier(telegram), dry_run=True, membership=_FIXTURE)
        silent.run()
        telegram.send.assert_not_called()
        forwarded = PaperTrader(
            broker, DryRunNotifier(telegram, forward=True), dry_run=True, membership=_FIXTURE
        )
        forwarded.run()
        assert any("lag warning" in call.args[0] for call in telegram.send.call_args_list)
        assert all(call.args[0].startswith("[DRY RUN] ") for call in telegram.send.call_args_list)

    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_a_passing_gate_adds_no_message(self, mock_signals):
        blend = _blend(buys=[], sells=[], as_of=_TEST_TODAY)
        blend["sleeves"]["sp500"]["membership_gate"] = {"applied": True, "level": "pass"}
        mock_signals.return_value = blend
        trader, _b, notifier = self._live_trader(_FIXTURE)
        trader.run()
        assert not any("Membership list" in c.args[0] for c in notifier.send.call_args_list)


@pytest.mark.parametrize("stop_enabled", [False, True])
@patch("ggTrader.paper.trader.generate_core_signals")
def test_pending_order_read_failure_stops_live_run(mock_signals, stop_enabled):
    mock_signals.return_value = _blend(buys=["MSFT"], sells=["NXPI"])
    trader, broker, notifier = _make_trader(
        positions={"NXPI": _losing_position(qty=100.0, cost_basis=10000.0, loss_pct=-0.40)}
    )
    with (
        patch.dict(
            "os.environ",
            {"CATASTROPHE_STOP_ENABLED": str(stop_enabled).lower(), "CASH_SWEEP_ENABLED": "true"},
        ),
        patch("ggTrader.paper.trader.get_pending_orders", side_effect=RuntimeError("DB down")),
        pytest.raises(RuntimeError, match="DB down"),
    ):
        trader.run()
    broker.submit_buy.assert_not_called()
    broker.submit_sell.assert_not_called()
    assert any("pending orders" in str(call).lower() for call in notifier.send.call_args_list)


def test_pending_order_read_failure_dry_run_still_rehearses():
    from ggTrader.paper.trader import PaperTrader

    _, broker, notifier = _make_trader()
    trader = PaperTrader(broker, notifier, dry_run=True, read_only=True)
    with (
        patch("ggTrader.paper.trader.get_pending_orders", side_effect=RuntimeError("DB down")),
        patch("ggTrader.paper.trader.generate_core_signals", return_value=_blend(["MSFT"], [])),
    ):
        result = trader.run()
    assert result["buys"] == ["MSFT"]
    broker.submit_buy.assert_not_called()
    broker.submit_sell.assert_not_called()
