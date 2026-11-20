"""Pins for WHEN the trader resolves its collaborators, written through the public trader only.

The old private forwards (`_submit_order`, `_poll_orders`, `_missed_exits`, ...) rebuilt the
order executor and the corporate-actions step from the trader's CURRENT state at every call. A
collaborator or flag the trader replaced mid-run therefore applied to the next call. These pins
use only `PaperTrader`, `InMemoryLedger` and fakes (no step-object API), so the same file runs
against the code before the forwards were retired and after.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper.corporate_actions import CorporateActions
from ggTrader.paper.ledger import InMemoryLedger
from ggTrader.paper.trader import PaperTrader
from tests.paper.support import seed_pending_orders

_TODAY = "2026-06-19"
_AAPL = {"qty": 10.0, "market_value": 1500.0, "avg_entry": 145.0, "unrealized_pl": 50.0}
_MSFT = {"qty": 5.0, "market_value": 2000.0, "avg_entry": 390.0, "unrealized_pl": 50.0}


@pytest.fixture(autouse=True)
def _pinned_environment():
    from ggTrader.paper.gates import PT

    with (
        patch("ggTrader.paper.trader._today_et", return_value=date.fromisoformat(_TODAY)),
        patch(
            "ggTrader.paper.trader._now_pt",
            return_value=datetime(2026, 6, 19, 12, 46, tzinfo=PT),
        ),
        patch("ggTrader.paper.trader.refresh_benchmark_tape", return_value=[]),
        patch("ggTrader.paper.trader.require_live_membership_spec"),
    ):
        yield


def _signals(sells: list[str], last_exit: dict[str, str] | None = None) -> dict:
    return {
        "sleeves": {
            "sp500": {
                "buys": [],
                "sells": sells,
                "as_of": _TODAY,
                "universe_size": 100,
                "last_exit": last_exit or {},
            }
        },
        "weights": {"sp500": 1.0},
        "scale": 1.0,
        "rebalanced_today": False,
        "fallback_used": False,
    }


def _broker(positions: dict) -> MagicMock:
    broker = MagicMock()
    broker.get_account.return_value = {
        "cash": 50000.0,
        "portfolio_value": 100000.0,
        "last_equity": 100000.0,
        "buying_power": 100000.0,
    }
    broker.get_positions.return_value = positions
    broker.get_untradable.return_value = set()
    broker.get_split_corrections.return_value = {}
    broker.get_split_evidence.return_value = {"corp_splits": {}, "activity_applied": set()}
    broker.get_dividend_corrections.return_value = {"corp_dividends": {}, "credited_keys": set()}
    broker.get_clock.return_value = {"is_open": True}
    return broker


class _CountingLedger(InMemoryLedger):
    """An in-memory ledger that counts the reads the corporate-action steps make."""

    WATCHED = ("get_last_buy_dates", "get_total_dividend_accrual", "get_open_split_states")

    def __init__(self) -> None:
        super().__init__()
        self.reads: dict[str, int] = dict.fromkeys(self.WATCHED, 0)

    def get_last_buy_dates(self, *args, **kwargs):
        self.reads["get_last_buy_dates"] += 1
        return super().get_last_buy_dates(*args, **kwargs)

    def get_total_dividend_accrual(self, *args, **kwargs):
        self.reads["get_total_dividend_accrual"] += 1
        return super().get_total_dividend_accrual(*args, **kwargs)

    def get_open_split_states(self, *args, **kwargs):
        self.reads["get_open_split_states"] += 1
        return super().get_open_split_states(*args, **kwargs)


@patch("ggTrader.paper.trader.generate_core_signals")
def test_a_read_only_flag_set_by_the_first_sell_rejects_the_second_order(mock_signals):
    """The second order is built from the trader's state when it is submitted, not the first's."""
    mock_signals.return_value = _signals(sells=["AAPL", "MSFT"])
    broker = _broker({"AAPL": dict(_AAPL), "MSFT": dict(_MSFT)})
    ledger = InMemoryLedger()
    notifier = MagicMock()
    trader = PaperTrader(broker, notifier, dry_run=False, ledger=ledger)

    def first_sell_flips_the_flag(symbol, amount, client_id=None):
        trader._read_only = True
        return "sell-1"

    broker.submit_sell.side_effect = first_sell_flips_the_flag

    with pytest.raises(RuntimeError, match="read-only trader must not submit orders"):
        trader.run()

    assert broker.submit_sell.call_count == 1
    assert len(InMemoryLedger.get_unresolved_order_intents(ledger)) == 1  # no intent for #2


@patch("ggTrader.paper.trader.generate_core_signals")
def test_b_a_ledger_replaced_by_get_positions_is_the_one_the_corporate_actions_read(
    mock_signals,
):
    """Missed-exit, split and dividend handling read the ledger current when each runs."""
    mock_signals.return_value = _signals(sells=[], last_exit={"AAPL": "2026-06-10"})
    broker = _broker({"AAPL": dict(_AAPL)})
    first, second = _CountingLedger(), _CountingLedger()
    # Only the FIRST ledger records an earlier buy, so only it would call the exit missed.
    InMemoryLedger.log_trade(first, "2026-06-05", "BUY", "AAPL", 1.0, "AAPL")
    trader = PaperTrader(broker, MagicMock(), dry_run=True, read_only=False, ledger=first)

    def positions_then_swap_ledger():
        trader._ledger = second
        return {"AAPL": dict(_AAPL)}

    broker.get_positions.side_effect = positions_then_swap_ledger
    for name in _CountingLedger.WATCHED:
        first.reads[name] = 0  # forget what the run read before the swap

    result = trader.run()

    assert first.reads == dict.fromkeys(_CountingLedger.WATCHED, 0)
    assert second.reads["get_last_buy_dates"] == 1
    assert second.reads["get_total_dividend_accrual"] >= 1
    assert second.reads["get_open_split_states"] >= 1
    assert result["sells"] == []  # the second ledger has no buy record, so no missed exit


@pytest.mark.parametrize(
    ("failing_read", "error"),
    [("get_account", RuntimeError("account down")), ("get_positions", RuntimeError("no data"))],
)
@patch("ggTrader.paper.trader.generate_core_signals")
def test_c_a_corporate_actions_constructor_that_raises_does_not_preempt_the_read_halt(
    mock_signals, failing_read, error
):
    """A broker read failure halts the run with its own exception and alert, as it always did."""
    mock_signals.return_value = _signals(sells=[])
    broker = _broker({"AAPL": dict(_AAPL)})
    getattr(broker, failing_read).side_effect = error
    notifier = MagicMock()
    trader = PaperTrader(broker, notifier, dry_run=True, read_only=False, ledger=InMemoryLedger())

    with patch.object(CorporateActions, "__init__", side_effect=ValueError("constructor")):
        with pytest.raises(RuntimeError) as raised:
            trader.run()

    assert raised.value is error
    assert notifier.send.called  # the halt alert went out


@patch("ggTrader.paper.trader.generate_core_signals")
def test_d_the_stale_order_alert_goes_to_the_notifier_current_when_it_is_raised(mock_signals):
    """The alert is built when it fires: a notifier swapped during the broker lookup gets it."""
    mock_signals.return_value = _signals(sells=[])
    broker = _broker({})
    ledger = InMemoryLedger()
    seed_pending_orders(
        ledger,
        [
            {
                "order_id": "old-1",
                "run_date": "2026-01-02",
                "side": "BUY",
                "symbol": "MSFT",
                "notional": 1000.0,
                "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            }
        ],
    )
    before, after = MagicMock(), MagicMock()
    trader = PaperTrader(broker, before, dry_run=False, ledger=ledger)

    def still_working(order_id):
        trader._notifier = after
        return {"id": order_id, "status": "new", "filled_qty": 0.0, "filled_avg_price": None}

    broker.get_order.side_effect = still_working
    trader.run()

    def stale_alerts(notifier):
        return [c for c in notifier.send.call_args_list if "Stale pending order" in str(c)]

    assert len(stale_alerts(after)) == 1
    assert stale_alerts(before) == []
