"""In-memory paper trader test helpers."""

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

from ggTrader.paper.ledger import TRADER_NAMES, InMemoryLedger, PaperLedger

PERSIST_NAMES = TRADER_NAMES


class SpyLedger(InMemoryLedger):
    """Record public ledger calls while retaining in-memory behavior."""

    def __init__(self):
        super().__init__()
        self._calls = []

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name not in PaperLedger.__dict__ or name.startswith("_"):
            return value

        def record(*args, **kwargs):
            self._calls.append((name, args, kwargs))
            return value(*args, **kwargs)

        return record

    def calls(self, name=None):
        return [call for call in self._calls if name is None or call[0] == name]


def seed_snapshots(ledger, snapshots):
    """Seed snapshots without recording setup as trader activity."""
    for snapshot in snapshots:
        InMemoryLedger.log_snapshot(ledger, *snapshot)


def seed_peak(ledger, peak):
    """Seed the persisted peak."""
    InMemoryLedger.save_peak_value(ledger, peak)


def seed_pending_orders(ledger, orders):
    """Seed legacy pending orders."""
    for order in orders:
        InMemoryLedger.log_pending_order(ledger, **order)


class UnstubbedPersistCall(AssertionError):
    """A trader test reached an unpatched persistence function."""


@contextmanager
def persist_guard():
    """Fail on unstubbed calls, including exceptions swallowed by the trader."""
    calls = []

    def guard(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            raise UnstubbedPersistCall(name)

        return fail

    try:
        with ExitStack() as stack:
            for name in PERSIST_NAMES:
                stack.enter_context(patch(f"ggTrader.paper.trader.{name}", guard(name)))
            yield calls
    finally:
        if calls:
            raise UnstubbedPersistCall(f"Unstubbed persist calls: {calls}")


def make_trader(positions=None, portfolio_value=100000.0, cash=50000.0):
    from ggTrader.paper.trader import PaperTrader

    broker = MagicMock()
    broker.get_account.return_value = {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "last_equity": portfolio_value,
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

    trader = PaperTrader(broker, notifier, dry_run=False, ledger=SpyLedger())
    trader._submit_order = MagicMock(
        side_effect=lambda side, symbol, amount, reason: (
            broker.submit_buy(symbol, amount)
            if side == "BUY"
            else broker.submit_sell(symbol, amount)
        )
    )
    return trader, broker, notifier
