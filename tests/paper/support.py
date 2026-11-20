"""In-memory paper trader test helpers."""

import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from ggTrader.paper.ledger import InMemoryLedger, PaperLedger

if TYPE_CHECKING:
    from ggTrader.paper.trader import PaperTrader

Call = tuple[str, tuple[Any, ...], dict[str, Any]]
LEDGER_NAMES = frozenset(
    name
    for name, value in PaperLedger.__dict__.items()
    if not name.startswith("_") and callable(value)
)


_REAL_BUILDER = "_real_executor_for_call"


def forward_to_broker(broker: Any) -> MagicMock:
    """A stand-in for `OrderExecutor.submit_order` that calls the broker and writes no intent."""
    return MagicMock(
        side_effect=lambda side, symbol, amount, reason: (
            broker.submit_buy(symbol, amount)
            if side == "BUY"
            else broker.submit_sell(symbol, amount)
        )
    )


def stub_submit_order(trader: "PaperTrader", stub: MagicMock) -> MagicMock:
    """Make every executor `trader` builds from now on submit through `stub`.

    The replacement point is `OrderExecutor.submit_order` on the executor instance the trader's
    `_executor_for_call` builds; the stub is also kept as `trader.submit_order_mock` so a test can
    assert the arguments of each order. `restore_submit_order` undoes it.
    """
    build = trader.__dict__.setdefault(_REAL_BUILDER, trader._executor_for_call)

    def build_with_stub() -> Any:
        executor = build()
        executor.submit_order = stub
        return executor

    trader._executor_for_call = build_with_stub  # type: ignore[method-assign]
    trader.submit_order_mock = stub  # type: ignore[attr-defined]
    return stub


def restore_submit_order(trader: "PaperTrader") -> None:
    """Undo `stub_submit_order`: the trader's executors submit through the real method again."""
    trader.__dict__.pop(_REAL_BUILDER, None)
    trader.__dict__.pop("_executor_for_call", None)
    trader.__dict__.pop("submit_order_mock", None)


class SpyLedger(InMemoryLedger):
    """Record public ledger calls while retaining in-memory behavior."""

    def __init__(self) -> None:
        super().__init__()
        self._calls: list[Call] = []

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name not in PaperLedger.__dict__ or name.startswith("_"):
            return value

        @wraps(value)
        def record(*args: Any, **kwargs: Any) -> Any:
            self._calls.append((name, deepcopy(args), deepcopy(kwargs)))
            return value(*args, **kwargs)

        return record

    def calls(self, name: str | None = None) -> list[Call]:
        """Return recorded calls; use unbound InMemoryLedger methods to inspect state."""
        if name is not None and name not in LEDGER_NAMES:
            raise ValueError(f"unknown ledger method: {name}")
        return [call for call in self._calls if name is None or call[0] == name]


def seed_snapshots(ledger: SpyLedger, snapshots: Iterable[tuple[Any, ...]]) -> None:
    """Seed snapshots without recording setup as trader activity."""
    for snapshot in snapshots:
        InMemoryLedger.log_snapshot(ledger, *snapshot)


def seed_peak(ledger: SpyLedger, peak: float) -> None:
    """Seed the persisted peak."""
    InMemoryLedger.save_peak_value(ledger, peak)


def seed_pending_orders(ledger: InMemoryLedger, orders: Iterable[Mapping[str, Any]]) -> None:
    """Seed complete legacy pending rows without recording setup."""
    required = {"order_id", "run_date", "side", "symbol", "notional"}
    allowed = required | {"created_at", "flagged_stale", "reason"}
    for order in orders:
        if not required <= order.keys() or not order.keys() <= allowed:
            raise ValueError("pending order has missing or unknown fields")
        row = {"created_at": None, "flagged_stale": False, "reason": None, **deepcopy(order)}
        ledger._pending.setdefault(row["order_id"], row)


class UnstubbedPersistCall(AssertionError):
    """A trader test reached an unpatched persistence function."""


@contextmanager
def persist_guard() -> Iterator[list[Call]]:
    """Fail on unstubbed calls, including exceptions swallowed by the trader."""
    calls: list[Call] = []

    def guard(name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> None:
            calls.append((name, args, kwargs))
            raise UnstubbedPersistCall(name)

        return fail

    class GuardedLedger:
        """Reject ledger operations attempted without an injected test ledger."""

        def __getattr__(self, name: str) -> Any:
            if name not in LEDGER_NAMES:
                raise AttributeError(name)
            return guard(name)

    try:
        with patch("ggTrader.paper.trader.PgPaperLedger", GuardedLedger):
            yield calls
    finally:
        if calls:
            raise UnstubbedPersistCall(f"Unstubbed persist calls: {calls}") from sys.exc_info()[1]


def make_trader(
    positions: dict[str, dict[str, Any]] | None = None,
    portfolio_value: float = 100000.0,
    cash: float = 50000.0,
    *,
    ledger: PaperLedger | None = None,
    dry_run: bool = False,
    read_only: bool | None = None,
) -> tuple["PaperTrader", MagicMock, MagicMock, PaperLedger]:
    """Build a trader with an exposed in-memory ledger and broker fakes."""
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

    ledger = SpyLedger() if ledger is None else ledger
    trader = PaperTrader(broker, notifier, dry_run=dry_run, read_only=read_only, ledger=ledger)
    stub_submit_order(trader, forward_to_broker(broker))
    return trader, broker, notifier, ledger
