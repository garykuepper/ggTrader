"""The paper ledger as a protocol, with a module-lookup form, a read-only form and a test double.

Public names: ``RunLock``, ``PaperLedger``, ``WRITE_METHODS``, ``TRADER_NAMES``, ``ModuleLedger``,
``ReadOnlyLedger`` and ``InMemoryLedger``.

``PaperLedger`` has one method per public function of ``paper/persist.py``: same name, same
parameters, same return annotation (``log_snapshot`` takes ``dict[str, Position]`` and
``paper_run_lock`` returns ``AbstractContextManager[RunLock]``). The names are the same on purpose,
so the skipped-write record of a read-only run stays name-for-name what it is today.

Nothing imports this module yet; the trader is wired to it in a later task.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import date
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from ggTrader.paper.constants import TERMINAL_STATUSES
from ggTrader.paper.models import Position

#: The 15 persist functions that write (each has a ``commit`` or an ``INSERT`` in the SQL golden,
#: ``tests/paper/golden/persist_sql.json``). The three ``get_*`` readers of intents, pending
#: orders and state, and ``paper_run_lock``, are not writes.
WRITE_METHODS: frozenset[str] = frozenset(
    {
        "init_paper_schema",
        "create_order_intent",
        "attach_broker_order",
        "settle_order_intent",
        "settle_legacy_pending_order",
        "log_trade",
        "log_pending_order",
        "clear_pending_order",
        "mark_pending_order_stale",
        "log_snapshot",
        "save_peak_value",
        "save_rebalance_state",
        "log_dividend_accrual",
        "save_split_correction",
        "clear_split_correction",
    }
)

#: The 26 persist names ``paper/trader.py`` binds in its own namespace. The trader's tests patch
#: them there by dotted string, so ``ModuleLedger`` looks these up in the trader's namespace.
TRADER_NAMES: tuple[str, ...] = (
    "attach_broker_order",
    "clear_pending_order",
    "clear_split_correction",
    "create_order_intent",
    "get_accrued_dividend_keys",
    "get_earliest_snapshot",
    "get_last_buy_dates",
    "get_latest_snapshot",
    "get_open_split_states",
    "get_peak_value",
    "get_pending_orders",
    "get_snapshot_history",
    "get_total_dividend_accrual",
    "get_trade_history_dates",
    "get_unresolved_order_intents",
    "init_paper_schema",
    "log_dividend_accrual",
    "log_pending_order",
    "log_snapshot",
    "log_trade",
    "mark_pending_order_stale",
    "paper_run_lock",
    "save_peak_value",
    "save_split_correction",
    "settle_legacy_pending_order",
    "settle_order_intent",
)

_TERMINAL_STATUSES = frozenset(TERMINAL_STATUSES)

#: The keys of a stored trade that follow ``run_date`` in the tuple a fill is compared with.
_TRADE_TAIL: tuple[str, ...] = ("side", "symbol", "amount", "reason")


class RunLock(Protocol):
    """A held run lock whose ownership can be checked before an order is submitted."""

    def verify(self) -> None: ...


@runtime_checkable
class PaperLedger(Protocol):
    """Durable paper-trading state: order intents, trades, snapshots and risk state.

    ``persist.py`` holds the SQL implementation. The signatures below are the ones of the 31
    public persist functions.
    """

    def init_paper_schema(self) -> None: ...

    def create_order_intent(
        self,
        client_order_id: str,
        run_date: str,
        side: str,
        symbol: str,
        amount: float,
        amount_kind: str,
        reason: str | None,
    ) -> None: ...

    def get_unresolved_order_intents(self) -> list[dict]: ...

    def attach_broker_order(self, client_order_id: str, order_id: str) -> None: ...

    def settle_order_intent(
        self,
        client_order_id: str,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float | None,
        terminal_status: str,
    ) -> None: ...

    def paper_run_lock(self) -> AbstractContextManager[RunLock]: ...

    def log_trade(
        self,
        run_date: str,
        side: str,
        symbol: str,
        amount: float,
        order_id: str,
        reason: str | None = None,
    ) -> None: ...

    def settle_legacy_pending_order(
        self,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float | None,
        terminal_status: str,
    ) -> None: ...

    def log_pending_order(
        self,
        run_date: str,
        side: str,
        symbol: str,
        notional: float,
        order_id: str,
        reason: str | None = None,
    ) -> None: ...

    def get_pending_orders(self) -> list[dict]: ...

    def clear_pending_order(self, order_id: str) -> None: ...

    def mark_pending_order_stale(self, order_id: str) -> None: ...

    def get_latest_snapshot(self, *, before: date | None = None) -> float | None: ...

    def get_earliest_snapshot(self) -> float | None: ...

    def log_snapshot(
        self, run_date: str, portfolio_value: float, cash: float, positions: dict[str, Position]
    ) -> None: ...

    def get_latest_snapshot_positions(self) -> dict | None: ...

    def get_latest_snapshot_run_date(self) -> str | None: ...

    def get_peak_value(self) -> float | None: ...

    def save_peak_value(self, peak_value: float) -> None: ...

    def get_rebalance_state(self) -> dict | None: ...

    def get_snapshot_history(self) -> list[tuple]: ...

    def get_trade_history_dates(self, symbol: str) -> list: ...

    def get_last_buy_dates(self) -> dict[str, date]: ...

    def get_accrued_dividend_keys(self) -> set[tuple]: ...

    def log_dividend_accrual(
        self, symbol: str, ex_date, rate: float, qty: float, amount: float
    ) -> None: ...

    def get_total_dividend_accrual(self) -> float: ...

    def save_rebalance_state(
        self, rebalance_date: str, weights: dict[str, float], scale: float
    ) -> None: ...

    def get_open_split_corrections(self) -> dict[str, float]: ...

    def get_open_split_states(self) -> dict[str, dict]: ...

    def save_split_correction(self, symbol: str, ex_date: str, factor: float) -> None: ...

    def clear_split_correction(self, symbol: str) -> None: ...


class ModuleLedger:
    """A ledger that looks each function up in a module namespace at the moment it is used.

    The trader's tests patch the persist functions by name in the trader module
    (``patch("ggTrader.paper.trader.log_trade")``). Resolving ``TRADER_NAMES`` in that namespace at
    every access, never caching, keeps those patches working however late they are applied. The
    five persist names the trader module does not bind come from ``fallback``. A name neither
    module has raises ``AttributeError``.

    ``isinstance(x, PaperLedger)`` is false for this class on Python 3.12 and later, which check
    the class for the methods without calling ``__getattr__``.
    """

    def __init__(self, namespace: ModuleType, fallback: ModuleType) -> None:
        self._namespace = namespace
        self._fallback = fallback

    def __getattr__(self, name: str) -> Any:
        # `__dict__` is found by ordinary lookup, so a copy made without `__init__` raises
        # `AttributeError` below instead of recursing into this method.
        state = self.__dict__
        if "_namespace" not in state:
            raise AttributeError(name)
        if name in TRADER_NAMES:
            return getattr(state["_namespace"], name)
        return getattr(state["_fallback"], name)


class ReadOnlyLedger:
    """A ledger that skips every write and passes every other name through to ``inner``.

    A skipped write appends the method name to ``skipped``, logs
    ``read-only: skipped write <name>`` at INFO through ``log`` and returns ``None`` without
    touching ``inner``. The caller passes its own module logger, so a handler or filter set by
    logger name sees the same logger as before. It never catches an exception.
    """

    def __init__(self, inner: PaperLedger, skipped: list[str], log: logging.Logger) -> None:
        self._inner = inner
        self._skipped = skipped
        self._log = log

    def __getattr__(self, name: str) -> Any:
        state = self.__dict__
        if "_inner" not in state:
            raise AttributeError(name)
        if name not in WRITE_METHODS:
            return getattr(state["_inner"], name)
        skipped, log = state["_skipped"], state["_log"]

        def skip(*args: Any, **kwargs: Any) -> None:
            skipped.append(name)
            log.info("read-only: skipped write %s", name)
            return None

        skip.__name__ = name
        return skip


class _MemoryRunLock:
    """The lock object of ``InMemoryLedger``: counts its ``verify`` calls."""

    def __init__(self, ledger: InMemoryLedger) -> None:
        self._ledger = ledger

    def verify(self) -> None:
        self._ledger.lock_verifications += 1


def _iso(value: Any) -> str:
    """A date or its ISO string, as the ISO string."""
    return value.isoformat() if isinstance(value, date) else str(value)


def _trade_key(trade: dict) -> tuple[Any, ...]:
    """A stored trade as the tuple a new fill is compared with."""
    return (_iso(trade["run_date"]), *(trade[key] for key in _TRADE_TAIL))


def _as_date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _validate_terminal_fill(
    filled_qty: float | None, filled_avg_price: float | None, terminal_status: str
) -> float:
    """Raise the SQL versions' ``ValueError`` for a bad terminal fill; return the fill amount."""
    if (
        filled_qty is None
        or not math.isfinite(filled_qty)
        or filled_qty < 0
        or terminal_status not in _TERMINAL_STATUSES
        or (terminal_status == "filled" and filled_qty == 0)
        or (
            filled_qty > 0
            and (
                filled_avg_price is None
                or not math.isfinite(filled_avg_price)
                or filled_avg_price <= 0
            )
        )
        or (filled_avg_price is not None and not math.isfinite(filled_avg_price))
    ):
        raise ValueError("Invalid terminal fill")
    amount = filled_qty * filled_avg_price if filled_qty > 0 else 0
    if not math.isfinite(amount):
        raise ValueError("Invalid fill value")
    return amount


class InMemoryLedger:
    """A test double that keeps the paper ledger in dicts and lists.

    The order-intent lifecycle (create, attach, get unresolved, settle) raises the same
    ``ValueError`` messages as the SQL functions, and the reads return what was seeded or
    written. It is modeled on the transcript harness's ``IntentStore``. It does not claim SQL
    parity: there are no transactions, no row locks, no constraints (a ``side`` or ``amount_kind``
    the table would reject is accepted), ``created_at`` is always ``None``, and the checks it
    makes are the ones listed above. The SQL golden and the SQLite runs in ``test_paper_persist``
    stay the authority on the SQL.

    Attributes:
        trades: The booked trades, oldest first.
        lock_entries: How many times ``paper_run_lock`` was entered.
        lock_verifications: How many times a lock object's ``verify`` was called.
        schema_initialized: Whether ``init_paper_schema`` has run.
    """

    def __init__(self) -> None:
        self.schema_initialized = False
        self.trades: list[dict] = []
        self.lock_entries = 0
        self.lock_verifications = 0
        self._intents: dict[str, dict] = {}
        self._pending: dict[str, dict] = {}
        self._snapshots: dict[str, dict] = {}
        self._peak: float | None = None
        self._rebalance: dict | None = None
        self._dividends: dict[tuple[str, str], dict] = {}
        self._splits: dict[str, dict] = {}

    # -- schema and lock -----------------------------------------------------------------

    def init_paper_schema(self) -> None:
        self.schema_initialized = True

    @contextmanager
    def paper_run_lock(self) -> Iterator[_MemoryRunLock]:
        self.lock_entries += 1
        yield _MemoryRunLock(self)

    # -- order intents -------------------------------------------------------------------

    def create_order_intent(
        self,
        client_order_id: str,
        run_date: str,
        side: str,
        symbol: str,
        amount: float,
        amount_kind: str,
        reason: str | None,
    ) -> None:
        if client_order_id in self._intents:
            raise ValueError("Duplicate client order id")
        self._intents[client_order_id] = {
            "client_order_id": client_order_id,
            "run_date": run_date,
            "side": side,
            "symbol": symbol,
            "amount": amount,
            "amount_kind": amount_kind,
            "reason": reason,
            "broker_order_id": None,
            "state": "intent",
            "terminal_status": None,
            "filled_qty": None,
            "filled_avg_price": None,
        }

    def get_unresolved_order_intents(self) -> list[dict]:
        return [copy.deepcopy(row) for row in self._intents.values() if row["state"] != "resolved"]

    def _reject_pending(self, order_id: str) -> None:
        if order_id in self._pending:
            raise ValueError("Order has both pending and intent representations")

    def attach_broker_order(self, client_order_id: str, order_id: str) -> None:
        self._reject_pending(order_id)
        row = self._intents.get(client_order_id)
        if row is None or row["broker_order_id"] not in (None, order_id):
            raise ValueError("Missing intent or conflicting broker order")
        if row["broker_order_id"] is None:
            if row["state"] != "intent":
                raise ValueError("Invalid intent state")
            row["broker_order_id"] = order_id
            row["state"] = "submitted"

    def settle_order_intent(
        self,
        client_order_id: str,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float | None,
        terminal_status: str,
    ) -> None:
        amount = _validate_terminal_fill(filled_qty, filled_avg_price, terminal_status)
        self._reject_pending(order_id)
        row = self._intents.get(client_order_id)
        if row is None or row["broker_order_id"] not in (None, order_id):
            raise ValueError("Missing intent or conflicting broker order")
        if row["state"] == "resolved":
            settled = (
                row["broker_order_id"],
                row["filled_qty"],
                row["filled_avg_price"],
                row["terminal_status"],
            )
            if settled != (order_id, filled_qty, filled_avg_price, terminal_status):
                raise ValueError("Conflicting settlement")
            return
        self._book_fill(
            order_id,
            filled_qty,
            amount,
            (row["run_date"], row["side"], row["symbol"], row["reason"]),
        )
        row.update(
            broker_order_id=order_id,
            state="resolved",
            terminal_status=terminal_status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
        )

    def _trades_of(self, order_id: str) -> list[dict]:
        return [t for t in self.trades if t["order_id"] == order_id]

    def _book_fill(
        self, order_id: str, filled_qty: float, amount: float, identity: tuple[Any, ...]
    ) -> None:
        """Insert the trade of a fill unless an identical one exists; raise on a conflicting one."""
        run_date, side, symbol, reason = identity
        existing = self._trades_of(order_id)
        expected = (_iso(run_date), side, symbol, amount, reason)
        if len(existing) > 1 or (
            existing and (filled_qty == 0 or _trade_key(existing[0]) != expected)
        ):
            raise ValueError("Conflicting historical trade")
        if filled_qty > 0 and not existing:
            self.trades.append(
                {
                    "run_date": run_date,
                    "side": side,
                    "symbol": symbol,
                    "amount": amount,
                    "order_id": order_id,
                    "reason": reason,
                }
            )

    # -- legacy pending orders and trades --------------------------------------------------

    def log_trade(
        self,
        run_date: str,
        side: str,
        symbol: str,
        amount: float,
        order_id: str,
        reason: str | None = None,
    ) -> None:
        existing = self._trades_of(order_id)
        expected = (_iso(run_date), side, symbol, amount, reason)
        if len(existing) > 1 or (existing and _trade_key(existing[0]) != expected):
            raise ValueError("Conflicting historical trade")
        if existing:
            return
        self.trades.append(
            {
                "run_date": run_date,
                "side": side,
                "symbol": symbol,
                "amount": amount,
                "order_id": order_id,
                "reason": reason,
            }
        )

    def settle_legacy_pending_order(
        self,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float | None,
        terminal_status: str,
    ) -> None:
        amount = _validate_terminal_fill(filled_qty, filled_avg_price, terminal_status)
        row = self._pending.get(order_id)
        if row is None:
            raise ValueError("Missing legacy pending order")
        if any(i["broker_order_id"] == order_id for i in self._intents.values()):
            raise ValueError("Order has both pending and intent representations")
        self._book_fill(
            order_id,
            filled_qty,
            amount,
            (row["run_date"], row["side"], row["symbol"], row["reason"]),
        )
        del self._pending[order_id]

    def log_pending_order(
        self,
        run_date: str,
        side: str,
        symbol: str,
        notional: float,
        order_id: str,
        reason: str | None = None,
    ) -> None:
        self._pending.setdefault(
            order_id,
            {
                "order_id": order_id,
                "run_date": run_date,
                "side": side,
                "symbol": symbol,
                "notional": notional,
                "created_at": None,
                "flagged_stale": False,
                "reason": reason,
            },
        )

    def get_pending_orders(self) -> list[dict]:
        return [copy.deepcopy(row) for row in self._pending.values()]

    def clear_pending_order(self, order_id: str) -> None:
        self._pending.pop(order_id, None)

    def mark_pending_order_stale(self, order_id: str) -> None:
        if order_id in self._pending:
            self._pending[order_id]["flagged_stale"] = True

    # -- snapshots, peak and rebalance state ---------------------------------------------

    def get_latest_snapshot(self, *, before: date | None = None) -> float | None:
        days = sorted(d for d in self._snapshots if before is None or d < _iso(before))
        return float(self._snapshots[days[-1]]["portfolio_value"]) if days else None

    def get_earliest_snapshot(self) -> float | None:
        days = sorted(self._snapshots)
        return float(self._snapshots[days[0]]["portfolio_value"]) if days else None

    def log_snapshot(
        self, run_date: str, portfolio_value: float, cash: float, positions: dict[str, Position]
    ) -> None:
        self._snapshots[_iso(run_date)] = {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "positions": copy.deepcopy(positions),
        }

    def get_latest_snapshot_positions(self) -> dict | None:
        days = sorted(self._snapshots)
        return copy.deepcopy(self._snapshots[days[-1]]["positions"]) if days else None

    def get_latest_snapshot_run_date(self) -> str | None:
        days = sorted(self._snapshots)
        return days[-1] if days else None

    def get_peak_value(self) -> float | None:
        return self._peak

    def save_peak_value(self, peak_value: float) -> None:
        self._peak = float(peak_value)

    def get_rebalance_state(self) -> dict | None:
        return copy.deepcopy(self._rebalance)

    def save_rebalance_state(
        self, rebalance_date: str, weights: dict[str, float], scale: float
    ) -> None:
        self._rebalance = {
            "rebalance_date": _iso(rebalance_date),
            "weights": copy.deepcopy(weights),
            "scale": float(scale),
        }

    def get_snapshot_history(self) -> list[tuple]:
        return [
            (date.fromisoformat(day), copy.deepcopy(self._snapshots[day]["positions"]))
            for day in sorted(self._snapshots)
        ]

    # -- trade history and dividends -----------------------------------------------------

    def get_trade_history_dates(self, symbol: str) -> list:
        return sorted(_as_date(t["run_date"]) for t in self.trades if t["symbol"] == symbol)

    def get_last_buy_dates(self) -> dict[str, date]:
        latest: dict[str, date] = {}
        for trade in self.trades:
            if trade["side"] == "BUY":
                day = _as_date(trade["run_date"])
                latest[trade["symbol"]] = max(day, latest.get(trade["symbol"], day))
        return latest

    def get_accrued_dividend_keys(self) -> set[tuple]:
        return {(row["symbol"], row["ex_date"]) for row in self._dividends.values()}

    def log_dividend_accrual(
        self, symbol: str, ex_date, rate: float, qty: float, amount: float
    ) -> None:
        self._dividends.setdefault(
            (symbol, _iso(ex_date)),
            {"symbol": symbol, "ex_date": ex_date, "rate": rate, "qty": qty, "amount": amount},
        )

    def get_total_dividend_accrual(self) -> float:
        return float(sum(row["amount"] for row in self._dividends.values()))

    # -- split corrections ---------------------------------------------------------------

    def get_open_split_corrections(self) -> dict[str, float]:
        return {symbol: float(row["factor"]) for symbol, row in self._splits.items()}

    def get_open_split_states(self) -> dict[str, dict]:
        return {
            symbol: {"factor": float(row["factor"]), "ex_date": _as_date(row["ex_date"])}
            for symbol, row in self._splits.items()
        }

    def save_split_correction(self, symbol: str, ex_date: str, factor: float) -> None:
        # Like the SQL upsert, a repeat for a symbol refreshes the factor and keeps the ex-date.
        if symbol in self._splits:
            self._splits[symbol]["factor"] = factor
        else:
            self._splits[symbol] = {"ex_date": ex_date, "factor": factor}

    def clear_split_correction(self, symbol: str) -> None:
        self._splits.pop(symbol, None)
