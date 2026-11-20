"""The paper ledger protocol, its read-only form and its in-memory test double.

Public names: ``RunLock``, ``PaperLedger``, ``WRITE_METHODS``, ``ReadOnlyLedger``,
``InMemoryLedger``, ``NonDurableLedgerError``, ``ALLOW_NON_DURABLE_LEDGER`` and
``require_durable_for_live``.

``PaperLedger`` has one method per public function of ``paper/persist.py``: same name, same
parameters, same return annotation (``log_snapshot`` takes ``dict[str, Position]`` and
``paper_run_lock`` returns ``AbstractContextManager[RunLock]``). The names are the same on purpose,
so the skipped-write record of a read-only run stays name-for-name what it is today.

The trader uses these adapters to access persistence and suppress read-only writes.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import date
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

#: Lifts the live-run refusal of a non-durable ledger. Only the tests turn it on (the paper
#: tests' ``conftest.py`` sets it per test and restores it); production code never does. It is
#: read through this module at call time, so a fixture that rebinds it takes effect.
ALLOW_NON_DURABLE_LEDGER: bool = False

_TERMINAL_STATUSES = frozenset(TERMINAL_STATUSES)

#: The keys of a stored trade that follow ``run_date`` in the tuple a fill is compared with.
_TRADE_TAIL: tuple[str, ...] = ("side", "symbol", "amount", "reason")


class NonDurableLedgerError(RuntimeError):
    """A live run was given a ledger whose state does not outlive the process."""


class RunLock(Protocol):
    """A held run lock whose ownership can be checked before an order is submitted."""

    def verify(self) -> None:
        """Raise if the caller no longer owns the lock (the SQL lock raises ``RuntimeError``)."""
        ...


@runtime_checkable
class PaperLedger(Protocol):
    """Durable paper-trading state: order intents, trades, snapshots and risk state.

    ``persist.py`` holds the SQL implementation. The signatures below are the ones of the 31
    public persist functions. Every implementation also carries a class attribute
    ``durable: bool`` (True only when its state survives the process); it is not a protocol member
    because a ledger without it is treated as non-durable (``require_durable_for_live``).
    """

    def init_paper_schema(self) -> None:
        """Create the paper tables if they do not exist yet."""
        ...

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
        """Record an order intent under its client order id before the order reaches the broker.

        The intent starts in state ``intent``.
        """
        ...

    def get_unresolved_order_intents(self) -> list[dict]:
        """Return the intents that are not yet resolved, as dicts."""
        ...

    def attach_broker_order(self, client_order_id: str, order_id: str) -> None:
        """Attach a broker order id to an intent and mark it submitted.

        Raise ``ValueError`` for an order id that is also a legacy pending order (checked first), a
        missing intent, another broker id on the intent, or, on a first attachment, an intent not in
        state ``intent``. Attaching the same id again changes nothing. The SQL ledger's unique
        broker-id column also refuses an id already attached to another intent.
        """
        ...

    def settle_order_intent(
        self,
        client_order_id: str,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float | None,
        terminal_status: str,
    ) -> None:
        """Resolve an intent with its terminal fill and book the fill as a trade.

        Raise ``ValueError`` for an invalid fill, a missing or conflicting intent, a conflicting
        trade or an order id that is also a legacy pending order. A trade is booked only for a
        positive fill quantity, and an identical trade or identical earlier settlement is kept as it
        is.
        """
        ...

    def paper_run_lock(self) -> AbstractContextManager[RunLock]:
        """Return a context manager that holds the run lock and yields a ``RunLock``.

        The SQL ledger raises ``RuntimeError`` when another run holds the lock; the in-memory
        ledger only counts entries and ``verify`` calls and never refuses.
        """
        ...

    def log_trade(
        self,
        run_date: str,
        side: str,
        symbol: str,
        amount: float,
        order_id: str,
        reason: str | None = None,
    ) -> None:
        """Record a trade under its order id.

        A repeat of an identical trade changes nothing; a different trade under the same order id
        raises ``ValueError``.
        """
        ...

    def settle_legacy_pending_order(
        self,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float | None,
        terminal_status: str,
    ) -> None:
        """Book a legacy pending order's terminal fill as a trade and delete the pending order.

        Raise ``ValueError`` for an invalid fill, a missing pending order, an order id that is also
        an intent's broker order, or a conflicting trade. A trade is booked only for a positive fill
        quantity.
        """
        ...

    def log_pending_order(
        self,
        run_date: str,
        side: str,
        symbol: str,
        notional: float,
        order_id: str,
        reason: str | None = None,
    ) -> None:
        """Record an order still working at run end, maybe part-filled; a repeat id is ignored."""
        ...

    def get_pending_orders(self) -> list[dict]:
        """Return the legacy pending orders, as dicts."""
        ...

    def clear_pending_order(self, order_id: str) -> None:
        """Delete a pending order; an unknown order id does nothing."""
        ...

    def mark_pending_order_stale(self, order_id: str) -> None:
        """Mark a pending order as flagged stale; an unknown order id does nothing."""
        ...

    def get_latest_snapshot(self, *, before: date | None = None) -> float | None:
        """Return the portfolio value of the latest snapshot, or None when there is none.

        With ``before``, only snapshots strictly before that date count.
        """
        ...

    def get_earliest_snapshot(self) -> float | None:
        """Return the portfolio value of the earliest snapshot, or None when there is none."""
        ...

    def log_snapshot(
        self, run_date: str, portfolio_value: float, cash: float, positions: dict[str, Position]
    ) -> None:
        """Store the snapshot of a run date, replacing the one already stored for that date."""
        ...

    def get_latest_snapshot_positions(self) -> dict | None:
        """Return the positions of the latest snapshot, or None when there is none."""
        ...

    def get_latest_snapshot_run_date(self) -> str | None:
        """Return the latest snapshot's run date as an ISO string, or None when there is none."""
        ...

    def get_peak_value(self) -> float | None:
        """Return the stored portfolio peak, or None when none is stored."""
        ...

    def save_peak_value(self, peak_value: float) -> None:
        """Store the portfolio peak, replacing any earlier one."""
        ...

    def get_rebalance_state(self) -> dict | None:
        """Return the stored rebalance date, weights and scale, or None when none is stored."""
        ...

    def get_snapshot_history(self) -> list[tuple]:
        """Return every snapshot as ``(run_date, positions)``, oldest first."""
        ...

    def get_trade_history_dates(self, symbol: str) -> list:
        """Return the run dates of the trades in ``symbol``, oldest first."""
        ...

    def get_last_buy_dates(self) -> dict[str, date]:
        """Return the run date of the latest BUY trade of each symbol."""
        ...

    def get_accrued_dividend_keys(self) -> set[tuple]:
        """Return the ``(symbol, ex_date)`` keys of the stored dividend accruals."""
        ...

    def log_dividend_accrual(
        self, symbol: str, ex_date: date | str, rate: float, qty: float, amount: float
    ) -> None:
        """Store a dividend accrual per symbol and ex-date; a repeat of the pair is ignored."""
        ...

    def get_total_dividend_accrual(self) -> float:
        """Return the sum of the stored dividend accrual amounts (0.0 when there are none)."""
        ...

    def save_rebalance_state(
        self, rebalance_date: str, weights: dict[str, float], scale: float
    ) -> None:
        """Store the rebalance date, weights and scale, replacing any earlier state."""
        ...

    def get_open_split_corrections(self) -> dict[str, float]:
        """Return the stored split correction factor of each symbol."""
        ...

    def get_open_split_states(self) -> dict[str, dict]:
        """Return each symbol's stored split correction.

        The result maps each symbol to a dict with a float ``factor`` and a date ``ex_date``.
        """
        ...

    def save_split_correction(self, symbol: str, ex_date: str, factor: float) -> None:
        """Store a split correction for ``symbol``.

        A repeat for the same symbol updates the factor and keeps the stored ex-date.
        """
        ...

    def clear_split_correction(self, symbol: str) -> None:
        """Delete the stored split correction of ``symbol``; none stored does nothing."""
        ...


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

    @property
    def durable(self) -> bool:
        """Whether the wrapped ledger is durable (the wrapper adds nothing of its own)."""
        return bool(getattr(self._inner, "durable", False))

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


def require_durable_for_live(ledger: Any, *, dry_run: bool) -> None:
    """Raise ``NonDurableLedgerError`` for a live run on a ledger that is not durable.

    A dry run (read-only or not) is never refused. The flag ``ALLOW_NON_DURABLE_LEDGER`` is read
    from this module at call time. A ledger with no ``durable`` attribute counts as non-durable.
    """
    if dry_run or ALLOW_NON_DURABLE_LEDGER or getattr(ledger, "durable", False):
        return
    raise NonDurableLedgerError(
        f"a live run needs a durable ledger, got {type(ledger).__name__}: it would trade with "
        "no state that survives the process (ggTrader.paper.ledger.ALLOW_NON_DURABLE_LEDGER is "
        "for tests only)"
    )


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
        durable: Always False: nothing here outlives the process, so a live run refuses it.
    """

    durable: bool = False

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
        """Mark the schema as initialized."""
        self.schema_initialized = True

    @contextmanager
    def paper_run_lock(self) -> Iterator[_MemoryRunLock]:
        """Count one entry in ``lock_entries`` and yield a lock that counts its ``verify`` calls.

        Nothing is held and nothing is refused.
        """
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
        """Store a new intent; raise ``ValueError`` when its client order id already exists."""
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
        """Return deep copies of the intents that are not resolved, in creation order."""
        return [copy.deepcopy(row) for row in self._intents.values() if row["state"] != "resolved"]

    def _reject_pending(self, order_id: str) -> None:
        if order_id in self._pending:
            raise ValueError("Order has both pending and intent representations")

    def attach_broker_order(self, client_order_id: str, order_id: str) -> None:
        """Attach a broker order id to an intent and mark it submitted.

        Raise ``ValueError`` when the order id is a pending order (checked first), the intent is
        missing, it has another broker id, or, on a first attachment, it is not in state
        ``intent``. Attaching the same id again changes nothing.
        """
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
        """Resolve an intent with its terminal fill and book the fill as a trade.

        Raise ``ValueError`` for an invalid fill, a pending order with this id, a missing or
        conflicting intent, a conflicting settlement or a conflicting trade. A trade is booked only
        for a positive fill quantity. Settling an already resolved intent with the same fill changes
        nothing.
        """
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
        """Append the trade unless an identical one exists under the order id.

        A different trade under the same order id raises ``ValueError``.
        """
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
        """Book the terminal fill of a pending order as a trade and delete the pending order.

        Raise ``ValueError`` for an invalid fill, a missing pending order, an intent with this
        broker order id, or a conflicting trade. A trade is booked only for a positive fill
        quantity.
        """
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
        """Store a pending order; a repeat of the order id keeps the first one."""
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
        """Return deep copies of the pending orders, in the order they were stored."""
        return [copy.deepcopy(row) for row in self._pending.values()]

    def clear_pending_order(self, order_id: str) -> None:
        """Delete a pending order; an unknown order id does nothing."""
        self._pending.pop(order_id, None)

    def mark_pending_order_stale(self, order_id: str) -> None:
        """Flag a pending order as stale; an unknown order id does nothing."""
        if order_id in self._pending:
            self._pending[order_id]["flagged_stale"] = True

    # -- snapshots, peak and rebalance state ---------------------------------------------

    def get_latest_snapshot(self, *, before: date | None = None) -> float | None:
        """Return the portfolio value of the latest snapshot, or None when there is none.

        With ``before``, only snapshots strictly before that date count.
        """
        days = sorted(d for d in self._snapshots if before is None or d < _iso(before))
        return float(self._snapshots[days[-1]]["portfolio_value"]) if days else None

    def get_earliest_snapshot(self) -> float | None:
        """Return the portfolio value of the earliest snapshot, or None when there is none."""
        days = sorted(self._snapshots)
        return float(self._snapshots[days[0]]["portfolio_value"]) if days else None

    def log_snapshot(
        self, run_date: str, portfolio_value: float, cash: float, positions: dict[str, Position]
    ) -> None:
        """Store a snapshot (positions deep-copied) per run date, replacing any earlier one."""
        self._snapshots[_iso(run_date)] = {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "positions": copy.deepcopy(positions),
        }

    def get_latest_snapshot_positions(self) -> dict | None:
        """Return a deep copy of the latest snapshot's positions, or None when there is none."""
        days = sorted(self._snapshots)
        return copy.deepcopy(self._snapshots[days[-1]]["positions"]) if days else None

    def get_latest_snapshot_run_date(self) -> str | None:
        """Return the latest snapshot's run date as an ISO string, or None when there is none."""
        days = sorted(self._snapshots)
        return days[-1] if days else None

    def get_peak_value(self) -> float | None:
        """Return the stored portfolio peak, or None when none is stored."""
        return self._peak

    def save_peak_value(self, peak_value: float) -> None:
        """Store the portfolio peak as a float, replacing any earlier one."""
        self._peak = float(peak_value)

    def get_rebalance_state(self) -> dict | None:
        """Return a deep copy of the stored rebalance state, or None when none is stored."""
        return copy.deepcopy(self._rebalance)

    def save_rebalance_state(
        self, rebalance_date: str, weights: dict[str, float], scale: float
    ) -> None:
        """Replace all earlier rebalance state with the date (as an ISO string), weights and scale.

        The weights are stored as a copy.
        """
        self._rebalance = {
            "rebalance_date": _iso(rebalance_date),
            "weights": copy.deepcopy(weights),
            "scale": float(scale),
        }

    def get_snapshot_history(self) -> list[tuple]:
        """Return every snapshot as ``(run_date, positions)``, oldest first.

        The run date is a ``date`` and the positions are deep copies.
        """
        return [
            (date.fromisoformat(day), copy.deepcopy(self._snapshots[day]["positions"]))
            for day in sorted(self._snapshots)
        ]

    # -- trade history and dividends -----------------------------------------------------

    def get_trade_history_dates(self, symbol: str) -> list:
        """Return the run dates of the trades in ``symbol``, oldest first."""
        return sorted(_as_date(t["run_date"]) for t in self.trades if t["symbol"] == symbol)

    def get_last_buy_dates(self) -> dict[str, date]:
        """Return the run date of the latest BUY trade of each symbol."""
        latest: dict[str, date] = {}
        for trade in self.trades:
            if trade["side"] == "BUY":
                day = _as_date(trade["run_date"])
                latest[trade["symbol"]] = max(day, latest.get(trade["symbol"], day))
        return latest

    def get_accrued_dividend_keys(self) -> set[tuple]:
        """Return the ``(symbol, ex_date)`` keys of the stored dividend accruals."""
        return {(row["symbol"], row["ex_date"]) for row in self._dividends.values()}

    def log_dividend_accrual(
        self, symbol: str, ex_date: date | str, rate: float, qty: float, amount: float
    ) -> None:
        """Store a dividend accrual per symbol and ex-date; a repeat of the pair is ignored."""
        self._dividends.setdefault(
            (symbol, _iso(ex_date)),
            {"symbol": symbol, "ex_date": ex_date, "rate": rate, "qty": qty, "amount": amount},
        )

    def get_total_dividend_accrual(self) -> float:
        """Return the sum of the stored dividend accrual amounts as a float."""
        return float(sum(row["amount"] for row in self._dividends.values()))

    # -- split corrections ---------------------------------------------------------------

    def get_open_split_corrections(self) -> dict[str, float]:
        """Return the stored split correction factor of each symbol."""
        return {symbol: float(row["factor"]) for symbol, row in self._splits.items()}

    def get_open_split_states(self) -> dict[str, dict]:
        """Return each symbol's stored split correction.

        The result maps each symbol to a dict with a float ``factor`` and a date ``ex_date``.
        """
        return {
            symbol: {"factor": float(row["factor"]), "ex_date": _as_date(row["ex_date"])}
            for symbol, row in self._splits.items()
        }

    def save_split_correction(self, symbol: str, ex_date: str, factor: float) -> None:
        """Store a split correction for ``symbol``.

        A repeat for the same symbol updates the factor and keeps the stored ex-date.
        """
        # Like the SQL upsert, a repeat for a symbol refreshes the factor and keeps the ex-date.
        if symbol in self._splits:
            self._splits[symbol]["factor"] = factor
        else:
            self._splits[symbol] = {"ex_date": ex_date, "factor": factor}

    def clear_split_correction(self, symbol: str) -> None:
        """Delete the stored split correction of ``symbol``; none stored does nothing."""
        self._splits.pop(symbol, None)
