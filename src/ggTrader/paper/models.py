"""Typed records of the paper trader; `TypedDict`s describe mappings that are stored as JSON,
dataclasses describe values the trader builds.

Public names: ``Account``, ``Position``, ``OrderStatus``, ``MarketClock``, ``SleeveSignals`` and
``SignalBundle`` (the mappings that cross the broker and signal boundaries, as
``TypedDict``s so the JSON snapshots and the modules that take plain mappings are unchanged),
and ``SubmittedOrder`` and ``RunResult`` (the values the trader builds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class Account(TypedDict, total=False):
    """The keys of ``AlpacaBroker.get_account``; ``total=False`` because tests build accounts
    without ``multiplier``."""

    cash: float
    portfolio_value: float
    last_equity: float | None
    buying_power: float
    multiplier: float


class Position(TypedDict, total=False):
    """One entry of ``AlpacaBroker.get_positions``. A ``TypedDict``, not a dataclass: snapshots
    store positions as JSON, and ``split_check``, ``dividend_check``, ``catastrophe_stop`` and the
    studies take plain mappings."""

    qty: float
    market_value: float
    current_price: float
    avg_entry: float
    cost_basis: float
    unrealized_pl: float | None
    unrealized_plpc: float | None
    change_today: float | None


class OrderStatus(TypedDict, total=False):
    """The normalized order of ``AlpacaBroker.get_order``; the fill fields are ``None`` until the
    broker reports them."""

    id: str
    client_order_id: str
    symbol: str
    side: str
    status: str
    qty: float | None
    notional: float | None
    filled_qty: float | None
    filled_avg_price: float | None


class MarketClock(TypedDict, total=False):
    """``AlpacaBroker.get_clock``; ``total=False`` because the trader reads it with ``.get`` and
    fails open."""

    is_open: bool
    next_open: str
    next_close: str


class SleeveSignals(TypedDict, total=False):
    """One sleeve's signals, the return of ``signal_runner.generate_signals``."""

    buys: list[str]
    sells: list[str]
    as_of: str
    universe_size: int
    membership_gate: dict[str, Any]
    last_exit: dict[str, str]
    engine: dict[str, Any]
    bars: dict[str, Any]


class SignalBundle(TypedDict, total=False):
    """The return of ``signal_runner.generate_core_signals`` (the shape the blend also uses)."""

    sleeves: dict[str, SleeveSignals]
    weights: dict[str, float]
    scale: float
    rebalanced_today: bool
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class SubmittedOrder:
    """An order handed to the broker this run, kept until it settles.

    Attributes:
        order_id: The broker's id for the order.
        side: ``"BUY"`` or ``"SELL"``.
        symbol: The ticker.
        amount: Intended notional for a BUY or sweep funding SELL; position market value for
            other SELLs. It is what the alert shows when the order has not filled.
        reason: ``None`` for a strategy order, otherwise the ledger tag:
            ``cash_sweep.SWEEP_TRADE_REASON`` or ``catastrophe_stop.CATASTROPHE_STOP_REASON``.
    """

    order_id: str
    side: Literal["BUY", "SELL"]
    symbol: str
    amount: float
    reason: str | None


@dataclass(frozen=True)
class RunResult:
    """What one trader run returns; `as_dict()` is the mapping `run()` hands its callers.

    Attributes:
        buys: Symbols bought (or planned, in a dry run).
        sells: Symbols sold (or planned).
        errors: Messages of the problems the run met.
        gate: The gate that ended the run early, when one did (``"late_run"``).
        untradable: Symbols skipped as untradable; only the final result carries it.
        skipped_writes: Names of the writes a read-only dry run skipped; only the final result
            carries it.
    """

    buys: list[str]
    sells: list[str]
    errors: list[str]
    gate: str | None = None
    untradable: list[str] | None = None
    skipped_writes: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return today's keys in today's order; a field that is None is absent.

        The lists are the fields themselves, not copies, as the callers have always received them.
        """
        result: dict[str, Any] = {"buys": self.buys, "sells": self.sells, "errors": self.errors}
        if self.gate is not None:
            result["gate"] = self.gate
        if self.untradable is not None:
            result["untradable"] = self.untradable
        if self.skipped_writes is not None:
            result["skipped_writes"] = self.skipped_writes
        return result
