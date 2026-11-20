"""Typed records of the paper trader; `TypedDict`s describe mappings that are stored as JSON,
dataclasses describe values the trader builds.

Public names: ``Account``, ``Position``, ``OrderStatus``, ``MarketClock``, ``SleeveSignals`` and
``SignalBundle`` (the mappings that cross the broker and signal boundaries, as
``TypedDict``s so the JSON snapshots and the modules that take plain mappings are unchanged),
and ``SubmittedOrder`` and ``RunResult`` (the values the trader builds).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from uuid import UUID as UUID

if TYPE_CHECKING:
    from ggTrader.paper.config import TraderConfig


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


class IntentKind(StrEnum):
    """Order categories."""

    STOP = "stop"
    EXIT = "exit"
    SWEEP_SELL = "sweep_sell"
    BUY = "buy"
    SWEEP_BUY = "sweep_buy"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """An order proposed by a planner."""

    kind: IntentKind
    side: Literal["BUY", "SELL"]
    symbol: str
    size: float
    reason: str | None
    book_amount: float | None
    sleeve: str | None = None
    unrealized_pct: float | None = None
    stop_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class Skip:
    """An already formatted log message."""

    level: int
    message: str


@dataclass(frozen=True, slots=True)
class InvalidExit:
    """An exit that blocks buys."""

    symbol: str
    error: str


Decision = OrderIntent | Skip | InvalidExit


@dataclass(frozen=True)
class StrategySignals:
    """Ordered buys and sorted unique sells with immutable metadata."""

    buys: tuple[tuple[str, str], ...]
    sells: tuple[str, ...]
    last_exit: Mapping[str, str]
    as_of: str
    weights: Mapping[str, float]
    scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "last_exit", MappingProxyType(dict(self.last_exit)))
        # A non-mapping value stays as the signal source returned it: the code before the
        # phase 5 split failed with AttributeError at the first `weights.items()` (the slot-cap
        # step), after any stop-loss sells, and coercing it here would change that outcome.
        if isinstance(self.weights, Mapping):
            object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


@dataclass
class ExecutionBook:
    """Mutable execution accumulator."""

    buys: list[str] = field(default_factory=list)
    sells: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    orders: list[SubmittedOrder] = field(default_factory=list)
    filled: dict[str, OrderStatus] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskVerdict:
    """Hard stop and ordered buy halts."""

    hard_stop: str | None
    buy_halts: tuple[str, ...]

    def with_halt(self, reason: str) -> RiskVerdict:
        """Return a verdict with an additional halt."""
        return RiskVerdict(self.hard_stop, (*self.buy_halts, reason))


@dataclass(frozen=True)
class RunContext:
    """Settings and date for one run."""

    today_et: date
    config: TraderConfig
    pending_symbols: frozenset[str]


@dataclass
class FlowState:
    """Interim order flow state."""

    strategy_positions: dict[str, Position] = field(default_factory=dict)
    stopped: set[str] = field(default_factory=set)
    invalid_exit_symbols: set[str] = field(default_factory=set)
    all_buys: list[tuple[str, str]] = field(default_factory=list)
    loss_halts: list[str] = field(default_factory=list)
    slot_caps: dict[str, int] = field(default_factory=dict)
    sleeve_open_initial: dict[str, int] = field(default_factory=dict)
    buys_by_sleeve: dict[str, list[str]] = field(default_factory=dict)
    sleeve_notional_by_universe: dict[str, float] = field(default_factory=dict)
    book: ExecutionBook = field(default_factory=ExecutionBook)
    regular_sell_count: int = 0
