"""Pure signal and exit planning for the paper trader."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Protocol

from ggTrader.paper import catastrophe_stop
from ggTrader.paper.models import (
    Decision,
    IntentKind,
    InvalidExit,
    OrderIntent,
    Position,
    SignalBundle,
    Skip,
    StrategySignals,
)


class Planner(Protocol):
    """A stage that produces one decision at a time."""

    def advance(self) -> Decision | None:
        """Return the next decision or None once exhausted."""
        ...


def strategy_signals(bundle: SignalBundle) -> StrategySignals:
    """Collect signals in sleeve order."""
    weights, scale = bundle["weights"], bundle["scale"]
    buys = []
    sells = []
    last_exit = {}
    for universe, sleeve in bundle["sleeves"].items():
        for symbol in sleeve["buys"]:
            buys.append((symbol, universe))
        sells.extend(sleeve["sells"])
        last_exit.update(sleeve.get("last_exit", {}))
    as_of = next(iter(bundle["sleeves"].values()))["as_of"]
    return StrategySignals(tuple(buys), tuple(sorted(set(sells))), last_exit, as_of, weights, scale)


def assess_stops(corrected_positions: Mapping[str, Position]) -> frozenset[str]:
    """Find positions whose catastrophe stop cannot be assessed."""
    unassessable = set()
    for symbol, position in corrected_positions.items():
        try:
            if catastrophe_stop.unrealized_pct(position) is None:
                unassessable.add(symbol)
        except ValueError:
            unassessable.add(symbol)
    return frozenset(unassessable)


def _invalid_exit(symbol: str) -> InvalidExit:
    return InvalidExit(
        symbol,
        f"Cannot submit risk-reducing SELL {symbol}: invalid long quantity; "
        "new strategy buys blocked",
    )


def _invalid_quantity(qty: float) -> bool:
    return qty is None or not math.isfinite(qty) or qty <= 0


class StopPlanner:
    """Produce catastrophe-stop decisions on demand."""

    def __init__(
        self, assessable, strategy_positions, corrected_positions, threshold, pending_symbols
    ):
        self.assessable = assessable
        self.strategy_positions = strategy_positions
        self.corrected_positions = corrected_positions
        self.threshold = threshold
        self.pending_symbols = pending_symbols
        self.cursor = None

    def advance(self) -> Decision | None:
        """Produce the next stop decision."""
        if self.cursor is None:
            self.cursor = iter(
                catastrophe_stop.find_catastrophe_stops(self.assessable, self.threshold)
            )
        symbol = next(self.cursor, None)
        if symbol is None:
            return None
        qty = self.strategy_positions[symbol]["qty"]
        if _invalid_quantity(qty):
            return _invalid_exit(symbol)
        if symbol in self.pending_symbols:
            return Skip(logging.INFO, f"Skipping catastrophe stop {symbol}: order already pending")
        pct = catastrophe_stop.unrealized_pct(self.corrected_positions[symbol])
        return OrderIntent(
            IntentKind.STOP,
            "SELL",
            symbol,
            qty,
            catastrophe_stop.CATASTROPHE_STOP_REASON,
            None,
            unrealized_pct=pct,
            stop_threshold=self.threshold,
        )


def plan_stops(
    *,
    assessable: Mapping[str, Position],
    strategy_positions: Mapping[str, Position],
    corrected_positions: Mapping[str, Position],
    threshold: float,
    pending_symbols: frozenset[str],
) -> StopPlanner:
    """Construct a lazy catastrophe-stop planner."""
    return StopPlanner(
        assessable, strategy_positions, corrected_positions, threshold, pending_symbols
    )


class ExitValidationPlanner:
    """Validate requested exits before funding calculations."""

    def __init__(
        self,
        sells: Iterable[str],
        strategy_positions: Mapping[str, Position],
        invalid_symbols: frozenset[str],
    ) -> None:
        self.cursor = iter(sells)
        self.strategy_positions = strategy_positions
        self.invalid_symbols = set(invalid_symbols)

    def advance(self) -> Decision | None:
        """Return the next invalid exit, if any."""
        while (symbol := next(self.cursor, None)) is not None:
            if symbol not in self.strategy_positions or symbol in self.invalid_symbols:
                continue
            qty = self.strategy_positions[symbol]["qty"]
            if _invalid_quantity(qty):
                self.invalid_symbols.add(symbol)
                return _invalid_exit(symbol)
        return None


def validate_exits(
    *,
    sells: Iterable[str],
    strategy_positions: Mapping[str, Position],
    invalid_symbols: frozenset[str],
) -> ExitValidationPlanner:
    """Construct an exit validation planner."""
    return ExitValidationPlanner(sells, strategy_positions, invalid_symbols)


class ExitPlanner:
    """Produce regular exit decisions on demand."""

    def __init__(
        self,
        sells: Iterable[str],
        strategy_positions: Mapping[str, Position],
        invalid_symbols: frozenset[str],
        pending_symbols: frozenset[str],
    ) -> None:
        self.cursor = iter(sells)
        self.strategy_positions = strategy_positions
        self.invalid_symbols = invalid_symbols
        self.pending_symbols = pending_symbols

    def advance(self) -> Decision | None:
        """Return the next regular exit decision."""
        while (symbol := next(self.cursor, None)) is not None:
            if symbol not in self.strategy_positions or symbol in self.invalid_symbols:
                continue
            if symbol in self.pending_symbols:
                return Skip(logging.INFO, f"Skipping SELL {symbol}: order already pending")
            qty = self.strategy_positions[symbol]["qty"]
            return OrderIntent(IntentKind.EXIT, "SELL", symbol, qty, None, None)
        return None


def plan_exits(
    *,
    sells: Iterable[str],
    strategy_positions: Mapping[str, Position],
    invalid_symbols: frozenset[str],
    pending_symbols: frozenset[str],
) -> ExitPlanner:
    """Construct a regular exit planner."""
    return ExitPlanner(sells, strategy_positions, invalid_symbols, pending_symbols)
