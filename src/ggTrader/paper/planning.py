"""Pure signal and exit planning for the paper trader."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Protocol

from ggTrader.paper import cash_sweep, catastrophe_stop
from ggTrader.paper.config import TraderConfig
from ggTrader.paper.models import (
    Account,
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
        self,
        assessable: Mapping[str, Position],
        strategy_positions: Mapping[str, Position],
        corrected_positions: Mapping[str, Position],
        threshold: float,
        pending_symbols: frozenset[str],
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


class RiskLimits(Protocol):
    """Risk sizing and concentration checks used by planning."""

    def max_new_positions(self, current_count: int) -> int: ...

    def sleeve_slot_caps(self, weights: dict[str, float]) -> dict[str, int]: ...

    def sleeve_position_notional(
        self, portfolio_value: float, sleeve_weight: float, scale: float
    ) -> float: ...

    def check_concentration(
        self,
        symbol: str,
        positions: dict[str, dict],
        portfolio_value: float,
        prospective_notional: float = 0.0,
    ) -> bool: ...


def plan_sweep_sell(
    *,
    cfg: TraderConfig,
    account: Account,
    positions: Mapping[str, Position],
    strategy_positions: Mapping[str, Position],
    sells: Iterable[str],
    pending_symbols: frozenset[str],
    risk: RiskLimits,
    buys_by_sleeve: Mapping[str, list[str]],
    notional_by_sleeve: Mapping[str, float],
    slot_caps: Mapping[str, int],
    portfolio_value: float,
) -> OrderIntent | None:
    """Size a sweep sale to fund anticipated buys and restore the reserve."""
    anticipated = [s for s in sells if s in strategy_positions and s not in pending_symbols]
    slots = risk.max_new_positions(len(strategy_positions) - len(anticipated))
    prospective = cash_sweep.estimate_prospective_buy_notional(
        buys_by_sleeve, notional_by_sleeve, slot_caps, slots
    )
    position = positions.get(cfg.sweep_symbol)
    value = position["market_value"] if position else 0.0
    action = cash_sweep.compute_sweep_sell_for_funding(
        cash_available=account["cash"],
        portfolio_value=portfolio_value,
        prospective_buy_notional=prospective,
        current_sweep_position_value=value,
        reserve_pct=cfg.reserve_pct(),
    )
    if action.side == "sell" and position:
        price = position["current_price"]
        qty = round(action.notional / price, 4) if price > 0 else 0.0
        if qty > 0:
            return OrderIntent(
                IntentKind.SWEEP_SELL,
                "SELL",
                cfg.sweep_symbol,
                qty,
                cash_sweep.SWEEP_TRADE_REASON,
                action.notional,
            )
    return None


class BuyPlanner:
    """Produce strategy buys lazily, retaining stage-local slot counts."""

    def __init__(
        self,
        *,
        buys_by_sleeve: Mapping[str, list[str]],
        notional_by_sleeve: Mapping[str, float],
        slot_caps: Mapping[str, int],
        strategy_positions: Mapping[str, Position],
        corrected_positions: Mapping[str, Position],
        stopped: frozenset[str],
        pending_symbols: frozenset[str],
        slots_available: int,
        max_positions: int,
        risk: RiskLimits,
        portfolio_value: float,
        open_by_sleeve: Mapping[str, int],
    ):
        self.sleeves = iter(buys_by_sleeve.items())
        self.symbols = iter(())
        self.notional_by_sleeve = notional_by_sleeve
        self.slot_caps = slot_caps
        self.strategy_positions = strategy_positions
        self.corrected_positions = corrected_positions
        self.stopped = stopped
        self.pending_symbols = pending_symbols
        self.slots_available = slots_available
        self.max_positions = max_positions
        self.risk = risk
        self.portfolio_value = portfolio_value
        self.open_by_sleeve = dict(open_by_sleeve)
        self.buys_attempted = 0

    def advance(self) -> Decision | None:
        """Return one decision, without inspecting later candidates."""
        while True:
            symbol = next(self.symbols, None)
            if symbol is None:
                sleeve = next(self.sleeves, None)
                if sleeve is None:
                    return None
                self.universe, symbols = sleeve
                self.notional = self.notional_by_sleeve[self.universe]
                self.cap = self.slot_caps.get(self.universe, 0)
                self.symbols = iter(symbols)
                continue
            if symbol in self.strategy_positions:
                continue
            if symbol in self.stopped:
                return Skip(logging.INFO, f"Skipping BUY {symbol}: catastrophe-stopped this run")
            if symbol in self.pending_symbols:
                return Skip(logging.INFO, f"Skipping BUY {symbol}: order already pending")
            if self.buys_attempted >= self.slots_available:
                self.symbols = iter(())
                return Skip(
                    logging.INFO, f"Max positions reached ({self.max_positions}), skipping {symbol}"
                )
            if self.open_by_sleeve[self.universe] >= self.cap:
                self.symbols = iter(())
                return Skip(
                    logging.INFO, f"Sleeve {self.universe} slot cap reached, skipping {symbol}"
                )
            if self.risk.check_concentration(
                symbol,
                self.corrected_positions,
                self.portfolio_value,
                prospective_notional=self.notional,
            ):
                return Skip(logging.INFO, f"Concentration limit for {symbol}, skipping")
            self.buys_attempted += 1
            self.open_by_sleeve[self.universe] += 1
            return OrderIntent(
                IntentKind.BUY,
                "BUY",
                symbol,
                self.notional,
                None,
                self.notional,
                sleeve=self.universe,
            )


def plan_buys(
    *,
    buys_by_sleeve: Mapping[str, list[str]],
    notional_by_sleeve: Mapping[str, float],
    slot_caps: Mapping[str, int],
    strategy_positions: Mapping[str, Position],
    corrected_positions: Mapping[str, Position],
    stopped: frozenset[str],
    pending_symbols: frozenset[str],
    slots_available: int,
    max_positions: int,
    risk: RiskLimits,
    portfolio_value: float,
    open_by_sleeve: Mapping[str, int],
) -> BuyPlanner:
    """Construct a lazy strategy-buy planner."""
    return BuyPlanner(
        buys_by_sleeve=buys_by_sleeve,
        notional_by_sleeve=notional_by_sleeve,
        slot_caps=slot_caps,
        strategy_positions=strategy_positions,
        corrected_positions=corrected_positions,
        stopped=stopped,
        pending_symbols=pending_symbols,
        slots_available=slots_available,
        max_positions=max_positions,
        risk=risk,
        portfolio_value=portfolio_value,
        open_by_sleeve=open_by_sleeve,
    )


def plan_sweep_buy(*, cfg: TraderConfig, mid_account: Account) -> OrderIntent | None:
    """Size a sweep purchase from the post-trade account."""
    action = cash_sweep.compute_sweep_buy(
        cash_after_strategy_orders=mid_account["cash"],
        portfolio_value=mid_account["portfolio_value"],
        reserve_pct=cfg.reserve_pct(),
        min_clip=cfg.min_clip_usd(),
        buy_trigger_pct=cfg.buy_trigger_pct(),
    )
    if action.side == "buy":
        return OrderIntent(
            IntentKind.SWEEP_BUY,
            "BUY",
            cfg.sweep_symbol,
            action.notional,
            cash_sweep.SWEEP_TRADE_REASON,
            action.notional,
        )
    return None
