"""Coordinate paper-trading order stages."""

import logging

from ggTrader.paper.constants import TRADER_LOGGER_NAME
from ggTrader.paper.execution import Executor
from ggTrader.paper.models import (
    FlowState,
    InvalidExit,
    RiskVerdict,
    RunContext,
    Skip,
    StrategySignals,
    SubmittedOrder,
)
from ggTrader.paper.planning import (
    assess_stops,
    plan_exits,
    plan_stops,
    plan_sweep_sell,
    validate_exits,
)
from ggTrader.paper.portfolio import PortfolioView
from ggTrader.paper.ports import Broker, Notifier
from ggTrader.paper.risk import RiskGuard

_log = logging.getLogger(TRADER_LOGGER_NAME)


class OrderFlow:
    """Apply order decisions in their original execution order."""

    def __init__(
        self,
        *,
        executor: Executor,
        broker: Broker,
        notifier: Notifier,
        risk: RiskGuard,
        ctx: RunContext,
        stop_threshold: float | None,
    ) -> None:
        self.executor = executor
        self.broker = broker
        self.notifier = notifier
        self.risk = risk
        self.ctx = ctx
        self.stop_threshold = stop_threshold

    def run_stage_one(
        self, view: PortfolioView, signals: StrategySignals, verdict: RiskVerdict
    ) -> FlowState:
        """Execute stops, validate exits, and compute buy limits."""
        state = FlowState(
            strategy_positions=dict(view.strategy_positions),
            all_buys=[] if verdict.buy_halts else list(view.buy_candidates),
            loss_halts=list(verdict.buy_halts),
        )
        if self.stop_threshold is not None:
            unassessable = assess_stops(view.corrected_positions)
            if unassessable:
                reason = "Cannot assess catastrophe stop for: " + ", ".join(sorted(unassessable))
                verdict = verdict.with_halt(reason)
                state.loss_halts.append(reason)
                state.all_buys = []
                _log.warning("%s; new buys and sweeps blocked", reason)
                try:
                    self.notifier.send(f"<b>🛑 HALTED buys; exits only:</b> {reason}")
                except Exception:
                    _log.exception("Could not send unassessable catastrophe-stop alert")
                if self.ctx.pending_symbols:
                    try:
                        self.notifier.send(
                            "<b>⚠️ Pending orders may still fill:</b> "
                            "no cancellations were attempted."
                        )
                    except Exception:
                        _log.exception("Could not send pending-order notice")
            assessable = {
                s: p for s, p in view.corrected_positions.items() if s not in unassessable
            }
            planner = plan_stops(
                assessable=assessable,
                strategy_positions=state.strategy_positions,
                corrected_positions=view.corrected_positions,
                threshold=self.stop_threshold,
                pending_symbols=self.ctx.pending_symbols,
            )
            while (decision := planner.advance()) is not None:
                if isinstance(decision, Skip):
                    _log.log(decision.level, decision.message)
                elif isinstance(decision, InvalidExit):
                    state.book.errors.append(decision.error)
                    state.invalid_exit_symbols.add(decision.symbol)
                else:
                    oid = self.executor.submit(decision)
                    state.stopped.add(decision.symbol)
                    state.book.sells.append(decision.symbol)
                    if self.executor.submits_orders:
                        amount = state.strategy_positions[decision.symbol]["market_value"]
                        state.book.orders.append(
                            SubmittedOrder(oid, "SELL", decision.symbol, amount, decision.reason)
                        )
                    self.executor.announce(decision)
            if state.stopped:
                state.strategy_positions = {
                    s: p for s, p in state.strategy_positions.items() if s not in state.stopped
                }
        validation = validate_exits(
            sells=view.sells,
            strategy_positions=state.strategy_positions,
            invalid_symbols=frozenset(state.invalid_exit_symbols),
        )
        while (decision := validation.advance()) is not None:
            state.book.errors.append(decision.error)
            state.invalid_exit_symbols.add(decision.symbol)
        if state.invalid_exit_symbols:
            state.all_buys = []
        state.slot_caps = self.risk.sleeve_slot_caps(signals.weights)
        state.sleeve_open_initial = {u: 0 for u in signals.weights}
        for symbol, universe in state.all_buys:
            state.buys_by_sleeve.setdefault(universe, []).append(symbol)
        state.sleeve_notional_by_universe = {
            universe: self.risk.sleeve_position_notional(
                view.portfolio_value, signals.weights.get(universe, 0.0), signals.scale
            )
            for universe in state.buys_by_sleeve
        }
        return state

    def run_stage_two(self, view: PortfolioView, state: FlowState) -> FlowState:
        """Execute the funding and strategy sells, then settle them."""
        if view.sweep_on and not state.loss_halts:
            intent = plan_sweep_sell(
                cfg=self.ctx.config,
                account=view.account,
                positions=view.positions,
                strategy_positions=state.strategy_positions,
                sells=view.sells,
                pending_symbols=self.ctx.pending_symbols,
                risk=self.risk,
                buys_by_sleeve=state.buys_by_sleeve,
                notional_by_sleeve=state.sleeve_notional_by_universe,
                slot_caps=state.slot_caps,
                portfolio_value=view.portfolio_value,
            )
            if intent is not None:
                oid = self.executor.submit(intent)
                if self.executor.submits_orders:
                    state.book.orders.append(
                        SubmittedOrder(
                            oid, "SELL", intent.symbol, intent.book_amount, intent.reason
                        )
                    )
                self.executor.announce(intent)
        planner = plan_exits(
            sells=view.sells,
            strategy_positions=state.strategy_positions,
            invalid_symbols=frozenset(state.invalid_exit_symbols),
            pending_symbols=self.ctx.pending_symbols,
        )
        while (decision := planner.advance()) is not None:
            if isinstance(decision, Skip):
                _log.log(decision.level, decision.message)
                continue
            oid = self.executor.submit(decision)
            state.book.sells.append(decision.symbol)
            state.regular_sell_count += 1
            if self.executor.submits_orders:
                amount = state.strategy_positions[decision.symbol]["market_value"]
                state.book.orders.append(SubmittedOrder(oid, "SELL", decision.symbol, amount, None))
            self.executor.announce(decision)
        state.book.filled = self.executor.poll([o for o in state.book.orders if o.side == "SELL"])
        self.executor.after_sells(state.book.filled)
        if self.executor.submits_orders:
            for error in self.executor.unfilled_exit_errors(
                state.book.orders, state.book.filled, view.positions
            ):
                state.book.errors.append(error)
                state.buys_by_sleeve.clear()
        return state
