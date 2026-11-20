"""Read and prepare the portfolio for a paper-trading run."""

import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType

from ggTrader.paper.constants import SPLIT_LOOKBACK_DAYS, TRADER_LOGGER_NAME
from ggTrader.paper.models import Account, Position, RunContext, StrategySignals
from ggTrader.paper.ports import Broker, Notifier
from ggTrader.paper.run_helpers import account_read_halt
from ggTrader.paper.seams import TraderSeams
from ggTrader.paper.split_check import apply_corrections_to_positions
from ggTrader.paper.untradable import resolve_untradable

_log = logging.getLogger(TRADER_LOGGER_NAME)


def _positions_copy(positions: Mapping[str, Position]) -> Mapping[str, Position]:
    return MappingProxyType(
        {symbol: MappingProxyType(deepcopy(dict(pos))) for symbol, pos in positions.items()}
    )


@dataclass(frozen=True)
class PortfolioView:
    """A snapshot of broker positions and the strategy's filtered holdings."""

    account: Account
    positions: Mapping[str, Position]
    strategy_positions: Mapping[str, Position]
    corrected_positions: Mapping[str, Position]
    split_corrections: Mapping[str, float]
    sweep_on: bool
    sweep_symbol: str | None
    untradable_seen: frozenset[str]
    buy_candidates: tuple[tuple[str, str], ...]
    sells: tuple[str, ...]
    dividends: Mapping[str, object]

    @property
    def portfolio_value(self) -> float:
        """Return the account's reported portfolio value."""
        return self.account["portfolio_value"]

    @classmethod
    def load(
        cls,
        *,
        broker: Broker,
        notifier: Notifier,
        seams: TraderSeams,
        ctx: RunContext,
        signals: StrategySignals,
    ) -> "PortfolioView":
        """Read the account and prepare strategy positions in trading order."""
        try:
            account = broker.get_account()
        except Exception as exc:
            account_read_halt(notifier, exc)
            raise
        try:
            positions = broker.get_positions()
        except Exception as exc:
            _log.error("Position read failed (%s); stopping before orders", type(exc).__name__)
            try:
                notifier.send("<b>🛑 HALTED:</b> Position data unavailable; no orders submitted")
            except Exception:
                _log.exception("Could not send position read failure notice")
            raise
        account["portfolio_value"]
        cfg = ctx.config
        sweep_on, sweep_sym = cfg.sweep_enabled, cfg.sweep_symbol
        strategy_positions = (
            {sym: pos for sym, pos in positions.items() if sym != sweep_sym}
            if sweep_on
            else positions
        )
        all_buys = list(signals.buys)
        untradable_seen = resolve_untradable(
            broker, set(strategy_positions) | {sym for sym, _ in all_buys}, cfg.untradable
        )
        frozen = untradable_seen & set(strategy_positions)
        if frozen:
            _log.warning("Untradable holdings excluded from strategy: %s", sorted(frozen))
            notifier.send(
                "<b>⚠️ Untradable holdings:</b> "
                f"{', '.join(sorted(frozen))} (broker tradable=false; excluded from "
                "slots and sells — resolve manually)"
            )
            strategy_positions = {s: p for s, p in strategy_positions.items() if s not in frozen}
        buy_symbols = {sym for sym, _ in all_buys}
        blocked_buys = sorted((untradable_seen & buy_symbols) - frozen)
        if blocked_buys:
            _log.warning("Untradable buy candidates skipped: %s", blocked_buys)
            notifier.send(
                f"<b>⚠️ Untradable buy candidates skipped:</b> {', '.join(blocked_buys)} "
                "(asset not tradable at the broker)"
            )
        all_buys = [(sym, u) for sym, u in all_buys if sym not in untradable_seen]
        split_since = ctx.today_et - timedelta(days=SPLIT_LOOKBACK_DAYS)
        split_corrections = seams.split_corrections(positions, split_since, ctx.today_et)
        if split_corrections:
            _log.warning("Unapplied broker split corrections: %s", split_corrections)
        corrected_positions = apply_corrections_to_positions(strategy_positions, split_corrections)
        sells = list(signals.sells)
        missed = seams.missed_exits(strategy_positions, dict(signals.last_exit), set(sells))
        if missed:
            _log.info("Catching up missed exits: %s", missed)
            sells = sorted(set(sells) | set(missed))
        dividends = seams.accrue_dividends(positions)
        return cls(
            MappingProxyType(deepcopy(account)),
            _positions_copy(positions),
            _positions_copy(strategy_positions),
            _positions_copy(corrected_positions),
            MappingProxyType(dict(split_corrections)),
            sweep_on,
            sweep_sym,
            frozenset(untradable_seen),
            tuple(all_buys),
            tuple(sells),
            MappingProxyType(deepcopy(dividends)),
        )
