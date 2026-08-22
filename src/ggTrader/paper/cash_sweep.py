"""Idle-cash sweep: deploy undeployed cash into a passive index ETF.

The ensemble strategy typically leaves ~60% of the paper account in idle
cash (it only ever holds as many positions as it has signals for), which
drags total return relative to simply holding the S&P 500 with the same
capital. This module computes the sizing math for a cash sweep that buys a
passive ETF (default SPY) with cash left over after the strategy's own
orders for the day, holding a small reserve buffer, and sells the sweep
position down first if the strategy's buys need more cash than is on hand.

Feature-flagged via `CASH_SWEEP_ENABLED` (env var), **default OFF** -- the
paper trader is a live-money production path (see paper/AGENTS.md-adjacent
docs), so this must not change behavior until explicitly enabled. All the
sizing functions here are pure and side-effect free; `trader.py` is
responsible for calling the broker and persisting the resulting trades.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Reason tag written to `paper_trades.reason` / `paper_pending_orders.reason`
#: for sweep-originated orders, so they can be distinguished from strategy
#: trades in the ledger (e.g. for PnL attribution) without a schema that
#: otherwise has no notion of "why" a trade happened.
SWEEP_TRADE_REASON = "cash_sweep"

_ENABLED_ENV_VAR = "CASH_SWEEP_ENABLED"
_SYMBOL_ENV_VAR = "SWEEP_SYMBOL"
_RESERVE_PCT_ENV_VAR = "SWEEP_CASH_RESERVE_PCT"
_MIN_CLIP_ENV_VAR = "SWEEP_MIN_CLIP_USD"

DEFAULT_SWEEP_SYMBOL = "SPY"
DEFAULT_RESERVE_PCT = 0.05
DEFAULT_MIN_CLIP_USD = 500.0


def sweep_enabled() -> bool:
    """Whether the cash sweep is turned on. Default OFF."""
    return os.environ.get(_ENABLED_ENV_VAR, "false").strip().lower() in ("1", "true", "yes", "on")


def sweep_symbol() -> str:
    """The passive ETF symbol the sweep buys/sells. Default SPY."""
    return os.environ.get(_SYMBOL_ENV_VAR, DEFAULT_SWEEP_SYMBOL).strip().upper()


def reserve_pct() -> float:
    """Fraction of portfolio value kept as a cash buffer, never swept."""
    raw = os.environ.get(_RESERVE_PCT_ENV_VAR, "").strip()
    return float(raw) if raw else DEFAULT_RESERVE_PCT


def min_clip_usd() -> float:
    """Minimum sweep-buy notional; below this, skip the trade (avoids
    submitting tiny/noise orders for a few dollars of leftover cash)."""
    raw = os.environ.get(_MIN_CLIP_ENV_VAR, "").strip()
    return float(raw) if raw else DEFAULT_MIN_CLIP_USD


@dataclass(frozen=True)
class SweepAction:
    """A sizing decision: `side` is "buy", "sell", or None (no action)."""

    side: str | None
    notional: float


def compute_sweep_buy(
    cash_after_strategy_orders: float,
    portfolio_value: float,
    reserve_pct: float = DEFAULT_RESERVE_PCT,
    min_clip: float = DEFAULT_MIN_CLIP_USD,
) -> SweepAction:
    """Size a sweep BUY from cash left over after the day's strategy orders.

    `sweep_target = max(0, cash_after_strategy_orders - reserve)` where
    `reserve = reserve_pct * portfolio_value`. Below `min_clip`, no trade.
    """
    if portfolio_value <= 0:
        return SweepAction(None, 0.0)
    reserve = reserve_pct * portfolio_value
    sweep_target = max(0.0, cash_after_strategy_orders - reserve)
    if sweep_target < min_clip:
        return SweepAction(None, 0.0)
    return SweepAction("buy", round(sweep_target, 2))


def compute_sweep_sell_for_funding(
    cash_available: float,
    portfolio_value: float,
    prospective_buy_notional: float,
    current_sweep_position_value: float,
    reserve_pct: float = DEFAULT_RESERVE_PCT,
) -> SweepAction:
    """Size a sweep SELL to free cash for strategy buys that need more than
    is currently on hand.

    The strategy's buys, plus the untouchable reserve, must fit within
    `cash_available`; any shortfall is funded by selling down the existing
    sweep position (capped at what's actually held -- never oversells).
    Returns `SweepAction(None, 0.0)` when there's no shortfall or nothing
    to sell.
    """
    if portfolio_value <= 0 or current_sweep_position_value <= 0:
        return SweepAction(None, 0.0)
    reserve = reserve_pct * portfolio_value
    shortfall = (prospective_buy_notional + reserve) - cash_available
    if shortfall <= 0:
        return SweepAction(None, 0.0)
    sell_notional = min(shortfall, current_sweep_position_value)
    if sell_notional <= 0:
        return SweepAction(None, 0.0)
    return SweepAction("sell", round(sell_notional, 2))


def estimate_prospective_buy_notional(
    buys_by_sleeve: dict[str, list[str]],
    sleeve_notional: dict[str, float],
    slot_caps: dict[str, int],
    slots_available: int,
) -> float:
    """Upper-bound estimate of how much cash the day's strategy buys could
    consume, used to decide whether the sweep position must be sold down
    *before* those buys are submitted.

    This intentionally over-estimates rather than under-estimates: it
    ignores concentration-limit rejections (which only ever reduce actual
    spend) and applies sleeve/global slot caps independently rather than
    modeling their exact interaction, so it may free slightly more cash
    than strictly necessary. That is the safe direction for a funding
    check -- a shortfall would block a strategy buy outright, while a
    surplus just gets swept back in as a sweep buy at the end of the run.
    """
    total = 0.0
    remaining_slots = slots_available
    for universe, symbols in buys_by_sleeve.items():
        if remaining_slots <= 0:
            break
        cap = min(slot_caps.get(universe, 0), remaining_slots)
        n = min(len(symbols), cap)
        if n <= 0:
            continue
        total += sleeve_notional.get(universe, 0.0) * n
        remaining_slots -= n
    return total
