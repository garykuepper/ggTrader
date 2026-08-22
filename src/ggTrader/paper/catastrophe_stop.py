"""Catastrophe-stop backstop: force-exit a position the strategy's own RSI
exit never fires on.

The ensemble strategy only ever sells on an RSI-based exit signal. A position
that never produces one can decay indefinitely -- the live example that
motivated this module is NXPI, which sat at roughly -19% unrealized for eight
weeks with no exit signal firing. This is a risk guardrail, not a signal
change: it force-sells a position once its unrealized loss breaches a fixed
floor, independent of what the strategy itself thinks.

Feature-flagged via `CATASTROPHE_STOP_ENABLED` (env var), **default OFF** --
the paper trader is a live-money production path, so this must not change
behavior until explicitly enabled (pending WFO evidence on the threshold).
All the decision logic here is pure and side-effect free; `trader.py` is
responsible for calling the broker, logging the trade, and notifying.
"""

from __future__ import annotations

import os

#: Reason tag written to `paper_trades.reason` / `paper_pending_orders.reason`
#: for a catastrophe-stop sell, mirroring `cash_sweep.SWEEP_TRADE_REASON` --
#: distinguishes a risk-backstop exit from an ordinary strategy sell in the
#: ledger.
CATASTROPHE_STOP_REASON = "catastrophe_stop"

_ENABLED_ENV_VAR = "CATASTROPHE_STOP_ENABLED"
_PCT_ENV_VAR = "CATASTROPHE_STOP_PCT"

#: -25% unrealized. Negative by convention (a loss), so a position is stopped
#: out when its unrealized fraction is <= this value.
DEFAULT_CATASTROPHE_STOP_PCT = -0.25


def catastrophe_stop_enabled() -> bool:
    """Whether the catastrophe-stop backstop is turned on. Default OFF."""
    return os.environ.get(_ENABLED_ENV_VAR, "false").strip().lower() in ("1", "true", "yes", "on")


def catastrophe_stop_pct() -> float:
    """Unrealized-loss floor (negative fraction) that triggers a force-sell."""
    raw = os.environ.get(_PCT_ENV_VAR, "").strip()
    return float(raw) if raw else DEFAULT_CATASTROPHE_STOP_PCT


def unrealized_pct(position: dict) -> float | None:
    """Unrealized P&L as a fraction of `cost_basis`, or `None` if unknown.

    Computed from `cost_basis` rather than trusting the broker's own
    `unrealized_plpc` directly -- `cost_basis` is split-invariant (a split
    changes share count and price, not dollars originally paid in), so
    dividing by it is safe even for a position the broker has not yet
    applied a known split to, PROVIDED `position["unrealized_pl"]` has
    already been split-corrected (see `split_check.apply_corrections_to_positions`,
    whose output this is meant to be called on). `avg_entry_price` is NOT
    split-adjusted on this broker and must never be used for this
    computation instead.
    """
    cost_basis = position.get("cost_basis", 0.0)
    if cost_basis <= 0:
        return None
    return position.get("unrealized_pl", 0.0) / cost_basis


def find_catastrophe_stops(positions: dict[str, dict], threshold_pct: float) -> list[str]:
    """Return symbols whose unrealized loss has breached `threshold_pct`.

    `threshold_pct` is a negative fraction (e.g. -0.25 for -25%); a symbol is
    returned when `unrealized_pct(position) <= threshold_pct`. `positions`
    should be the split-corrected view (see `unrealized_pct`'s docstring) and
    should already exclude any sweep-symbol position -- this function has no
    opinion on either; it just evaluates whatever's handed to it. Sorted for
    deterministic ordering (dict iteration order is otherwise incidental).
    """
    return sorted(
        symbol
        for symbol, position in positions.items()
        if (pct := unrealized_pct(position)) is not None and pct <= threshold_pct
    )
