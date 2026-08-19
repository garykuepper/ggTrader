"""Compute corrections for stock splits the broker failed to apply to the account.

Alpaca's paper-trading environment has been observed marking a position to
the post-split market price while leaving its qty and avg_entry_price at the
pre-split values (seen on MNST's 2026-08-11 2-for-1 split). Left unhandled,
that silently corrupts reported market_value/unrealized_pl for the position
and any aggregate P&L built from it.

Alpaca's own API already has authoritative split data and can prove whether
a known split was actually applied to *this* account:

- Corporate actions (`AlpacaBroker.get_split_corrections`'s corp-actions
  query) tell us a split happened for a symbol, broker-wide.
- Account SPLIT activities tell us whether *this account* ever booked it.

A held symbol with a corp-action split and no matching account SPLIT
activity is unapplied -- that cross-reference is what this module computes
from (near-zero false positives, unlike inferring it from qty deltas, which
also can't distinguish "broker didn't apply it" from "position was trimmed").
"""

from __future__ import annotations

from datetime import date


def compute_split_corrections(
    corp_splits: dict[str, list[tuple[date, float]]],
    applied_symbols: set[str],
) -> dict[str, float]:
    """Return `{symbol: correction_factor}` for symbols with an unapplied split.

    `corp_splits` maps symbol -> list of `(ex_date, factor)` events reported
    by the broker's corporate-actions feed (already filtered to the lookback
    window and to currently-held symbols by the caller), where `factor` is
    `new_rate / old_rate` (2.0 for a 2-for-1 forward split, 0.1 for a
    1-for-10 reverse split). `applied_symbols` is the set of symbols with a
    matching account SPLIT activity in the same window -- for those, the
    broker already corrected the account, so they are never flagged. This is
    the common case, and deliberately not re-flagged: a qty-delta heuristic
    used to falsely re-flag a correctly-adjusted split for days afterward.

    Multiple splits for the same symbol in the window multiply together.
    """
    corrections: dict[str, float] = {}
    for symbol, events in corp_splits.items():
        if symbol in applied_symbols or not events:
            continue
        factor = 1.0
        for _ex_date, ratio in events:
            factor *= ratio
        if factor != 1.0:
            corrections[symbol] = factor
    return corrections


def corrected_market_value(market_value: float, factor: float) -> float:
    """True market_value once `factor` (new_rate/old_rate) is applied."""
    return market_value * factor


def corrected_unrealized_pl(market_value: float, factor: float, cost_basis: float) -> float:
    """True unrealized P&L: corrected market_value minus (unaffected) cost_basis."""
    return corrected_market_value(market_value, factor) - cost_basis


def apply_corrections_to_positions(
    positions: dict[str, dict], corrections: dict[str, float]
) -> dict[str, dict]:
    """Return a copy of `positions` with market_value/unrealized_pl/unrealized_plpc
    corrected for symbols with a known `corrections` factor.

    `qty` is deliberately left untouched -- it is what the broker will
    actually let you transact (see `AlpacaBroker.submit_sell`), and must
    stay the broker's real, uncorrected number. `cost_basis` is also left
    as-is: a split changes share count and price, not the dollars originally
    paid in, so it is split-invariant. Symbols with no correction factor are
    passed through unchanged (though still copied, not aliased).
    """
    result: dict[str, dict] = {}
    for symbol, info in positions.items():
        factor = corrections.get(symbol)
        if factor is None:
            result[symbol] = info
            continue
        cost_basis = info.get("cost_basis", 0.0)
        market_value = info.get("market_value", 0.0)
        new_pl = corrected_unrealized_pl(market_value, factor, cost_basis)
        new_plpc = (new_pl / cost_basis) if cost_basis else info.get("unrealized_plpc", 0.0)
        result[symbol] = {
            **info,
            "market_value": corrected_market_value(market_value, factor),
            "unrealized_pl": new_pl,
            "unrealized_plpc": new_plpc,
        }
    return result
