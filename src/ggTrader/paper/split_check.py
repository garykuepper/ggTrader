"""Compute corrections for stock splits the broker failed to apply to the account.

Alpaca's paper-trading environment has been observed marking a position to
the post-split market price while leaving its qty and avg_entry_price at the
pre-split values (seen on MNST's 2026-08-11 2-for-1 split). Left unhandled,
that silently corrupts reported market_value/unrealized_pl for the position
and any aggregate P&L built from it.

Alpaca's own API has authoritative corp-action data, and originally this
module treated the account's own `/v2/account/activities/SPLIT` feed as
proof a split had (or hadn't) been booked. That feed is a no-op in paper:
it returns `[]` even for MNST's real 2026-08-11 split, so the "already
applied?" guard built on it has never once exercised its protective
branch -- if Alpaca ever does apply a split, the old logic would double
it (see docs/next_steps.md, "The one real defect this audit did surface").

`find_split_applied_symbols` replaces the activities feed as *primary*
evidence with `paper_snapshots` history: compare a held symbol's qty on
the last snapshot before its split's ex-date against the first snapshot
on/after it. A jump of roughly the split factor, with no `paper_trades`
entry in that window to explain it another way, means the broker applied
the split; anything else (no qty jump, or a trade that could explain the
change) means it wasn't, or is inconclusive and falls back to correcting
as before. The account SPLIT activities feed (`compute_split_corrections`
below) is kept only as a secondary signal for when snapshot evidence is
unavailable.
"""

from __future__ import annotations

from datetime import date

#: Fractional tolerance for "qty changed by roughly the split factor",
#: to absorb fractional-share rounding noise rather than requiring an
#: exact match.
_SPLIT_QTY_TOLERANCE = 0.05


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


def _qty_around_ex_date(
    symbol: str, ex_date: date, snapshot_history: list[tuple[date, dict]]
) -> tuple[date | None, float | None, date | None, float | None]:
    """Return `(before_date, qty_before, after_date, qty_after)` bracketing
    `ex_date`: the last snapshot strictly before it, and the first snapshot
    on/after it.

    `snapshot_history` is `[(run_date, positions_dict), ...]` ascending by
    `run_date` (see `persist.get_snapshot_history`). Any of the four
    elements is `None` if no qualifying snapshot exists -- e.g. the split
    happened before this account's snapshot history begins, or hasn't
    reached the "after" side yet.
    """
    before_date = qty_before = after_date = qty_after = None
    for run_date, positions in snapshot_history:
        info = positions.get(symbol)
        qty = float(info["qty"]) if info else None
        if run_date < ex_date:
            before_date, qty_before = run_date, qty
        elif after_date is None:
            after_date, qty_after = run_date, qty
    return before_date, qty_before, after_date, qty_after


def find_split_applied_symbols(
    corp_splits: dict[str, list[tuple[date, float]]],
    snapshot_history: list[tuple[date, dict]],
    trade_dates_by_symbol: dict[str, list[date]],
    today: date,
    tolerance: float = _SPLIT_QTY_TOLERANCE,
) -> tuple[set[str], set[str]]:
    """Return `(applied, unresolved)` using `paper_snapshots` history as the
    primary evidence for whether a broker-known split was actually applied
    to this account -- see the module docstring.

    Only events whose `ex_date` is `<= today` are considered (a future
    split can't have been applied yet). For each symbol, the last snapshot
    before the (earliest, if several) ex-date is compared against the
    first snapshot on/after it:

    - No qualifying snapshot on either side -> no evidence either way; the
      symbol goes in `unresolved` so the caller falls back to its prior
      behavior (correct + log a warning) instead of guessing.
    - A `paper_trades` entry for the symbol falls inside that window ->
      the qty change could be an ordinary trim/exit, not a split, so it is
      never inferred as "applied" from qty alone (also not `unresolved`:
      there IS a before/after snapshot, just an inconclusive one -- callers
      should keep correcting, same as an unapplied split).
    - Otherwise, qty changing by ~`factor` (within `tolerance`, relative)
      means the broker applied the split -> added to `applied`.

    `trade_dates_by_symbol` is `{symbol: [run_date, ...]}` (see
    `persist.get_trade_history_dates`), only needed for symbols actually
    being checked here.
    """
    applied: set[str] = set()
    unresolved: set[str] = set()
    for symbol, events in corp_splits.items():
        past_events = [(ex_date, factor) for ex_date, factor in events if ex_date <= today]
        if not past_events:
            continue
        ex_date = min(ex for ex, _ in past_events)
        factor = 1.0
        for _ex, ratio in past_events:
            factor *= ratio
        if factor == 1.0:
            continue

        before_date, qty_before, after_date, qty_after = _qty_around_ex_date(
            symbol, ex_date, snapshot_history
        )
        if before_date is None or after_date is None or qty_before is None or qty_after is None:
            unresolved.add(symbol)
            continue

        trade_dates = trade_dates_by_symbol.get(symbol, [])
        traded_in_window = any(before_date < d <= after_date for d in trade_dates)
        if traded_in_window or qty_before == 0:
            continue

        ratio = qty_after / qty_before
        if abs(ratio - factor) / abs(factor) <= tolerance:
            applied.add(symbol)
    return applied, unresolved


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
