"""Compute cash-dividend accruals for dividends the broker never credited.

Verified 2026-08-18: the Alpaca PAPER account's entire non-trade activity
log contains only FEE entries (REG/TAF/CAT) -- `/v2/account/activities/DIV`
returns `[]` -- while Alpaca's own corporate-actions feed lists 21 cash
dividends on held symbols since 2026-05-15 that were never credited. This
mirrors the split-check design (`split_check.py` /
`AlpacaBroker.get_split_corrections`): cross-reference the broker's
authoritative corporate-actions feed against what the account itself has
booked, and correct only *reporting* -- never the tradable cash balance.

Unlike a split (a point-in-time multiplicative correction to a currently
held position), a dividend is a cash event tied to a specific date: the
amount owed is `rate * qty_held_on_ex_date`, which is *not* necessarily
today's qty (a position can be trimmed or closed after earning a
dividend, and still be owed it). That means:

- Accruals must be persisted per `(symbol, ex_date)` (see
  `persist.log_dividend_accrual` / `paper_dividend_accruals`) so a sold
  position still gets credit and a later run never double-credits the
  same event.
- Establishing "held on ex_date" requires historical holding data, not
  the broker's current position snapshot. This module reads that from
  `paper_snapshots` history (via `qty_held_at_ex_date`) rather than
  guessing -- a dividend that can't be attributed to a known holding is
  skipped and reported, never approximated.
"""

from __future__ import annotations

from datetime import date

# First date backfilling covers (see the module docstring / task report):
# the audit found the DIV activity gap started showing up around here.
DIVIDEND_BACKFILL_START = date(2026, 5, 15)


def normalize_since(since: date | str) -> date:
    """Coerce `since` to a `date`, accepting an ISO date string.

    Deliberately validates loudly (raises) on a bad type instead of
    quietly producing "nothing since" -- the split-check landmine: passing
    a `str` where `get_split_corrections` expects `date` gets swallowed by
    its fail-soft `except Exception` and silently returns `{}`, which reads
    identically to "the API found no corrections". Call this *before* any
    fail-soft try/except, not inside one.
    """
    if isinstance(since, date):
        return since
    if isinstance(since, str):
        return date.fromisoformat(since)
    raise TypeError(f"since must be a date or ISO date string, got {type(since)!r}")


def qty_held_at_ex_date(
    symbol: str, ex_date: date, snapshot_history: list[tuple[date, dict]]
) -> float | None:
    """Best-effort qty of `symbol` held as of `ex_date`, from persisted
    daily snapshots.

    `snapshot_history` is `[(run_date, positions_dict), ...]` ascending by
    `run_date` (see `persist.get_snapshot_history`). Returns the qty from
    the most recent snapshot at or before `ex_date`, or `None` if no such
    snapshot exists or the symbol wasn't held there (covers "bought after
    the ex-date" -- no snapshot at/before it will show the position, so
    this naturally returns `None` rather than a guess).
    """
    qty: float | None = None
    for run_date, positions in snapshot_history:
        if run_date > ex_date:
            break
        info = positions.get(symbol)
        qty = float(info["qty"]) if info and info.get("qty") else None
    return qty if qty else None


def compute_dividend_accruals(
    corp_dividends: dict[str, list[tuple[date, float]]],
    credited_keys: set[tuple[str, date]],
    already_accrued_keys: set[tuple[str, date]],
    snapshot_history: list[tuple[date, dict]],
) -> tuple[list[dict], list[dict]]:
    """Return `(accruals, skipped)` for cash-dividend events the broker
    knows about (`corp_dividends`, from the corporate-actions feed) but
    this account hasn't already resolved.

    `corp_dividends` maps symbol -> `[(ex_date, rate), ...]`.
    `credited_keys` is `(symbol, ex_date)` pairs the broker *has* already
    credited via a real DIV account activity -- these are never
    re-accrued (the case that would double-count if Alpaca's paper
    environment starts crediting dividends). `already_accrued_keys` is
    `(symbol, ex_date)` pairs already persisted in `paper_dividend_accruals`
    from a prior run -- idempotency.

    Each accrual dict is `{symbol, ex_date, rate, qty, amount}`.
    Each skip dict is `{symbol, ex_date, rate, reason}`, reason one of
    `"already_credited_by_broker"`, `"already_accrued"`,
    `"not_held_on_ex_date"`.
    """
    accruals: list[dict] = []
    skipped: list[dict] = []
    for symbol, events in corp_dividends.items():
        for ex_date, rate in events:
            key = (symbol, ex_date)
            if key in credited_keys:
                skipped.append(
                    {
                        "symbol": symbol,
                        "ex_date": ex_date,
                        "rate": rate,
                        "reason": "already_credited_by_broker",
                    }
                )
                continue
            if key in already_accrued_keys:
                skipped.append(
                    {
                        "symbol": symbol,
                        "ex_date": ex_date,
                        "rate": rate,
                        "reason": "already_accrued",
                    }
                )
                continue
            qty = qty_held_at_ex_date(symbol, ex_date, snapshot_history)
            if not qty:
                skipped.append(
                    {
                        "symbol": symbol,
                        "ex_date": ex_date,
                        "rate": rate,
                        "reason": "not_held_on_ex_date",
                    }
                )
                continue
            accruals.append(
                {
                    "symbol": symbol,
                    "ex_date": ex_date,
                    "rate": rate,
                    "qty": qty,
                    "amount": rate * qty,
                }
            )
    return accruals, skipped
