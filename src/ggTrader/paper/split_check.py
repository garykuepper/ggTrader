"""Detect stock splits the paper broker failed to reflect in a position's qty.

Alpaca's paper-trading environment has been observed marking a position to
the post-split market price while leaving its qty and avg_entry_price at the
pre-split values (seen on MNST's 2026-08-11 2-for-1 split). Left unhandled,
that silently halves reported market_value/unrealized_pl for the position and
corrupts any aggregate P&L built from it — this module flags it instead.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import yfinance as yf

_log = logging.getLogger(__name__)

_QTY_TOLERANCE = 0.01  # relative tolerance for float drift, not trading noise
_DEFAULT_TIMEOUT = 8.0
_MAX_WORKERS = 5


def is_split_unadjusted(qty_before: float, qty_after: float, split_ratio: float) -> bool:
    """True if `qty_after` was not multiplied by `split_ratio` from `qty_before`."""
    expected = qty_before * split_ratio
    if expected == 0:
        return False
    return abs(qty_after - expected) / expected > _QTY_TOLERANCE


def find_unadjusted_split_symbols(
    prev_positions: dict[str, dict],
    curr_positions: dict[str, dict],
    splits: dict[str, list[tuple[date, float]]],
    prev_snapshot_date: date,
    today: date,
) -> list[str]:
    """Return symbols whose broker-reported qty didn't reflect a known split.

    `prev_positions`/`curr_positions` are broker position dicts (as returned
    by `AlpacaBroker.get_positions()`) from before and after the split.
    `splits` maps symbol -> list of (ex_date, factor) events discovered in
    the lookback window (see `get_recent_splits`). A split event only
    applies to this run's gate window when its ex-date falls strictly after
    `prev_snapshot_date` and on/before `today` — this is what prevents a
    correctly-adjusted split from being falsely re-flagged on every run for
    the rest of the lookback window (the day after a correct adjustment,
    `prev_snapshot_date` has already rolled past the ex-date, so the event no
    longer applies and the symbol is compared with an implicit 1.0 ratio,
    i.e. skipped as "no split in this window").

    A symbol not held on both sides of the split is skipped — there's
    nothing to compare, and guessing would risk a false positive.
    """
    flagged = []
    for symbol, events in splits.items():
        if symbol not in prev_positions or symbol not in curr_positions:
            continue
        ratio = 1.0
        for ex_date, factor in events:
            if prev_snapshot_date < ex_date <= today:
                ratio *= factor
        if ratio == 1.0:
            continue  # no applicable split event in this run's gate window
        qty_before = prev_positions[symbol].get("qty", 0.0)
        qty_after = curr_positions[symbol].get("qty", 0.0)
        if is_split_unadjusted(qty_before, qty_after, ratio):
            flagged.append(symbol)
    return flagged


def _fetch_one(symbol: str, since: str) -> list[tuple[date, float]]:
    """Fetch split events for one symbol. Raises on failure (caller handles)."""
    hist = yf.Ticker(symbol).history(start=since)
    events = hist.loc[hist["Stock Splits"] != 0, "Stock Splits"]
    return [(ts.date(), float(factor)) for ts, factor in events.items()]


def get_recent_splits(
    symbols: list[str], since: str, timeout: float = _DEFAULT_TIMEOUT
) -> tuple[dict[str, list[tuple[date, float]]], bool]:
    """Fetch split events (ex-date, ratio) for `symbols` since `since`.

    `since` is an ISO date string (e.g. "2026-08-01"). Symbols with no split
    in the window are omitted from the returned dict. Lookups run in a small
    thread pool with an overall `timeout` budget so one hung/slow symbol
    can't stall the run; a symbol whose lookup errors or doesn't complete in
    time is logged and skipped rather than raised.

    Returns `(splits_by_symbol, all_failed)`. `all_failed` is True only when
    `symbols` is non-empty and every symbol's lookup failed/timed out — the
    caller should treat that as "split check unavailable" rather than
    silently reporting "no splits found", since the two are indistinguishable
    from an empty dict alone.
    """
    splits: dict[str, list[tuple[date, float]]] = {}
    if not symbols:
        return splits, False

    failures = 0
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(symbols))) as pool:
        futures = {pool.submit(_fetch_one, sym, since): sym for sym in symbols}
        try:
            for fut in as_completed(futures, timeout=timeout):
                symbol = futures[fut]
                try:
                    events = fut.result()
                except Exception as exc:
                    _log.warning("Could not check %s for splits (non-fatal): %s", symbol, exc)
                    failures += 1
                    continue
                if events:
                    splits[symbol] = events
        except TimeoutError:
            not_done = [futures[f] for f in futures if not f.done()]
            for symbol in not_done:
                _log.warning("Split check for %s timed out (non-fatal)", symbol)
                failures += 1

    all_failed = failures == len(symbols)
    return splits, all_failed
