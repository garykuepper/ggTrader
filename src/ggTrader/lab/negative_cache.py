"""Negative cache: skip re-fetching symbols that repeatedly return no data.

Permanently-delisted tickers (e.g. TWTR, SIVB, ATVI) return empty from every
data provider, yet are retried on every run — the Tiingo fallback alone wastes
~12 min per research run on them. This records such symbols in the DB with a
timestamp and skips re-fetching them for a TTL window.

Opt-in only (``use_negative_cache=True``): the live paper trader must always
try every symbol, so a transient provider outage can never silently drop an
active symbol from trading. Only the research/lab CLI enables the cache.
"""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import text

from ggTrader.lab.persist import get_engine

DEFAULT_TTL_DAYS = 14

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv_no_data (
    symbol     text        NOT NULL,
    interval   text        NOT NULL,
    checked_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, interval)
)
"""


def ensure_schema() -> None:
    """Create the negative-cache table if it does not exist."""
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))


def load_skip_symbols(interval: str, ttl_days: int = DEFAULT_TTL_DAYS) -> set[str]:
    """Return symbols recorded as no-data within the TTL window — safe to skip.

    Entries older than ``ttl_days`` are ignored so a symbol that re-lists (or a
    past transient failure) is retried again after the window elapses.
    """
    ensure_schema()
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol FROM ohlcv_no_data "
                "WHERE interval = :iv "
                "AND checked_at >= now() - make_interval(days => :ttl)"
            ),
            {"iv": interval, "ttl": ttl_days},
        ).fetchall()
    return {r[0] for r in rows}


def record_no_data(symbols: Iterable[str], interval: str) -> None:
    """Upsert symbols that returned no data from any provider (refreshes checked_at)."""
    syms = sorted(set(symbols))
    if not syms:
        return
    ensure_schema()
    with get_engine().begin() as conn:
        for sym in syms:
            conn.execute(
                text(
                    "INSERT INTO ohlcv_no_data (symbol, interval, checked_at) "
                    "VALUES (:s, :iv, now()) "
                    "ON CONFLICT (symbol, interval) DO UPDATE SET checked_at = now()"
                ),
                {"s": sym, "iv": interval},
            )
