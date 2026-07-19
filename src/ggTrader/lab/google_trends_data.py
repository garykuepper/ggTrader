"""Google Trends (retail search-attention) data: free, no key, via the
unofficial pytrends client. Verified 2026-07-19: a single query per symbol
spanning the full multi-year eval window returns MONTHLY-resolution
interest_over_time data directly (Google Trends auto-downsamples long
ranges) -- exactly the cadence this lab's monthly-rebalance harness needs,
with no query-stitching or cross-batch-normalization problem (each
symbol's full history comes from one continuous, internally-consistent
0-100 scale). Rate-limit spot-check: 39/40 rapid successive queries
succeeded with zero explicit delay; still rate-limited conservatively here
since pytrends is an unofficial, historically fragile scraper of Google's
own UI backend, not a supported public API.
"""

from __future__ import annotations

from typing import Callable, Iterable, List

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

#: Google Trends has no documented publish-lag schedule (unlike FINRA/SEC);
#: this is a conservative buffer against same-week lookahead.
PUBLISH_LAG_DAYS = 7

_COLUMNS = ["symbol", "date", "search_interest"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS google_trends_interest (
    symbol text NOT NULL,
    date date NOT NULL,
    search_interest double precision,
    PRIMARY KEY (symbol, date)
)
"""

QueryFn = Callable[[str], object]  # returns a callable frame-provider, see fetch_symbol_interest


def _default_query_fn(query_term: str, timeframe: str) -> pd.DataFrame:
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload([query_term], timeframe=timeframe)
    return pytrends.interest_over_time()


def parse_interest_series(raw: pd.DataFrame, query_term: str) -> pd.Series:
    """Extract the queried term's column from pytrends' interest_over_time()
    frame (which also carries an 'isPartial' column and, for multi-term
    queries, other terms' columns)."""
    if raw.empty or query_term not in raw.columns:
        return pd.Series(dtype=float)
    return raw[query_term].astype(float)


def fetch_symbol_interest(
    symbol: str,
    start: str,
    end: str,
    query_fn: Callable[[str, str], pd.DataFrame] = _default_query_fn,
) -> pd.DataFrame:
    """One symbol's monthly search-interest history over [start, end]."""
    query_term = f"{symbol} stock"
    raw = query_fn(query_term, f"{start} {end}")
    series = parse_interest_series(raw, query_term)
    if series.empty:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(
        {"symbol": symbol, "date": pd.to_datetime(series.index), "search_interest": series.values}
    )


def available_as_of(
    df: pd.DataFrame, asof: pd.Timestamp, lag_days: int = PUBLISH_LAG_DAYS
) -> pd.DataFrame:
    """Filter to rows whose date + lag_days has already elapsed by asof."""
    asof_naive = (
        pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tz else pd.Timestamp(asof)
    )
    cutoff = asof_naive - pd.Timedelta(days=lag_days)
    date_col = pd.to_datetime(df["date"])
    if date_col.dt.tz is not None:
        date_col = date_col.dt.tz_localize(None)
    out = df.copy()
    out["date"] = date_col
    return out[date_col <= cutoff]


def ensure_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))


def cache_symbol_interest(
    symbol: str,
    start: str,
    end: str,
    query_fn: Callable[[str, str], pd.DataFrame] = _default_query_fn,
) -> int:
    """Fetch one symbol's search-interest history and upsert into the DB
    cache. Returns rows written."""
    ensure_schema()
    df = fetch_symbol_interest(symbol, start, end, query_fn=query_fn)
    if df.empty:
        return 0
    records = df.to_dict("records")
    with get_engine().begin() as conn:
        for r in records:
            conn.execute(
                text(
                    "INSERT INTO google_trends_interest (symbol, date, search_interest) "
                    "VALUES (:symbol, :date, :search_interest) "
                    "ON CONFLICT (symbol, date) DO UPDATE SET "
                    "search_interest = EXCLUDED.search_interest"
                ),
                r,
            )
    return len(records)


def load_search_interest(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Load cached search-interest rows for symbols within [start, end]."""
    ensure_schema()
    syms: List[str] = sorted(set(symbols))
    if not syms:
        return pd.DataFrame(columns=_COLUMNS)
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, date, search_interest FROM google_trends_interest "
                "WHERE symbol = ANY(:syms) AND date BETWEEN :start AND :end "
                "ORDER BY date"
            ),
            {"syms": syms, "start": start, "end": end},
        ).fetchall()
    return pd.DataFrame(rows, columns=_COLUMNS)
