"""Post-earnings-announcement-drift (PEAD) surprise data: free, deep
historical (back to ~2002 for most symbols) via yfinance's per-ticker
``get_earnings_dates`` -- unlike its estimate-revision fields (eps_trend,
eps_revisions), this endpoint genuinely exposes a queryable point-in-time
history of realized actual-vs-consensus EPS surprises, not just a rolling
current-quarter snapshot (verified 2026-07-17; see the analyst-revision-
momentum candidate's write-up for the contrasting infeasible case).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

#: Conservative fixed lag between an earnings report and when its surprise
#: is safely tradeable -- most reports land after market close (or before
#: open), so same-day price action already reflects the news; this project's
#: monthly rebalance cadence makes the exact intraday timing immaterial.
REPORT_LAG_DAYS = 1

_COLUMNS = ["symbol", "earnings_date", "eps_estimate", "eps_reported", "surprise_pct"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS earnings_surprise (
    symbol text NOT NULL,
    earnings_date date NOT NULL,
    eps_estimate double precision,
    eps_reported double precision,
    surprise_pct double precision,
    PRIMARY KEY (symbol, earnings_date)
)
"""

EarningsDatesFn = Callable[[str, int], Optional[pd.DataFrame]]


def _default_earnings_dates_fn(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    import yfinance as yf

    return yf.Ticker(symbol).get_earnings_dates(limit=limit)


def fetch_symbol_surprises(
    symbol: str,
    limit: int = 100,
    earnings_dates_fn: EarningsDatesFn = _default_earnings_dates_fn,
) -> pd.DataFrame:
    """One symbol's realized earnings-surprise history, deepest-first as
    yfinance returns it. Drops any row with no Reported EPS yet (a future or
    not-yet-processed earnings date -- not a realized surprise)."""
    raw = earnings_dates_fn(symbol, limit)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_COLUMNS)
    df = raw.reset_index()
    df = df.rename(
        columns={
            "Earnings Date": "earnings_date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_reported",
            "Surprise(%)": "surprise_pct",
        }
    )
    df = df.dropna(subset=["eps_reported", "surprise_pct"])
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    df["symbol"] = symbol
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.tz_localize(None)
    return df[_COLUMNS]


def available_as_of(
    df: pd.DataFrame, asof: pd.Timestamp, lag_days: int = REPORT_LAG_DAYS
) -> pd.DataFrame:
    """Filter to rows whose earnings_date + lag_days has already elapsed by
    asof -- point-in-time gate, same convention as short_interest_data.py's
    available_as_of. Both sides normalized to tz-naive before comparing."""
    asof_naive = (
        pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tz else pd.Timestamp(asof)
    )
    cutoff = asof_naive - pd.Timedelta(days=lag_days)
    earnings_date = pd.to_datetime(df["earnings_date"])
    if earnings_date.dt.tz is not None:
        earnings_date = earnings_date.dt.tz_localize(None)
    out = df.copy()
    out["earnings_date"] = earnings_date
    return out[earnings_date <= cutoff]


def ensure_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))


def cache_symbol_surprises(
    symbol: str,
    limit: int = 100,
    earnings_dates_fn: EarningsDatesFn = _default_earnings_dates_fn,
) -> int:
    """Fetch one symbol's surprise history and upsert into the DB cache.
    Returns rows written."""
    ensure_schema()
    df = fetch_symbol_surprises(symbol, limit=limit, earnings_dates_fn=earnings_dates_fn)
    if df.empty:
        return 0
    records = df.to_dict("records")
    with get_engine().begin() as conn:
        for r in records:
            conn.execute(
                text(
                    "INSERT INTO earnings_surprise (symbol, earnings_date, eps_estimate, "
                    "eps_reported, surprise_pct) "
                    "VALUES (:symbol, :earnings_date, :eps_estimate, :eps_reported, :surprise_pct) "
                    "ON CONFLICT (symbol, earnings_date) DO UPDATE SET "
                    "eps_estimate = EXCLUDED.eps_estimate, "
                    "eps_reported = EXCLUDED.eps_reported, "
                    "surprise_pct = EXCLUDED.surprise_pct"
                ),
                r,
            )
    return len(records)


def load_earnings_surprises(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Load cached earnings-surprise rows for symbols within [start, end]."""
    ensure_schema()
    syms: List[str] = sorted(set(symbols))
    if not syms:
        return pd.DataFrame(columns=_COLUMNS)
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, earnings_date, eps_estimate, eps_reported, surprise_pct "
                "FROM earnings_surprise "
                "WHERE symbol = ANY(:syms) AND earnings_date BETWEEN :start AND :end "
                "ORDER BY earnings_date"
            ),
            {"syms": syms, "start": start, "end": end},
        ).fetchall()
    return pd.DataFrame(rows, columns=_COLUMNS)
