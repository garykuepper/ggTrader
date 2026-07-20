"""FRED (Federal Reserve Economic Data) series: free, no API key required
via the public fredgraph.csv endpoint (a plain CSV download, distinct from
FRED's key-gated JSON API). Used for policy-rate and CPI series backing the
fx_hedge_overlay strategy's carry and value signals.
"""

from __future__ import annotations

import io
from typing import Callable

import pandas as pd
import requests
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: Conservative fixed lag for monthly macro series (CPI, short-term policy
#: rates): typically published 2-6 weeks after the reference month. Point-in-
#: time gate: never treat an observation as knowable before date + this lag.
PUBLISH_LAG_DAYS = 45

HttpFetch = Callable[[str], str]


def _default_http_fetch(series_id: str) -> str:
    # urllib.request.urlopen intermittently 404s/times out against this
    # host in this environment even for URLs that work fine via curl or
    # requests -- switched to requests after reproducing the failure
    # directly (same URL, same headers, only the client library differed).
    resp = requests.get(
        FRED_CSV_URL, params={"id": series_id}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
    )
    resp.raise_for_status()
    return resp.text


def parse_fred_csv(raw: str, series_id: str) -> pd.DataFrame:
    """Parse fredgraph.csv text into a (date, value) frame.

    FRED encodes missing observations as a literal "." -- dropped, not
    coerced to NaN-via-crash.
    """
    df = pd.read_csv(io.StringIO(raw))
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df = df.rename(columns={"observation_date": "date", series_id: "value"})
    df = df[df["value"] != "."]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].astype(float)
    return df[["date", "value"]].reset_index(drop=True)


def fetch_fred_series(series_id: str, http_fetch: HttpFetch | None = None) -> pd.DataFrame:
    """Fetch and parse one FRED series by id."""
    fetch = http_fetch or _default_http_fetch
    return parse_fred_csv(fetch(series_id), series_id=series_id)


def available_as_of(df: pd.DataFrame, asof: pd.Timestamp, lag_days: int) -> pd.DataFrame:
    """Filter to observations knowable by ``asof`` given the publication lag.

    Normalizes both sides to tz-naive before comparing (FRED dates parse
    tz-naive; lab asof timestamps are tz-aware UTC) and returns a tz-naive
    "date" column so downstream datetime arithmetic never crashes on a
    naive/aware mismatch.
    """
    asof_naive = (
        pd.Timestamp(asof).tz_localize(None)
        if pd.Timestamp(asof).tz is not None
        else pd.Timestamp(asof)
    )
    out = df.copy()
    date_naive = out["date"].dt.tz_localize(None) if out["date"].dt.tz is not None else out["date"]
    out["date"] = date_naive
    mask = date_naive + pd.Timedelta(days=lag_days) <= asof_naive
    return out[mask].reset_index(drop=True)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS fred_series (
    series_id text NOT NULL,
    date date NOT NULL,
    value double precision,
    PRIMARY KEY (series_id, date)
)
"""


def ensure_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))


def _rows_for_upsert(series_id: str, df: pd.DataFrame) -> list[dict]:
    """Build (series_id, date, value) rows for the upsert.

    Uses to_dict("records") rather than DataFrame.iterrows() -- iterrows()
    unifies each row into a single-dtype Series, and over a frame mixing a
    datetime64 "date" column with a float64 "value" column this can coerce
    some rows' values to NaT once enough rows are present (a real pandas
    dtype-unification gotcha, reproduced and regression-tested).
    """
    return [
        {"series_id": series_id, "date": rec["date"].date(), "value": float(rec["value"])}
        for rec in df.to_dict("records")
    ]


def cache_series(series_id: str, http_fetch: HttpFetch | None = None) -> int:
    """Fetch one FRED series and upsert it into the DB cache."""
    df = fetch_fred_series(series_id, http_fetch=http_fetch)
    if df.empty:
        return 0
    ensure_schema()
    rows = _rows_for_upsert(series_id, df)
    upsert_sql = text(
        """
        INSERT INTO fred_series (series_id, date, value)
        VALUES (:series_id, :date, :value)
        ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value
        """
    )
    with get_engine().begin() as conn:
        conn.execute(upsert_sql, rows)
    return len(rows)


def load_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Load a cached FRED series for [start, end] from the DB."""
    query = text(
        """
        SELECT date, value FROM fred_series
        WHERE series_id = :series_id AND date >= :start AND date <= :end
        ORDER BY date
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(query, {"series_id": series_id, "start": start, "end": end}).fetchall()
    df = pd.DataFrame(rows, columns=["date", "value"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
