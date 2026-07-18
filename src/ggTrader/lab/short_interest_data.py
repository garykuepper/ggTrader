"""FINRA consolidated short interest: free, historical (~April 2020+),
bi-monthly settlement-date short-position data via FINRA's public Query API
(no API key). This is the free-data-only cut of the hard-to-borrow /
short-interest candidate -- no borrow-fee/utilization data (that's paid);
just consolidated short position + days-to-cover per settlement cycle.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable, Iterable, List

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
PAGE_SIZE = 5000

#: FINRA's actual settlement->publish reporting lag varies by cycle; this is
#: a conservative fixed estimate (member firms report within a few days of
#: settlement, FINRA publishes roughly two weeks after). Point-in-time gate:
#: never treat a settlement's data as knowable before settlement_date +
#: PUBLISH_LAG_DAYS has passed.
PUBLISH_LAG_DAYS = 15

_COLUMNS = [
    "symbol",
    "settlement_date",
    "current_short_position",
    "previous_short_position",
    "days_to_cover",
    "avg_daily_volume",
    "change_percent",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS short_interest (
    symbol text NOT NULL,
    settlement_date date NOT NULL,
    current_short_position bigint,
    previous_short_position bigint,
    days_to_cover double precision,
    avg_daily_volume bigint,
    change_percent double precision,
    PRIMARY KEY (symbol, settlement_date)
)
"""

HttpFetch = Callable[[dict], list]


def _default_http_fetch(payload: dict) -> list:
    req = urllib.request.Request(
        FINRA_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
        data=json.dumps(payload).encode(),
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else []


def _parse_rows(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "symbol": df["symbolCode"],
            "settlement_date": pd.to_datetime(df["settlementDate"]),
            "current_short_position": df["currentShortPositionQuantity"],
            "previous_short_position": df["previousShortPositionQuantity"],
            "days_to_cover": df["daysToCoverQuantity"],
            "avg_daily_volume": df["averageDailyVolumeQuantity"],
            "change_percent": df["changePercent"],
        }
    )


def fetch_settlement_date(
    settlement_date: str, http_fetch: HttpFetch = _default_http_fetch
) -> pd.DataFrame:
    """All symbols reported for one settlement date, paginated (FINRA caps
    each response page at PAGE_SIZE rows -- a full page means there's more,
    a short page means this was the last one)."""
    rows: list = []
    offset = 0
    while True:
        page = http_fetch(
            {
                "limit": PAGE_SIZE,
                "offset": offset,
                "dateRangeFilters": [
                    {
                        "fieldName": "settlementDate",
                        "startDate": settlement_date,
                        "endDate": settlement_date,
                    }
                ],
            }
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return _parse_rows(rows)


def discover_settlement_dates(
    start: str,
    end: str,
    anchor_symbol: str = "AAPL",
    http_fetch: HttpFetch = _default_http_fetch,
) -> List[str]:
    """The ACTUAL settlement dates FINRA reports in [start, end], discovered
    from data rather than assumed from the calendar. Settlement dates
    nominally fall on the 15th and last calendar day of each month, but
    shift to the nearest preceding business day when that date lands on a
    weekend/holiday -- guessing the fixed calendar rule silently misses
    roughly half of all real cycles. Uses one always-present, highly liquid
    anchor symbol as a proxy for "which dates does FINRA actually have data
    for" rather than paginating the entire OTC market just to find out.
    """
    rows: list = []
    offset = 0
    while True:
        page = http_fetch(
            {
                "limit": PAGE_SIZE,
                "offset": offset,
                "dateRangeFilters": [
                    {"fieldName": "settlementDate", "startDate": start, "endDate": end}
                ],
                "compareFilters": [
                    {"fieldName": "symbolCode", "fieldValue": anchor_symbol, "compareType": "EQUAL"}
                ],
            }
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return sorted({r["settlementDate"] for r in rows})


def available_as_of(
    df: pd.DataFrame, asof: pd.Timestamp, lag_days: int = PUBLISH_LAG_DAYS
) -> pd.DataFrame:
    """Filter to rows whose settlement_date + lag_days has already elapsed by
    asof -- the point-in-time gate against FINRA's real reporting lag, so a
    strategy can never see a settlement's short-interest figure before it
    would actually have been published.

    settlement_date parses tz-naive (FINRA has no timezone concept for a
    calendar settlement date); every OHLCV-derived asof in this lab is
    tz-aware UTC. Both sides are normalized to tz-naive before comparing --
    correct at the calendar-date granularity this data actually has.
    """
    asof_naive = (
        pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tz else pd.Timestamp(asof)
    )
    cutoff = asof_naive - pd.Timedelta(days=lag_days)
    settlement = pd.to_datetime(df["settlement_date"])
    if settlement.dt.tz is not None:
        settlement = settlement.dt.tz_localize(None)
    return df[settlement <= cutoff]


def ensure_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))


def cache_settlement_date(
    settlement_date: str,
    symbols: Iterable[str] | None = None,
    http_fetch: HttpFetch = _default_http_fetch,
) -> int:
    """Fetch one settlement date and upsert into the DB cache, optionally
    filtered to a symbol universe first (reduces write volume -- FINRA
    reports the whole OTC market, most of which is outside any equity
    universe this lab trades). Returns rows written."""
    ensure_schema()
    df = fetch_settlement_date(settlement_date, http_fetch=http_fetch)
    if symbols is not None:
        keep = set(symbols)
        df = df[df["symbol"].isin(keep)]
    if df.empty:
        return 0
    records = df.to_dict("records")
    with get_engine().begin() as conn:
        for r in records:
            conn.execute(
                text(
                    "INSERT INTO short_interest (symbol, settlement_date, "
                    "current_short_position, previous_short_position, days_to_cover, "
                    "avg_daily_volume, change_percent) "
                    "VALUES (:symbol, :settlement_date, :current_short_position, "
                    ":previous_short_position, :days_to_cover, :avg_daily_volume, "
                    ":change_percent) "
                    "ON CONFLICT (symbol, settlement_date) DO UPDATE SET "
                    "current_short_position = EXCLUDED.current_short_position, "
                    "previous_short_position = EXCLUDED.previous_short_position, "
                    "days_to_cover = EXCLUDED.days_to_cover, "
                    "avg_daily_volume = EXCLUDED.avg_daily_volume, "
                    "change_percent = EXCLUDED.change_percent"
                ),
                r,
            )
    return len(records)


def load_short_interest(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Load cached short-interest rows for symbols within [start, end]."""
    ensure_schema()
    syms: List[str] = sorted(set(symbols))
    if not syms:
        return pd.DataFrame(columns=_COLUMNS)
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, settlement_date, current_short_position, "
                "previous_short_position, days_to_cover, avg_daily_volume, change_percent "
                "FROM short_interest "
                "WHERE symbol = ANY(:syms) AND settlement_date BETWEEN :start AND :end "
                "ORDER BY settlement_date"
            ),
            {"syms": syms, "start": start, "end": end},
        ).fetchall()
    return pd.DataFrame(rows, columns=_COLUMNS)
