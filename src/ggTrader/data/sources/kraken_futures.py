"""Kraken Futures historical data ingester.

Bypasses ccxt and talks directly to Kraken Futures' public REST API. ccxt's
coverage for this venue is unreliable — it cannot fetch OHLCV for expired
contracts, only currently-listed ones, and the funding-rate endpoint isn't
exposed through ccxt at all. Direct REST is the only path that works.

Endpoints used
--------------
1. ``GET /api/charts/v1/trade/{symbol}/{resolution}?from=<sec>&to=<sec>``
   OHLCV candles. ``resolution`` ∈ {1m,5m,15m,1h,4h,12h,1d,1w}. Returns up
   to 2000 candles per call. ``from``/``to`` are UNIX seconds (UTC).
2. ``GET /derivatives/api/v4/historicalfundingrates?symbol={sym}``
   Funding rates. Returns a fixed window (~1 year) of hourly funding events
   — does NOT accept date-range params. Each row has
   ``timestamp`` (ISO 8601), ``fundingRate`` (USD/contract per interval),
   and ``relativeFundingRate`` (decimal per-interval rate).

History depth
-------------
- Funding rates: Kraken's API only returns ~1 year of recent data. The
  2022-2024 funding history is not accessible by any public endpoint.
- OHLCV: PF_XBTUSD listed 2022-03-23; full hourly history back to listing
  is available via paginated ``from``/``to`` calls.

Rate limiting
-------------
Kraken Futures public endpoints document no explicit rate cap but advise
"reasonable" use. The ingester sleeps 250 ms between calls (4 req/s ceiling)
which is conservative and gets us through a 3-year hourly backfill in under
30 seconds total wall time.

Idempotency
-----------
Both ingesters write via ``INSERT ... ON CONFLICT DO NOTHING``. Re-running
is safe; only new rows land.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
import psycopg2.extras
import requests

KRAKEN_FUTURES_BASE = "https://futures.kraken.com"
CHART_BASE = f"{KRAKEN_FUTURES_BASE}/api/charts/v1/trade"
FUNDING_BASE = f"{KRAKEN_FUTURES_BASE}/derivatives/api/v4/historicalfundingrates"

CHART_MAX_CANDLES = 2000
REQUEST_SLEEP_SEC = 0.25
HTTP_TIMEOUT_SEC = 20


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "ggTrader/0.1 (historical backfill)"})
    return s


def _get_json(session: requests.Session, url: str) -> dict[str, Any]:
    """GET with one retry on 429 / 5xx."""
    for attempt in range(3):
        r = session.get(url, timeout=HTTP_TIMEOUT_SEC)
        if r.status_code == 200:
            payload: dict[str, Any] = r.json()
            return payload
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed after retries: {url}")


# --------------------------------------------------------------------------- OHLCV


@dataclass(frozen=True)
class OHLCVBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _resolution_seconds(resolution: str) -> int:
    return {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "12h": 43200,
        "1d": 86400,
        "1w": 604800,
    }[resolution]


def fetch_chart_window(
    session: requests.Session,
    symbol: str,
    resolution: str,
    start: datetime,
    end: datetime,
) -> list[OHLCVBar]:
    """Paginate the chart endpoint across ``[start, end]``. Returns bars in
    chronological order, deduped on timestamp."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be tz-aware UTC")
    if end <= start:
        return []

    step_seconds = _resolution_seconds(resolution) * CHART_MAX_CANDLES
    cursor = start
    bars: dict[int, OHLCVBar] = {}

    while cursor < end:
        window_end = min(cursor + timedelta(seconds=step_seconds), end)
        url = (
            f"{CHART_BASE}/{symbol}/{resolution}"
            f"?from={int(cursor.timestamp())}&to={int(window_end.timestamp())}"
        )
        body = _get_json(session, url)
        candles = body.get("candles", [])
        if not candles:
            # Empty page: jump forward to avoid an infinite loop on perpetually-empty ranges
            cursor = window_end
            time.sleep(REQUEST_SLEEP_SEC)
            continue
        for c in candles:
            ts_ms = int(c["time"])
            bars[ts_ms] = OHLCVBar(
                ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c.get("volume", 0.0)),
            )
        last_ts_ms = candles[-1]["time"]
        next_cursor = datetime.fromtimestamp(last_ts_ms / 1000, tz=timezone.utc) + timedelta(
            seconds=_resolution_seconds(resolution)
        )
        cursor = max(next_cursor, cursor + timedelta(seconds=_resolution_seconds(resolution)))
        time.sleep(REQUEST_SLEEP_SEC)

    return [bars[k] for k in sorted(bars.keys())]


def write_perp_ohlcv(
    conn: psycopg2.extensions.connection,
    symbol: str,
    resolution: str,
    bars: list[OHLCVBar],
) -> int:
    if not bars:
        return 0
    rows = [
        (
            b.ts.replace(tzinfo=None),
            symbol,
            resolution,
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
        )
        for b in bars
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO perp_ohlcv
                ("timestamp", symbol, "interval", open, high, low, close, volume)
            VALUES %s
            ON CONFLICT ("timestamp", symbol, "interval") DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


# --------------------------------------------------------------------------- Funding


@dataclass(frozen=True)
class FundingRow:
    ts: datetime
    funding_rate: float
    relative_funding_rate: float


def fetch_funding_history(session: requests.Session, symbol: str) -> list[FundingRow]:
    """Pull the full funding-rate window the API will return.

    Kraken returns a fixed window (~1 year at the time of writing) with no
    date-range params honored. The caller should pin the snapshot date in
    their run log for reproducibility.
    """
    url = f"{FUNDING_BASE}?symbol={symbol}"
    body = _get_json(session, url)
    rows = body.get("rates", [])
    out: list[FundingRow] = []
    for r in rows:
        ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        out.append(
            FundingRow(
                ts=ts,
                funding_rate=float(r["fundingRate"]),
                relative_funding_rate=float(r["relativeFundingRate"]),
            )
        )
    out.sort(key=lambda x: x.ts)
    return out


def write_funding_rates(
    conn: psycopg2.extensions.connection, symbol: str, rows: list[FundingRow]
) -> int:
    if not rows:
        return 0
    payload = [
        (
            r.ts.replace(tzinfo=None),
            symbol,
            r.funding_rate,
            r.relative_funding_rate,
        )
        for r in rows
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO funding_rates
                ("timestamp", symbol, funding_rate, relative_funding_rate)
            VALUES %s
            ON CONFLICT ("timestamp", symbol) DO NOTHING
            """,
            payload,
        )
    conn.commit()
    return len(payload)


# --------------------------------------------------------------------------- Top-level


def backfill_perp(
    conn: psycopg2.extensions.connection,
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: str = "1h",
) -> int:
    session = _make_session()
    bars = fetch_chart_window(session, symbol, resolution, start, end)
    return write_perp_ohlcv(conn, symbol, resolution, bars)


def backfill_funding(
    conn: psycopg2.extensions.connection,
    symbol: str,
) -> tuple[int, datetime, datetime]:
    """Returns (rows_inserted, earliest_ts, latest_ts) for run-log pinning."""
    session = _make_session()
    rows = fetch_funding_history(session, symbol)
    inserted = write_funding_rates(conn, symbol, rows)
    if not rows:
        return (
            0,
            datetime.min.replace(tzinfo=timezone.utc),
            datetime.min.replace(tzinfo=timezone.utc),
        )
    return inserted, rows[0].ts, rows[-1].ts


def refresh_views(conn: psycopg2.extensions.connection) -> None:
    """Refresh derived materialized views after a backfill batch."""
    with conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW funding_apr")
        cur.execute("REFRESH MATERIALIZED VIEW basis_series")
    conn.commit()


__all__ = [
    "OHLCVBar",
    "FundingRow",
    "backfill_funding",
    "backfill_perp",
    "fetch_chart_window",
    "fetch_funding_history",
    "refresh_views",
    "write_funding_rates",
    "write_perp_ohlcv",
]
