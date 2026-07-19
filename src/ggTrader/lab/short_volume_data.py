"""FINRA daily short-sale volume (Reg SHO): free, structured, no key, via
FINRA's public CDN (one pipe-delimited file per trading day). This is a
DIFFERENT dataset than short_interest_data.py's consolidated short
INTEREST (bi-monthly settlement snapshots) -- this one is daily short
SALE VOLUME, the free-data-only proxy candidate #5 ("stealthy shorts")
needs, since the paper's actual liquidity-demand/supply decomposition
requires intraday tick data this project doesn't have (see
short_volume_ratio.py for how the proxy is constructed from this data).

Verified 2026-07-19 via bisection: the CDN only hosts files from
2018-08-01 onward (a clean retention boundary -- everything before returns
403, everything after 200, confirmed on multiple weekday spot-checks).
"""

from __future__ import annotations

import io
from typing import Callable, Iterable, List

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

DAILY_FILE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
USER_AGENT = "ggTrader research contact@example.com"

#: Verified empirically (bisection) -- the FINRA CDN's earliest hosted file.
EARLIEST_DATE = "2018-08-01"

#: FINRA publishes each day's file after that day's close; a strategy
#: should never treat a day's short-volume figure as knowable before then.
PUBLISH_LAG_DAYS = 1

_COLUMNS = ["symbol", "date", "short_volume", "total_volume", "short_volume_ratio"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS short_volume (
    symbol text NOT NULL,
    date date NOT NULL,
    short_volume bigint,
    total_volume bigint,
    short_volume_ratio double precision,
    PRIMARY KEY (symbol, date)
)
"""

HttpFetch = Callable[[str], str]


def _default_http_fetch(url: str) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def parse_daily_file(raw: str) -> List[dict]:
    """Every symbol's short-volume row in one day's FINRA file."""
    df = pd.read_csv(io.StringIO(raw), sep="|")
    if df.empty:
        return []
    df = df.dropna(subset=["Date", "Symbol"])
    rows: List[dict] = []
    for _, row in df.iterrows():
        total = float(row["TotalVolume"])
        short = float(row["ShortVolume"])
        ratio = (short / total) if total > 0 else float("nan")
        rows.append(
            {
                "symbol": str(row["Symbol"]),
                "date": pd.Timestamp(str(int(row["Date"]))),
                "short_volume": short,
                "total_volume": total,
                "short_volume_ratio": ratio,
            }
        )
    return rows


def available_as_of(
    df: pd.DataFrame, asof: pd.Timestamp, lag_days: int = PUBLISH_LAG_DAYS
) -> pd.DataFrame:
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


def cache_daily_file(
    date: str,
    symbols: Iterable[str] | None = None,
    http_fetch: HttpFetch = _default_http_fetch,
) -> int:
    """Fetch one day's short-volume file and upsert into the DB cache,
    optionally filtered to a symbol universe first. Returns rows written."""
    ensure_schema()
    date_compact = date.replace("-", "")
    raw = http_fetch(DAILY_FILE_URL.format(date=date_compact))
    rows = parse_daily_file(raw)
    if symbols is not None:
        keep = set(symbols)
        rows = [r for r in rows if r["symbol"] in keep]
    if not rows:
        return 0
    with get_engine().begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    "INSERT INTO short_volume (symbol, date, short_volume, total_volume, "
                    "short_volume_ratio) "
                    "VALUES (:symbol, :date, :short_volume, :total_volume, :short_volume_ratio) "
                    "ON CONFLICT (symbol, date) DO UPDATE SET "
                    "short_volume = EXCLUDED.short_volume, "
                    "total_volume = EXCLUDED.total_volume, "
                    "short_volume_ratio = EXCLUDED.short_volume_ratio"
                ),
                r,
            )
    return len(rows)


def load_short_volume(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    ensure_schema()
    syms: List[str] = sorted(set(symbols))
    if not syms:
        return pd.DataFrame(columns=_COLUMNS)
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, date, short_volume, total_volume, short_volume_ratio "
                "FROM short_volume "
                "WHERE symbol = ANY(:syms) AND date BETWEEN :start AND :end "
                "ORDER BY date"
            ),
            {"syms": syms, "start": start, "end": end},
        ).fetchall()
    return pd.DataFrame(rows, columns=_COLUMNS)
