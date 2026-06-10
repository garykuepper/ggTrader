"""Point-in-time S&P 500 constituent membership.

Source: fja05680/sp500 "S&P 500 Historical Components & Changes (Updated)"
— one snapshot row per membership-change date (1996 -> present), with tickers
mapped to *current* symbology so they resolve against yfinance. The committed
copy lives at ``data/universe/sp500_constituents_history.csv.gz``; refresh it
with ``download_sp500_history()`` (manual, occasional).

Honest limitation: point-in-time membership removes selection-survivorship
(picking from today's index), but yfinance lacks price data for many delisted
tickers, so residual survivorship remains in the *data*. Quantify it per
period with ``coverage_stats()`` rather than ignoring it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SP500_HISTORY_CSV = PROJECT_ROOT / "data" / "universe" / "sp500_constituents_history.csv.gz"
SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

_history_cache: Optional[pd.DataFrame] = None


def download_sp500_history(dest: Path = SP500_HISTORY_CSV, url: str = SP500_HISTORY_URL) -> Path:
    """Refresh the committed constituent-history file from GitHub (gzipped)."""
    import gzip
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wb") as f:
        f.write(raw)
    global _history_cache
    _history_cache = None
    return dest


def load_sp500_history(csv_path: Path = SP500_HISTORY_CSV) -> pd.DataFrame:
    """Load the snapshot table: DatetimeIndex (UTC) -> sorted list of tickers."""
    global _history_cache
    if _history_cache is not None and csv_path == SP500_HISTORY_CSV:
        return _history_cache

    if not csv_path.exists():
        raise FileNotFoundError(
            f"S&P 500 constituent history not found at {csv_path}. "
            "Run download_sp500_history() once to fetch it."
        )
    raw = pd.read_csv(csv_path)
    if list(raw.columns) != ["date", "tickers"]:
        raise ValueError(f"Unexpected constituent CSV columns: {list(raw.columns)}")
    out = pd.DataFrame(
        {
            "tickers": [
                sorted(t.strip() for t in row.split(",") if t.strip()) for row in raw["tickers"]
            ]
        },
        index=pd.DatetimeIndex(pd.to_datetime(raw["date"]), tz="UTC", name="date"),
    ).sort_index()

    if csv_path == SP500_HISTORY_CSV:
        _history_cache = out
    return out


def sp500_members_asof(asof: pd.Timestamp, history: Optional[pd.DataFrame] = None) -> List[str]:
    """Members at ``asof``: the latest snapshot row dated on or before it."""
    hist = history if history is not None else load_sp500_history()
    ts = pd.Timestamp(asof)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    eligible = hist.index[hist.index <= ts]
    if len(eligible) == 0:
        raise ValueError(
            f"asof={asof} predates constituent history (starts {hist.index[0].date()})"
        )
    return list(hist.loc[eligible[-1], "tickers"])


def all_members_between(
    start: pd.Timestamp, end: pd.Timestamp, history: Optional[pd.DataFrame] = None
) -> List[str]:
    """Union of members over [start, end] — the data-download universe.

    Includes the snapshot in force at ``start`` (the latest row <= start), plus
    every change snapshot inside the window.
    """
    hist = history if history is not None else load_sp500_history()

    def _utc(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    start_ts, end_ts = _utc(start), _utc(end)
    union: set[str] = set(sp500_members_asof(start_ts, hist))
    in_window = hist[(hist.index > start_ts) & (hist.index <= end_ts)]
    for tickers in in_window["tickers"]:
        union.update(tickers)
    return sorted(union)


def normalize_yf_ticker(ticker: str) -> str:
    """Map index-style tickers to yfinance form ('BRK.B' -> 'BRK-B')."""
    return ticker.strip().upper().replace(".", "-")


def coverage_stats(members: Iterable[str], available: Iterable[str]) -> Dict[str, object]:
    """Quantify how much of a point-in-time universe has price data.

    ``available`` is compared in yfinance ticker form. The residual
    survivorship bias of a backtest is bounded by the missing fraction.
    """
    member_set = {normalize_yf_ticker(m) for m in members}
    avail_set = {normalize_yf_ticker(a) for a in available}
    missing = sorted(member_set - avail_set)
    n = len(member_set)
    return {
        "n_members": n,
        "n_with_data": n - len(missing),
        "coverage_pct": 100.0 * (n - len(missing)) / n if n else 0.0,
        "missing": missing,
    }
