"""Data access for the lab bench: OHLCV from the DB, PIT universe, rebalance dates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ggTrader.data.core.index_constituents import (
    all_members_between,
    coverage_stats,
    normalize_yf_ticker,
    sp500_members_asof,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.research.equity_wfo import fetch_stock_ohlcv


def load_ohlcv(symbols: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """DB-first daily OHLCV as a (symbol, field) MultiIndex frame."""
    df = fetch_stock_ohlcv(symbols, start=start, end=end, interval="1d", use_db_cache=True)
    df.columns.names = ["symbol", "field"]
    return df


def equity_universe_between(eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> List[str]:
    """All yfinance-normalized S&P 500 members that existed anywhere in the span."""
    return sorted({normalize_yf_ticker(t) for t in all_members_between(eval_start, eval_end)})


def rebalance_dates(
    index: pd.DatetimeIndex, eval_start: pd.Timestamp, eval_end: pd.Timestamp
) -> List[pd.Timestamp]:
    """Last trading day of each month in [eval_start, eval_end), excluding the final
    month (selecting there would leave no forward period to trade)."""
    idx = index[(index >= eval_start) & (index <= eval_end)]
    if len(idx) == 0:
        return []
    series = pd.Series(idx, index=idx)
    month_ends = series.groupby(idx.tz_localize(None).to_period("M")).max().tolist()
    return month_ends[:-1]


def eligible_at(
    asof: pd.Timestamp, past: pd.DataFrame, cfg: LabConfig
) -> Tuple[List[str], Dict[str, Any]]:
    """PIT members at asof with enough history in ``past`` (data <= asof)."""
    members = [normalize_yf_ticker(m) for m in sp500_members_asof(asof)]
    have = set(past.columns.get_level_values(0).unique())
    eligible: List[str] = []
    for sym in members:
        if sym in have and len(past[sym]["close"].dropna()) >= cfg.min_history_bars:
            eligible.append(sym)
    if cfg.max_stocks is not None:
        eligible = sorted(eligible)[: cfg.max_stocks]
    coverage = coverage_stats(members, eligible)
    coverage["asof"] = str(asof.date())
    return eligible, coverage
