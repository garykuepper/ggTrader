"""Data access for the lab bench: OHLCV from the DB, PIT universe, rebalance dates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ggTrader.data.core.index_constituents import (
    coverage_stats,
    normalize_yf_ticker,
    universe_all_between,
    universe_members_asof,
)
from ggTrader.lab.strategy import LabConfig

#: Default research config for daily-bar US equities.
STOCK_BASE_CONFIG: dict[str, Any] = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.0,
    "SLIPPAGE": 0.0005,
    "FREQ": "1d",
    "USE_CASH_SHARING": False,
    "TRAIN_METRIC": "composite",
    "MIN_CLOSED_TRADES_TRAIN": 0,
    "MIN_TRADES_PER_TRAIN_FOLD": 8,
    "MAX_TRAIN_DRAWDOWN_PCT": 75,
    "BENCHMARK_SYMBOL": "SPY",
}

DEFAULT_UNIVERSE = "sp500"


def fetch_stock_ohlcv(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    use_db_cache: bool = True,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ``symbols`` as a (symbol, field) MultiIndex frame.

    DB-first via CachedYFinanceLoader (TimescaleDB) when reachable; falls back
    to plain yfinance. Symbols absent from the cached result are fetched for
    the full range and persisted.
    """
    tickers = sorted({normalize_yf_ticker(s) for s in symbols})
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    from ggTrader.data.live.yfinance_loader import YFinanceDataLoader

    loader: Any = None
    df = pd.DataFrame()
    if use_db_cache:
        try:
            from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader

            loader = CachedYFinanceLoader()
            df = loader.fetch_ohlcv(
                tickers, interval, start_date=start_ts, end_date=end_ts, limit=None
            )
        except Exception as exc:
            print(f"  [data] DB cache unavailable ({exc!r}); falling back to yfinance only")
            loader = None

    plain: YFinanceDataLoader | None = None
    if df.empty:
        plain = YFinanceDataLoader()
        df = plain.fetch_ohlcv(tickers, interval, start_date=start_ts, end_date=end_ts)
        if df.empty:
            raise ValueError("yfinance returned no data for the requested universe")

    have = set(df.columns.get_level_values(0).unique())
    missing = [t for t in tickers if t not in have]
    if missing:
        print(f"  [data] fetching {len(missing)} symbols missing from cache...")
        if plain is None:
            plain = YFinanceDataLoader()
        extra = plain.fetch_ohlcv(missing, interval, start_date=start_ts, end_date=end_ts)
        if not extra.empty:
            if loader is not None:
                try:
                    loader._cache_to_db(extra, interval)
                except Exception as exc:
                    print(f"  [data] failed to persist gap fetch: {exc!r}")
            df = pd.concat([df, extra], axis=1)
            df.sort_index(axis=1, inplace=True)

    df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]

    if min_coverage > 0.0:
        keep = [
            sym
            for sym in df.columns.get_level_values(0).unique()
            if df[sym]["close"].notna().mean() >= min_coverage
        ]
        df = df[keep]

    n_syms = len(df.columns.get_level_values(0).unique())
    print(f"  [data] {len(df)} rows x {n_syms} symbols ({interval})")
    df.columns.names = ["symbol", "field"]
    return df


def load_ohlcv(symbols: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """DB-first daily OHLCV as a (symbol, field) MultiIndex frame."""
    return fetch_stock_ohlcv(symbols, start=start, end=end, interval="1d", use_db_cache=True)


def equity_universe_between(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    universe: str = DEFAULT_UNIVERSE,
) -> List[str]:
    """All yfinance-normalized members that existed anywhere in the span."""
    return sorted(
        {normalize_yf_ticker(t) for t in universe_all_between(universe, eval_start, eval_end)}
    )


def rebalance_dates(
    index: pd.DatetimeIndex, eval_start: pd.Timestamp, eval_end: pd.Timestamp
) -> List[pd.Timestamp]:
    """Last trading day of each month in [eval_start, eval_end), excluding the final
    month (selecting there would leave no forward period to trade)."""
    idx = index[(index >= eval_start) & (index <= eval_end)]
    if len(idx) == 0:
        return []
    series = pd.Series(idx, index=idx)
    month_ends = series.groupby(idx.tz_convert(None).to_period("M")).max().tolist()
    return month_ends[:-1]


def eligible_at(
    asof: pd.Timestamp,
    past: pd.DataFrame,
    cfg: LabConfig,
    universe: str = DEFAULT_UNIVERSE,
) -> Tuple[List[str], Dict[str, Any]]:
    """Members at asof with enough history in ``past`` (data <= asof)."""
    members = [normalize_yf_ticker(m) for m in universe_members_asof(universe, asof)]
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
