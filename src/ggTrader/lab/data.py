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
    # 3% per entry: the 2026-06-24 lever diagnostic (scripts/lever_diagnostic.py)
    # showed the 5-voter book sits ~61% in cash at the old 0.02 default. Raising
    # to 0.03 deploys that idle cash (no leverage) and beats SPY outright in the
    # 17-fold WFO: OOS CAGR 16.2% / Sharpe 1.09 / DD -11% (vs 0.02: 10.5% / 0.89).
    # Matches the live paper trader's position_pct (risk.py, 0.033).
    "SIGNAL_POSITION_SIZE": 0.03,
    "USE_CASH_SHARING": False,
    "TRAIN_METRIC": "composite",
    "MIN_CLOSED_TRADES_TRAIN": 0,
    "MIN_TRADES_PER_TRAIN_FOLD": 8,
    "MAX_TRAIN_DRAWDOWN_PCT": 75,
    "BENCHMARK_SYMBOL": "SPY",
}

#: Default research config for daily-bar cryptocurrencies (low-volume baseline).
CRYPTO_BASE_CONFIG: dict[str, Any] = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.0040,  # 0.40% baseline taker fee (Kraken Pro <$10k tier)
    "SLIPPAGE": 0.0015,  # 0.15% baseline slippage / bid-ask spread penalty
    "FREQ": "1d",
    "SIGNAL_POSITION_SIZE": 0.03,
    "USE_CASH_SHARING": True,  # Default to cash sharing for portfolio-level routing
    "TRAIN_METRIC": "composite",
    "MIN_CLOSED_TRADES_TRAIN": 0,
    "MIN_TRADES_PER_TRAIN_FOLD": 8,
    "MAX_TRAIN_DRAWDOWN_PCT": 75,
    "BENCHMARK_SYMBOL": "BTC",
}

DEFAULT_UNIVERSE = "sp500"


def fetch_stock_ohlcv(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    use_db_cache: bool = True,
    min_coverage: float = 0.0,
    use_negative_cache: bool = False,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ``symbols`` as a (symbol, field) MultiIndex frame.

    DB-first via CachedYFinanceLoader (TimescaleDB) when reachable; falls back
    to plain yfinance. Symbols absent from the cached result are fetched for
    the full range and persisted.

    When ``use_negative_cache`` is set, symbols recorded as no-data within the
    TTL window (permanently-delisted tickers) are skipped to avoid the slow
    per-run yfinance/Tiingo retries. Opt-in only — the live paper trader must
    keep the default (always try) so a transient outage never silently drops
    an active symbol.
    """
    tickers = sorted({normalize_yf_ticker(s) for s in symbols})
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    skip: set[str] = set()
    if use_negative_cache:
        try:
            from ggTrader.lab.negative_cache import load_skip_symbols

            skip = load_skip_symbols(interval)
            if skip:
                print(f"  [data] negative cache: skipping {len(skip)} known no-data symbols")
        except Exception as exc:
            print(f"  [data] negative cache unavailable ({exc!r}); fetching all")
            skip = set()

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
    missing = [t for t in tickers if t not in have and t not in skip]
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

    have = set(df.columns.get_level_values(0).unique())
    still_missing = [t for t in tickers if t not in have and t not in skip]
    if still_missing:
        try:
            from ggTrader.data.live.tiingo_loader import TiingoDataLoader

            tiingo = TiingoDataLoader()
            print(f"  [data] trying Tiingo for {len(still_missing)} symbols yfinance missed...")
            tiingo_extra = tiingo.fetch_ohlcv(
                still_missing, interval, start_date=start_ts, end_date=end_ts
            )
            if not tiingo_extra.empty:
                if loader is not None:
                    try:
                        loader._cache_to_db(tiingo_extra, interval)
                    except Exception as exc:
                        print(f"  [data] failed to persist Tiingo data: {exc!r}")
                df = pd.concat([df, tiingo_extra], axis=1)
                df.sort_index(axis=1, inplace=True)
        except Exception as exc:
            print(f"  [data] Tiingo fallback unavailable: {exc!r}")

    # Record symbols we tried (not skipped) that no provider could supply, so
    # future opt-in runs skip them for the TTL window.
    if use_negative_cache:
        have = set(df.columns.get_level_values(0).unique())
        newly_dead = [t for t in tickers if t not in have and t not in skip]
        if newly_dead:
            try:
                from ggTrader.lab.negative_cache import record_no_data

                record_no_data(newly_dead, interval)
                print(f"  [data] negative cache: recorded {len(newly_dead)} no-data symbols")
            except Exception as exc:
                print(f"  [data] negative cache write failed ({exc!r})")

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


def collapse_daily_duplicates(wide: pd.DataFrame) -> pd.DataFrame:
    """Keep only the dominant timestamp convention in a daily OHLCV frame.

    The `ohlcv` table holds daily bars under two mutually-inconsistent
    timezone conventions, and mixing them corrupts the index:

    * **Canonical** (all ~1,410 equity symbols): 16:00/17:00, and the bar
      is labelled with the *previous* calendar day -- 2023 shows AAPL with
      zero Friday bars and 45 Sunday bars, i.e. a uniform -1 day shift.
      Internally consistent, so backtests align correctly against it.
    * **yfinance-native** (`SPY` only, 826 rows from 2022-12-27): 04:00/
      05:00, labelled with the true session date.

    Because the research harness loads ``universe + ["SPY"]`` into one
    frame (`lab/cli.py`, `lab/blend.py`), the pivot's union index gained a
    second row per trading day from 2023 on -- every non-SPY symbol NaN on
    the duplicate -- which deflated every equity Sharpe covering 2023+ by
    ~1/sqrt(2) (measured: AAPL buy-and-hold 0.848 vs 1.187 clean; CAGR and
    MaxDD unaffected, as they read endpoints and the drawdown path rather
    than per-row statistics). See
    `docs/research/2026-07-25-strategy-implementation-audit.md` §2.0.

    The two conventions are offset by one session, so they cannot be
    merged by date -- that would align SPY's Friday close against every
    other symbol's Thursday close. Instead keep the modal time-of-day and
    drop the minority convention: SPY's sessions are all present in the
    canonical rows anyway, just labelled the same way as everything else.

    The equivalent fix is deliberately not made in the shared yfinance
    loader -- that module is live-trader runtime code.
    """
    if wide.empty:
        return wide
    idx = pd.DatetimeIndex(wide.index)
    tods = pd.Index(idx.strftime("%H:%M"))
    if tods.nunique() <= 1:
        return wide
    if not pd.Series(idx.normalize()).duplicated().any():
        return wide
    # Identify the convention by cross-sectional coverage rather than row
    # count: rows written by the stray ingest path carry data for that one
    # symbol only, so they are near-empty across columns, while canonical
    # rows are populated for the whole universe. Coverage also survives
    # DST, which legitimately splits the canonical convention across two
    # times-of-day (16:00 and 17:00).
    coverage = wide.notna().mean(axis=1).groupby(tods).mean()
    keep = set(coverage[coverage >= coverage.max() * 0.5].index)
    return wide[tods.isin(keep)]


def load_ohlcv(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    use_negative_cache: bool = False,
) -> pd.DataFrame:
    """DB-first daily OHLCV as a (symbol, field) MultiIndex frame."""
    wide = fetch_stock_ohlcv(
        symbols,
        start=start,
        end=end,
        interval="1d",
        use_db_cache=True,
        use_negative_cache=use_negative_cache,
    )
    return collapse_daily_duplicates(wide)


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
