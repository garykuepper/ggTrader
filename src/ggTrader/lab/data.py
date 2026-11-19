"""Data access for the lab bench: OHLCV from the DB, PIT universe, rebalance dates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ggTrader.data.core.index_constituents import (
    coverage_stats,
    normalize_yf_ticker,
    universe_all_between,
    universe_members_asof,
)
from ggTrader.data.live import fetch_chain
from ggTrader.data.membership import MembershipSource
from ggTrader.lab.strategy import LabConfig

# PORTFOLIO_SHARE, TRAIN_METRIC, MIN_CLOSED_TRADES_TRAIN,
# MIN_TRADES_PER_TRAIN_FOLD, MAX_TRAIN_DRAWDOWN_PCT, and BENCHMARK_SYMBOL
# are not read in src/. Keep them because lab/cli.py copies this dict into
# run records and scripts/legacy_reference_runs.py uses the legacy config.
#: Default research config for daily-bar US equities.
STOCK_BASE_CONFIG: dict[str, Any] = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.0,
    "SLIPPAGE": 0.0005,
    "FREQ": "1D",
    # 3% per entry: the 2026-06-24 lever diagnostic
    # (research/programme/vectorbt-engine/legacy-scripts/lever_diagnostic.py)
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

# PORTFOLIO_SHARE, TRAIN_METRIC, MIN_CLOSED_TRADES_TRAIN,
# MIN_TRADES_PER_TRAIN_FOLD, MAX_TRAIN_DRAWDOWN_PCT, and BENCHMARK_SYMBOL
# are not read in src/. Keep them because lab/cli.py copies this dict into
# run records and scripts/legacy_reference_runs.py uses the legacy config.
#: Default research config for daily-bar cryptocurrencies (low-volume baseline).
CRYPTO_BASE_CONFIG: dict[str, Any] = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.0040,  # 0.40% baseline taker fee (Kraken Pro <$10k tier)
    "SLIPPAGE": 0.0015,  # 0.15% baseline slippage / bid-ask spread penalty
    "FREQ": "1D",
    "SIGNAL_POSITION_SIZE": 0.03,
    "USE_CASH_SHARING": True,  # Default to cash sharing for portfolio-level routing
    "TRAIN_METRIC": "composite",
    "MIN_CLOSED_TRADES_TRAIN": 0,
    "MIN_TRADES_PER_TRAIN_FOLD": 8,
    "MAX_TRAIN_DRAWDOWN_PCT": 75,
    "BENCHMARK_SYMBOL": "BTC",
}

DEFAULT_UNIVERSE = "sp500"


class _LabNegativeCache:
    """The lab's no-data list as the fetch chain's ``NegativeCache``.

    The module is imported inside each method, so it is resolved when the chain calls it and
    inside the chain's own guard, as before; an import error is "negative cache unavailable".
    """

    def load_skip(self, interval: str) -> set[str]:
        """The symbols recorded as no-data for ``interval`` within the TTL window."""
        from ggTrader.lab import negative_cache

        return negative_cache.load_skip_symbols(interval)

    def record(self, symbols: list[str], interval: str) -> None:
        """Remember ``symbols`` as no-data for ``interval``."""
        from ggTrader.lab import negative_cache

        negative_cache.record_no_data(symbols, interval)


def fetch_stock_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None = None,
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
    negative_cache = _LabNegativeCache() if use_negative_cache else None
    return fetch_chain.fetch_stock_ohlcv(
        symbols,
        start,
        end,
        interval=interval,
        use_db_cache=use_db_cache,
        min_coverage=min_coverage,
        negative_cache=negative_cache,
    )


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
    `research/audits/implementation/2026-07-25-strategy-implementation-audit.md` §2.0.

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
    symbols: list[str],
    start: str,
    end: str | None = None,
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
    membership: MembershipSource | None = None,
) -> list[str]:
    """All yfinance-normalized members that existed anywhere in the span.

    ``membership`` is the source of the S&P 500 list (``None``: the one selected by
    ``GGT_MEMBERSHIP``, else an error); static universes ignore it.
    """
    return sorted(
        {
            normalize_yf_ticker(t)
            for t in universe_all_between(universe, eval_start, eval_end, membership=membership)
        }
    )


def rebalance_dates(
    index: pd.DatetimeIndex, eval_start: pd.Timestamp, eval_end: pd.Timestamp
) -> list[pd.Timestamp]:
    """Rebalance/decision dates: window start, then the last trading day of
    each subsequent month in [eval_start, eval_end), excluding the final
    month (selecting there would leave no forward period to trade).

    Previously this returned *only* month-ends, so a weight-strategy window
    had no decision at all until the first month-end -- up to ~1 month spent
    in pure cash at the start of every fold (item 12,
    research/audits/rebaseline/2026-08-18-wfo-anchor-leakage-fix.md). Prepending the
    window's first trading day gives a decision immediately at (or right
    after) window start while keeping full causality: like every other
    rebalance date, its target takes effect on the first bar *after* it, and
    the strategy's own `select()` only sees data up to and including it
    (``data.loc[:asof]`` / ``data.index < d`` depending on the strategy --
    see allocation.py's convention). Skipped if the window is a single bar
    (no forward period to trade) or already equals the first month-end.
    """
    idx = index[(index >= eval_start) & (index <= eval_end)]
    if len(idx) == 0:
        return []
    series = pd.Series(idx, index=idx)
    month_ends = series.groupby(idx.tz_convert(None).to_period("M")).max().tolist()
    dates = month_ends[:-1]
    first_bar = idx[0]
    if first_bar != idx[-1] and (not dates or dates[0] != first_bar):
        dates = [first_bar] + dates
    return dates


def eligible_at(
    asof: pd.Timestamp,
    past: pd.DataFrame,
    cfg: LabConfig,
    universe: str = DEFAULT_UNIVERSE,
    membership: MembershipSource | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Members at asof with enough history in ``past`` (data <= asof).

    ``membership`` as in :func:`equity_universe_between`.
    """
    members = [
        normalize_yf_ticker(m) for m in universe_members_asof(universe, asof, membership=membership)
    ]
    have = set(past.columns.get_level_values(0).unique())
    eligible: list[str] = []
    for sym in members:
        if sym in have and len(past[sym]["close"].dropna()) >= cfg.min_history_bars:
            eligible.append(sym)
    if cfg.max_stocks is not None:
        eligible = sorted(eligible)[: cfg.max_stocks]
    coverage = coverage_stats(members, eligible)
    coverage["asof"] = str(asof.date())
    return eligible, coverage
