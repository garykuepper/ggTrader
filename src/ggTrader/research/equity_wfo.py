"""Shared equity WFO research library.

Replaces the five near-identical research scripts (stock/nasdaq100/russell2000/
blended/quick). Each stock runs a full (entry, exit) strategy tournament: every
combo gets its own walk-forward optimization over the per-strategy param grid,
combos are ranked by out-of-sample (OOS) robustness, and the winner is reported
per stock.

IMPORTANT: ``run_combined_validation`` selects top-N stocks using OOS scores
computed over the SAME period it then backtests — that is in-sample selection,
useful only as a smoke test. The honest estimate comes from
``ggTrader.research.monthly_walkforward``, which re-selects inside the loop.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.benchmarking import _enrich_final_stats_with_cagr_and_benchmark
from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.orchestrator_utils import _default_params_from_grid
from ggTrader.core.wfo import (
    _calculate_oos_robustness,
    _calculate_robustness,
    _execute_wfo_loop,
)
from ggTrader.pipeline.param_grids import (
    COARSE_ENTRY_PARAM_GRIDS,
    DETAILED_ENTRY_PARAM_GRIDS,
    DETAILED_EXIT_AXIS_GRIDS,
    build_param_grid,
)

#: Default research config for daily-bar US equities.
STOCK_BASE_CONFIG: Dict[str, Any] = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.0,  # Alpaca commission-free; slippage still modeled
    "SLIPPAGE": 0.0005,
    "FREQ": "1d",
    "USE_CASH_SHARING": False,
    "TRAIN_METRIC": "composite",
    "MIN_CLOSED_TRADES_TRAIN": 0,
    "MIN_TRADES_PER_TRAIN_FOLD": 8,
    "MAX_TRAIN_DRAWDOWN_PCT": 75,
    "BENCHMARK_SYMBOL": "SPY",
}


def grid_books(book: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return (entry_book, exit_book) for a named grid book ('coarse'/'detailed')."""
    if book == "coarse":
        # The coarse entry book covers a subset of strategies; entries missing
        # from it fall back to their detailed grid so the tournament stays full.
        merged = {**DETAILED_ENTRY_PARAM_GRIDS, **COARSE_ENTRY_PARAM_GRIDS}
        return merged, DETAILED_EXIT_AXIS_GRIDS
    if book == "detailed":
        return DETAILED_ENTRY_PARAM_GRIDS, DETAILED_EXIT_AXIS_GRIDS
    raise ValueError(f"Unknown grid book: {book!r} (expected 'coarse' or 'detailed')")


def normalize_yf_ticker(ticker: str) -> str:
    """Map index-style tickers to yfinance form ('BRK.B' -> 'BRK-B')."""
    return ticker.strip().upper().replace(".", "-")


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
    to plain yfinance. Symbols entirely absent from the cached result (e.g.
    tickers newly added to an existing cache) are fetched separately for the
    full range and persisted.

    Args:
        min_coverage: drop symbols with less than this fraction of non-NaN
            closes over the requested range (0.0 keeps everything — the monthly
            harness applies its own per-month history requirements instead).
    """
    tickers = sorted({normalize_yf_ticker(s) for s in symbols})
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    from ggTrader.data.live.yfinance_loader import YFinanceDataLoader

    loader: Any = None
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
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    plain = YFinanceDataLoader()
    if df.empty:
        df = plain.fetch_ohlcv(tickers, interval, start_date=start_ts, end_date=end_ts)
        if df.empty:
            raise ValueError("yfinance returned no data for the requested universe")

    # Per-symbol gap fix: cached incremental fetches miss tickers that were not
    # in the DB yet — fetch those for the full range and persist them.
    have = set(df.columns.get_level_values(0).unique())
    missing = [t for t in tickers if t not in have]
    if missing:
        print(f"  [data] fetching {len(missing)} symbols missing from cache...")
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
        keep = []
        for sym in df.columns.get_level_values(0).unique():
            cov = df[sym]["close"].notna().mean()
            if cov >= min_coverage:
                keep.append(sym)
        df = df[keep]

    n_syms = len(df.columns.get_level_values(0).unique())
    print(f"  [data] {len(df)} rows x {n_syms} symbols ({interval})")
    return df


def wfo_strategy_tournament_one_stock(
    sym_ohlcv: pd.DataFrame,
    config: Dict[str, Any],
    entries: List[str],
    exits: List[str],
    entry_book: Dict[str, Dict[str, Any]],
    exit_book: Dict[str, Dict[str, Any]],
    n_splits: int = 5,
    test_ratio: float = 3.0,
) -> Dict[str, Any]:
    """Run a full (entry, exit) WFO tournament on one stock.

    For each combo: rolling-window WFO over the combo's param grid, params
    selected per fold by in-sample (IS) robustness, scored by recency-weighted
    OOS robustness. Returns the winning combo plus all combo scores.

    Everything here uses only the data in ``sym_ohlcv`` — callers control the
    window, so the monthly harness can pass data ending at the selection date.
    """
    combo_results: List[Dict[str, Any]] = []

    for entry_name, exit_name in product(entries, exits):
        param_grid = build_param_grid(entry_name, exit_name, entry_book, exit_book)
        if not any(k in entry_book.get(entry_name, {}) for k in param_grid):
            continue  # entry has no grid in this book
        param_names = list(param_grid.keys())
        config_combo = {
            **config,
            "ENTRY_STRATEGY": entry_name,
            "EXIT_STRATEGY": exit_name,
        }

        try:
            wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
                sym_ohlcv,
                None,
                param_grid,
                config_combo,
                param_names,
                n_splits,
                test_ratio,
                False,
                None,
            )
        except Exception as exc:
            print(f"    [{entry_name}+{exit_name}] WFO failed: {exc!r}")
            continue

        valid = [
            (i + 1, s)
            for i, s in enumerate(wfo_stats)
            if not s.get("_skipped_vectorized_failure") and not s.get("_skipped_insufficient_bars")
        ]
        if not valid:
            continue

        oos_metrics = {f: s.get("oos_sharpe", float("nan")) for f, s in valid}
        oos_bear = {f: s.get("oos_is_bear", False) for f, s in valid}
        oos_rob, fold_cons = _calculate_oos_robustness(
            oos_metrics, config=config_combo, oos_bear_by_fold=oos_bear
        )

        robust_top5, best_params = _calculate_robustness(
            is_metrics_by_fold,
            param_names,
            param_grid,
            debug_metrics=False,
            config=config_combo,
        )
        if not best_params:
            best_params = _default_params_from_grid(param_grid)

        combo_results.append(
            {
                "entry": entry_name,
                "exit": exit_name,
                "params": best_params,
                "oos_robustness": float(oos_rob) if np.isfinite(oos_rob) else float("nan"),
                "fold_consistency": float(fold_cons),
                "is_robustness": (
                    float(robust_top5[0]["robustness_score"]) if robust_top5 else float("nan")
                ),
                "wfo_stats": wfo_stats,
            }
        )

    if not combo_results:
        return {"combos": [], "best": None}

    def _score(c: Dict[str, Any]) -> float:
        v = c["oos_robustness"]
        return v if np.isfinite(v) else float("-inf")

    combo_results.sort(key=_score, reverse=True)
    return {"combos": combo_results, "best": combo_results[0]}


def _tournament_worker(args: Tuple) -> Tuple[str, Dict[str, Any]]:
    """Process-pool worker: tournament + full-period replay stats for one stock."""
    (symbol, sym_ohlcv, config, entries, exits, entry_book, exit_book, n_splits, test_ratio) = args

    result = wfo_strategy_tournament_one_stock(
        sym_ohlcv, config, entries, exits, entry_book, exit_book, n_splits, test_ratio
    )
    best = result["best"]
    if best is None:
        return symbol, {"best": None, "combos": []}

    # Full-period replay with the winning combo (diagnostic; in-sample by design).
    replay_cfg = {
        **config,
        "ENTRY_STRATEGY": best["entry"],
        "EXIT_STRATEGY": best["exit"],
    }
    try:
        engine = FastBacktest(sym_ohlcv, best["params"], config=replay_cfg)
        engine.run(show_progress=False)
        full_stats = engine.get_stats()
    except Exception as exc:
        print(f"    [{symbol}] full-period replay failed: {exc!r}")
        full_stats = {}

    # Keep the payload light for pickling back: drop per-fold train_metrics blobs.
    slim_combos = [{k: v for k, v in c.items() if k != "wfo_stats"} for c in result["combos"]]
    return symbol, {
        "best": {k: v for k, v in best.items() if k != "wfo_stats"},
        "combos": slim_combos,
        "full_period_stats": full_stats,
        "avg_holding_days": full_stats.get("avg_holding_days", float("nan")),
    }


def run_wfo_per_stock(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
    entries: List[str],
    exits: List[str],
    entry_book: Dict[str, Dict[str, Any]],
    exit_book: Dict[str, Dict[str, Any]],
    n_splits: int = 5,
    test_ratio: float = 3.0,
    n_jobs: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Per-stock strategy tournament across the universe (optionally in parallel)."""
    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    per_stock: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()

    jobs = [
        (sym, ohlcv[[sym]], config, entries, exits, entry_book, exit_book, n_splits, test_ratio)
        for sym in symbols
    ]

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for i, (sym, res) in enumerate(pool.map(_tournament_worker, jobs, chunksize=1)):
                per_stock[sym] = res
                _print_stock_progress(i, len(symbols), sym, res, t0)
    else:
        for i, job in enumerate(jobs):
            sym, res = _tournament_worker(job)
            per_stock[sym] = res
            _print_stock_progress(i, len(symbols), sym, res, t0)

    print(f"\nTournament complete: {len(symbols)} stocks in {time.time() - t0:.1f}s")
    return per_stock


def _print_stock_progress(i: int, total: int, symbol: str, res: Dict[str, Any], t0: float) -> None:
    best = res.get("best")
    if best is None:
        print(f"[{i + 1}/{total}] {symbol}: no valid combo")
        return
    print(
        f"[{i + 1}/{total}] {symbol}: {best['entry']}+{best['exit']} "
        f"OOS={best['oos_robustness']:.3f} fold_cons={best['fold_consistency']:.0%} "
        f"hold={res.get('avg_holding_days', float('nan')):.1f}d "
        f"({time.time() - t0:.0f}s elapsed)"
    )


def run_combined_validation(
    ohlcv: pd.DataFrame,
    per_stock: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    max_position_pct: float = 0.02,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Combined portfolio replay with per-stock winning combos.

    WARNING: when ``top_n`` is set, stocks are picked by OOS robustness scores
    computed over the same period this replays — IN-SAMPLE selection. Smoke
    test only; never quote these numbers as an out-of-sample estimate.
    """
    candidates = {s: r for s, r in per_stock.items() if r.get("best")}
    symbols = [s for s in ohlcv.columns.get_level_values(0).unique() if s in candidates]

    if top_n is not None:
        symbols = sorted(
            symbols,
            key=lambda s: (
                candidates[s]["best"]["oos_robustness"]
                if np.isfinite(candidates[s]["best"]["oos_robustness"])
                else float("-inf")
            ),
            reverse=True,
        )[:top_n]
        print(
            f"  -> Top {top_n} by full-period OOS score (IN-SAMPLE selection — "
            f"smoke test only): {symbols}"
        )

    all_entries, all_exits, all_close = [], [], []
    for symbol in symbols:
        best = candidates[symbol]["best"]
        sym_ohlcv = ohlcv[[symbol]]
        cfg = {**config, "ENTRY_STRATEGY": best["entry"], "EXIT_STRATEGY": best["exit"]}
        try:
            engine = FastBacktest(sym_ohlcv, best["params"], config=cfg)
            engine.run(show_progress=False)
        except Exception as exc:
            print(f"    [{symbol}] combined replay failed: {exc!r}")
            continue
        all_entries.append(engine.entries.droplevel("param_combo", axis=1))
        all_exits.append(engine.exits.droplevel("param_combo", axis=1))
        all_close.append(sym_ohlcv.xs("close", axis=1, level=1, drop_level=True))

    if not all_entries:
        return {}

    entries_df = pd.concat(all_entries, axis=1)
    exits_df = pd.concat(all_exits, axis=1)
    close_df = pd.concat(all_close, axis=1)

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_df.fillna(False),
        exits=exits_df.fillna(False),
        init_cash=float(config["START_CASH"]),
        fees=float(config["FEES"]),
        slippage=float(config["SLIPPAGE"]),
        freq=config["FREQ"],
        size=max_position_pct,
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(entries_df.shape[1], 0),
    ).copy()

    def _safe(val: Any) -> float:
        try:
            v = float(val)
            return 0.0 if (np.isnan(v) or np.isinf(v)) else v
        except (TypeError, ValueError):
            return 0.0

    dur = np.array(pf.trades.duration.values, dtype=np.float64, copy=True)
    stats = {
        "total_value": _safe(pf.final_value().sum()),
        "total_profit": _safe(pf.total_profit().sum()),
        "profit_pct": _safe((pf.total_profit().sum() / float(config["START_CASH"])) * 100),
        "total_trades": int(pf.trades.count().sum()),
        "win_rate": _safe(pf.trades.win_rate().mean()) * 100,
        "sharpe": _safe(pf.sharpe_ratio().mean()),
        "sortino": _safe(pf.sortino_ratio().mean()),
        "max_drawdown": _safe(pf.max_drawdown().min()) * 100,
        "avg_holding_days": _safe(dur.mean()) if dur.size else 0.0,
        "n_stocks": len(symbols),
        "selection": "IN-SAMPLE top-N (smoke test only)" if top_n else "all stocks",
    }
    _enrich_final_stats_with_cagr_and_benchmark(stats, close_df, config)
    return stats


def print_tournament_summary(per_stock: Dict[str, Dict[str, Any]], top: int = 20) -> None:
    """Readable summary: winners per stock + strategy-combo win counts."""
    rows = []
    for sym, r in per_stock.items():
        best = r.get("best")
        if not best:
            continue
        rows.append(
            {
                "symbol": sym,
                "combo": f"{best['entry']}+{best['exit']}",
                "oos": best["oos_robustness"],
                "fold_cons": best["fold_consistency"],
                "hold_days": r.get("avg_holding_days", float("nan")),
                "trades": r.get("full_period_stats", {}).get("total_trades", 0),
            }
        )
    if not rows:
        print("No stocks produced a valid tournament winner.")
        return

    df = pd.DataFrame(rows).sort_values("oos", ascending=False)

    print("\n" + "=" * 78)
    print(f"TOP {top} STOCKS BY OOS ROBUSTNESS (winning combo per stock)")
    print("=" * 78)
    for _, row in df.head(top).iterrows():
        print(
            f"  {row['symbol']:>6} | {row['combo']:<38} | OOS={row['oos']:>7.3f} | "
            f"cons={row['fold_cons']:>4.0%} | hold={row['hold_days']:>5.1f}d | "
            f"trades={row['trades']:>4}"
        )

    print("\nStrategy-combo win counts (which combos win the per-stock tournament):")
    counts = df["combo"].value_counts()
    for combo, n in counts.items():
        print(f"  {combo:<38} {n:>4} stocks")

    finite_hold = df["hold_days"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_hold):
        print(
            f"\nHolding-period distribution (days): "
            f"p25={finite_hold.quantile(0.25):.1f} median={finite_hold.median():.1f} "
            f"p75={finite_hold.quantile(0.75):.1f} "
            f"(goal: 2-10 trading days average)"
        )
