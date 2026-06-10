"""Rolling monthly walk-forward portfolio harness — the honest S&P 500 backtest.

At each month-end T:
  1. Universe = S&P 500 members AT T (point-in-time) with enough price history.
  2. Per stock: full (entry, exit) strategy tournament via WFO on the trailing
     ``lookback_bars`` ending at T. Nothing after T is visible.
  3. Select the top-N stocks by OOS robustness; record combo/params/holding.
  4. Trade the following month with those frozen selections (signals warmed up
     on pre-T data, entries restricted to the forward month).
  5. Checkpoint the month; stitch all months into one equity curve vs SPY.

Known limitation (documented, bounded by short target holds): positions are
not carried across month boundaries — an entry near month-end is effectively
liquidated at the last bar of the month at close, and pre-month entries are
not re-opened.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.orchestrator_utils import _to_native
from ggTrader.data.core.index_constituents import (
    all_members_between,
    coverage_stats,
    normalize_yf_ticker,
    sp500_members_asof,
)
from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY
from ggTrader.research.equity_wfo import (
    STOCK_BASE_CONFIG,
    fetch_stock_ohlcv,
    grid_books,
    wfo_strategy_tournament_one_stock,
)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class MonthlyHarnessConfig:
    eval_start: str = "2021-01-31"  # first selection date T (trade Feb 2021 onward)
    eval_end: Optional[str] = None  # default: last completed month in the data
    lookback_bars: int = 504  # ~2y of daily bars per selection window
    n_splits: int = 5  # -> 63-bar OOS folds at test_ratio=3
    test_ratio: float = 3.0
    top_n: int = 50
    max_position_pct: float = 0.02
    entries: List[str] = field(default_factory=lambda: list(ENTRY_REGISTRY.keys()))
    exits: List[str] = field(default_factory=lambda: list(EXIT_REGISTRY.keys()))
    grid_book: str = "detailed"
    warmup_bars: int = 200  # indicator warmup ahead of the forward month
    n_jobs: int = 8
    min_history_bars: int = 400  # required non-NaN closes inside the lookback
    refit_every_n_months: int = 1  # 3 = quarterly re-selection (compute escape valve)
    max_stocks: Optional[int] = None  # cap universe per month (deterministic; --quick)
    checkpoint_dir: str = "results/monthly_wf"
    run_id: str = "sp500_monthly"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _select_worker(args: Tuple) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Tournament for one stock on its trailing window (process-pool worker)."""
    (
        symbol,
        sym_ohlcv,
        base_config,
        entries,
        exits,
        entry_book,
        exit_book,
        n_splits,
        test_ratio,
    ) = args
    try:
        res = wfo_strategy_tournament_one_stock(
            sym_ohlcv, base_config, entries, exits, entry_book, exit_book, n_splits, test_ratio
        )
    except Exception as exc:
        print(f"    [{symbol}] tournament failed: {exc!r}")
        return symbol, None
    best = res.get("best")
    if best is None or not np.isfinite(best.get("oos_robustness", float("nan"))):
        return symbol, None

    # Trailing-window replay with the winning combo for the holding-period
    # diagnostic. Uses only data <= T (the window itself).
    avg_hold = float("nan")
    try:
        replay_cfg = {
            **base_config,
            "ENTRY_STRATEGY": best["entry"],
            "EXIT_STRATEGY": best["exit"],
        }
        engine = FastBacktest(sym_ohlcv, best["params"], config=replay_cfg)
        engine.run(show_progress=False)
        avg_hold = engine.get_stats().get("avg_holding_days", float("nan"))
    except Exception:
        pass

    return symbol, {
        "symbol": symbol,
        "entry": best["entry"],
        "exit": best["exit"],
        "params": _to_native(best["params"]),
        "oos_robustness": float(best["oos_robustness"]),
        "fold_consistency": float(best["fold_consistency"]),
        "is_robustness": float(best["is_robustness"]),
        "avg_holding_days": float(avg_hold) if np.isfinite(avg_hold) else None,
    }


def select_for_month(
    asof: pd.Timestamp,
    ohlcv: pd.DataFrame,
    cfg: MonthlyHarnessConfig,
    base_config: Dict[str, Any],
    entry_book: Dict[str, Dict[str, Any]],
    exit_book: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rank the point-in-time universe at ``asof`` and return (top-N, coverage).

    Only data with index <= asof is ever passed to the tournament.
    """
    members = [normalize_yf_ticker(m) for m in sp500_members_asof(asof)]
    have = set(ohlcv.columns.get_level_values(0).unique())
    past = ohlcv.loc[:asof]

    eligible: List[str] = []
    for sym in members:
        if sym not in have:
            continue
        closes = past[sym]["close"].dropna()
        if len(closes) >= cfg.min_history_bars:
            eligible.append(sym)
    if cfg.max_stocks is not None:
        eligible = sorted(eligible)[: cfg.max_stocks]

    coverage = coverage_stats(members, eligible)
    coverage["asof"] = str(asof.date())

    jobs = []
    for sym in eligible:
        window = past[[sym]].tail(cfg.lookback_bars)
        jobs.append(
            (
                sym,
                window,
                base_config,
                cfg.entries,
                cfg.exits,
                entry_book,
                exit_book,
                cfg.n_splits,
                cfg.test_ratio,
            )
        )

    results: List[Dict[str, Any]] = []
    if cfg.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as pool:
            for _sym, rec in pool.map(_select_worker, jobs, chunksize=1):
                if rec is not None:
                    results.append(rec)
    else:
        for job in jobs:
            _sym, rec = _select_worker(job)
            if rec is not None:
                results.append(rec)

    results.sort(key=lambda r: r["oos_robustness"], reverse=True)
    return results[: cfg.top_n], coverage


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


def simulate_forward_month(
    ohlcv: pd.DataFrame,
    selections: List[Dict[str, Any]],
    asof: pd.Timestamp,
    month_end: pd.Timestamp,
    cfg: MonthlyHarnessConfig,
    base_config: Dict[str, Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Trade (asof, month_end] with the frozen selections.

    Signals are generated on [asof - warmup, month_end] so indicators are warm,
    then entries are masked to the forward month. Returns the month's daily
    portfolio returns plus diagnostics.
    """
    month_index = ohlcv.index[(ohlcv.index > asof) & (ohlcv.index <= month_end)]
    if len(month_index) == 0 or not selections:
        return pd.Series(dtype=float), {"n_positions": 0, "n_trades": 0}

    all_entries, all_exits, all_close = [], [], []
    for sel in selections:
        sym = sel["symbol"]
        if sym not in ohlcv.columns.get_level_values(0):
            continue
        window = ohlcv[[sym]].loc[:month_end].tail(cfg.warmup_bars + len(month_index))
        sig_cfg = {
            **base_config,
            "ENTRY_STRATEGY": sel["entry"],
            "EXIT_STRATEGY": sel["exit"],
        }
        try:
            engine = FastBacktest(window, sel["params"], config=sig_cfg)
            engine.run(show_progress=False)
        except Exception as exc:
            print(f"    [{sym}] forward signal generation failed: {exc!r}")
            continue
        entries = engine.entries.droplevel("param_combo", axis=1)
        exits = engine.exits.droplevel("param_combo", axis=1)
        entries.loc[entries.index <= asof] = False  # no pre-month positions
        all_entries.append(entries)
        all_exits.append(exits)
        all_close.append(window.xs("close", axis=1, level=1, drop_level=True))

    if not all_entries:
        return pd.Series(dtype=float), {"n_positions": 0, "n_trades": 0}

    entries_df = pd.concat(all_entries, axis=1).fillna(False)
    exits_df = pd.concat(all_exits, axis=1).fillna(False)
    close_df = pd.concat(all_close, axis=1)

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_df,
        exits=exits_df,
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        size=cfg.max_position_pct,
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(entries_df.shape[1], 0),
    ).copy()

    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    month_returns = returns.loc[(returns.index > asof) & (returns.index <= month_end)]

    dur = np.array(pf.trades.duration.values, dtype=np.float64, copy=True)
    diags = {
        "n_positions": int(len(selections)),
        "n_trades": int(pf.trades.count().sum()),
        "avg_holding_days": float(dur.mean()) if dur.size else None,
        "month_return_pct": float((1.0 + month_returns).prod() - 1.0) * 100,
    }
    return month_returns, diags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _month_end_selection_dates(
    index: pd.DatetimeIndex, eval_start: pd.Timestamp, eval_end: pd.Timestamp
) -> List[pd.Timestamp]:
    """Last trading day of each month in [eval_start, eval_end)."""
    idx = index[(index >= eval_start) & (index <= eval_end)]
    if len(idx) == 0:
        return []
    series = pd.Series(idx, index=idx)
    month_ends = series.groupby(idx.tz_localize(None).to_period("M")).max().tolist()
    # The final month end is the *end* of the eval span; selecting there would
    # leave no forward month to trade.
    return month_ends[:-1]


def stitch_equity_curve(month_returns: List[pd.Series], start_cash: float) -> pd.Series:
    rets = pd.concat([r for r in month_returns if len(r)]).sort_index()
    rets = rets[~rets.index.duplicated(keep="first")]
    return start_cash * (1.0 + rets).cumprod()


def benchmark_vs_spy(equity: pd.Series, spy_close: pd.Series, start_cash: float) -> Dict[str, Any]:
    """Strategy vs SPY buy-and-hold over the stitched span."""

    def _curve_stats(curve: pd.Series) -> Dict[str, float]:
        rets = curve.pct_change().dropna()
        years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
        total = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
        ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = (
            float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if rets.std() > 0
            else float("nan")
        )
        downside = rets[rets < 0]
        sortino = (
            float(rets.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if len(downside) and downside.std() > 0
            else float("nan")
        )
        dd = float((curve / curve.cummax() - 1.0).min())
        return {
            "total_return_pct": total * 100,
            "cagr_pct": ((1 + total) ** (1 / years) - 1) * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "ann_vol_pct": ann_vol * 100,
            "max_drawdown_pct": dd * 100,
        }

    spy = spy_close.reindex(equity.index).ffill().dropna()
    spy_curve = start_cash * (spy / spy.iloc[0])
    eq = equity.reindex(spy_curve.index)

    strat_m = eq.resample("ME").last().pct_change().dropna()
    spy_m = spy_curve.resample("ME").last().pct_change().dropna()
    common = strat_m.index.intersection(spy_m.index)
    hit_rate = float((strat_m.loc[common] > spy_m.loc[common]).mean()) if len(common) else None

    return {
        "strategy": _curve_stats(eq.dropna()),
        "spy": _curve_stats(spy_curve),
        "monthly_hit_rate_vs_spy": hit_rate,
        "n_months": int(len(common)),
    }


def selection_turnover(selections_by_month: List[List[Dict[str, Any]]]) -> List[float]:
    """Fraction of the book replaced at each re-selection."""
    out: List[float] = []
    prev: Optional[set] = None
    for sels in selections_by_month:
        cur = {s["symbol"] for s in sels}
        if prev is not None and cur:
            out.append(1.0 - len(cur & prev) / max(len(cur), 1))
        prev = cur
    return out


def run_monthly_walkforward(
    cfg: MonthlyHarnessConfig, base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    base_config = {**STOCK_BASE_CONFIG, **(base_config or {})}
    entry_book, exit_book = grid_books(cfg.grid_book)
    run_dir = Path(cfg.checkpoint_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_start = pd.Timestamp(cfg.eval_start, tz="UTC")
    eval_end = (
        pd.Timestamp(cfg.eval_end, tz="UTC")
        if cfg.eval_end
        else pd.Timestamp.now(tz="UTC").normalize()
    )

    # Data span: lookback before eval_start (in calendar days, ~1.5x trading days).
    data_start = eval_start - pd.Timedelta(days=int(cfg.lookback_bars * 1.6) + 30)
    universe = all_members_between(eval_start, eval_end)
    universe_yf = sorted({normalize_yf_ticker(t) for t in universe})
    print(
        f"Universe: {len(universe_yf)} point-in-time members over "
        f"{eval_start.date()} -> {eval_end.date()}; data from {data_start.date()}"
    )
    ohlcv = fetch_stock_ohlcv(
        universe_yf + ["SPY"], start=str(data_start.date()), end=str(eval_end.date())
    )
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]

    sel_dates = _month_end_selection_dates(ohlcv.index, eval_start, eval_end)
    print(
        f"{len(sel_dates)} monthly selection dates: {sel_dates[0].date()} -> {sel_dates[-1].date()}"
    )

    month_returns: List[pd.Series] = []
    selections_by_month: List[List[Dict[str, Any]]] = []
    active_selections: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, asof in enumerate(sel_dates):
        month_end = sel_dates[i + 1] if i + 1 < len(sel_dates) else ohlcv.index[-1]
        tag = str(asof.date())[:7]
        month_dir = run_dir / f"month={tag}"
        ret_path = month_dir / "month_returns.parquet"
        sel_path = month_dir / "selections.json"

        if ret_path.exists() and sel_path.exists():
            month_returns.append(pd.read_parquet(ret_path)["ret"])
            cached = json.loads(sel_path.read_text())
            selections_by_month.append(cached["selections"])
            active_selections = cached["selections"]
            print(f"[{i + 1}/{len(sel_dates)}] {tag}: checkpoint found, skipping")
            continue

        refit = (i % max(cfg.refit_every_n_months, 1) == 0) or not active_selections
        if refit:
            selections, coverage = select_for_month(
                asof, ohlcv, cfg, base_config, entry_book, exit_book
            )
            active_selections = selections
        else:
            selections, coverage = active_selections, {"reused_previous_selection": True}

        rets, diags = simulate_forward_month(ohlcv, selections, asof, month_end, cfg, base_config)

        month_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ret": rets}).to_parquet(ret_path)
        sel_path.write_text(
            json.dumps(
                {"asof": str(asof.date()), "refit": refit, "selections": selections},
                indent=1,
                default=str,
            )
        )
        (month_dir / "coverage.json").write_text(
            json.dumps({**coverage, "diagnostics": diags}, indent=1, default=str)
        )

        month_returns.append(rets)
        selections_by_month.append(selections)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(sel_dates) - i - 1)
        print(
            f"[{i + 1}/{len(sel_dates)}] {tag}: {len(selections)} stocks, "
            f"month_ret={diags.get('month_return_pct', float('nan')):+.2f}% "
            f"trades={diags.get('n_trades', 0)} "
            f"(elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.0f}m)"
        )

    equity = stitch_equity_curve(month_returns, float(base_config["START_CASH"]))
    if equity.empty:
        raise RuntimeError("No month produced any returns — nothing to report.")
    report = benchmark_vs_spy(equity, spy_close, float(base_config["START_CASH"]))

    turnover = selection_turnover(selections_by_month)
    holds = [
        s.get("avg_holding_days")
        for sels in selections_by_month
        for s in sels
        if s.get("avg_holding_days") is not None
    ]
    combo_counts: Dict[str, int] = {}
    for sels in selections_by_month:
        for s in sels:
            key = f"{s['entry']}+{s['exit']}"
            combo_counts[key] = combo_counts.get(key, 0) + 1

    summary = {
        "config": {**cfg.__dict__},
        "report": report,
        "avg_monthly_turnover": float(np.mean(turnover)) if turnover else None,
        "holding_days": {
            "p25": float(np.percentile(holds, 25)) if holds else None,
            "median": float(np.percentile(holds, 50)) if holds else None,
            "p75": float(np.percentile(holds, 75)) if holds else None,
        },
        "combo_selection_counts": dict(
            sorted(combo_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "n_months": len(sel_dates),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    equity.to_frame("equity").to_parquet(run_dir / "equity_curve.parquet")
    print(f"\nSummary written to {run_dir / 'summary.json'}")
    return summary


def leak_check(cfg: MonthlyHarnessConfig, base_config: Optional[Dict[str, Any]] = None) -> bool:
    """Selections at T must be identical with and without post-T data loaded."""
    base_config = {**STOCK_BASE_CONFIG, **(base_config or {})}
    entry_book, exit_book = grid_books(cfg.grid_book)

    eval_start = pd.Timestamp(cfg.eval_start, tz="UTC")
    eval_end = (
        pd.Timestamp(cfg.eval_end, tz="UTC")
        if cfg.eval_end
        else pd.Timestamp.now(tz="UTC").normalize()
    )
    data_start = eval_start - pd.Timedelta(days=int(cfg.lookback_bars * 1.6) + 30)
    universe = sorted({normalize_yf_ticker(t) for t in all_members_between(eval_start, eval_end)})
    ohlcv = fetch_stock_ohlcv(universe, start=str(data_start.date()), end=str(eval_end.date()))

    sel_dates = _month_end_selection_dates(ohlcv.index, eval_start, eval_end)
    asof = sel_dates[len(sel_dates) // 2]
    print(f"Leak check at {asof.date()}...")

    full, _ = select_for_month(asof, ohlcv, cfg, base_config, entry_book, exit_book)
    truncated, _ = select_for_month(asof, ohlcv.loc[:asof], cfg, base_config, entry_book, exit_book)

    ok = json.dumps(full, sort_keys=True, default=str) == json.dumps(
        truncated, sort_keys=True, default=str
    )
    print("LEAK CHECK:", "PASS — selections identical" if ok else "FAIL — selections differ!")
    return ok
