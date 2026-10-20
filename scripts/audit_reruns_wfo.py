"""Re-runs on the corrected harness for xs_momentum, dual_momentum, and fomc_drift.

Production replica audit Task 8:
- xs_momentum and dual_momentum: pinned PIT sp500, 17 folds, run_core on 2021-01-31 -> 2026-04-30
  with excess_stats against SPY. Pre-tax pass requires t >= 3.0 and CAGR diff >= 1.0 pp.
- fomc_drift: walk-forward optimization on 2012-01-01 -> 2026-04-30, universe fomc_treasury,
  plus event study on the corrected return attribution mask. Pass requires net t >= 3.0 on
  (entry_offset_days=1, hold_days=0) and aggregate WFE >= 0.50.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from anchor_fix_reproduction_wfo import run_core  # noqa: E402

from ggTrader.lab.data import (  # noqa: E402
    STOCK_BASE_CONFIG,
    eligible_at,
    equity_universe_between,
    load_ohlcv,
)
from ggTrader.lab.event_study import (  # noqa: E402
    event_study_test,
    in_position_mask,
    run_event_study_from_signals,
    split_event_returns,
)
from ggTrader.lab.metrics import curve_stats  # noqa: E402
from ggTrader.lab.significance import (  # noqa: E402
    CAGR_BAR_PP,
    NW_LAGS,
    T_BAR,
    excess_stats,
    passes_pretax_bar,
)
from ggTrader.lab.strategies import STRATEGY_REGISTRY  # noqa: E402
from ggTrader.lab.strategies.fomc_drift import (  # noqa: E402
    FomcDriftStrategy,
    _event_signals,
)
from ggTrader.lab.strategies.indicators import extract_close  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from ggTrader.lab.sweep import build_grid  # noqa: E402
from ggTrader.lab.wfo import run_wfo  # noqa: E402

MOMENTUM_EVAL_START = "2021-01-31"
MOMENTUM_EVAL_END = "2026-04-30"

FOMC_EVAL_START = "2012-01-01"
FOMC_EVAL_END = "2026-04-30"
FOMC_WARMUP_DAYS = 420
FOMC_UNIVERSE = "fomc_treasury"
FOMC_ROUNDTRIP_COST = 0.0010
FOMC_WFE_TARGET = 0.50

STRATEGY_CHOICES = ["xs_momentum", "dual_momentum", "fomc_drift"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-runs on the corrected harness for xs_momentum, dual_momentum, and fomc_drift."
        )
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["xs_momentum", "dual_momentum", "fomc_drift"],
        choices=STRATEGY_CHOICES,
        help="Strategies to re-run (default: xs_momentum dual_momentum fomc_drift).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (default: docs/research/_audit_reruns_<YYYYMMDD>.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-running strategies even if already recorded in output JSON.",
    )
    return parser


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert data to strict JSON-compliant primitives:
    - NaN, Inf, -Inf -> None (serialized as null, no invented 0)
    - numpy scalars/arrays -> Python int, float, bool, list
    - dict keys -> str
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write data as strict JSON using a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_data = _sanitize_for_json(data)
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, allow_nan=False)
        f.write("\n")
    tmp_path.replace(path)


def run_momentum_rerun(strategy_name: str) -> dict[str, Any]:
    """Run a momentum strategy (xs_momentum or dual_momentum) on the corrected harness."""
    t0 = time.time()
    core_result = run_core(
        MOMENTUM_EVAL_START,
        MOMENTUM_EVAL_END,
        strategy=strategy_name,
        return_equity=True,
    )
    if not core_result.get("ok", False):
        core_result.pop("oos_equity", None)
        core_result.pop("spy_close", None)
        core_result["verdict"] = "ERROR"
        return core_result

    # Must pop both Series before any JSON dump
    oos_equity: pd.Series = core_result.pop("oos_equity")
    spy_close: pd.Series = core_result.pop("spy_close")

    spy_common = spy_close.reindex(oos_equity.index).ffill().dropna()
    start_cash = float(oos_equity.iloc[0])
    spy_curve = start_cash * (spy_common / spy_common.iloc[0])

    stats = excess_stats(oos_equity, spy_curve, lags=NW_LAGS)
    passes = passes_pretax_bar(stats, t_bar=T_BAR, cagr_bar_pp=CAGR_BAR_PP)

    verdict = "pretax pass — needs its own after-tax brief" if passes else "NO-GO stands"

    fold_rows = core_result.get("fold_rows", [])
    fold_winners = [r["winner_combo"] for r in fold_rows]
    dsr_gate_passes = sum(1 for r in fold_rows if r.get("dsr_passed"))
    n_gate_pass = core_result.get("n_gate_pass", sum(1 for r in fold_rows if r.get("gates_passed")))
    n_folds = len(fold_rows)

    return {
        "strategy": strategy_name,
        "eval_start": MOMENTUM_EVAL_START,
        "eval_end": MOMENTUM_EVAL_END,
        "verdict": verdict,
        "passes_pretax_bar": passes,
        "excess_stats": {
            "n_obs": stats.n_obs,
            "t_stat": stats.t_stat,
            "ann_excess_pct": stats.ann_excess_pct,
            "tracking_error_pct": stats.tracking_error_pct,
            "information_ratio": stats.information_ratio,
            "cagr_diff_pp": stats.cagr_diff_pp,
            "t_bar": T_BAR,
            "cagr_bar_pp": CAGR_BAR_PP,
            "nw_lags": NW_LAGS,
        },
        "oos_metrics": {
            "sharpe": core_result.get("sharpe"),
            "cagr_pct": core_result.get("cagr_pct"),
            "max_drawdown_pct": core_result.get("max_drawdown_pct"),
            "spy_sharpe": core_result.get("spy_sharpe"),
        },
        "fold_winners": fold_winners,
        "dsr_gate_passes": dsr_gate_passes,
        "n_gate_pass": n_gate_pass,
        "n_folds": n_folds,
        "gate_pass_str": core_result.get("gate_pass_str"),
        "fold_rows": fold_rows,
        "elapsed_sec": time.time() - t0,
        "ok": True,
    }


def run_fomc_drift_rerun() -> dict[str, Any]:
    """Run fomc_drift walk-forward and event study on the corrected tape and mask."""
    t0 = time.time()
    cfg = LabConfig()
    es = pd.Timestamp(FOMC_EVAL_START, tz="UTC")
    ee = pd.Timestamp(FOMC_EVAL_END, tz="UTC")
    data_start = str((es - pd.Timedelta(days=FOMC_WARMUP_DAYS)).date())

    universe = equity_universe_between(es, ee, universe=FOMC_UNIVERSE)
    print(f"  [universe] {FOMC_UNIVERSE}: {len(universe)} symbols", flush=True)
    ohlcv = load_ohlcv(universe + ["SPY"], data_start, FOMC_EVAL_END, use_negative_cache=True)
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv_stocks = ohlcv[sym_cols]

    strategy_cls = STRATEGY_REGISTRY["fomc_drift"]
    grid = build_grid(strategy_cls)
    print(f"WFO: fomc_drift | {len(grid)} param combos", flush=True)

    wfo_result = run_wfo(
        "fomc_drift",
        strategy_cls,
        cfg,
        ohlcv_stocks,
        spy_close,
        eval_start=FOMC_EVAL_START,
        eval_end=FOMC_EVAL_END,
        market="stock",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
        universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe=FOMC_UNIVERSE)[0],
    )

    valid_wfes = [r["wfe"] for r in wfo_result.fold_results if r.get("wfe") is not None]
    avg_wfe: float | None = (
        float(np.mean(valid_wfes)) if valid_wfes and np.isfinite(valid_wfes).all() else None
    )

    fold_rows = []
    for r in wfo_result.fold_results:
        fold_rows.append(
            {
                "fold_num": r["fold_num"],
                "train_start": str(pd.Timestamp(r["train_start"]).date()),
                "train_end": str(pd.Timestamp(r["train_end"]).date()),
                "test_start": str(pd.Timestamp(r["test_start"]).date()),
                "test_end": str(pd.Timestamp(r["test_end"]).date()),
                "winner_combo": r.get("winner_combo"),
                "winner_params": r.get("winner_params"),
                "ndh_passed": bool(r.get("ndh_passed", False)),
                "dsr_passed": bool(r.get("dsr_passed", False)),
                "gates_passed": bool(r.get("gates_passed", False)),
                "wfe": r.get("wfe"),
                "oos_sharpe": r.get("oos_sharpe"),
                "used_anchor": bool(r.get("used_anchor", False)),
            }
        )

    fold_winners = [r["winner_combo"] for r in fold_rows]
    dsr_gate_passes = sum(1 for r in fold_rows if r["dsr_passed"])
    n_gate_pass = sum(1 for r in fold_rows if r["gates_passed"])
    oos_metrics = curve_stats(wfo_result.oos_equity) if len(wfo_result.oos_equity) > 0 else {}

    # Event study on the full universe, restricted strictly to registered calendar bounds
    # [FOMC_EVAL_START, FOMC_EVAL_END] inclusive, preserving the 16:00 UTC last session
    symbols = sorted(
        [s for s in universe if s in ohlcv_stocks.columns.get_level_values(0).unique()]
    )
    close = extract_close(ohlcv_stocks, symbols)
    strat = FomcDriftStrategy(cfg)
    events = strat._load_events()

    eval_start_d = pd.Timestamp(FOMC_EVAL_START).date()
    eval_end_d = pd.Timestamp(FOMC_EVAL_END).date()
    if hasattr(close.index, "tz") and close.index.tz is not None:
        idx_dates = close.index.tz_convert("UTC").date
    else:
        idx_dates = pd.to_datetime(close.index).date
    eval_mask = (idx_dates >= eval_start_d) & (idx_dates <= eval_end_d)
    close_eval = close.loc[eval_mask]

    pair_results: dict[str, Any] = {}
    for combo in grid:
        entry_offset = int(combo["entry_offset_days"])
        hold = int(combo["hold_days"])
        signals = _event_signals(close_eval.index, symbols, events, entry_offset, hold)
        gross_res = run_event_study_from_signals(close_eval, signals.entries, signals.exits)

        # Net: subtract 0.0010 from the return of each event's first in-position day
        mask = in_position_mask(signals.entries, signals.exits)
        first_held_mask = mask & (~mask.shift(1, fill_value=False))
        returns_eval = close_eval.pct_change()
        net_returns = returns_eval.copy()
        net_returns[first_held_mask] = net_returns[first_held_mask] - FOMC_ROUNDTRIP_COST
        event_net_returns, nonevent_net_returns = split_event_returns(net_returns, mask)
        net_res = event_study_test(event_net_returns, nonevent_net_returns)

        pair_key = f"entry_{entry_offset}_hold_{hold}"
        pair_results[pair_key] = {
            "entry_offset_days": entry_offset,
            "hold_days": hold,
            "gross": gross_res._asdict(),
            "net": net_res._asdict(),
        }

    # Primary pair: (entry_offset_days=1, hold_days=0)
    primary_pair_key = "entry_1_hold_0"
    primary_pair = pair_results[primary_pair_key]
    net_t = primary_pair["net"]["t_stat"]
    passes_fomc = bool(net_t >= T_BAR and avg_wfe is not None and avg_wfe >= FOMC_WFE_TARGET)
    verdict = "pretax pass — needs its own after-tax brief" if passes_fomc else "NO-GO stands"

    context_pairs = {k: v for k, v in pair_results.items() if k != primary_pair_key}

    return {
        "strategy": "fomc_drift",
        "eval_start": FOMC_EVAL_START,
        "eval_end": FOMC_EVAL_END,
        "verdict": verdict,
        "passes_bar": passes_fomc,
        "primary_pair": primary_pair_key,
        "primary_event_study": primary_pair,
        "context_pairs": context_pairs,
        "wfo": {
            "aggregate_wfe": avg_wfe,
            "wfe_target": FOMC_WFE_TARGET,
            "n_folds": len(fold_rows),
            "n_gate_pass": n_gate_pass,
            "dsr_gate_passes": dsr_gate_passes,
            "fold_winners": fold_winners,
            "oos_metrics": oos_metrics,
            "fold_rows": fold_rows,
            "table": wfo_result.table,
        },
        "elapsed_sec": time.time() - t0,
        "ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for audit re-runs driver."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    output_path = args.output or (
        REPO_ROOT / "docs" / "research" / f"_audit_reruns_{today_str}.json"
    )

    results: dict[str, Any] = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded existing checkpoint with {len(results)} strategies from {output_path}")
        except Exception as exc:
            print(
                f"Error: checkpoint exists at {output_path} but could not be parsed: {exc}",
                file=sys.stderr,
            )
            return 1

    any_failed = False
    for strat in args.strategies:
        if strat in results and not args.force:
            if results[strat].get("ok", False):
                print(
                    f"Skipping {strat}: already recorded in {output_path} (use --force to re-run)"
                )
                continue
            else:
                print(f"Retrying {strat}: previous run recorded error in {output_path}")

        print(f"\n=== Running {strat} ===")
        try:
            if strat in ("xs_momentum", "dual_momentum"):
                strat_res = run_momentum_rerun(strat)
            elif strat == "fomc_drift":
                strat_res = run_fomc_drift_rerun()
            else:
                raise ValueError(f"Unsupported strategy: {strat}")

            # Ensure pd.Series are popped before saving
            strat_res.pop("oos_equity", None)
            strat_res.pop("spy_close", None)

            results[strat] = strat_res
            _atomic_write_json(output_path, results)
            print(
                f"Checkpoint saved: {strat} -> {output_path} (verdict: {strat_res.get('verdict')})"
            )
            if not strat_res.get("ok", False):
                any_failed = True
        except Exception as exc:
            traceback.print_exc()
            results[strat] = {
                "strategy": strat,
                "ok": False,
                "verdict": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            _atomic_write_json(output_path, results)
            print(f"Checkpoint saved (error): {strat} -> {output_path}")
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
