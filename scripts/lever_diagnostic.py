#!/usr/bin/env python
"""Lever-selection diagnostic: compare three CAGR levers on the 17-fold WFO.

Decision gate for the regime-gated-exposure feature. Compares:
  - exposure scalar (flat): SIGNAL_POSITION_SIZE in {0.02, 0.03, 0.04, 0.05}
  - min_agree: grid override min_agree in {2, 3}
  - vol-target: vol_target in {0.15, 0.20, 0.25} with vol_cap=2.0

Uses the 3-way parallel ProcessPoolExecutor(fork) pattern from ablation_voters.py.
After the sweep, regime-conditions the best exposure setting's OOS returns via
classify_regime to check whether payoff is regime-dependent.

Run:
    source .venv/bin/activate
    python scripts/lever_diagnostic.py 2>&1 | tee lever_diagnostic.log
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ggTrader.lab.data import (
    DEFAULT_UNIVERSE,
    STOCK_BASE_CONFIG,
    equity_universe_between,
    load_ohlcv,
)
from ggTrader.lab.regime import classify_regime
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid, combo_name, split_params
from ggTrader.lab.wfo import run_wfo

# parse_table lives in a sibling script — add scripts/ to path if needed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ablation_voters import parse_table  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────
EVAL_START = "2021-01-31"
EVAL_END = None  # -> now
UNIVERSE = DEFAULT_UNIVERSE
WORKERS = 2  # 3 workers OOM'd on 32GB box; 2 is safe (~11GB per worker)
LOG_DIR = "lever_diagnostic_logs"

# Module globals for fork-inherited copy-on-write.
_OHLCV: pd.DataFrame | None = None
_SPY: pd.Series | None = None
_CFG: LabConfig | None = None
_BASE_GRID: List[Dict[str, Any]] | None = None
_EVAL_START: str = ""
_EVAL_END: str = ""


# ── Lever families ────────────────────────────────────────────────────
def _build_lever_configs() -> List[Tuple[str, Dict[str, Any], Dict[str, list] | None]]:
    """Return (label, base_config_overrides, grid_overrides) per lever setting.

    - exposure scalar: override base_config SIGNAL_POSITION_SIZE
    - min_agree: override the grid's min_agree axis
    - vol-target: override base_config vol_target + vol_cap
    """
    configs: List[Tuple[str, Dict[str, Any], Dict[str, list] | None]] = []

    # Exposure scalar family
    for size in (0.02, 0.03, 0.04, 0.05):
        configs.append(
            (
                f"exposure_{size:.2f}",
                {"SIGNAL_POSITION_SIZE": size},
                None,
            )
        )

    # min_agree family (grid override — force a single min_agree value)
    for ma in (2, 3):
        configs.append(
            (
                f"min_agree_{ma}",
                {},
                {"min_agree": [ma]},
            )
        )

    # vol-target family
    for vt in (0.15, 0.20, 0.25):
        configs.append(
            (
                f"vol_target_{vt:.2f}",
                {"vol_target": vt, "vol_cap": 2.0},
                None,
            )
        )

    return configs


# ── Worker (runs in a forked child) ──────────────────────────────────
def _run_lever(
    label: str,
    bc_overrides: Dict[str, Any],
    grid_overrides: Dict[str, list] | None,
) -> Dict[str, Any]:
    """Run one lever setting's full WFO. Reads inherited globals."""
    assert _OHLCV is not None and _SPY is not None
    assert _CFG is not None and _BASE_GRID is not None

    # Apply base_config overrides
    bc = {**dict(STOCK_BASE_CONFIG), **bc_overrides}

    # Apply grid overrides
    if grid_overrides:
        grid = build_grid(EnsembleSignal, overrides=grid_overrides)
    else:
        grid = list(_BASE_GRID)

    log_path = os.path.join(LOG_DIR, f"{label}.log")
    t0 = time.time()
    with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
        print(f"=== {label}  overrides={bc_overrides}  grid_ov={grid_overrides} ===", flush=True)
        result = run_wfo(
            "ensemble",
            EnsembleSignal,
            _CFG,
            _OHLCV,
            _SPY,
            eval_start=_EVAL_START,
            eval_end=_EVAL_END,
            market="equity",
            base_config=bc,
            grid=grid,
        )
    elapsed = time.time() - t0
    parsed = parse_table(result.table)
    parsed.update(
        {
            "label": label,
            "bc_overrides": str(bc_overrides),
            "grid_overrides": str(grid_overrides),
            "n_combos": len(grid),
            "elapsed_s": round(elapsed, 1),
        }
    )
    return parsed


# ── Regime-conditioning breakdown ────────────────────────────────────
def _regime_breakdown(
    bc_overrides: Dict[str, Any],
) -> pd.DataFrame:
    """Reconstruct OOS daily returns for the best exposure setting and
    break down by regime label.

    This re-runs a lightweight single-config WFO fold-by-fold to extract
    the concatenated OOS equity curve, then joins with classify_regime.
    """
    assert _OHLCV is not None and _SPY is not None and _CFG is not None
    from ggTrader.lab.wfo import _sweep_fold, composite_score, compute_anchor_set, generate_folds

    bc = {**dict(STOCK_BASE_CONFIG), **bc_overrides}
    grid = list(_BASE_GRID)
    eval_start_ts = pd.Timestamp(_EVAL_START, tz="UTC")
    eval_end_ts = pd.Timestamp(_EVAL_END, tz="UTC")
    folds = generate_folds(eval_start_ts, eval_end_ts)

    strat = EnsembleSignal(_CFG)
    start_cash = float(bc["START_CASH"])

    # Compute anchor
    anchor = compute_anchor_set("ensemble", EnsembleSignal, _CFG, _OHLCV, bc, grid)

    oos_curves: List[pd.Series] = []
    oos_running_value = start_cash

    from ggTrader.lab.gates import dsr_check, ndh_check
    from ggTrader.lab.wfo import (
        REGIME_PARAMS,
        WfoState,
        _compute_dsr_inputs,
        _extract_grid_arrays,
        check_circuit_breaker,
        check_shadow_reentry,
        compute_wfe,
    )

    wfo_state = WfoState()

    for i, fold in enumerate(folds):
        train_metrics, train_eq = _sweep_fold(
            "ensemble",
            strat,
            _OHLCV,
            fold.train_start,
            fold.train_end,
            bc,
            grid,
        )
        if not train_metrics:
            continue

        scores = composite_score(train_metrics)
        best_idx = max(range(len(scores)), key=lambda j: scores[j])
        winner = train_metrics[best_idx]

        # Gate checks (simplified — mirror run_wfo logic)
        ndh_passed = dsr_passed = False
        sharpe_grid, exp_grid, grid_shape, r2g = _extract_grid_arrays(
            train_metrics,
            grid,
            "ensemble",
        )
        if best_idx in r2g and len(grid_shape) > 0:
            param_keys = sorted(grid[0].keys())
            neighbor_axes = tuple(idx for idx, k in enumerate(param_keys) if k not in REGIME_PARAMS)
            ndh_result = ndh_check(
                peak_idx=r2g[best_idx],
                sharpe_grid=sharpe_grid,
                expectancy_grid=exp_grid,
                grid_shape=grid_shape,
                density_threshold=0.85,
                neighbor_axes=neighbor_axes,
            )
            ndh_passed = ndh_result.passed

        winner_key = winner["combo"]
        eq_is = train_eq.get(winner_key, pd.Series(dtype=float))
        eq_is = eq_is.loc[fold.train_start : fold.train_end].dropna()
        if len(eq_is) > 10:
            n_obs, skew_val, kurt_val = _compute_dsr_inputs(eq_is)
            dsr_result = dsr_check(
                observed_sr=winner.get("sharpe", 0.0),
                n_obs=n_obs,
                n_trials=len(grid),
                skew=skew_val,
                kurtosis_excess=kurt_val,
                threshold=0.80,
            )
            dsr_passed = dsr_result.passed

        gates_passed = ndh_passed and dsr_passed
        deploy_params = winner["params"]
        if not gates_passed or wfo_state.halted:
            deploy_params = anchor.params

        # Test fold — extract OOS equity
        test_metrics, _test_eq = _sweep_fold(
            "ensemble",
            strat,
            _OHLCV,
            fold.test_start,
            fold.test_end,
            bc,
            [deploy_params],
        )
        if test_metrics:
            test_ohlcv = _OHLCV.loc[: fold.test_end]
            symbols = sorted(test_ohlcv.columns.get_level_values(0).unique())
            prices = pd.concat({s: test_ohlcv[s]["close"] for s in symbols}, axis=1)
            signal_combos = [split_params(deploy_params)[0]]
            _, stop_p = split_params(deploy_params)
            targets = strat.sweep_signals(signal_combos, symbols, test_ohlcv)
            key = combo_name("ensemble", signal_combos[0])
            full_key = combo_name("ensemble", deploy_params)
            sim_config = {**bc, **stop_p}
            ohlcv_arg = test_ohlcv if "atr_mult" in stop_p else None
            _r, eq, _d = simulate_signals(
                {full_key: targets[key]},
                prices,
                sim_config,
                ohlcv=ohlcv_arg,
            )
            eq_test = eq[full_key].loc[fold.test_start : fold.test_end].dropna()
            if len(eq_test) > 0:
                normalized = oos_running_value * (eq_test / eq_test.iloc[0])
                oos_curves.append(normalized)
                oos_running_value = float(normalized.iloc[-1])

        # Update circuit breaker state
        is_sharpe = winner.get("sharpe", float("nan"))
        oos_sharpe = float("nan")
        if test_metrics:
            oos_sharpe = test_metrics[0].get("sharpe", float("nan"))
        wfe_val = compute_wfe(is_sharpe, oos_sharpe)
        wfo_state.wfe_history.append(wfe_val)
        if np.isfinite(oos_sharpe):
            wfo_state.oos_sharpes.append(oos_sharpe)
        if not wfo_state.halted:
            wfo_state = check_circuit_breaker(wfo_state)
        else:
            wfo_state = check_shadow_reentry(wfo_state, ndh_passed, dsr_passed, wfe_val)

    if not oos_curves:
        print("[regime] No OOS curves to analyze", flush=True)
        return pd.DataFrame()

    oos_equity = pd.concat(oos_curves)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
    oos_returns = oos_equity.pct_change(fill_method=None).dropna()

    # Regime conditioning
    reg = classify_regime(_SPY)["label"].reindex(oos_returns.index).ffill()
    by_regime = oos_returns.groupby(reg).agg(["mean", "count"])
    return by_regime


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    cfg = LabConfig()
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = (
        pd.Timestamp(EVAL_END, tz="UTC") if EVAL_END else pd.Timestamp.now(tz="UTC").normalize()
    )

    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    universe = equity_universe_between(eval_start, eval_end, universe=UNIVERSE)
    print(f"[universe] {UNIVERSE}: {len(universe)} symbols", flush=True)

    print("[data] loading OHLCV (one-time)...", flush=True)
    ohlcv = load_ohlcv(
        universe + ["SPY"],
        str(data_start.date()),
        str(eval_end.date()),
        use_negative_cache=True,
    )
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]

    # Publish to module globals for fork-inherited copy-on-write
    global _OHLCV, _SPY, _CFG, _BASE_GRID, _EVAL_START, _EVAL_END
    _OHLCV = ohlcv
    _SPY = spy_close
    _CFG = cfg
    _BASE_GRID = build_grid(EnsembleSignal)
    _EVAL_START = str(eval_start.date())
    _EVAL_END = str(eval_end.date())

    lever_configs = _build_lever_configs()
    os.makedirs(LOG_DIR, exist_ok=True)
    print(
        f"[lever diagnostic] {len(lever_configs)} configs, {WORKERS} workers "
        f"(per-config logs in {LOG_DIR}/)\n",
        flush=True,
    )

    # Run all lever settings in parallel (fork for copy-on-write OHLCV)
    ctx = mp.get_context("fork")
    by_label: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        futures = {
            pool.submit(_run_lever, label, bc_ov, grid_ov): label
            for label, bc_ov, grid_ov in lever_configs
        }
        done = 0
        for fut in as_completed(futures):
            parsed = fut.result()
            by_label[parsed["label"]] = parsed
            done += 1
            print(
                f"[{done}/{len(lever_configs)}] {parsed['label']}: "
                f"OOS Sharpe {parsed.get('oos_sharpe')} "
                f"CAGR {parsed.get('oos_cagr')}% "
                f"gate {parsed.get('gate_pass')}/{parsed.get('n_folds')} "
                f"({parsed.get('elapsed_s')}s)",
                flush=True,
            )

    # Re-order to match config order
    results = [by_label[label] for label, _, _ in lever_configs if label in by_label]
    _print_summary(results)

    # Step 2: Regime-conditioning breakdown for the best exposure setting
    print("\n" + "=" * 80)
    print("REGIME-CONDITIONING BREAKDOWN")
    print("=" * 80)

    # Find the best exposure setting by CAGR
    exposure_results = [r for r in results if r["label"].startswith("exposure_")]
    if exposure_results:
        best_exposure = max(
            exposure_results,
            key=lambda r: r.get("oos_cagr", float("-inf")),
        )
        print(
            f"\nBest exposure setting: {best_exposure['label']} "
            f"(CAGR {best_exposure.get('oos_cagr')}%)"
        )
        print("\nRe-running best exposure WFO to extract OOS equity for regime analysis...")

        # Parse the SIGNAL_POSITION_SIZE from the label
        size_str = best_exposure["label"].replace("exposure_", "")
        best_size = float(size_str)
        breakdown = _regime_breakdown({"SIGNAL_POSITION_SIZE": best_size})
        if not breakdown.empty:
            print(f"\nOOS daily returns by regime (exposure={best_size}):\n")
            print(breakdown.to_string())
        else:
            print("No OOS equity data available for regime breakdown.")
    else:
        print("No exposure results to analyze.")

    print("\n" + "=" * 80, flush=True)


def _print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("LEVER DIAGNOSTIC SUMMARY")
    print("=" * 100)
    print("\nBaseline reference: static 5-voter  CAGR 10.5% / Sharpe 0.89 / MaxDD -10.5%")
    print("SPY baseline:                       CAGR 13.0% / Sharpe 0.58 / MaxDD -22.1%\n")
    hdr = f"{'lever':<20}{'cmb':>5}{'OOS Sh':>8}{'CAGR%':>8}{'MaxDD%':>8}{'WFE':>6}{'gate':>7}"
    print(hdr)
    print("-" * 100)

    # Group by family
    for family in ("exposure", "min_agree", "vol_target"):
        family_results = [r for r in results if r["label"].startswith(family)]
        for r in family_results:
            print(
                f"{r['label']:<20}{r.get('n_combos', ''):>5}"
                f"{r.get('oos_sharpe', float('nan')):>8.2f}"
                f"{r.get('oos_cagr', float('nan')):>8.1f}"
                f"{r.get('oos_maxdd', float('nan')):>8.1f}"
                f"{r.get('wfe', float('nan')):>6.2f}"
                f"{str(r.get('gate_pass', '')) + '/' + str(r.get('n_folds', '')):>7}"
            )
        print("-" * 100)

    if results:
        spy = results[0]
        print(
            f"{'SPY baseline':<20}{'':>5}"
            f"{spy.get('spy_sharpe', float('nan')):>8.2f}"
            f"{spy.get('spy_cagr', float('nan')):>8.1f}"
            f"{spy.get('spy_maxdd', float('nan')):>8.1f}"
        )
    print("=" * 100, flush=True)


if __name__ == "__main__":
    sys.exit(main())
