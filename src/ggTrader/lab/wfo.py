"""Walk-forward optimization: rolling train/test folds with composite scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Type

import numpy as np
import pandas as pd

from ggTrader.lab.gates import dsr_check, ndh_check
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import combo_name, split_params

TRAIN_MONTHS = 12
TEST_MONTHS = 3


# ── WFE & Circuit Breaker ──────────────────────────────────────────────


def compute_wfe(
    is_sharpe: float,
    oos_sharpe: float,
    is_floor: float = 0.4,
) -> float | None:
    """Walk-Forward Efficiency: OOS Sharpe / IS Sharpe.

    Returns None for neutral windows (IS Sharpe < floor) — these are excluded
    from the rolling WFE average, not counted as zero.
    """
    if is_sharpe < is_floor:
        return None
    return oos_sharpe / is_sharpe


@dataclass
class WfoState:
    """Tracks WFE history and circuit breaker state across WFO folds."""

    wfe_history: List[float | None] = field(default_factory=list)
    oos_sharpes: List[float] = field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None
    shadow_strikes: int = 0


def check_circuit_breaker(
    state: WfoState,
    window_size: int = 4,
    wfe_floor: float = 0.25,
) -> WfoState:
    """Check the OR-gate circuit breaker.

    Triggers halt if EITHER:
    1. Chronic decay: trailing window_size non-None WFE average < wfe_floor
    2. Acute failure: two consecutive negative OOS Sharpe Ratios
    """
    new_state = WfoState(
        wfe_history=list(state.wfe_history),
        oos_sharpes=list(state.oos_sharpes),
        halted=state.halted,
        halt_reason=state.halt_reason,
        shadow_strikes=state.shadow_strikes,
    )

    # Chronic decay: trailing non-None WFE avg
    valid_wfes = [w for w in state.wfe_history if w is not None]
    recent = valid_wfes[-window_size:] if len(valid_wfes) >= window_size else valid_wfes
    if len(recent) >= window_size:
        avg_wfe = sum(recent) / len(recent)
        if avg_wfe < wfe_floor:
            new_state.halted = True
            new_state.halt_reason = (
                f"Chronic decay: trailing {window_size}-window WFE avg {avg_wfe:.3f} < {wfe_floor}"
            )
            return new_state

    # Acute failure: 2 consecutive negative OOS Sharpes
    if len(state.oos_sharpes) >= 2:
        if state.oos_sharpes[-1] < 0 and state.oos_sharpes[-2] < 0:
            new_state.halted = True
            new_state.halt_reason = (
                f"Acute failure: 2 consecutive negative OOS Sharpe"
                f" ({state.oos_sharpes[-2]:.2f}, {state.oos_sharpes[-1]:.2f})"
            )
            return new_state

    return new_state


def check_shadow_reentry(
    state: WfoState,
    ndh_passed: bool,
    dsr_passed: bool,
    wfe: float | None,
    wfe_healthy: float = 0.5,
) -> WfoState:
    """Check shadow re-entry: 2 consecutive clean windows restore live trading.

    A clean window requires all three: NDH pass, DSR pass, WFE >= wfe_healthy.
    """
    new_state = WfoState(
        wfe_history=list(state.wfe_history),
        oos_sharpes=list(state.oos_sharpes),
        halted=state.halted,
        halt_reason=state.halt_reason,
        shadow_strikes=state.shadow_strikes,
    )

    clean = ndh_passed and dsr_passed and wfe is not None and wfe >= wfe_healthy
    if clean:
        new_state.shadow_strikes += 1
        if new_state.shadow_strikes >= 2:
            new_state.halted = False
            new_state.halt_reason = None
            new_state.shadow_strikes = 0
    else:
        new_state.shadow_strikes = 0

    return new_state


class AnchorSet(NamedTuple):
    """Global min-drawdown parameter set for defensive fallback."""

    combo: str
    params: Dict[str, Any]
    max_drawdown_pct: float
    cagr_pct: float
    sharpe: float


class Fold(NamedTuple):
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> List[Fold]:
    """Rolling fixed-width folds. Slides forward by test_months each step."""
    folds: List[Fold] = []
    cursor = eval_start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > eval_end:
            break
        folds.append(Fold(cursor, train_end, train_end, test_end))
        cursor += pd.DateOffset(months=test_months)
    return folds


def _min_max_normalize(values: List[float]) -> List[float]:
    """Min-max scale to [0, 1]. Returns all 0.0 if min == max."""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def composite_score(metrics_list: List[Dict[str, float]]) -> List[float]:
    """Composite rank: 0.5*norm(sharpe) + 0.3*norm(sortino) - 0.2*norm(|maxdd|).

    NaN values are replaced with the worst value in each metric's range.
    """
    sharpes: List[float] = []
    sortinos: List[float] = []
    drawdowns: List[float] = []
    for m in metrics_list:
        sharpes.append(m.get("sharpe", float("nan")))
        sortinos.append(m.get("sortino", float("nan")))
        drawdowns.append(abs(m.get("max_drawdown_pct", 0.0)))

    def _floor_nan(vals: List[float]) -> List[float]:
        finite = [v for v in vals if not math.isnan(v)]
        floor = min(finite) if finite else 0.0
        return [floor if math.isnan(v) else v for v in vals]

    sharpes = _floor_nan(sharpes)
    sortinos = _floor_nan(sortinos)
    drawdowns = _floor_nan(drawdowns)

    ns = _min_max_normalize(sharpes)
    no = _min_max_normalize(sortinos)
    nd = _min_max_normalize(drawdowns)

    return [0.5 * ns[i] + 0.3 * no[i] - 0.2 * nd[i] for i in range(len(metrics_list))]


def _sweep_fold(
    strategy_name: str,
    strat_instance: Any,
    ohlcv: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    """Run all combos on a single time window and return per-combo metrics.

    Signal generation uses all data up to window_end (for EMA warmup).
    Scoring uses only [window_start, window_end).
    Returns (results, all_eq) where results is a list of dicts with keys:
    'combo', 'params', and all curve_stats keys; all_eq maps combo keys
    to full equity series.
    """
    from ggTrader.lab.strategies.indicators import extract_close
    from ggTrader.lab.sweep import group_by_stop_config, sweep_signal_group

    ohlcv_window = ohlcv.loc[:window_end]
    symbols = sorted(ohlcv_window.columns.get_level_values(0).unique())
    prices = extract_close(ohlcv_window, symbols)

    start_cash = float(base_config["START_CASH"])

    all_eq: Dict[str, pd.Series] = {}
    for stop_key, group_combos in group_by_stop_config(grid).items():
        eq_dict, _ = sweep_signal_group(
            strategy_name,
            strat_instance,
            stop_key,
            group_combos,
            symbols,
            ohlcv_window,
            prices,
            base_config,
        )
        all_eq.update(eq_dict)

    # Score each combo over the scoring window only
    results: List[Dict[str, Any]] = []
    for key, eq_series in all_eq.items():
        eq_window = eq_series.loc[window_start:window_end].dropna()
        if len(eq_window) < 2:
            continue
        # Rescale to start at start_cash for consistent metrics
        eq_scaled = start_cash * (eq_window / eq_window.iloc[0])
        metrics = curve_stats(eq_scaled)
        combo_params = next(c for c in grid if combo_name(strategy_name, c) == key)
        results.append({"combo": key, "params": combo_params, **metrics})
    return results, all_eq


def compute_anchor_set(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    risk_free_rate: float = 4.0,
) -> AnchorSet:
    """Derive the anchor set: minimize max drawdown subject to CAGR > risk-free.

    Runs all grid combos over the full available history and picks the combo
    with the smallest absolute drawdown among those with CAGR > risk_free_rate.
    Falls back to the least-drawdown combo if none clears the CAGR constraint.
    """
    strat_instance = strategy_cls(cfg)
    full_start = ohlcv.index[0]
    full_end = ohlcv.index[-1]

    metrics_list, _eq = _sweep_fold(
        strategy_name,
        strat_instance,
        ohlcv,
        full_start,
        full_end,
        base_config,
        grid,
    )
    if not metrics_list:
        return AnchorSet(
            combo="none",
            params={},
            max_drawdown_pct=0.0,
            cagr_pct=0.0,
            sharpe=float("nan"),
        )

    # Filter: CAGR > risk-free rate
    viable = [m for m in metrics_list if m.get("cagr_pct", 0.0) > risk_free_rate]
    candidates = viable if viable else metrics_list

    # Sort by least drawdown (max_drawdown_pct is negative, so max = least drawdown)
    best = max(candidates, key=lambda m: m.get("max_drawdown_pct", float("-inf")))

    return AnchorSet(
        combo=best["combo"],
        params=best["params"],
        max_drawdown_pct=best.get("max_drawdown_pct", 0.0),
        cagr_pct=best.get("cagr_pct", 0.0),
        sharpe=best.get("sharpe", float("nan")),
    )


def _extract_grid_arrays(
    train_metrics: List[Dict[str, Any]],
    grid: List[Dict[str, Any]],
    strategy_name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], Dict[int, int]]:
    """Map sweep results to N-dim grid arrays for NDH.

    Returns:
        sharpe_grid: 1D array indexed by flat grid position
        expectancy_grid: 1D array indexed by flat grid position
        grid_shape: shape of the N-dim grid
        result_to_grid: mapping from train_metrics index to grid flat index
    """
    if not grid:
        return np.array([]), np.array([]), (), {}

    # Requires uniform keys across all grid combos (Cartesian product)
    param_keys = sorted(grid[0].keys())
    if any(sorted(c.keys()) != param_keys for c in grid):
        return np.array([]), np.array([]), (), {}

    axes: Dict[str, List[Any]] = {k: sorted(set(c[k] for c in grid)) for k in param_keys}
    shape = tuple(len(axes[k]) for k in param_keys)

    n_cells = 1
    for s in shape:
        n_cells *= s

    sharpe_arr = np.full(n_cells, float("nan"))
    expectancy_arr = np.full(n_cells, float("nan"))

    # Build lookup: combo params -> flat index
    axis_idx = {k: {v: i for i, v in enumerate(axes[k])} for k in param_keys}

    def _flat_idx(params: Dict[str, Any]) -> int:
        coords = tuple(axis_idx[k][params[k]] for k in param_keys)
        return int(np.ravel_multi_index(coords, shape))

    # Map results to grid
    result_to_grid: Dict[int, int] = {}
    for ri, m in enumerate(train_metrics):
        combo_params = m["params"]
        flat = _flat_idx(combo_params)
        sharpe_arr[flat] = m.get("sharpe", float("nan"))
        # Trade expectancy: approximate as total_return / max(1, n_trades)
        # For now use total_return_pct as proxy since we don't have n_trades
        expectancy_arr[flat] = m.get("total_return_pct", 0.0) / 100.0
        result_to_grid[ri] = flat

    # Replace NaN with 0 for the gate check (missing combos count as zero-edge)
    sharpe_arr = np.nan_to_num(sharpe_arr, nan=0.0)
    expectancy_arr = np.nan_to_num(expectancy_arr, nan=0.0)

    return sharpe_arr, expectancy_arr, shape, result_to_grid


def _compute_dsr_inputs(eq_series: pd.Series) -> tuple[int, float, float]:
    """Extract (n_obs, skew, excess_kurtosis) from an equity curve for DSR."""
    rets = eq_series.pct_change().dropna()
    n_obs = len(rets)
    if n_obs < 3:
        return n_obs, 0.0, 0.0
    skew = float(rets.skew())
    kurt = float(rets.kurtosis())  # pandas .kurtosis() returns excess kurtosis
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    return n_obs, skew, kurt


def run_wfo(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    ndh_threshold: float = 0.85,
    dsr_threshold: float = 0.80,
) -> str:
    """Main WFO entry point: fold, train, test, concatenate, report."""
    eval_start_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    folds = generate_folds(eval_start_ts, eval_end_ts)
    if not folds:
        return (
            f"WFO: {strategy_name} | no valid folds (need >= {TRAIN_MONTHS + TEST_MONTHS} months)"
        )

    strat_instance = strategy_cls(cfg)
    start_cash = float(base_config["START_CASH"])
    fold_results: List[Dict[str, Any]] = []
    oos_curves: List[pd.Series] = []
    oos_running_value = start_cash
    fold_winners: List[Dict[str, Any]] = []
    wfo_state = WfoState()

    anchor = compute_anchor_set(
        strategy_name,
        strategy_cls,
        cfg,
        ohlcv,
        base_config,
        grid,
    )
    print(
        f"  Anchor set: {anchor.combo}"
        f" (MaxDD {anchor.max_drawdown_pct:.1f}%, CAGR {anchor.cagr_pct:.1f}%)"
    )

    for i, fold in enumerate(folds):
        # Train: sweep all combos on train window
        train_metrics, train_eq = _sweep_fold(
            strategy_name,
            strat_instance,
            ohlcv,
            fold.train_start,
            fold.train_end,
            base_config,
            grid,
        )
        if not train_metrics:
            continue

        # Pick winner by composite score
        scores = composite_score(train_metrics)
        best_idx = max(range(len(scores)), key=lambda j: scores[j])
        winner = train_metrics[best_idx]

        # --- Gate checks ---
        ndh_passed = False
        dsr_passed = False
        ndh_density = 0.0
        dsr_value = 0.0

        sharpe_grid, exp_grid, grid_shape, r2g = _extract_grid_arrays(
            train_metrics,
            grid,
            strategy_name,
        )
        if best_idx in r2g and len(grid_shape) > 0:
            ndh_result = ndh_check(
                peak_idx=r2g[best_idx],
                sharpe_grid=sharpe_grid,
                expectancy_grid=exp_grid,
                grid_shape=grid_shape,
                density_threshold=ndh_threshold,
            )
            ndh_passed = ndh_result.passed
            ndh_density = ndh_result.density

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
                threshold=dsr_threshold,
            )
            dsr_passed = dsr_result.passed
            dsr_value = dsr_result.dsr_value

        gates_passed = ndh_passed and dsr_passed

        # Fallback to anchor when gates fail or system is halted
        deploy_params = winner["params"]
        used_anchor = False
        if not gates_passed or wfo_state.halted:
            deploy_params = anchor.params
            used_anchor = True

        # Test: simulate deploy params on data up to test_end, score test window
        winner_grid = [deploy_params]
        test_metrics, _test_eq = _sweep_fold(
            strategy_name,
            strat_instance,
            ohlcv,
            fold.test_start,
            fold.test_end,
            base_config,
            winner_grid,
        )
        oos_score = 0.0
        if test_metrics:
            oos_score = composite_score(test_metrics)[0]

            # Build continuous OOS equity curve
            test_ohlcv = ohlcv.loc[: fold.test_end]
            symbols = sorted(test_ohlcv.columns.get_level_values(0).unique())
            prices = pd.concat({s: test_ohlcv[s]["close"] for s in symbols}, axis=1)
            signal_combos = [split_params(deploy_params)[0]]
            _, stop_p = split_params(deploy_params)
            targets = strat_instance.sweep_signals(signal_combos, symbols, test_ohlcv)
            key = combo_name(strategy_name, signal_combos[0])
            full_key = combo_name(strategy_name, deploy_params)
            sim_config = {**base_config, **stop_p}
            ohlcv_arg = test_ohlcv if "atr_mult" in stop_p else None
            _r, eq, _d = simulate_signals(
                {full_key: targets[key]}, prices, sim_config, ohlcv=ohlcv_arg
            )
            eq_test = eq[full_key].loc[fold.test_start : fold.test_end].dropna()
            if len(eq_test) > 0:
                normalized = oos_running_value * (eq_test / eq_test.iloc[0])
                oos_curves.append(normalized)
                oos_running_value = float(normalized.iloc[-1])

        # --- WFE + circuit breaker ---
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

        fold_results.append(
            {
                "fold_num": i + 1,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "winner_combo": winner["combo"],
                "winner_params": winner["params"],
                "train_score": scores[best_idx],
                "oos_score": oos_score,
                "ndh_passed": ndh_passed,
                "dsr_passed": dsr_passed,
                "gates_passed": gates_passed,
                "ndh_density": ndh_density,
                "dsr_value": dsr_value,
                "wfe": wfe_val,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "halted": wfo_state.halted,
                "used_anchor": used_anchor,
            }
        )
        fold_winners.append(winner)

    # Concatenate OOS curves and score
    if oos_curves:
        oos_equity = pd.concat(oos_curves)
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
        oos_metrics = curve_stats(oos_equity)
        spy_oos = spy_close.reindex(oos_equity.index).ffill().dropna()
        if len(spy_oos) > 1:
            spy_curve = start_cash * (spy_oos / spy_oos.iloc[0])
            spy_metrics = curve_stats(spy_curve)
        else:
            spy_metrics = {
                "sharpe": float("nan"),
                "cagr_pct": float("nan"),
                "max_drawdown_pct": float("nan"),
            }
    else:
        oos_metrics = {
            "sharpe": float("nan"),
            "cagr_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
        }
        spy_metrics = oos_metrics.copy()

    # Recommended live params
    live = select_live_params(
        strategy_name,
        strategy_cls,
        cfg,
        ohlcv,
        eval_end,
        base_config,
        grid,
        fold_winners,
    )

    table = format_wfo_table(
        fold_results,
        oos_metrics,
        spy_metrics,
        live,
        strategy_name,
        len(grid),
        len(folds),
        halted=wfo_state.halted,
        anchor=anchor,
    )
    print(table)
    return table


def select_live_params(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    eval_end: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    fold_winners: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Train on the most recent TRAIN_MONTHS window and pick the composite winner."""
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    live_train_start = eval_end_ts - pd.DateOffset(months=TRAIN_MONTHS)
    strat_instance = strategy_cls(cfg)

    train_metrics, _train_eq = _sweep_fold(
        strategy_name,
        strat_instance,
        ohlcv,
        live_train_start,
        eval_end_ts,
        base_config,
        grid,
    )
    if not train_metrics:
        return {"combo": "none", "params": {}, "train_metrics": {}, "stability": 0}

    scores = composite_score(train_metrics)
    best_idx = max(range(len(scores)), key=lambda j: scores[j])
    winner = train_metrics[best_idx]

    # Stability: count how many WFO folds selected the same combo
    stability = sum(1 for fw in fold_winners if fw["combo"] == winner["combo"])

    return {
        "combo": winner["combo"],
        "params": winner["params"],
        "train_metrics": {
            k: winner[k] for k in ("sharpe", "cagr_pct", "max_drawdown_pct") if k in winner
        },
        "stability": stability,
        "train_start": live_train_start,
        "train_end": eval_end_ts,
    }


def format_wfo_table(
    fold_results: List[Dict[str, Any]],
    oos_metrics: Dict[str, float],
    spy_metrics: Dict[str, float],
    live_params: Dict[str, Any],
    strategy_name: str,
    n_combos: int,
    n_folds: int,
    halted: bool = False,
    anchor: AnchorSet | None = None,
) -> str:
    """Render per-fold table + OOS aggregate + recommended live params."""
    lines = [
        f"WFO: {strategy_name} | {n_combos} combos x {n_folds} folds"
        f" | rolling {TRAIN_MONTHS}mo/{TEST_MONTHS}mo",
        "",
        f"{'Fold':<6}{'Train Window':<20}{'Test Window':<20}"
        f"{'Winner':<26}{'Train':>6}{'OOS':>6}  {'Gate':<4} {'WFE':>5}",
        "─" * 96,
    ]
    for r in fold_results:
        ts = r["train_start"].strftime("%Y-%m")
        te = r["train_end"].strftime("%Y-%m")
        os_ = r["test_start"].strftime("%Y-%m")
        oe = r["test_end"].strftime("%Y-%m")
        short = r["winner_combo"].replace(f"{strategy_name}__", "")
        if len(short) > 24:
            short = short[:21] + "..."
        gate_str = "PASS" if r.get("gates_passed", False) else "FAIL"
        wfe_str = f"{r['wfe']:.2f}" if r.get("wfe") is not None else " n/a"
        halt_str = " [H]" if r.get("halted") else ""
        anchor_str = " [A]" if r.get("used_anchor") else ""
        lines.append(
            f"{r['fold_num']:<6}{ts} → {te:<13}{os_} → {oe:<13}"
            f"{short:<26}{r['train_score']:>6.2f}{r['oos_score']:>6.2f}"
            f"  {gate_str:<4} {wfe_str:>5}{halt_str}{anchor_str}"
        )

    lines.append("")
    lines.append(
        f"OOS Aggregate: Sharpe {oos_metrics.get('sharpe', float('nan')):.2f}"
        f" | CAGR {oos_metrics.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {oos_metrics.get('max_drawdown_pct', float('nan')):.1f}%"
    )
    lines.append(
        f"SPY baseline:  Sharpe {spy_metrics.get('sharpe', float('nan')):.2f}"
        f" | CAGR {spy_metrics.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {spy_metrics.get('max_drawdown_pct', float('nan')):.1f}%"
    )

    valid_wfes = [r["wfe"] for r in fold_results if r.get("wfe") is not None]
    avg_wfe = sum(valid_wfes) / len(valid_wfes) if valid_wfes else float("nan")
    lines.append(f"Aggregate WFE: {avg_wfe:.2f} (target >= 0.50)")
    if halted:
        lines.append("!! REGIME HALT ACTIVE -- trading on anchor params")

    # Recommended live params
    lines.append("")
    lines.append("── Recommended Live Params " + "─" * 71)
    ts = live_params.get("train_start")
    te = live_params.get("train_end")
    ts_str = ts.strftime("%Y-%m") if ts else "?"
    te_str = te.strftime("%Y-%m") if te else "?"
    lines.append(f"Train window: {ts_str} → {te_str}")
    lines.append(f"Winner:       {live_params.get('combo', 'none')}")
    tm = live_params.get("train_metrics", {})
    lines.append(
        f"Train Sharpe: {tm.get('sharpe', float('nan')):.2f}"
        f" | CAGR {tm.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {tm.get('max_drawdown_pct', float('nan')):.1f}%"
    )
    lines.append(
        f"Stability:    selected in {live_params.get('stability', 0)}/{len(fold_results)} folds"
    )

    if anchor is not None:
        lines.append("")
        lines.append("── Anchor Set (Defensive Fallback) " + "─" * 63)
        lines.append(f"Combo:    {anchor.combo}")
        lines.append(
            f"MaxDD:    {anchor.max_drawdown_pct:.1f}%"
            f" | CAGR {anchor.cagr_pct:.1f}%"
            f" | Sharpe {anchor.sharpe:.2f}"
        )

    return "\n".join(lines)
