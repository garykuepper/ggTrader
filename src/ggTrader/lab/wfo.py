"""Walk-forward optimization: rolling train/test folds with composite scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Type

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ggTrader.lab.gates import dsr_check, ndh_check
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_combo_lookup, combo_name

UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]

TRAIN_MONTHS = 12
TEST_MONTHS = 3

#: Minimum number of walk-forward folds a combo must have won to be eligible as
#: a live-param recommendation. Guards against deploying a combo that scores
#: best only on the most recent window but never proved out-of-sample (a
#: classic overfit-to-recent-regime trap). Set to 0 to disable the filter.
MIN_LIVE_STABILITY = 1

#: Discrete "regime" params where a ±1 grid step is a different strategy rather
#: than a small perturbation. Excluded from the NDH plateau neighborhood.
REGIME_PARAMS: frozenset = frozenset({"min_agree", "min_agree_exit"})

#: combo_params keys that are absorbed into the merged LabConfig instance
#: passed to weight-strategy constructors in _sweep_fold_weights. Any other
#: key present in a combo (e.g. IdioVolStrategy's reg_window/quintile) is a
#: constructor kwarg and must be forwarded separately, or it's silently
#: dropped and every combo ends up built with identical defaults.
LAB_CONFIG_COMBO_KEYS: frozenset = frozenset({"top_n", "lookback", "skip"})


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
    #: Rolling clean/dirty flags for the last 3 shadow windows during a halt.
    shadow_window: List[bool] = field(default_factory=list)


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
        shadow_window=list(state.shadow_window),
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
    window_size: int = 3,
    clean_required: int = 2,
) -> WfoState:
    """Check shadow re-entry: 2 of the last 3 clean windows restore live trading.

    A clean window requires all three: NDH pass, DSR pass, WFE >= wfe_healthy.

    The original rule required 2 *consecutive* clean windows. On a noisy
    universe (e.g. MidCap 400) genuinely-good strategies alternate clean/dirty
    windows, so a tripped halt never released and the system stayed pinned to
    defensive anchor params indefinitely. Using "k of the last n" over a rolling
    window tolerates that noise while still demanding real, recent recovery.
    """
    new_state = WfoState(
        wfe_history=list(state.wfe_history),
        oos_sharpes=list(state.oos_sharpes),
        halted=state.halted,
        halt_reason=state.halt_reason,
        shadow_window=list(state.shadow_window),
    )

    clean = ndh_passed and dsr_passed and wfe is not None and wfe >= wfe_healthy
    new_state.shadow_window.append(clean)
    new_state.shadow_window = new_state.shadow_window[-window_size:]

    if sum(new_state.shadow_window) >= clean_required:
        new_state.halted = False
        new_state.halt_reason = None
        new_state.shadow_window = []

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


class WfoResult(NamedTuple):
    """Structured result of a walk-forward run (the table is also printed)."""

    oos_equity: pd.Series
    fold_results: List[Dict[str, Any]]
    live_params: Dict[str, Any]
    table: str


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
    all_diags: Dict[str, Dict[str, Any]] = {}
    for stop_key, group_combos in group_by_stop_config(grid).items():
        eq_dict, diag_dict = sweep_signal_group(
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
        all_diags.update(diag_dict)

    # Score each combo over the scoring window only
    results: List[Dict[str, Any]] = []
    combo_lookup = build_combo_lookup(strategy_name, grid)
    for key, eq_series in all_eq.items():
        eq_window = eq_series.loc[window_start:window_end].dropna()
        if len(eq_window) < 2:
            continue
        # Rescale to start at start_cash for consistent metrics
        eq_scaled = start_cash * (eq_window / eq_window.iloc[0])
        metrics = curve_stats(eq_scaled)
        metrics["n_trades"] = all_diags.get(key, {}).get("n_trades", 0)
        combo_params = combo_lookup[key]
        results.append({"combo": key, "params": combo_params, **metrics})
    return results, all_eq


def _run_one_weight_combo(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    combo_params: Dict[str, Any],
    dates: List[pd.Timestamp],
    ohlcv_window: pd.DataFrame,
    universe_fn: UniverseFn,
) -> tuple[str, pd.DataFrame]:
    """One grid combo's plan + to_targets: fresh strategy instance, select()
    at every rebalance date, then to_targets(). Split out of
    _sweep_fold_weights so joblib can run combos across processes -- each
    combo builds its own strategy instance/state with no cross-combo
    dependency (per-strategy memoization, e.g. pairs_stat_arb's module-level
    pair-candidate cache, is keyed by args that are identical across combos
    on the same window, so each process rebuilding it independently is
    redundant work, not a correctness risk)."""
    merged = {
        "top_n": cfg.top_n,
        "lookback": cfg.lookback,
        "skip": cfg.skip,
        **combo_params,
    }
    combo_cfg = LabConfig(
        top_n=int(merged.get("top_n", cfg.top_n)),
        lookback=int(merged.get("lookback", cfg.lookback)),
        skip=int(merged.get("skip", cfg.skip)),
        min_history_bars=cfg.min_history_bars,
        max_stocks=cfg.max_stocks,
        max_sector_count=cfg.max_sector_count,
    )
    extra_kwargs = {k: v for k, v in combo_params.items() if k not in LAB_CONFIG_COMBO_KEYS}
    strat = strategy_cls(combo_cfg, **extra_kwargs)
    plans: Dict[pd.Timestamp, Any] = {}
    for asof in dates:
        past = ohlcv_window.loc[:asof]
        eligible = universe_fn(asof, past)
        plans[asof] = strat.select(asof, past, eligible)
    target = strat.to_targets(plans, ohlcv_window)
    key = combo_name(strategy_name, combo_params)
    return key, target


def _sweep_fold_weights(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    universe_fn: UniverseFn,
) -> tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    """Weight-strategy analog of _sweep_fold: plan + simulate every combo on one window.

    Mirrors sweep.py's weight branch (run_sweep), scoped to [window_start,
    window_end] instead of the full eval span, using a caller-supplied
    universe_fn (the same UniverseFn protocol harness.py's walkforward() uses)
    instead of a hardcoded universe string.
    """
    from ggTrader.lab.data import rebalance_dates
    from ggTrader.lab.simulate import simulate_weights
    from ggTrader.lab.strategies.indicators import extract_close

    ohlcv_window = ohlcv.loc[:window_end]
    symbols = sorted(ohlcv_window.columns.get_level_values(0).unique())
    prices = extract_close(ohlcv_window, symbols)

    dates = rebalance_dates(ohlcv_window.index, window_start, window_end)
    if not dates:
        return [], {}

    # Combos are fully independent of each other (each builds its own
    # strategy instance/state) -- run them across processes to use all
    # cores. n_jobs=1 for trivial grids skips pool-spawn overhead entirely.
    n_jobs = 1 if len(grid) <= 1 else -1
    combo_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_weight_combo)(
            strategy_name, strategy_cls, cfg, combo_params, dates, ohlcv_window, universe_fn
        )
        for combo_params in grid
    )
    all_targets: Dict[str, pd.DataFrame] = dict(combo_results)

    start_cash = float(base_config["START_CASH"])
    _rets, eq_df, diags = simulate_weights(all_targets, prices, base_config)

    results: List[Dict[str, Any]] = []
    all_eq: Dict[str, pd.Series] = {}
    combo_lookup = build_combo_lookup(strategy_name, grid)
    for key in all_targets:
        eq_series = eq_df[key]
        all_eq[key] = eq_series
        eq_window = eq_series.loc[window_start:window_end].dropna()
        if len(eq_window) < 2:
            continue
        eq_scaled = start_cash * (eq_window / eq_window.iloc[0])
        metrics = curve_stats(eq_scaled)
        n_trades = diags.get(key, {}).get("n_trades", 0)
        results.append({"combo": key, "params": combo_lookup[key], "n_trades": n_trades, **metrics})
    return results, all_eq


def _sweep_fold_dispatch(
    strategy_name: str,
    strat_instance: Any,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    universe_fn: "UniverseFn | None",
) -> tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    """Route to the signal or weight fold-sweep path by strategy target_kind."""
    if getattr(strat_instance, "target_kind", "signals") == "weights":
        if universe_fn is None:
            raise ValueError(
                f"{strategy_name}: weight strategies require universe_fn "
                "(e.g. lambda asof, past: eligible_at(asof, past, cfg, universe=...)[0])"
            )
        return _sweep_fold_weights(
            strategy_name,
            strategy_cls,
            cfg,
            ohlcv,
            window_start,
            window_end,
            base_config,
            grid,
            universe_fn,
        )
    return _sweep_fold(
        strategy_name, strat_instance, ohlcv, window_start, window_end, base_config, grid
    )


def compute_anchor_set(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    risk_free_rate: float = 4.0,
    universe_fn: "UniverseFn | None" = None,
) -> AnchorSet:
    """Derive the anchor set: minimize max drawdown subject to CAGR > risk-free.

    Runs all grid combos over the full available history and picks the combo
    with the smallest absolute drawdown among those with CAGR > risk_free_rate.
    Falls back to the least-drawdown combo if none clears the CAGR constraint.
    """
    strat_instance = strategy_cls(cfg)
    full_start = ohlcv.index[0]
    full_end = ohlcv.index[-1]

    metrics_list, _eq = _sweep_fold_dispatch(
        strategy_name,
        strat_instance,
        strategy_cls,
        cfg,
        ohlcv,
        full_start,
        full_end,
        base_config,
        grid,
        universe_fn,
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

    # Axes are categorical coordinates for the NDH lattice; the sort only needs
    # to be deterministic. None ("no stop", e.g. td_stop/tp_stop) is not
    # orderable against numbers, so key it to sort last without comparing values.
    axes: Dict[str, List[Any]] = {
        k: sorted({c[k] for c in grid}, key=lambda v: (v is None, v)) for k in param_keys
    }
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
        # Real per-trade expectancy: total_return / n_trades. n_trades is
        # threaded from simulate_weights/simulate_signals' diags (added
        # 2026-07-20 -- previously this used raw total_return_pct as a
        # stand-in, which conflates "many small wins" with "one lucky
        # trade" and is especially misleading for sparse event-driven
        # strategies where a handful of trades dominate the whole curve).
        n_trades = max(1, int(m.get("n_trades", 0)))
        expectancy_arr[flat] = (m.get("total_return_pct", 0.0) / 100.0) / n_trades
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
    universe_fn: "UniverseFn | None" = None,
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
        universe_fn=universe_fn,
    )
    print(
        f"  Anchor set: {anchor.combo}"
        f" (MaxDD {anchor.max_drawdown_pct:.1f}%, CAGR {anchor.cagr_pct:.1f}%)",
        flush=True,
    )

    for i, fold in enumerate(folds):
        print(
            f"  Fold {i + 1}/{len(folds)}: train {fold.train_start.date()}→{fold.train_end.date()}"
            f" | test {fold.test_start.date()}→{fold.test_end.date()}",
            end="",
            flush=True,
        )
        # Train: sweep all combos on train window
        train_metrics, train_eq = _sweep_fold_dispatch(
            strategy_name,
            strat_instance,
            strategy_cls,
            cfg,
            ohlcv,
            fold.train_start,
            fold.train_end,
            base_config,
            grid,
            universe_fn,
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
        ndh_variance = float("nan")
        dsr_value = 0.0

        sharpe_grid, exp_grid, grid_shape, r2g = _extract_grid_arrays(
            train_metrics,
            grid,
            strategy_name,
        )
        if best_idx in r2g and len(grid_shape) > 0:
            # Restrict the NDH neighborhood to smooth tuning axes — a ±1 step in
            # a regime param (min_agree) is a different strategy, not a small
            # perturbation, and would wrongly drag the plateau density down.
            param_keys = sorted(grid[0].keys())
            neighbor_axes = tuple(i for i, k in enumerate(param_keys) if k not in REGIME_PARAMS)
            ndh_result = ndh_check(
                peak_idx=r2g[best_idx],
                sharpe_grid=sharpe_grid,
                expectancy_grid=exp_grid,
                grid_shape=grid_shape,
                density_threshold=ndh_threshold,
                neighbor_axes=neighbor_axes,
            )
            ndh_passed = ndh_result.passed
            ndh_density = ndh_result.density
            ndh_variance = ndh_result.variance_ratio

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
        test_metrics, test_eq = _sweep_fold_dispatch(
            strategy_name,
            strat_instance,
            strategy_cls,
            cfg,
            ohlcv,
            fold.test_start,
            fold.test_end,
            base_config,
            winner_grid,
            universe_fn,
        )
        oos_score = 0.0
        if test_metrics:
            oos_score = composite_score(test_metrics)[0]
            full_key = combo_name(strategy_name, deploy_params)
            eq_test = test_eq.get(full_key, pd.Series(dtype=float))
            eq_test = eq_test.loc[fold.test_start : fold.test_end].dropna()
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
        oos_s = f" OOS Sharpe {oos_sharpe:.2f}" if np.isfinite(oos_sharpe) else ""
        wfe_s = f" WFE {wfe_val:.2f}" if wfe_val is not None else ""
        gate_s = (
            f" [NDH dens {ndh_density:.2f} var {ndh_variance:.2f}/0.20 "
            f"{'✓' if ndh_passed else '✗'} DSR {dsr_value:.2f}{'✓' if dsr_passed else '✗'}]"
        )
        print(f" → done{oos_s}{wfe_s}{gate_s}", flush=True)

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
        oos_equity = pd.Series(dtype=float)
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
        universe_fn=universe_fn,
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
    return WfoResult(
        oos_equity=oos_equity,
        fold_results=fold_results,
        live_params=live,
        table=table,
    )


def _pick_live_winner(
    train_metrics: List[Dict[str, Any]],
    scores: List[float],
    fold_win_counts: Dict[str, int],
    min_stability: int = MIN_LIVE_STABILITY,
) -> tuple[int, int]:
    """Pick the live-param combo, preferring out-of-sample durability.

    Among combos that won at least ``min_stability`` walk-forward folds, return
    the one with the best composite score on the recent training window. If no
    combo cleared that bar (e.g. every fold failed its gates, so there are no
    fold winners), fall back to the global best composite score so a
    recommendation is still produced.

    Returns ``(winner_index, stability)`` where ``stability`` is the winner's
    fold-win count.
    """
    durable = [
        j
        for j in range(len(train_metrics))
        if fold_win_counts.get(train_metrics[j]["combo"], 0) >= min_stability
    ]
    pool = durable or list(range(len(train_metrics)))
    best_idx = max(pool, key=lambda j: scores[j])
    stability = fold_win_counts.get(train_metrics[best_idx]["combo"], 0)
    return best_idx, stability


def select_live_params(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    eval_end: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    fold_winners: List[Dict[str, Any]],
    universe_fn: "UniverseFn | None" = None,
) -> Dict[str, Any]:
    """Train on the most recent TRAIN_MONTHS window and pick the durable winner.

    Prefers combos proven across walk-forward folds over those that merely score
    best on the most recent window — see :func:`_pick_live_winner`.
    """
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    live_train_start = eval_end_ts - pd.DateOffset(months=TRAIN_MONTHS)
    strat_instance = strategy_cls(cfg)

    train_metrics, _train_eq = _sweep_fold_dispatch(
        strategy_name,
        strat_instance,
        strategy_cls,
        cfg,
        ohlcv,
        live_train_start,
        eval_end_ts,
        base_config,
        grid,
        universe_fn,
    )
    if not train_metrics:
        return {"combo": "none", "params": {}, "train_metrics": {}, "stability": 0}

    scores = composite_score(train_metrics)
    fold_win_counts: Dict[str, int] = {}
    for fw in fold_winners:
        fold_win_counts[fw["combo"]] = fold_win_counts.get(fw["combo"], 0) + 1

    best_idx, stability = _pick_live_winner(train_metrics, scores, fold_win_counts)
    winner = train_metrics[best_idx]

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
