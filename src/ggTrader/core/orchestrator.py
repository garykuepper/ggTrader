"""Centralized orchestration logic for backtesting, sensitivity analysis, and WFO."""

import gc
import itertools
import time
from math import prod
import traceback
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from tabulate import tabulate

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.pipeline.exit_tournament import parse_exit_tournament
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers

try:
    import psutil
except ImportError:
    psutil = None


def _eta_str(seconds: float) -> str:
    """Format seconds as HH:MM:SS for ETA display."""
    return str(timedelta(seconds=int(max(0, seconds))))


def _wall_clock_eta(seconds: float) -> str:
    """Return wall-clock estimate (e.g., '1:30 PM')."""
    from datetime import datetime, timedelta
    finish = datetime.now() + timedelta(seconds=max(0, seconds))
    return finish.strftime("%I:%M %p")


def _log_memory_usage(label: str) -> None:
    """Log current process memory for debugging."""
    if psutil is None:
        return
    try:
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / (1024 ** 2)
        print(f"    [{label}] Memory: {mem_mb:.1f} MB")
    except Exception:
        pass


def _to_native(val: Any) -> Any:
    """Ensure basic types are JSON serializable. Converts NaNs to None."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_native(x) for x in val]
    return val


def _coerce_metric_float(x: Any) -> float:
    """Coerce WFO/OOS metrics to float; None or invalid -> nan for numpy reductions."""
    if x is None:
        return float("nan")
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return float("nan")
        return xf
    except (TypeError, ValueError):
        return float("nan")


def _align_grouped_combo_series(agg: pd.Series, sh_index: pd.Index) -> pd.Series:
    """Map ``groupby`` on column MultiIndex levels onto metric index (e.g. 0 -> (0,))."""
    if agg.index.equals(sh_index):
        return agg
    if isinstance(sh_index, pd.MultiIndex):
        vals: list[float] = []
        for k in sh_index:
            v = float("nan")
            if k in agg.index:
                v = float(agg.loc[k])
            elif isinstance(k, tuple) and len(k) >= 1 and k[0] in agg.index:
                v = float(agg.loc[k[0]])
            vals.append(v if np.isfinite(v) else float("nan"))
        return pd.Series(vals, index=sh_index, dtype=float)
    return agg.reindex(sh_index)


def _trade_counts_for_train_gate(pf: Any, sharpe_series: Any) -> pd.Series:
    """Closed-trade counts indexed like ``sharpe_series`` (sum per param combo if needed).

    VectorBT usually aligns ``trades.count()`` with ``sharpe_ratio()`` for ``group_by``
    portfolios. If the raw count Series is per underlying column (MultiIndex), aggregate
    by summing across the symbol level so gating matches portfolio-level Sharpe.
    """
    sh = sharpe_series if isinstance(sharpe_series, pd.Series) else pd.Series([sharpe_series])
    raw = pf.trades.count()
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])
    if raw.index.equals(sh.index):
        return raw.astype(float)
    cols = pf.wrapper.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2 and len(raw) == len(cols):
        per_col = pd.Series(np.asarray(raw, dtype=float).ravel(), index=cols)
        agg = per_col.groupby(level=list(range(cols.nlevels - 1))).sum()
        aligned = _align_grouped_combo_series(agg, sh.index)
        return aligned.fillna(0.0).astype(float)
    out = raw.reindex(sh.index)
    return out.fillna(0.0).astype(float)


def _open_position_count_end_for_gate(pf: Any, sharpe_series: Any) -> pd.Series:
    """Open-position counts at last bar, aligned to ``sharpe_series.index`` (max per combo)."""
    sh = sharpe_series if isinstance(sharpe_series, pd.Series) else pd.Series([sharpe_series])
    try:
        raw = pf.positions.open.count()
    except Exception:
        return pd.Series(0.0, index=sh.index, dtype=float)
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])
    if raw.index.equals(sh.index):
        return raw.astype(float)
    cols = pf.wrapper.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2 and len(raw) == len(cols):
        per_col = pd.Series(np.asarray(raw, dtype=float).ravel(), index=cols)
        agg = per_col.groupby(level=list(range(cols.nlevels - 1))).max()
        aligned = _align_grouped_combo_series(agg, sh.index)
        return aligned.fillna(0.0).astype(float)
    out = raw.reindex(sh.index)
    return out.fillna(0.0).astype(float)


def _calmar_ratio_series(pf_train: Any) -> pd.Series:
    """Per-combo Calmar-like ratio: total_return / abs(max_drawdown)."""
    tr = pf_train.total_return()
    mdd = pf_train.max_drawdown()
    if isinstance(tr, pd.Series) and isinstance(mdd, pd.Series):
        denom = mdd.abs().replace(0, np.nan)
        m = tr / denom
    else:
        tr_f = float(tr) if np.isfinite(float(tr)) else float("nan")
        mdd_f = float(mdd) if np.isfinite(float(mdd)) else float("nan")
        m = tr_f / abs(mdd_f) if mdd_f != 0 else float("nan")
        m = pd.Series([m])
    if not isinstance(m, pd.Series):
        m = pd.Series([m])
    return m


def _profit_factor_series(pf_train: Any) -> pd.Series:
    """Per-combo profit factor, mean-centred: (gross_profit/gross_loss - 1), clipped [-3, 3].

    0.0 = breakeven, 1.0 = 2:1 reward ratio, inf (all-win) clipped to 3.0.
    NaN (no trades) propagates — the outer ``m.where(sh_f.notna(), nan)`` gate then
    removes those combos, consistent with the Sharpe NaN convention.
    """
    raw = pf_train.trades.profit_factor()
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])
    return (raw - 1.0).clip(lower=-3.0, upper=3.0).astype(float)


def _train_metric_series(pf_train: Any, config: Dict[str, Any]) -> pd.Series:
    """In-sample metric Series used to pick best params on the train window."""
    name = str(config.get("TRAIN_METRIC", "sharpe")).lower().strip()
    if name == "sortino":
        m = pf_train.sortino_ratio()
    elif name == "calmar":
        m = _calmar_ratio_series(pf_train)
    elif name == "composite":
        raw_w = config.get("TRAIN_METRIC_COMPOSITE_WEIGHTS") or {}
        ws = float(raw_w.get("sharpe", 1.0 / 3.0))
        wso = float(raw_w.get("sortino", 1.0 / 3.0))
        wc = float(raw_w.get("calmar", 1.0 / 3.0))
        wpf = float(raw_w.get("profit_factor", 0.0))
        s = ws + wso + wc + wpf
        if s <= 0:
            ws, wso, wc, wpf = 0.25, 0.25, 0.25, 0.25
        else:
            ws, wso, wc, wpf = ws / s, wso / s, wc / s, wpf / s
        sh = pf_train.sharpe_ratio()
        so = pf_train.sortino_ratio()
        ca = _calmar_ratio_series(pf_train)
        if not isinstance(sh, pd.Series):
            sh = pd.Series([float(sh)])
        if not isinstance(so, pd.Series):
            so = pd.Series([float(so)])
        so = so.reindex(sh.index)
        ca = ca.reindex(sh.index)
        # Clip calmar to [-5, 5] to prevent extreme values (near-zero drawdown) from
        # dominating the composite and artificially inflating a strategy's score.
        ca_clipped = ca.clip(lower=-5.0, upper=5.0)
        # When calmar is NaN but sharpe is valid (e.g. fold with zero drawdown),
        # treat calmar as 0 rather than propagating NaN to the whole composite.
        sh_f = sh.astype(float)
        so_f = so.astype(float)
        ca_fin = ca_clipped.where(ca_clipped.notna(), other=0.0)
        # profit_factor: (GP/GL - 1), clipped [-3, 3]; NaN → 0 (same as calmar convention).
        pf_s = _profit_factor_series(pf_train).reindex(sh.index)
        pf_fin = pf_s.where(pf_s.notna(), other=0.0)
        m = ws * sh_f + wso * so_f + wc * ca_fin + wpf * pf_fin
        # Preserve NaN for combos where sharpe itself is NaN (no trades / gated out).
        m = m.where(sh_f.notna(), other=float("nan"))
    else:
        m = pf_train.sharpe_ratio()
    if not isinstance(m, pd.Series):
        m = pd.Series([m])
    return m


def _max_drawdown_for_train_gate(pf_train: Any, sharpe_series: Any) -> pd.Series:
    """Max drawdown per combo, aligned to sharpe index (more negative = deeper DD)."""
    sh = sharpe_series if isinstance(sharpe_series, pd.Series) else pd.Series([sharpe_series])
    raw = pf_train.max_drawdown()
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])
    if raw.index.equals(sh.index):
        return raw.astype(float)
    cols = pf_train.wrapper.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2 and len(raw) == len(cols):
        per_col = pd.Series(np.asarray(raw, dtype=float).ravel(), index=cols)
        agg = per_col.groupby(level=list(range(cols.nlevels - 1))).min()
        return _align_grouped_combo_series(agg, sh.index)
    return raw.reindex(sh.index)


def _print_wfo_fold_all_rejected_diagnostics(
    fold_idx: int,
    train_ohlcv: pd.DataFrame,
    train_metrics_before_gates: pd.Series,
    trade_for_gate: pd.Series,
    config: Dict[str, Any],
    pf_train: Any,
) -> None:
    """Log structured diagnostics when every train combo is NaN after gating."""
    n_bars = len(train_ohlcv)
    t0, t1 = train_ohlcv.index[0], train_ohlcv.index[-1]
    n_combo = len(train_metrics_before_gates)
    min_closed = int(config.get("MIN_CLOSED_TRADES_TRAIN", 1) or 0)

    tfg_alg = trade_for_gate.reindex(train_metrics_before_gates.index).fillna(0.0)
    tfg_arr = np.asarray(tfg_alg, dtype=float).ravel()
    raw_arr = np.asarray(train_metrics_before_gates, dtype=float).ravel()

    if min_closed > 0:
        pass_trade = tfg_arr >= float(min_closed)
    else:
        pass_trade = np.ones(len(tfg_arr), dtype=bool)
    n_pass_trade = int(np.sum(pass_trade))
    n_fin_on_pass = int(np.sum(np.isfinite(raw_arr[pass_trade])))
    raw_fin_total = int(np.sum(np.isfinite(raw_arr)))
    tfg_min = float(np.nanmin(tfg_arr)) if tfg_arr.size else 0.0
    tfg_max = float(np.nanmax(tfg_arr)) if tfg_arr.size else 0.0

    raw_tc = pf_train.trades.count()
    idx_match: Optional[bool] = None
    align_note = "trade_count_index=n/a"
    if isinstance(raw_tc, pd.Series) and isinstance(
        train_metrics_before_gates, pd.Series
    ):
        idx_match = bool(raw_tc.index.equals(train_metrics_before_gates.index))
        align_note = f"trade_count_index_equals_metric_index={idx_match}"
        if not idx_match:
            align_note += " (_trade_counts_for_train_gate uses groupby/reindex)"

    if min_closed > 0 and n_pass_trade == 0:
        hypothesis = "all_combos_below_MIN_CLOSED_TRADES_TRAIN"
    elif n_pass_trade > 0 and n_fin_on_pass == 0:
        hypothesis = "TRAIN_METRIC_non_finite_despite_enough_closed_trades"
    elif n_pass_trade > 0 and n_fin_on_pass > 0:
        dd_on = config.get("MAX_TRAIN_DRAWDOWN_PCT") is not None
        open_on = int(config.get("REJECT_OPEN_END_IF_CLOSED_LT", 0) or 0) > 0
        if dd_on or open_on:
            hypothesis = "likely_drawdown_or_open_position_gate"
        else:
            hypothesis = "unexpected_all_nan_check_alignment"
    else:
        hypothesis = "unknown"

    print(
        f"  WARNING: Fold {fold_idx} - All param combos rejected after train gates. "
        f"Diagnostics: train_bars={n_bars} [{t0}..{t1}] combos={n_combo} | "
        f"closed_trades min={tfg_min:.0f} max={tfg_max:.0f} "
        f"count_ge_min_closed={n_pass_trade} (min_closed={min_closed}) | "
        f"raw_{config.get('TRAIN_METRIC', 'sharpe')}_finite_total={raw_fin_total} "
        f"finite_on_trade_ok={n_fin_on_pass} | {align_note} | hypothesis={hypothesis}"
    )


def _extract_params(
    idx: Any,
    metric_series: pd.Series,
    param_names: List[str],
    full_param_grid: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper to extract native parameters from VectorBT Index/MultiIndex."""
    extracted = {}
    if isinstance(idx, tuple):
        # MultiIndex case
        names = (
            metric_series.index.names
            if hasattr(metric_series.index, "names") and metric_series.index.names[0]
            else param_names
        )
        for name, val in zip(names, idx):
            clean_name = name.replace("sf_", "") if name else "unknown"
            if clean_name in param_names:
                extracted[clean_name] = val
    else:
        # Single index case
        name = metric_series.index.name
        clean_name = name.replace("sf_", "") if name else param_names[0]
        extracted[clean_name] = idx

    # Fill defaults for any missing from extraction (constants in the grid)
    for k, v in full_param_grid.items():
        if k not in extracted:
            extracted[k] = v[0] if isinstance(v, list) else v
    return extracted


def _first_grid_value(param_grid: Dict[str, Any], key: str) -> Any:
    """Return a single default from the grid for key (first list element or scalar)."""
    v = param_grid.get(key)
    if v is None:
        return None
    return v[0] if isinstance(v, list) else v


def _default_params_from_grid(param_grid: Dict[str, Any]) -> Dict[str, Any]:
    """Scalar defaults for every key (first list element or scalar)."""
    return {k: _first_grid_value(param_grid, k) for k in param_grid}


def _wfo_per_coin_fallback_triple(
    strategy_param_grids: Dict[str, Dict[str, Any]],
    exit_tournament: List[str],
) -> Tuple[str, str, Dict[str, Any]]:
    """First entry strategy, first exit in tournament, and grid first-value params."""
    if not strategy_param_grids or not exit_tournament:
        raise ValueError("strategy_param_grids and exit_tournament must be non-empty")
    first_strategy = next(iter(strategy_param_grids))
    first_exit = exit_tournament[0]
    grid = strategy_param_grids[first_strategy]
    return first_strategy, first_exit, _default_params_from_grid(grid)


def _is_better_robustness(candidate: float, best: float) -> bool:
    """True if candidate should replace best (finite candidate beats non-finite or lower best)."""
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(best):
        return True
    return candidate > best


def _format_robustness_metric(x: Any) -> str:
    """Human-readable robustness for logs (handles nan, +/-inf)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf):
        return "nan"
    if np.isinf(xf):
        return "-inf" if xf < 0 else "inf"
    return f"{xf:.4f}"


def _is_bad_engine_param(val: Any) -> bool:
    """True if val is None or a non-finite float (would break float() in signal code)."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        return bool(np.isnan(val) or np.isinf(val))
    return False


def _coerce_strategy_params_for_engine(
    extracted: Dict[str, Any], param_grid: Dict[str, Any]
) -> Dict[str, Any]:
    """Replace None/NaN with grid defaults so FastBacktest never gets JSON-sanitized Nones."""
    out = dict(extracted)
    for k in param_grid:
        if k not in out or _is_bad_engine_param(out.get(k)):
            out[k] = _first_grid_value(param_grid, k)
    return out


def run_backtest_orchestrator(
    config: Dict[str, Any],
    params: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate a single backtest run.

    Args:
        config: Portfolio and data configuration.
        params: Strategy parameters.
        save_results: Whether to persist results via ResultsManager.

    Returns:
        Dictionary containing 'portfolio', 'stats', and 'rm' (if saved).
    """
    rm = ResultsManager("run_backtest") if save_results else None

    ohlcv, mover_mask = load_data_with_movers(config)

    print("Running backtest...")
    engine = FastBacktest(ohlcv, params, config=config, mover_mask=mover_mask)
    pf = engine.run(show_progress=show_progress)
    stats = engine.get_stats()

    if save_results and rm:
        rm.save_run_results(params=params, metrics=stats, metadata=config)
        rm.save_vbt_dashboard(pf, "dashboard")
        rm.print_summary(stats)
        print(f"Results saved to: {rm.run_dir}")

    return {"portfolio": pf, "stats": stats, "results_manager": rm}


# =============================================================================
# Sensitivity Analysis Helpers & Orchestrator
# =============================================================================


def _apply_sensitivity_train_gates(
    sharpe_series: pd.Series,
    trade_for_gate: pd.Series,
    pf: Any,
    config: Dict[str, Any],
) -> pd.Series:
    """NaN Sharpe for combos that fail closed-trade / open-position gates."""
    out = sharpe_series.copy()
    min_closed = config.get("MIN_CLOSED_TRADES_TRAIN", 1)
    if min_closed > 0:
        incomplete_mask = trade_for_gate < min_closed
        if incomplete_mask.any():
            out = out.copy()
            out[incomplete_mask] = np.nan

    reject_open_lt = config.get("REJECT_OPEN_END_IF_CLOSED_LT", 0)
    if reject_open_lt > 0:
        try:
            open_end = _open_position_count_end_for_gate(pf, out)
            hold_mask = (open_end > 0) & (trade_for_gate < reject_open_lt)
            if hold_mask.any():
                out = out.copy()
                out[hold_mask] = np.nan
        except Exception:
            pass
    return out


def _cleanup_after_heavy_vectorized_run() -> None:
    """Best-effort memory trim after large vectorized Portfolio runs."""
    try:
        import numba

        numba.core.registry.CPUTarget.clear()
    except Exception:
        pass
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass


def _combo_index_keys(param_keys: List[str], combos: List[Dict[str, Any]]) -> List[str]:
    """Keys present in every combo dict, stable order (``param_keys`` first, then sorted extras).

    WFO/sensitivity may pass a superset ``param_grid`` (all EXIT_TOURNAMENT axes merged) while
    ``_last_param_combos`` rows only contain keys for the active exit — avoid KeyError.
    """
    if not combos:
        return param_keys
    common = set(combos[0].keys())
    for c in combos[1:]:
        common &= set(c.keys())
    ordered = [k for k in param_keys if k in common]
    extras = sorted(common - set(ordered))
    return ordered + extras


def _metric_series_from_vectorized_pf(
    pf: Any,
    engine: Any,
    param_keys: List[str],
    config: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series]:
    """Align train metric (Sharpe/Sortino/Calmar) to ``_last_param_combos`` MultiIndex."""
    combos = getattr(engine, "_last_param_combos", None) or []
    if not combos:
        raise RuntimeError("Vectorized run produced no _last_param_combos metadata.")

    train_metrics_raw = _train_metric_series(pf, config)
    if not isinstance(train_metrics_raw, pd.Series):
        train_metrics_raw = pd.Series([float(train_metrics_raw)])

    mvals = np.asarray(train_metrics_raw, dtype=float).ravel()
    if mvals.size != len(combos):
        raise RuntimeError(
            f"Metric length {mvals.size} != param combos {len(combos)}; "
            "portfolio grouping may not match vectorized columns."
        )

    index_keys = _combo_index_keys(param_keys, combos)
    mi = pd.MultiIndex.from_tuples(
        [tuple(combo[k] for k in index_keys) for combo in combos],
        names=index_keys,
    )
    metric_series = pd.Series(mvals, index=mi)
    trade_counts = _trade_counts_for_train_gate(pf, metric_series)
    return metric_series, trade_counts


def _execute_sensitivity_vectorized(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    show_progress: bool,
    logger: Any = None,
) -> Tuple[pd.Series, pd.Series]:
    """Run the full parameter grid in one vectorized FastBacktest pass."""
    keys = list(param_grid.keys())
    grid_values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
    n_total = prod(len(v) for v in grid_values) if grid_values else 0

    msg = (
        f"Running vectorized sensitivity: {n_total} combinations in one pass "
        f"(set USE_VECTORIZED_SENSITIVITY=False to use the slower chunked path)."
    )
    print(msg)
    if logger:
        logger.update(msg)

    vec_cfg = {**config, "USE_VECTORIZED": True}
    engine = FastBacktest(ohlcv, param_grid, config=vec_cfg)
    pf = engine.run(show_progress=show_progress)
    sharpe_series, trade_for_gate = _metric_series_from_vectorized_pf(
        pf, engine, keys, config
    )
    closed_for_output = trade_for_gate.copy()
    sharpe_series = _apply_sensitivity_train_gates(
        sharpe_series, trade_for_gate, pf, config
    )

    del pf
    del engine
    _cleanup_after_heavy_vectorized_run()

    return sharpe_series, closed_for_output


def _process_sensitivity_chunk(
    chunk: List[Tuple],
    keys: List[str],
    config: Dict[str, Any],
    ohlcv: pd.DataFrame,
    show_progress: bool,
) -> Tuple[pd.Series, pd.Series]:
    """Helper to process a single chunk of sensitivity parameters.

    Returns:
        (sharpe_series, closed_trades_series) with identical index for reporting/gating.
    """
    chunk_params = {k: [c[j] for c in chunk] for j, k in enumerate(keys)}
    # Sensitivity analysis uses parallel lists (not grid), so must disable vectorized path
    # The vectorized strategy path expects grids and treats parallel lists as cross-product ranges
    chunk_config = {
        **config,
        "PARAM_PRODUCT": False,
        "USE_VECTORIZED": False,
        "N_JOBS": 2,  # Reduce from -1 (all cores) to 2 to avoid thread contention with NumBa
    }

    engine = FastBacktest(ohlcv, chunk_params, config=chunk_config)
    pf = engine.run(show_progress=show_progress)

    sharpe_series = pf.sharpe_ratio()

    # Handle case where sharpe_series is a scalar (single combo)
    if not isinstance(sharpe_series, pd.Series):
        sharpe_series = pd.Series([sharpe_series])

    trade_for_gate = _trade_counts_for_train_gate(pf, sharpe_series)
    closed_for_output = trade_for_gate.copy()
    sharpe_series = _apply_sensitivity_train_gates(sharpe_series, trade_for_gate, pf, config)

    # Aggressive memory cleanup to prevent accumulation between chunks
    del pf
    del engine

    # Clear NumBa JIT cache to free compiled function memory
    try:
        import numba
        numba.core.registry.CPUTarget.clear()
    except Exception:
        pass

    gc.collect()

    # Force C memory trimming (Linux/POSIX) to reclaim fragmented malloc'd memory
    try:
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass

    return sharpe_series, closed_for_output


def _execute_sensitivity_grid(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    show_progress: bool,
    logger: Any = None,
) -> Tuple[pd.Series, pd.Series]:
    """Generates combinations, splits into chunks, and executes the grid search."""
    if config.get("USE_VECTORIZED_SENSITIVITY", True):
        try:
            return _execute_sensitivity_vectorized(
                ohlcv, config, param_grid, show_progress, logger
            )
        except Exception as e:
            print(f"Vectorized sensitivity failed ({e!r}); falling back to chunked path.")

    keys = list(param_grid.keys())
    values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
    combinations = list(itertools.product(*values))
    total_total = len(combinations)

    chunk_size = config.get("CHUNK_SIZE", 500)
    total_chunks = (total_total + chunk_size - 1) // chunk_size

    print(
        f"Running chunked (non-vectorized) sensitivity in {total_chunks} chunks "
        f"({total_total} total combinations, chunk_size={chunk_size})..."
    )

    all_sharpe_series: List[pd.Series] = []
    all_closed_series: List[pd.Series] = []
    t0 = time.time()

    for i in range(0, total_total, chunk_size):
        chunk_idx = i // chunk_size + 1
        chunk = combinations[i : i + chunk_size]
        chunk_end = min(i + chunk_size, total_total)
        print(
            f"  > Processing chunk {chunk_idx}/{total_chunks} "
            f"(combos {i}-{chunk_end})..."
        )
        _log_memory_usage("chunk start")
        chunk_start = time.time()

        sharpe_series, closed_series = _process_sensitivity_chunk(
            chunk, keys, config, ohlcv, show_progress
        )
        all_sharpe_series.append(sharpe_series)
        all_closed_series.append(closed_series)
        _log_memory_usage("chunk end")

        elapsed = time.time() - t0
        avg_per_chunk = elapsed / chunk_idx
        remaining_chunks = total_chunks - chunk_idx
        eta = remaining_chunks * avg_per_chunk
        chunk_msg = (
            f"Chunk {chunk_idx}/{total_chunks} done in {time.time()-chunk_start:.1f}s "
            f"| total elapsed {_eta_str(elapsed)} | ETA {_eta_str(eta)} (est. {_wall_clock_eta(eta)})"
        )
        print(f"  > {chunk_msg}")
        if logger:
            logger.update(f"  {chunk_msg}")

    return pd.concat(all_sharpe_series), pd.concat(all_closed_series)


def _save_sensitivity_results(
    rm: Any,
    best_pf: Any,
    best_stats: Dict[str, Any],
    best_params: Dict[str, Any],
    results_df: pd.DataFrame,
    param_names: List[str],
    config: Dict[str, Any],
) -> None:
    """Handles plotting and saving outputs for the sensitivity analysis."""
    from ggTrader.utils.plotting import plot_optimization_landscape

    rm.save_metrics(results_df, "sensitivity_results.csv", save_csv=True)

    print("\nTop 5 Parameter Combinations:")
    print(
        tabulate(
            results_df.sort_values("Sharpe Ratio", ascending=False).head(5),
            headers="keys",
            tablefmt="github",
        )
    )

    plot_optimization_landscape(
        results_df,
        params_to_plot=param_names,
        metric_name="Sharpe Ratio",
        results_manager=rm,
    )

    rm.save_run_results(
        params=best_params,
        metrics=best_stats,
        metadata={**config, "NOTE": "Best Case from Sensitivity"},
    )
    rm.save_vbt_dashboard(best_pf, "best_case_dashboard")
    print(f"Best Case Results saved to: {rm.run_dir}")


def run_sensitivity_orchestrator(
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = True,
    logger: Any = None,
) -> Dict[str, Any]:
    """Orchestrate a vectorized sensitivity analysis (grid search)."""
    rm = ResultsManager("run_sensitivity") if save_results else None

    ohlcv, _ = load_data_with_movers(config)
    param_names = list(param_grid.keys())

    # Execute grid search
    sharpe_series, closed_trades_series = _execute_sensitivity_grid(
        ohlcv, config, param_grid, show_progress, logger
    )

    grid_values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
    grid_keys = list(param_grid.keys())

    if sharpe_series.dropna().empty:
        print(
            "WARNING: All parameter combinations have NaN Sharpe "
            "(no completed round-trips found — all combos had buy-and-hold only paths); "
            "using first grid combination for best-case replay."
        )
        first_combo = next(iter(itertools.product(*grid_values)))
        best_params_engine = {grid_keys[j]: first_combo[j] for j in range(len(grid_keys))}
    else:
        best_idx = sharpe_series.idxmax()
        best_params_engine = _extract_params(best_idx, sharpe_series, param_names, param_grid)

    best_params_engine = _coerce_strategy_params_for_engine(best_params_engine, param_grid)
    best_params = _to_native(best_params_engine)

    results_df = sharpe_series.reset_index()
    param_col_labels = [str(col).replace("sf_", "") for col in results_df.columns[:-1]]
    results_df.columns = param_col_labels + ["Sharpe Ratio"]
    results_df.insert(
        len(param_col_labels),
        "Closed trades (agg)",
        closed_trades_series.reindex(sharpe_series.index).values,
    )

    # Evaluate best case (use engine dict — never _to_native, which maps NaN to None)
    best_engine = FastBacktest(ohlcv, best_params_engine, config=config)
    best_pf = best_engine.run(show_progress=show_progress)
    best_stats = best_engine.get_stats()

    if save_results and rm:
        _save_sensitivity_results(
            rm, best_pf, best_stats, best_params, results_df, param_names, config
        )

    return {
        "portfolio": best_pf,
        "results_df": results_df,
        "best_params": best_params,
        "results_manager": rm,
    }


# =============================================================================
# WFO Helpers & Orchestrator
# =============================================================================


def _vectorized_grid_metrics(
    pf: Any,
    engine: Any,
    param_keys: List[str],
    config: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series]:
    """WFO train path: same alignment as vectorized sensitivity (TRAIN_METRIC-aware)."""
    return _metric_series_from_vectorized_pf(pf, engine, param_keys, config)


def _process_wfo_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ohlcv: pd.DataFrame,
    mover_mask: Optional[pd.DataFrame],
    param_grid: Dict[str, Any],
    config: Dict[str, Any],
    show_progress: bool,
    param_names: List[str],
) -> Dict[str, Any]:
    """Helper to process a single WFO fold (Train & Test).

    Train window uses the vectorized registry path so ENTRY_STRATEGY /
    EXIT_STRATEGY in ``config`` are honoured.  Metric alignment relies on
    ``_vectorized_grid_metrics``.  Falls back to the legacy non-vectorized
    SignalFactory path if the vectorized pass raises.
    """
    train_ohlcv = ohlcv.loc[train_idx]
    test_ohlcv = ohlcv.loc[test_idx]

    train_mask = mover_mask.loc[train_idx] if mover_mask is not None else None
    test_mask = mover_mask.loc[test_idx] if mover_mask is not None else None

    # Try vectorized train path first (honours ENTRY/EXIT_STRATEGY).
    wfo_train_cfg = {**config, "USE_VECTORIZED": True}
    train_metrics: pd.Series
    pf_train: Any = None
    try:
        train_engine = FastBacktest(
            train_ohlcv, param_grid, config=wfo_train_cfg, mover_mask=train_mask
        )
        pf_train = train_engine.run(show_progress=show_progress)
        train_metrics, trade_for_gate = _vectorized_grid_metrics(
            pf_train, train_engine, param_names, config
        )
        train_metrics_before_gates = train_metrics.copy()
    except Exception as vec_exc:
        # Graceful fall-back to legacy SignalFactory path.
        print(
            f"  WFO fold {fold_idx}: vectorized train failed ({vec_exc!r}), "
            "falling back to non-vectorized path."
        )
        wfo_config_legacy = {**config, "USE_VECTORIZED": False}
        train_engine = FastBacktest(
            train_ohlcv, param_grid, config=wfo_config_legacy, mover_mask=train_mask
        )
        pf_train = train_engine.run(show_progress=show_progress)
        train_metrics = _train_metric_series(pf_train, config)
        if not isinstance(train_metrics, pd.Series):
            train_metrics = pd.Series([train_metrics])
        trade_for_gate = _trade_counts_for_train_gate(pf_train, train_metrics)
        train_metrics_before_gates = train_metrics.copy()

    # Gate: require MIN_CLOSED_TRADES_TRAIN completed round-trips (per combo).
    min_closed = config.get("MIN_CLOSED_TRADES_TRAIN", 1)
    if min_closed > 0:
        incomplete_mask = trade_for_gate < min_closed
        if incomplete_mask.any():
            train_metrics = train_metrics.copy()
            train_metrics[incomplete_mask] = np.nan

    # Optional: reject combos with train drawdown deeper than -MAX_TRAIN_DRAWDOWN_PCT%.
    dd_limit = config.get("MAX_TRAIN_DRAWDOWN_PCT")
    if dd_limit is not None:
        try:
            mdd = _max_drawdown_for_train_gate(pf_train, train_metrics)
            too_deep = mdd < -(float(dd_limit) / 100.0)
            if too_deep.any():
                train_metrics = train_metrics.copy()
                train_metrics[too_deep] = np.nan
        except Exception:
            pass

    # Optional stricter tier (REJECT_OPEN_END_IF_CLOSED_LT, default 0 = off).
    reject_open_lt = config.get("REJECT_OPEN_END_IF_CLOSED_LT", 0)
    if reject_open_lt > 0:
        try:
            open_end = _open_position_count_end_for_gate(pf_train, train_metrics)
            hold_mask = (open_end > 0) & (trade_for_gate < reject_open_lt)
            if hold_mask.any():
                train_metrics = train_metrics.copy()
                train_metrics[hold_mask] = np.nan
        except Exception:
            pass

    if train_metrics.isnull().all():
        _print_wfo_fold_all_rejected_diagnostics(
            fold_idx,
            train_ohlcv,
            train_metrics_before_gates,
            trade_for_gate,
            config,
            pf_train,
        )
        best_param_idx = train_metrics.index[0]
    else:
        best_param_idx = train_metrics.idxmax()

    fold_best_params = _extract_params(best_param_idx, train_metrics, param_names, param_grid)

    # Test: single-combo run; USE_VECTORIZED False is fine here (scalar params).
    wfo_test_cfg = {**config, "USE_VECTORIZED": False}
    test_engine = FastBacktest(test_ohlcv, fold_best_params, config=wfo_test_cfg, mover_mask=test_mask)
    pf_test = test_engine.run(show_progress=show_progress)

    return {
        "fold": fold_idx,
        "train_start": str(train_ohlcv.index[0]),
        "test_start": str(test_ohlcv.index[0]),
        "test_end": str(test_ohlcv.index[-1]),
        "params": _to_native(fold_best_params),
        "is_sharpe": _to_native(train_metrics.max()),
        "oos_sharpe": _to_native(pf_test.sharpe_ratio().mean()),
        "sortino": _to_native(pf_test.sortino_ratio().mean()),
        "profit": _to_native(pf_test.total_profit().sum()),
        "start_capital": _to_native(pf_test.init_cash.sum()),
        "end_capital": _to_native(pf_test.value().iloc[-1].sum()),
        "return_pct": _to_native(pf_test.total_return().mean() * 100),
        "train_metrics": train_metrics,
        "oos_returns": pf_test.returns(),
    }


def _wfo_train_metric_row_key(idx: Any) -> tuple[Any, ...]:
    """Hashable row label for train-metric Series (flatten MultiIndex tuples)."""
    if isinstance(idx, tuple):
        return tuple(idx)
    return (idx,)


def _weighted_robustness_series(
    is_metrics_by_fold: Dict[int, pd.Series],
    weights: Dict[int, float],
) -> pd.Series:
    """Fold-weighted mean per param combo keyed by flattened tuple rows.

    Avoids ``pd.DataFrame({fold: Series})`` on mixed MultiIndex depths (pandas can
    assert). Vectorized vs legacy WFO paths may return different row counts; we
    take the **union** of tuple keys and average each fold's value with fold
    weights, skipping NaNs (denominator = sum of weights for folds with finite
    values at that key).
    """
    fold_keys = sorted(weights.keys())
    if not fold_keys:
        raise ValueError("weights must be non-empty")

    flat: Dict[int, pd.Series] = {}
    for fk in fold_keys:
        s = is_metrics_by_fold[fk]
        keys = [_wfo_train_metric_row_key(i) for i in s.index]
        vals = np.asarray(s, dtype=float).ravel()
        if len(keys) != len(vals):
            raise ValueError(f"WFO fold {fk}: index/values length mismatch")
        flat[fk] = pd.Series(vals, index=pd.Index(keys, dtype=object))

    all_keys: set = set()
    for s in flat.values():
        all_keys.update(s.index.tolist())
    if not all_keys:
        return pd.Series(dtype=float)

    union_list = sorted(all_keys, key=lambda t: repr(t))
    union_idx = pd.Index(union_list, dtype=object)

    wvec = np.array([float(weights[fk]) for fk in fold_keys], dtype=float)
    if not np.any(wvec > 0):
        wvec = np.ones(len(fold_keys), dtype=float)

    fold_maps = [{k: float(v) for k, v in flat[fk].items()} for fk in fold_keys]

    combined = np.empty(len(union_idx), dtype=float)
    for i, key in enumerate(union_list):
        num = 0.0
        den = 0.0
        for j, fm in enumerate(fold_maps):
            if key not in fm:
                continue
            v = fm[key]
            if np.isfinite(v):
                w = float(wvec[j])
                num += v * w
                den += w
        combined[i] = num / den if den > 0.0 else float("nan")

    return pd.Series(combined, index=union_idx)


def _calculate_oos_robustness(
    oos_metrics_by_fold: Dict[int, float],
) -> Tuple[float, float]:
    """Recency-weighted mean OOS Sharpe and fold consistency fraction.

    Args:
        oos_metrics_by_fold: Dict mapping fold index → OOS Sharpe for the winning params.

    Returns:
        (oos_robustness_score, fold_consistency)
        - oos_robustness_score: recency-weighted mean OOS Sharpe (later folds count more).
        - fold_consistency: fraction of folds with positive OOS Sharpe (0.0 – 1.0).
    """
    if not oos_metrics_by_fold:
        return float("nan"), float("nan")
    fold_indices = sorted(oos_metrics_by_fold.keys())
    oos_vals = np.array(
        [float(oos_metrics_by_fold[f]) if oos_metrics_by_fold[f] is not None else float("nan")
         for f in fold_indices],
        dtype=float,
    )
    weights = np.array([float(f) for f in fold_indices], dtype=float)
    mask = np.isfinite(oos_vals)
    if not mask.any():
        return float("nan"), 0.0
    w_sum = float(weights[mask].sum())
    oos_rob = float(np.dot(oos_vals[mask], weights[mask]) / w_sum) if w_sum > 0 else float("nan")
    fold_cons = float(np.sum(oos_vals[mask] > 0) / int(mask.sum()))
    return oos_rob, fold_cons


def _calculate_robustness(
    is_metrics_by_fold: Dict[int, pd.Series],
    param_names: List[str],
    param_grid: Dict[str, Any],
    oos_metrics_by_fold: Optional[Dict[int, float]] = None,
    debug_metrics: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Calculates parameter robustness validation across folds.
    
    If oos_metrics_by_fold is provided, uses OOS Sharpe to weight each fold's
    contribution to robustness (penalizes parameter sets that don't generalize).
    Otherwise falls back to in-sample Sharpe with recency weighting.
    
    Args:
        is_metrics_by_fold: Dict mapping fold idx to in-sample Sharpe Series (all param combos)
        param_names: List of parameter names
        param_grid: Parameter grid definition
        oos_metrics_by_fold: Optional dict mapping fold idx to OOS Sharpe (for fold-level consistency)
    
    Returns:
        (robust_top_5, best_robust_params) tuples
    """
    # If OOS metrics provided, use them to measure generalization
    if oos_metrics_by_fold:
        fold_indices = sorted(oos_metrics_by_fold.keys())
        oos_values = np.array(
            [_coerce_metric_float(oos_metrics_by_fold.get(f)) for f in fold_indices],
            dtype=float,
        )
        # Min-max normalize OOS Sharpes to [0.1, 1.0] so folds with better OOS get
        # proportionally higher weight even when *all* OOS values are negative.
        # This prevents the old ratio-to-mean collapsing to a flat 0.5 weight when
        # oos_mean <= 0 (which made OOS weighting useless in poor-OOS regimes).
        oos_finite = oos_values[np.isfinite(oos_values)]
        oos_min_val = float(np.min(oos_finite)) if oos_finite.size > 0 else 0.0
        oos_max_val = float(np.max(oos_finite)) if oos_finite.size > 0 else 0.0
        oos_range_val = oos_max_val - oos_min_val

        weights = {}
        for f in fold_indices:
            oos_sharpe = _coerce_metric_float(oos_metrics_by_fold.get(f))
            recency_weight = float(f)
            if np.isfinite(oos_sharpe) and oos_range_val > 1e-8:
                consistency_weight = 0.1 + 0.9 * (oos_sharpe - oos_min_val) / oos_range_val
            else:
                consistency_weight = 0.5
            weights[f] = recency_weight * consistency_weight

        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights)
    else:
        # Original recency-weighted IS Sharpe (for backwards compatibility)
        weights = {f: float(f) for f in is_metrics_by_fold.keys()}
        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights)

    if debug_metrics:
        for fk, s in sorted(is_metrics_by_fold.items()):
            arr = np.asarray(s, dtype=float).ravel()
            n_fin = int(np.sum(np.isfinite(arr)))
            print(
                f"      [WFO_DEBUG] fold {fk}: train_metric len={len(s)} "
                f"finite={n_fin}"
            )
        if robustness_scores.size:
            rs_arr = np.asarray(robustness_scores, dtype=float).ravel()
            n_c = int(np.sum(np.isfinite(rs_arr)))
            print(
                f"      [WFO_DEBUG] combined robustness: len={len(robustness_scores)} "
                f"finite={n_c}"
            )
        else:
            print("      [WFO_DEBUG] combined robustness: empty")

    if robustness_scores.size == 0:
        return [], {}

    top_robust_idx = robustness_scores.sort_values(ascending=False).head(5)
    robust_top_5 = []

    for idx, score in top_robust_idx.items():
        extracted = _extract_params(idx, robustness_scores, param_names, param_grid)
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = float("nan")
        if not np.isfinite(score_f):
            score_f = float("nan")
        robust_top_5.append(
            {"params": _to_native(extracted), "robustness_score": score_f}
        )

    if not robust_top_5:
        return [], {}

    best_robust_params = robust_top_5[0]["params"]
    return robust_top_5, best_robust_params


def _calculate_wfo_bounds(
    total_len: int, n_splits: int, test_ratio: float
) -> List[Tuple[int, int, int, int]]:
    """Calculates exact integer index boundaries for WFO splits."""
    test_len = int(total_len / (test_ratio + n_splits))
    train_len = int(test_len * test_ratio)
    bounds = []

    for i in range(n_splits):
        start_idx = i * test_len
        train_end_idx = start_idx + train_len
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_len

        if i == n_splits - 1:
            test_end_idx = total_len

        bounds.append((start_idx, train_end_idx, test_start_idx, test_end_idx))

    return bounds


def _execute_wfo_loop(
    ohlcv: pd.DataFrame,
    mover_mask: Optional[pd.DataFrame],
    param_grid: Dict[str, Any],
    config: Dict[str, Any],
    param_names: List[str],
    n_splits: int,
    test_ratio: float,
    show_progress: bool,
    logger: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, pd.Series], List[pd.Series]]:
    """Iterates through the dataset and processes each WFO fold."""
    wfo_stats = []
    is_metrics_by_fold = {}
    oos_returns_list = []

    bounds = _calculate_wfo_bounds(len(ohlcv), n_splits, test_ratio)
    t0_wfo = time.time()

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(bounds):
        fold_idx = i + 1
        fold_start = time.time()

        train_idx = ohlcv.index[tr_start:tr_end]
        test_idx = ohlcv.index[te_start:te_end]

        fold_result = _process_wfo_fold(
            fold_idx,
            train_idx,
            test_idx,
            ohlcv,
            mover_mask,
            param_grid,
            config,
            show_progress,
            param_names,
        )

        is_metrics_by_fold[fold_idx] = fold_result.pop("train_metrics")
        oos_returns_list.append(fold_result.pop("oos_returns"))
        wfo_stats.append(fold_result)

        fold_elapsed = time.time() - fold_start
        total_elapsed = time.time() - t0_wfo
        avg_per_fold = total_elapsed / fold_idx
        eta = (n_splits - fold_idx) * avg_per_fold
        fold_msg = (
            f"Fold {fold_idx}/{n_splits} done in {fold_elapsed:.1f}s "
            f"| total {_eta_str(total_elapsed)} | ETA {_eta_str(eta)} (est. {_wall_clock_eta(eta)})"
        )
        print(f"  > {fold_msg}")
        if logger:
            logger.update(f"  {fold_msg}")

    return wfo_stats, is_metrics_by_fold, oos_returns_list


def _save_wfo_results(
    rm: Any,
    final_pf: Any,
    final_stats: Dict[str, Any],
    best_robust_params: Dict[str, Any],
    best_recent_params: Dict[str, Any],
    robust_top_5: List[Dict[str, Any]],
    wfo_stats: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """Handles persistence of metrics, parameters, and VectorBT dashboards."""
    rm.save_metrics(pd.DataFrame(wfo_stats), "wfo_results.csv")

    metadata = {
        **config,
        "wfo_fold_results": wfo_stats,
        "robustness_summary": {
            "top_robust": robust_top_5,
            "recent_vs_robust": {
                "recent": best_recent_params,
                "robust": best_robust_params,
                "is_equal": best_recent_params == best_robust_params,
            },
        },
    }

    rm.save_run_results(
        params=best_robust_params,
        metrics=_to_native(final_stats),
        metadata=_to_native(metadata),
    )
    rm.save_vbt_dashboard(final_pf, "final_robust_model_dashboard")
    print(f"WFO Results saved to: {rm.run_dir}")


def run_wfo_orchestrator(
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Central orchestration function for Walk-Forward Optimization."""
    rm = ResultsManager("run_wfo") if save_results else None
    from ggTrader.utils.plotting import plot_wfo_splits

    ohlcv, mover_mask = load_data_with_movers(config)
    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config.get("TEST_RATIO", 3.0)
    param_names = list(param_grid.keys())

    print(f"Starting WFO Loop ({n_splits} splits, Ratio: {test_ratio}:1)...")

    plot_wfo_splits(ohlcv, n_splits, test_ratio, results_manager=rm)

    wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
        ohlcv, mover_mask, param_grid, config, param_names, n_splits, test_ratio, show_progress, None
    )

    # Extract OOS Sharpe ratios from wfo_stats to measure generalization
    oos_metrics_by_fold = {fold_idx: stats["oos_sharpe"] for fold_idx, stats in enumerate(wfo_stats, 1)}
    
    dbg = bool(config.get("WFO_DEBUG_METRICS", False))
    robust_top_5, best_robust_params = _calculate_robustness(
        is_metrics_by_fold,
        param_names,
        param_grid,
        oos_metrics_by_fold,
        debug_metrics=dbg,
    )
    if not best_robust_params:
        best_robust_params = _default_params_from_grid(param_grid)
    best_recent_params = wfo_stats[-1]["params"]

    final_engine = FastBacktest(ohlcv, best_robust_params, config=config, mover_mask=mover_mask)
    final_pf = final_engine.run(show_progress=show_progress)
    final_stats = final_engine.get_stats()

    if save_results and rm:
        _save_wfo_results(
            rm,
            final_pf,
            final_stats,
            best_robust_params,
            best_recent_params,
            robust_top_5,
            wfo_stats,
            config,
        )

    return {
        "final_portfolio": final_pf,
        "wfo_stats": wfo_stats,
        "robust_top_5": robust_top_5,
        "best_robust_params": best_robust_params,
        "best_recent_params": best_recent_params,
        "results_manager": rm,
    }


def run_wfo_per_coin_orchestrator(
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Walk-Forward Optimization with per-coin independent optimization.

    Each symbol is optimized independently, then combined into a single portfolio.

    Args:
        config: Portfolio and data configuration.
        param_grid: Parameter grid for strategy.
        save_results: Whether to save results.
        show_progress: Whether to show progress bars.

    Returns:
        Dictionary with results including per-coin params and combined portfolio.
    """
    from ggTrader.utils.plotting import plot_wfo_splits

    rm = ResultsManager("run_wfo_per_coin") if save_results else None
    ohlcv, mover_mask = load_data_with_movers(config)
    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config.get("TEST_RATIO", 3.0)
    param_names = list(param_grid.keys())

    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    print(f"\nStarting Per-Coin WFO ({len(symbols)} symbols, {n_splits} splits)...")

    plot_wfo_splits(ohlcv, n_splits, test_ratio, results_manager=rm)

    # Dictionary to store best params per symbol
    per_coin_results: Dict[str, Dict[str, Any]] = {}

    # Optimize each symbol independently
    for symbol in symbols:
        print(f"\n--- Optimizing {symbol} ---")

        # Extract single-symbol data
        symbol_ohlcv = ohlcv[[symbol]]
        symbol_mover_mask = mover_mask[[symbol]] if mover_mask is not None else None

        wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
            symbol_ohlcv,
            symbol_mover_mask,
            param_grid,
            config,
            param_names,
            n_splits,
            test_ratio,
            show_progress,
            None,
        )

        # Extract OOS Sharpe ratios from wfo_stats to measure generalization
        oos_metrics_by_fold = {fold_idx: stats["oos_sharpe"] for fold_idx, stats in enumerate(wfo_stats, 1)}
        
        dbg = bool(config.get("WFO_DEBUG_METRICS", False))
        robust_top_5, best_robust_params = _calculate_robustness(
            is_metrics_by_fold,
            param_names,
            param_grid,
            oos_metrics_by_fold,
            debug_metrics=dbg,
        )
        if not best_robust_params:
            best_robust_params = _default_params_from_grid(param_grid)

        per_coin_results[symbol] = {
            "best_params": best_robust_params,
            "wfo_stats": wfo_stats,
            "robust_top_5": robust_top_5,
        }

        print(f"  > {symbol} Best Params: {best_robust_params}")

    # Build combined portfolio with per-coin params
    print("\nBuilding combined portfolio with per-coin parameters...")

    combined_entries_list = []
    combined_exits_list = []
    combined_close_list = []
    all_cols = []

    for symbol in symbols:
        best_params = per_coin_results[symbol]["best_params"]
        symbol_ohlcv = ohlcv[[symbol]]

        engine = FastBacktest(symbol_ohlcv, best_params, config=config)
        pf = engine.run(show_progress=False)

        # Extract signals
        close = symbol_ohlcv.xs("close", axis=1, level=1, drop_level=True)
        entries = pf.entries()
        exits = pf.exits()

        combined_entries_list.append(entries)
        combined_exits_list.append(exits)
        combined_close_list.append(close)
        all_cols.append(symbol)

    # Concatenate all symbols
    combined_entries = pd.concat(combined_entries_list, axis=1)
    combined_exits = pd.concat(combined_exits_list, axis=1)
    combined_close = pd.concat(combined_close_list, axis=1)

    # Create final portfolio
    final_pf = vbt.Portfolio.from_signals(
        close=combined_close,
        entries=combined_entries,
        exits=combined_exits,
        init_cash=float(config["START_CASH"]),
        fees=float(config["FEES"]),
        slippage=float(config["SLIPPAGE"]),
        freq=config["FREQ"],
        size=float(config["PORTFOLIO_SHARE"]),
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(combined_entries.shape[1], 0),
    ).copy()

    final_stats = {
        "total_value": _safe(final_pf.final_value().sum()),
        "total_profit": _safe(final_pf.total_profit().sum()),
        "profit_pct": _safe(
            (final_pf.total_profit().sum() / float(config["START_CASH"])) * 100
        ),
        "total_trades": int(final_pf.trades.count().sum()),
        "win_rate": _safe(final_pf.trades.win_rate().mean()) * 100,
        "sharpe": _safe(final_pf.sharpe_ratio().mean()),
        "sortino": _safe(final_pf.sortino_ratio().mean()),
        "max_drawdown": _safe(final_pf.max_drawdown().min()) * 100,
    }
    _enrich_final_stats_with_cagr_and_benchmark(final_stats, combined_close, config)

    if save_results and rm:
        # Save per-coin results
        per_coin_params_list = []
        for symbol, results in per_coin_results.items():
            params_dict = {**results["best_params"], "symbol": symbol}
            per_coin_params_list.append(params_dict)

        per_coin_df = pd.DataFrame(per_coin_params_list)
        rm.save_metrics(per_coin_df, "per_coin_params.csv")

        # Save overall results
        metadata = {
            **config,
            "per_coin_results": _to_native(per_coin_results),
        }

        rm.save_run_results(params={"per_coin": _to_native(per_coin_results)}, metrics=final_stats, metadata=metadata)
        rm.save_vbt_dashboard(final_pf, "combined_portfolio_dashboard")
        print(f"\nPer-Coin WFO Results saved to: {rm.run_dir}")

    return {
        "final_portfolio": final_pf,
        "per_coin_results": per_coin_results,
        "final_stats": final_stats,
        "results_manager": rm,
    }


def _safe(val: Any, default: float = 0.0) -> float:
    """Replace None, NaN, or Inf with default for JSON safety."""
    import math

    if val is None:
        return default
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(v) or math.isinf(v)) else v


def _as_optional_float(x: Any) -> Any:
    """Return a finite float or None (for JSON/report fields where 0 would mislead)."""
    import math

    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def _years_from_price_index(index: Any) -> float:
    """Calendar years between first and last bar (for CAGR)."""
    import math

    if index is None:
        return float("nan")
    try:
        idx = pd.to_datetime(pd.Index(index))
    except (TypeError, ValueError):
        return float("nan")
    if len(idx) < 2:
        return float("nan")
    delta = idx[-1] - idx[0]
    sec = float(delta.total_seconds())
    if sec <= 0 or not math.isfinite(sec):
        return float("nan")
    return sec / (365.25 * 24 * 3600.0)


def _cagr_percent(total_return_pct: float, years: float) -> float:
    """Annualized geometric return (%) from total return (%) over ``years``."""
    import math

    if not (years > 0 and math.isfinite(years)):
        return float("nan")
    try:
        mult = 1.0 + float(total_return_pct) / 100.0
    except (TypeError, ValueError):
        return float("nan")
    if mult <= 0:
        return float("nan")
    return (mult ** (1.0 / years) - 1.0) * 100.0


def _equal_weight_buy_hold_portfolio_stats(
    close: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Equal-weight spot B&H: buy all names on bar 0, sell on last bar; same vbt costs as WFO."""
    n = close.shape[1]
    rows = close.shape[0]
    empty: Dict[str, Any] = {
        "profit_pct": None,
        "cagr_pct": None,
        "sharpe": None,
        "max_drawdown": None,
        "total_trades": 0,
    }
    if n == 0 or rows < 2:
        return empty

    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    entries.iloc[0] = True
    exits.iloc[-1] = True

    bench_pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=float(config["START_CASH"]),
        fees=float(config["FEES"]),
        slippage=float(config["SLIPPAGE"]),
        freq=config["FREQ"],
        size=1.0 / n,
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(n, 0),
    ).copy()

    years = _years_from_price_index(close.index)
    init_cash = float(config["START_CASH"])
    profit_pct = float((bench_pf.total_profit().sum() / init_cash) * 100.0)
    cagr = _cagr_percent(profit_pct, years)
    sh = float(bench_pf.sharpe_ratio().mean())
    dd = float(bench_pf.max_drawdown().min()) * 100.0

    return {
        "profit_pct": _as_optional_float(profit_pct),
        "cagr_pct": _as_optional_float(cagr),
        "sharpe": _as_optional_float(sh),
        "max_drawdown": _as_optional_float(dd),
        "total_trades": int(bench_pf.trades.count().sum()),
    }


def _enrich_final_stats_with_cagr_and_benchmark(
    final_stats: Dict[str, Any],
    combined_close: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Add CAGR, calendar span, and equal-weight B&H benchmark fields to ``final_stats``."""
    years = _years_from_price_index(combined_close.index)
    rp = float(final_stats.get("profit_pct", 0.0) or 0.0)
    strat_cagr = _cagr_percent(rp, years)

    try:
        idx = pd.to_datetime(pd.Index(combined_close.index))
        final_stats["backtest_start"] = idx[0].strftime("%Y-%m-%d") if len(idx) else None
        final_stats["backtest_end"] = idx[-1].strftime("%Y-%m-%d") if len(idx) else None
    except (TypeError, ValueError, IndexError):
        final_stats["backtest_start"] = None
        final_stats["backtest_end"] = None

    final_stats["backtest_years"] = _as_optional_float(years)
    final_stats["cagr_pct"] = _as_optional_float(strat_cagr)

    bench = _equal_weight_buy_hold_portfolio_stats(combined_close, config)
    final_stats["benchmark_label"] = (
        "Equal-weight buy-and-hold: one buy per symbol on the first bar and one sell on the "
        "last bar; same START_CASH, FEES, SLIPPAGE, and bar frequency as the strategy run."
    )
    final_stats["benchmark_profit_pct"] = bench["profit_pct"]
    final_stats["benchmark_cagr_pct"] = bench["cagr_pct"]
    final_stats["benchmark_sharpe"] = bench["sharpe"]
    final_stats["benchmark_max_drawdown"] = bench["max_drawdown"]
    final_stats["benchmark_total_trades"] = bench["total_trades"]

    sc = final_stats["cagr_pct"]
    bc = bench["cagr_pct"]
    if sc is not None and bc is not None:
        final_stats["excess_cagr_pct"] = float(sc) - float(bc)
    else:
        final_stats["excess_cagr_pct"] = None


def analyze_sensitivity_results(
    results_df: pd.DataFrame,
    param_grid: Dict[str, Any],
    top_percentile: int = 50,
) -> Dict[str, Any]:
    """Analyze sensitivity results to identify impactful parameters and narrow ranges.

    For each parameter, computes the variance of Sharpe ratio across that parameter's
    unique values (marginal effect). Returns a narrowed parameter grid that keeps
    values producing top-N% Sharpe ratios.

    Args:
        results_df: DataFrame with parameter columns and 'Sharpe Ratio' column
        param_grid: Original parameter grid
        top_percentile: Keep parameter values in top N% by mean Sharpe ratio

    Returns:
        Narrowed parameter grid dict with reduced value ranges
    """
    if results_df.empty:
        return param_grid

    narrowed = {}

    for param_name in param_grid.keys():
        if param_name not in results_df.columns:
            # Parameter not varied in sensitivity; keep original
            narrowed[param_name] = param_grid[param_name]
            continue

        # Group by this parameter and compute mean Sharpe per value
        grouped = results_df.groupby(param_name)["Sharpe Ratio"].agg(["mean", "count"])
        grouped = grouped.sort_values("mean", ascending=False)

        # Keep top N% of values by mean Sharpe, with floor of at least 2 values
        threshold_idx = max(2, int(len(grouped) * (top_percentile / 100)))
        top_values = grouped.head(threshold_idx).index.tolist()

        narrowed[param_name] = top_values

        # Print summary
        print(
            f"  {param_name}: {len(grouped)} unique values -> {len(top_values)} "
            f"(top {top_percentile}%): {top_values}"
        )

    return narrowed


def run_frozen_params_combined_backtest(
    ohlcv: pd.DataFrame,
    per_coin_results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    *,
    exit_tournament: List[str],
    save_results: bool = False,
    results_manager: Any = None,
    phase_title: str = "PHASE 3: FINAL VALIDATION BACKTEST (FULL 3-YEAR RANGE)",
    combined_portfolio_label: str = "Final Combined Portfolio (Full 3-Year Range)",
    logger: Any = None,
) -> Dict[str, Any]:
    """Replay WFO-selected (entry, exit, params) on ``ohlcv`` and build one combined portfolio.

    Used for full-history Phase 3 and for optional recent-window validation (same frozen book).
    """
    print("\n" + "=" * 100)
    print(phase_title)
    print("=" * 100)

    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    combined_entries_list = []
    combined_exits_list = []
    combined_close_list = []
    per_coin_final_stats: Dict[str, Any] = {}

    n_sym_total = len(symbols)
    for sym_idx, symbol in enumerate(symbols, start=1):
        if symbol not in per_coin_results:
            print(f"\nSkipping {symbol}: no WFO selection (not in per_coin_results).")
            continue

        strategy_name = per_coin_results[symbol]["best_strategy"]
        exit_name = per_coin_results[symbol]["best_exit"]
        best_params = per_coin_results[symbol]["best_params"]
        robustness_score = per_coin_results[symbol]["robustness_score"]
        selection_reason = per_coin_results[symbol].get("selection_reason", "wfo_robustness")
        best_label = f"{strategy_name}+{exit_name}"
        if selection_reason != "wfo_robustness":
            print(
                f"\nGenerating signals for {symbol} with {best_label} "
                f"(selection={selection_reason})..."
            )
        else:
            print(f"\nGenerating signals for {symbol} with {best_label}...")

        symbol_ohlcv = ohlcv[[symbol]]

        config_for_final = {
            **config,
            "ENTRY_STRATEGY": strategy_name,
            "EXIT_STRATEGY": exit_name,
            "USE_VECTORIZED": False,
        }
        engine = FastBacktest(symbol_ohlcv, best_params, config=config_for_final)
        engine.run(show_progress=False)

        close = symbol_ohlcv.xs("close", axis=1, level=1, drop_level=True)
        entries, exits, _ = engine._generate_signals(show_progress=False)

        combined_entries_list.append(entries)
        combined_exits_list.append(exits)
        combined_close_list.append(close)

        stats = engine.get_stats()
        per_coin_final_stats[symbol] = {
            "strategy": strategy_name,
            "exit": exit_name,
            "params": _to_native(best_params),
            "selection_reason": selection_reason,
            **stats,
        }
        rs_disp = _format_robustness_metric(robustness_score)
        sel_note = (
            "" if selection_reason == "wfo_robustness" else f" | sel={selection_reason}"
        )
        phase3_msg = (
            f"  > {symbol} ({sym_idx}/{n_sym_total}): Best={best_label} | "
            f"Robustness={rs_disp}{sel_note} | Win Rate={stats['win_rate']:.2f}% | "
            f"Trades={stats['total_trades']}"
        )
        print(phase3_msg)
        if logger:
            logger.update(phase3_msg)

    if not combined_entries_list:
        raise ValueError(
            "Frozen-params backtest produced no symbols (empty OHLCV or no matching per_coin_results)."
        )

    combined_entries = pd.concat(combined_entries_list, axis=1)
    combined_exits = pd.concat(combined_exits_list, axis=1)
    combined_close = pd.concat(combined_close_list, axis=1)

    final_pf = vbt.Portfolio.from_signals(
        close=combined_close,
        entries=combined_entries,
        exits=combined_exits,
        init_cash=float(config["START_CASH"]),
        fees=float(config["FEES"]),
        slippage=float(config["SLIPPAGE"]),
        freq=config["FREQ"],
        size=float(config["PORTFOLIO_SHARE"]),
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(combined_entries.shape[1], 0),
    ).copy()

    final_stats = {
        "total_value": _safe(final_pf.final_value().sum()),
        "total_profit": _safe(final_pf.total_profit().sum()),
        "profit_pct": _safe(
            (final_pf.total_profit().sum() / float(config["START_CASH"])) * 100
        ),
        "total_trades": int(final_pf.trades.count().sum()),
        "win_rate": _safe(final_pf.trades.win_rate().mean()) * 100,
        "sharpe": _safe(final_pf.sharpe_ratio().mean()),
        "sortino": _safe(final_pf.sortino_ratio().mean()),
        "max_drawdown": _safe(final_pf.max_drawdown().min()) * 100,
    }
    _enrich_final_stats_with_cagr_and_benchmark(final_stats, combined_close, config)

    print(f"\n{combined_portfolio_label}:")
    print(f"  Total Return: {final_stats['profit_pct']:.2f}%")
    cagr_s = (
        f"{final_stats['cagr_pct']:.2f}%"
        if final_stats.get("cagr_pct") is not None
        else "n/a"
    )
    print(f"  CAGR: {cagr_s}")
    b_cagr = final_stats.get("benchmark_cagr_pct")
    b_cagr_s = f"{b_cagr:.2f}%" if b_cagr is not None else "n/a"
    print(f"  Benchmark CAGR (EW B&H): {b_cagr_s}")
    print(f"  Sharpe Ratio: {final_stats['sharpe']:.4f}")
    print(f"  Max Drawdown: {final_stats['max_drawdown']:.2f}%")
    print(f"  Total Trades: {final_stats['total_trades']}")
    print(f"  Win Rate: {final_stats['win_rate']:.2f}%")

    if save_results and results_manager:
        strategy_summary = []
        for symbol, results in per_coin_results.items():
            strategy_summary.append({
                "symbol": symbol,
                "strategy": results["best_strategy"],
                "exit": results.get("best_exit", "atr_trailing"),
                "robustness_score": results["robustness_score"],
                "selection_reason": results.get("selection_reason", "wfo_robustness"),
            })

        strategy_df = pd.DataFrame(strategy_summary)
        results_manager.save_metrics(strategy_df, "per_coin_strategy_selection.csv")

        final_stats_df = pd.DataFrame([
            {"symbol": sym, **stats} for sym, stats in per_coin_final_stats.items()
        ])
        results_manager.save_metrics(final_stats_df, "per_coin_final_stats.csv")

        metadata = {
            **config,
            "exit_tournament": exit_tournament,
            "per_coin_results": _to_native(per_coin_results),
            "per_coin_final_stats": _to_native(per_coin_final_stats),
        }

        results_manager.save_run_results(
            params={"per_coin": _to_native(per_coin_results)},
            metrics=final_stats,
            metadata=metadata,
        )
        results_manager.save_vbt_dashboard(final_pf, "combined_portfolio_final_dashboard")
        print(f"\nMulti-Strategy WFO Results saved to: {results_manager.run_dir}")

    return {
        "final_portfolio": final_pf,
        "per_coin_final_stats": per_coin_final_stats,
        "final_stats": final_stats,
    }


def run_multi_strategy_per_coin_wfo(
    config: Dict[str, Any],
    strategy_param_grids: Dict[str, Dict],
    save_results: bool = True,
    show_progress: bool = True,
    logger: Any = None,
) -> Dict[str, Any]:
    """Run WFO for each symbol across multiple (entry, exit) combinations.

    For each symbol the tournament tests every registered entry strategy against
    every exit in ``config["EXIT_TOURNAMENT"]`` (default from ``full_pipeline_config``
    is typically ``["atr_trailing"]`` only; use ``--dual-exits`` or two names for both).
    The winner is picked by robustness score.  Phase 3 replays the full range with the
    winning (entry, exit, params) triple.

    Args:
        config: Portfolio and data configuration.  Supports optional key
            ``EXIT_TOURNAMENT`` (list of exit names); parsed against ``EXIT_REGISTRY``.
        strategy_param_grids: Dict mapping entry strategy name → merged param grid
            (entry-only keys already merged with exit-axis keys by the caller via
            ``build_param_grid``).
        save_results: Whether to save results.
        show_progress: Whether to show progress bars.

    Returns:
        Dict with per_coin_results, final_portfolio, final_stats, results_manager.
    """
    from ggTrader.indicators.strategies import EXIT_REGISTRY

    exit_tournament = parse_exit_tournament(
        config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys())),
        EXIT_REGISTRY,
    )

    rm = ResultsManager("run_wfo_per_coin_multi_strategy") if save_results else None
    ohlcv, mover_mask = load_data_with_movers(config)

    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config.get("TEST_RATIO", 3.0)

    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    n_symbols = len(symbols)
    n_combos = len(strategy_param_grids) * len(exit_tournament)
    print(
        f"\nStarting Multi-Strategy Per-Coin WFO "
        f"({n_symbols} symbols, {n_splits} splits, "
        f"{len(strategy_param_grids)} entries × {len(exit_tournament)} exits = {n_combos} combos)..."
    )
    print(f"  Exit tournament: {exit_tournament}")

    # Dictionary to store best (entry, exit, params) per symbol.
    per_coin_results: Dict[str, Dict[str, Any]] = {}
    t0_wfo_coins = time.time()

    # Phase 2: Per-coin WFO loop — test all (entry, exit) combos for each coin.
    for coin_idx, symbol in enumerate(symbols):
        coin_start = time.time()
        print(f"\n--- Optimizing {symbol} ({coin_idx+1}/{n_symbols}) ---")

        symbol_ohlcv = ohlcv[[symbol]]
        symbol_mover_mask = mover_mask[[symbol]] if mover_mask is not None else None

        best_strategy: Optional[str] = None
        best_exit: Optional[str] = None
        best_robust_score = float("-inf")
        best_params_for_coin: Dict[str, Any] = {}
        best_wfo_stats: List[Dict] = []
        best_robust_top_5: List[Dict] = []
        best_is_robustness_score: float = float("-inf")
        best_oos_robustness_score: float = float("nan")
        best_fold_consistency: float = float("nan")
        debug_wfo = bool(config.get("WFO_DEBUG_METRICS", False))

        for strategy_name, param_grid in strategy_param_grids.items():
            for exit_name in exit_tournament:
                label = f"{strategy_name}+{exit_name}"
                print(f"  Testing: {label}")

                config_combo = {
                    **config,
                    "ENTRY_STRATEGY": strategy_name,
                    "EXIT_STRATEGY": exit_name,
                }

                wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
                    symbol_ohlcv,
                    symbol_mover_mask,
                    param_grid,
                    config_combo,
                    list(param_grid.keys()),
                    n_splits,
                    test_ratio,
                    show_progress,
                    logger,
                )

                oos_metrics_by_fold = {
                    fold_idx: stats["oos_sharpe"]
                    for fold_idx, stats in enumerate(wfo_stats, 1)
                }

                robust_top_5, best_robust_params = _calculate_robustness(
                    is_metrics_by_fold,
                    list(param_grid.keys()),
                    param_grid,
                    oos_metrics_by_fold,
                    debug_metrics=debug_wfo,
                )

                if robust_top_5:
                    robustness_score = float(robust_top_5[0]["robustness_score"])
                else:
                    robustness_score = float("-inf")

                # OOS-direct robustness: recency-weighted mean of per-fold OOS Sharpe.
                oos_rob_combo, fold_cons_combo = _calculate_oos_robustness(oos_metrics_by_fold)
                oos_blend_alpha = float(config.get("OOS_ROBUSTNESS_BLEND_ALPHA", 0.5))
                if np.isfinite(oos_rob_combo) and np.isfinite(robustness_score):
                    gate_score = (1.0 - oos_blend_alpha) * robustness_score + oos_blend_alpha * oos_rob_combo
                elif np.isfinite(oos_rob_combo):
                    gate_score = oos_rob_combo
                else:
                    gate_score = robustness_score

                print(
                    f"    {label} robustness: IS={_format_robustness_metric(robustness_score)} "
                    f"OOS={_format_robustness_metric(oos_rob_combo)} "
                    f"gate={_format_robustness_metric(gate_score)} "
                    f"consistency={fold_cons_combo:.0%}"
                )

                if _is_better_robustness(gate_score, best_robust_score):
                    best_robust_score = gate_score
                    best_strategy = strategy_name
                    best_exit = exit_name
                    best_params_for_coin = best_robust_params
                    best_wfo_stats = wfo_stats
                    best_robust_top_5 = robust_top_5
                    best_is_robustness_score = robustness_score
                    best_oos_robustness_score = oos_rob_combo
                    best_fold_consistency = fold_cons_combo

        selection_reason = "wfo_robustness"
        if best_strategy is None or not np.isfinite(best_robust_score):
            fb_s, fb_e, fb_p = _wfo_per_coin_fallback_triple(
                strategy_param_grids, exit_tournament
            )
            best_strategy = fb_s
            best_exit = fb_e
            best_params_for_coin = fb_p
            best_robust_score = float("-inf")
            best_robust_top_5 = []
            selection_reason = "fallback_no_finite_robustness"
            print(
                f"  WARNING: {symbol} — no finite WFO robustness winner; "
                f"using fallback {best_strategy}+{best_exit} with first grid values."
            )

        per_coin_results[symbol] = {
            "best_strategy":        best_strategy,
            "best_exit":            best_exit,
            "best_params":          best_params_for_coin,
            "robustness_score":     best_robust_score,       # blended gate score
            "is_robustness_score":  best_is_robustness_score,
            "oos_robustness_score": best_oos_robustness_score,
            "fold_consistency":     best_fold_consistency,
            "wfo_stats":            best_wfo_stats,
            "robust_top_5":         best_robust_top_5,
            "selection_reason":     selection_reason,
        }

        coin_elapsed = time.time() - coin_start
        total_elapsed = time.time() - t0_wfo_coins
        avg_per_coin = total_elapsed / (coin_idx + 1)
        eta = (n_symbols - coin_idx - 1) * avg_per_coin
        status_msg = (
            f"  > {symbol} ({coin_idx+1}/{n_symbols}): WFO complete in {coin_elapsed:.0f}s | "
            f"ETA {_eta_str(eta)} (est. {_wall_clock_eta(eta)}) "
            f"(full-range Best / Win Rate printed in Phase 3)"
        )
        print(status_msg)
        if logger:
            logger.update(status_msg)

    # Filter out coins that failed to meet the minimum robustness threshold.
    # Set MIN_ROBUSTNESS_SCORE=None in config to disable.
    min_robust_cfg = config.get("MIN_ROBUSTNESS_SCORE")
    if min_robust_cfg is not None:
        min_robust = float(min_robust_cfg)
        skipped = [
            sym for sym, r in per_coin_results.items()
            if not np.isfinite(r["robustness_score"]) or r["robustness_score"] < min_robust
        ]
        if skipped:
            print(
                f"\n  [Robustness gate] Dropping {len(skipped)} coin(s) below "
                f"MIN_ROBUSTNESS_SCORE={min_robust}: {skipped}"
            )
            per_coin_results = {sym: r for sym, r in per_coin_results.items() if sym not in skipped}
        if not per_coin_results:
            print("  WARNING: All coins dropped by robustness gate — lowering threshold or "
                  "setting MIN_ROBUSTNESS_SCORE=None is recommended.")

    phase3_out = run_frozen_params_combined_backtest(
        ohlcv,
        per_coin_results,
        config,
        exit_tournament=exit_tournament,
        save_results=save_results,
        results_manager=rm,
        logger=logger,
    )
    final_pf = phase3_out["final_portfolio"]
    per_coin_final_stats = phase3_out["per_coin_final_stats"]
    final_stats = phase3_out["final_stats"]

    return {
        "final_portfolio": final_pf,
        "per_coin_results": per_coin_results,
        "per_coin_final_stats": per_coin_final_stats,
        "final_stats": final_stats,
        "results_manager": rm,
    }
