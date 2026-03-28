"""Grid search and sensitivity analysis orchestration."""

import gc
import itertools
import time
from math import prod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.metrics import _apply_sensitivity_train_gates, _trade_counts_for_train_gate, _train_metric_series
from ggTrader.core.orchestrator_utils import (
    _coerce_strategy_params_for_engine,
    _eta_str,
    _extract_params,
    _log_memory_usage,
    _to_native,
    _wall_clock_eta,
)
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers


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


def _vectorized_grid_metrics(
    pf: Any,
    engine: Any,
    param_keys: List[str],
    config: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series]:
    """WFO train path: same alignment as vectorized sensitivity (TRAIN_METRIC-aware)."""
    return _metric_series_from_vectorized_pf(pf, engine, param_keys, config)


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
    sharpe_series, trade_for_gate = _metric_series_from_vectorized_pf(pf, engine, keys, config)
    closed_for_output = trade_for_gate.copy()
    sharpe_series = _apply_sensitivity_train_gates(sharpe_series, trade_for_gate, pf, config)

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
            return _execute_sensitivity_vectorized(ohlcv, config, param_grid, show_progress, logger)
        except Exception as e:
            print(
                f"WARNING: Vectorized sensitivity failed ({type(e).__name__}: {e}); "
                "falling back to chunked path (~100× slower). "
                "Set USE_VECTORIZED_SENSITIVITY=False to suppress this warning."
            )

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
        print(f"  > Processing chunk {chunk_idx}/{total_chunks} (combos {i}-{chunk_end})...")
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
            f"Chunk {chunk_idx}/{total_chunks} done in {time.time() - chunk_start:.1f}s "
            f"| total elapsed {_eta_str(elapsed)} | ETA {_eta_str(eta)} "
            f"(est. {_wall_clock_eta(eta)})"
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
    rm = (
        ResultsManager(
            "run_sensitivity",
            explicit_run_dir=config.get("EXPLICIT_RUN_DIR"),
            pipeline_stage=config.get("PIPELINE_STAGE"),
        )
        if save_results
        else None
    )

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
