"""Centralized orchestration logic for backtesting, sensitivity analysis, and WFO."""

import gc
import itertools
import traceback
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from tabulate import tabulate

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers


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


def _process_sensitivity_chunk(
    chunk: List[Tuple],
    keys: List[str],
    config: Dict[str, Any],
    ohlcv: pd.DataFrame,
    show_progress: bool,
) -> pd.Series:
    """Helper to process a single chunk of sensitivity parameters."""
    chunk_params = {k: [c[j] for c in chunk] for j, k in enumerate(keys)}
    chunk_config = {**config, "PARAM_PRODUCT": False}

    engine = FastBacktest(ohlcv, chunk_params, config=chunk_config)
    pf = engine.run(show_progress=show_progress)

    sharpe_series = pf.sharpe_ratio()

    # Apply filtering by trade count
    min_trades = config.get("MIN_TRADES", 0)
    if min_trades > 0:
        trade_counts = pf.trades.count()
        low_trade_mask = trade_counts < min_trades
        if low_trade_mask.any():
            sharpe_series[low_trade_mask] = np.nan

    # Free RAM
    del pf
    del engine
    gc.collect()

    return sharpe_series


def _execute_sensitivity_grid(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    show_progress: bool,
) -> pd.Series:
    """Generates combinations, splits into chunks, and executes the grid search."""
    keys = list(param_grid.keys())
    values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
    combinations = list(itertools.product(*values))
    total_total = len(combinations)

    chunk_size = config.get("CHUNK_SIZE", 500)
    total_chunks = (total_total + chunk_size - 1) // chunk_size

    print(
        f"Running Vectorized Sensitivity Analysis in {total_chunks} chunks "
        f"({total_total} total combinations, chunk_size={chunk_size})..."
    )

    all_sharpe_series = []

    for i in range(0, total_total, chunk_size):
        chunk_idx = i // chunk_size + 1
        chunk = combinations[i : i + chunk_size]
        print(
            f"  > Processing chunk {chunk_idx} of {total_chunks} "
            f"({i} to {min(i + chunk_size, total_total)})..."
        )

        sharpe_series = _process_sensitivity_chunk(chunk, keys, config, ohlcv, show_progress)
        all_sharpe_series.append(sharpe_series)

    return pd.concat(all_sharpe_series)


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
) -> Dict[str, Any]:
    """Orchestrate a vectorized sensitivity analysis (grid search)."""
    rm = ResultsManager("run_sensitivity") if save_results else None

    ohlcv, _ = load_data_with_movers(config)
    param_names = list(param_grid.keys())

    # Execute grid search
    sharpe_series = _execute_sensitivity_grid(ohlcv, config, param_grid, show_progress)

    best_idx = sharpe_series.idxmax()
    best_params = _to_native(_extract_params(best_idx, sharpe_series, param_names, param_grid))

    results_df = sharpe_series.reset_index()
    results_df.columns = [str(col).replace("sf_", "") for col in results_df.columns[:-1]] + [
        "Sharpe Ratio"
    ]

    # Evaluate best case
    best_engine = FastBacktest(ohlcv, best_params, config=config)
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
    """Helper to process a single WFO fold (Train & Test)."""
    train_ohlcv = ohlcv.loc[train_idx]
    test_ohlcv = ohlcv.loc[test_idx]

    train_mask = mover_mask.loc[train_idx] if mover_mask is not None else None
    test_mask = mover_mask.loc[test_idx] if mover_mask is not None else None

    # Train
    train_engine = FastBacktest(train_ohlcv, param_grid, config=config, mover_mask=train_mask)
    pf_train = train_engine.run(show_progress=show_progress)
    train_metrics = pf_train.sharpe_ratio()

    min_trades = config.get("MIN_TRADES", 0)
    if min_trades > 0:
        trade_counts = pf_train.trades.count()
        low_trade_mask = trade_counts < min_trades
        if low_trade_mask.any():
            train_metrics[low_trade_mask] = np.nan

    if train_metrics.isnull().all():
        print(f"  WARNING: Fold {fold_idx} - All param combos rejected (likely low trades).")
        best_param_idx = train_metrics.index[0]
    else:
        best_param_idx = train_metrics.idxmax()

    fold_best_params = _extract_params(best_param_idx, train_metrics, param_names, param_grid)

    # Test
    test_engine = FastBacktest(test_ohlcv, fold_best_params, config=config, mover_mask=test_mask)
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


def _calculate_robustness(
    is_metrics_by_fold: Dict[int, pd.Series],
    param_names: List[str],
    param_grid: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Calculates parameter robustness validation across folds."""
    df_is_all = pd.DataFrame(is_metrics_by_fold)
    weights = {f: f for f in is_metrics_by_fold.keys()}
    robustness_scores = (df_is_all * pd.Series(weights)).sum(axis=1) / sum(weights.values())

    top_robust_idx = robustness_scores.sort_values(ascending=False).head(5)
    robust_top_5 = []

    for idx, score in top_robust_idx.items():
        extracted = _extract_params(idx, robustness_scores, param_names, param_grid)
        robust_top_5.append(
            {"params": _to_native(extracted), "robustness_score": _to_native(score)}
        )

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
) -> Tuple[List[Dict[str, Any]], Dict[int, pd.Series], List[pd.Series]]:
    """Iterates through the dataset and processes each WFO fold."""
    wfo_stats = []
    is_metrics_by_fold = {}
    oos_returns_list = []

    bounds = _calculate_wfo_bounds(len(ohlcv), n_splits, test_ratio)

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(bounds):
        fold_idx = i + 1

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

        print(f"  > Fold {fold_idx} Complete.")

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
        ohlcv, mover_mask, param_grid, config, param_names, n_splits, test_ratio, show_progress
    )

    robust_top_5, best_robust_params = _calculate_robustness(
        is_metrics_by_fold, param_names, param_grid
    )
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
