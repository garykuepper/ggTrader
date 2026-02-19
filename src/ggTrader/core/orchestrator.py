"""Centralized orchestration logic for backtesting, sensitivity analysis, and WFO."""

import traceback
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Dict, Any, List, Optional, Tuple
import itertools
import gc

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers
from ggTrader.utils.utils import make_end_anchored_tscv, plot_cv_indices


def _to_native(val: Any) -> Any:
    """Ensure basic types are JSON serializable."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
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


def _process_sensitivity_chunk(
    chunk: List[Tuple],
    keys: List[str],
    config: Dict[str, Any],
    ohlcv: pd.DataFrame,
    show_progress: bool,
) -> pd.Series:
    """Helper to process a single chunk of sensitivity parameters."""
    # Build a "flat" param grid for this chunk
    chunk_params = {k: [c[j] for c in chunk] for j, k in enumerate(keys)}

    # Create config for this run ensuring PARAM_PRODUCT is False
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

    # Explicitly delete objects and collect garbage to free RAM
    del pf
    del engine
    gc.collect()

    return sharpe_series


def run_sensitivity_orchestrator(
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate a vectorized sensitivity analysis (grid search).
    """
    from ggTrader.utils.plotting import plot_optimization_landscape

    rm = ResultsManager("run_sensitivity") if save_results else None
    best_pf = None

    ohlcv, _ = load_data_with_movers(
        config
    )  # Mover mask not used in grid search usually unless specified?
    # Actually FastBacktest supports it, but sensitivity analysis usually runs on raw data first?
    # The original code loaded data and then ran FastBacktest without checking mover_mask explicitly
    # in the loop, EXCEPT that FastBacktest init took it.
    # WAIT: pure sensitivity didn't use mover_mask in original code?
    # Original code:
    # ohlcv = load_data_and_setup(config)
    # ...
    # engine = FastBacktest(ohlcv, chunk_params, config=chunk_config)
    # It did NOT pass mover_mask.
    # But wait, config might have USE_MOVERS.
    # If I use `load_data_with_movers`, I get it. I probably should pass it if it exists.
    # However, for consistency with *original* code, I should check if it was used.
    # In original `run_sensitivity_orchestrator`, it did NOT build mover_mask.
    # So I will ignore it here to match behavior, OR I could enable it.
    # Let's stick to original behavior: No mover mask in sensitivity (unless I missed it).
    # Ah, lines 126-127 of original: ohlcv = load_data_and_setup(config). No mover mask build block.
    # So effectively USE_MOVERS was ignored in sensitivity analysis in the original code.

    # 1. Generate full parameter product for chunking
    keys = list(param_grid.keys())
    values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
    combinations = list(itertools.product(*values))
    total_total = len(combinations)
    chunk_size = config.get("CHUNK_SIZE", 500)  # Default to 500 to be safe
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

    # Merge all chunk results
    sharpe_series = pd.concat(all_sharpe_series)

    best_idx = sharpe_series.idxmax()
    param_names = list(param_grid.keys())
    best_params = _to_native(_extract_params(best_idx, sharpe_series, param_names, param_grid))

    results_df = sharpe_series.reset_index()
    # Clean up column names (remove sf_ prefix from VectorBT)
    results_df.columns = [str(col).replace("sf_", "") for col in results_df.columns[:-1]] + [
        "Sharpe Ratio"
    ]

    # Run best-case backtest for dashboard and return value
    # Here we WOULD use mover mask if we wanted to be consistent with backtest,
    # but again, original didn't.
    best_engine = FastBacktest(ohlcv, best_params, config=config)
    best_pf = best_engine.run(show_progress=show_progress)
    best_stats = best_engine.get_stats()

    if save_results and rm:
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

    return {
        "portfolio": best_pf,
        "results_df": results_df,
        "best_params": best_params,
        "results_manager": rm,
    }


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
    # A. IN-SAMPLE
    train_ohlcv = ohlcv.iloc[train_idx]
    test_ohlcv = ohlcv.iloc[test_idx]

    # Slice mover_mask if exists
    train_mask = mover_mask.iloc[train_idx] if mover_mask is not None else None
    test_mask = mover_mask.iloc[test_idx] if mover_mask is not None else None

    # Run optimization on Train
    train_engine = FastBacktest(train_ohlcv, param_grid, config=config, mover_mask=train_mask)
    pf_train = train_engine.run(show_progress=show_progress)

    train_metrics = pf_train.sharpe_ratio()

    # Apply filtering by trade count (In-Sample)
    min_trades = config.get("MIN_TRADES", 0)
    if min_trades > 0:
        trade_counts = pf_train.trades.count()
        low_trade_mask = trade_counts < min_trades
        if low_trade_mask.any():
            train_metrics[low_trade_mask] = np.nan

    best_param_idx = train_metrics.idxmax()
    fold_best_params = _extract_params(best_param_idx, train_metrics, param_names, param_grid)

    # B. OUT-OF-SAMPLE
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
        "train_metrics": train_metrics,  # Needed for robustness
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


def run_wfo_orchestrator(
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate Walk-Forward Optimization with Robustness-First selection.
    """
    rm = ResultsManager("run_wfo") if save_results else None

    ohlcv, mover_mask = load_data_with_movers(config)

    # 1. Setup Splits
    tscv, _, _ = make_end_anchored_tscv(
        n_samples=len(ohlcv),
        n_splits=config["N_SPLITS"],
        test_ratio=config["TEST_RATIO"],
    )

    if save_results and rm:
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_cv_indices(tscv, ohlcv.index, ax, config["N_SPLITS"])
        rm.save_plot(fig, "wfo_splits.png")
        plt.close(fig)

    wfo_stats = []
    oos_returns_list = []
    is_metrics_by_fold = {}
    param_names = list(param_grid.keys())

    print(f"Starting WFO Loop ({config['N_SPLITS']} splits)...")

    for i, (train_idx, test_idx) in enumerate(tscv.split(ohlcv.index), 1):
        fold_result = _process_wfo_fold(
            i,
            train_idx,
            test_idx,
            ohlcv,
            mover_mask,
            param_grid,
            config,
            show_progress,
            param_names,
        )

        # Unpack extras we need
        is_metrics_by_fold[i] = fold_result.pop("train_metrics")
        oos_returns_list.append(fold_result.pop("oos_returns"))

        wfo_stats.append(fold_result)
        print(f"  > Fold {i} Complete.")

    # 2. Robustness Analysis
    robust_top_5, best_robust_params = _calculate_robustness(
        is_metrics_by_fold, param_names, param_grid
    )
    best_recent_params = wfo_stats[-1]["params"]

    # 3. Final Model & Persistence
    final_engine = FastBacktest(ohlcv, best_robust_params, config=config, mover_mask=mover_mask)
    final_pf = final_engine.run(show_progress=show_progress)
    final_stats = final_engine.get_stats()

    if save_results and rm:
        rm.save_metrics(pd.DataFrame(wfo_stats), "wfo_results.csv")
        rm.save_run_results(
            params=best_robust_params,
            metrics=final_stats,
            metadata={
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
            },
        )
        rm.save_vbt_dashboard(final_pf, "final_robust_model_dashboard")
        print(f"WFO Results saved to: {rm.run_dir}")

    return {
        "final_portfolio": final_pf,
        "wfo_stats": wfo_stats,
        "robust_top_5": robust_top_5,
        "best_robust_params": best_robust_params,
        "best_recent_params": best_recent_params,
        "results_manager": rm,
    }
