"""Centralized orchestration logic for backtesting, sensitivity analysis, and WFO."""

import traceback
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Dict, Any, List, Optional, Tuple

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_and_setup, build_mover_mask
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

    print("Loading data...")
    ohlcv = load_data_and_setup(config)

    # Dynamic mover mask
    mover_mask = None
    use_movers = config.get("USE_MOVERS", 0)
    if use_movers > 0:
        print(f"Building dynamic top-{use_movers} mover mask...")
        try:
            mover_mask = build_mover_mask(ohlcv, config, top_n=use_movers)
        except Exception as e:
            print(f"Warning: mover mask build failed: {e}")

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

    print("Loading data...")
    ohlcv = load_data_and_setup(config)

    print("Running Vectorized Sensitivity Analysis...")
    engine = FastBacktest(ohlcv, param_grid, config=config)
    pf = engine.run(show_progress=show_progress)

    sharpe_series = pf.sharpe_ratio()

    # Apply filtering by trade count
    min_trades = config.get("MIN_TRADES", 0)
    if min_trades > 0:
        trade_counts = pf.trades.count()
        # If vectorized, trade_counts is a series aligned with sharpe_series
        # Mask out those with low trades by setting sharpe to NaN
        low_trade_mask = trade_counts < min_trades
        if low_trade_mask.any():
            print(
                f"Filtering out {(low_trade_mask).sum()} parameter combinations with < {min_trades} trades."
            )
            sharpe_series[low_trade_mask] = np.nan

    best_idx = sharpe_series.idxmax()
    param_names = list(param_grid.keys())
    best_params = _to_native(
        _extract_params(best_idx, sharpe_series, param_names, param_grid)
    )

    results_df = sharpe_series.reset_index()
    # Clean up column names (remove sf_ prefix from VectorBT)
    results_df.columns = [
        str(col).replace("sf_", "") for col in results_df.columns[:-1]
    ] + ["Sharpe Ratio"]

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

        # Run best-case backtest for dashboard
        best_engine = FastBacktest(ohlcv, best_params, config=config)
        best_pf = best_engine.run(show_progress=show_progress)
        best_stats = best_engine.get_stats()
        rm.save_run_results(
            params=best_params,
            metrics=best_stats,
            metadata={**config, "NOTE": "Best Case from Sensitivity"},
        )
        rm.save_vbt_dashboard(best_pf, "best_case_dashboard")
        print(f"Best Case Results saved to: {rm.run_dir}")

    return {
        "portfolio": pf,
        "results_df": results_df,
        "best_params": best_params,
        "results_manager": rm,
    }


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

    print("Loading data...")
    ohlcv = load_data_and_setup(config)

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
        # A. IN-SAMPLE
        train_ohlcv = ohlcv.iloc[train_idx]
        test_ohlcv = ohlcv.iloc[test_idx]

        train_engine = FastBacktest(train_ohlcv, param_grid, config=config)
        pf_train = train_engine.run(show_progress=show_progress)

        train_metrics = pf_train.sharpe_ratio()

        # Apply filtering by trade count (In-Sample)
        min_trades = config.get("MIN_TRADES", 0)
        if min_trades > 0:
            trade_counts = pf_train.trades.count()
            low_trade_mask = trade_counts < min_trades
            if low_trade_mask.any():
                train_metrics[low_trade_mask] = np.nan

        is_metrics_by_fold[i] = train_metrics

        best_param_idx = train_metrics.idxmax()
        fold_best_params = _extract_params(
            best_param_idx, train_metrics, param_names, param_grid
        )

        # B. OUT-OF-SAMPLE
        test_engine = FastBacktest(test_ohlcv, fold_best_params, config=config)
        pf_test = test_engine.run(show_progress=show_progress)

        oos_returns_list.append(pf_test.returns())

        wfo_stats.append(
            {
                "fold": i,
                "train_start": str(train_ohlcv.index[0]),
                "test_start": str(test_ohlcv.index[0]),
                "best_params": _to_native(fold_best_params),
                "is_sharpe": _to_native(train_metrics.max()),
                "oos_sharpe": _to_native(pf_test.sharpe_ratio().mean()),
                "profit": _to_native(pf_test.total_profit().sum()),
            }
        )
        print(f"  > Fold {i} Complete.")

    # 2. Robustness Analysis
    df_is_all = pd.DataFrame(is_metrics_by_fold)
    weights = {f: f for f in is_metrics_by_fold.keys()}
    robustness_scores = (df_is_all * pd.Series(weights)).sum(axis=1) / sum(
        weights.values()
    )

    top_robust_idx = robustness_scores.sort_values(ascending=False).head(5)
    robust_top_5 = []
    for idx, score in top_robust_idx.items():
        extracted = _extract_params(idx, robustness_scores, param_names, param_grid)
        robust_top_5.append(
            {"params": _to_native(extracted), "robustness_score": _to_native(score)}
        )

    best_robust_params = robust_top_5[0]["params"]
    best_recent_params = wfo_stats[-1]["best_params"]

    # 3. Final Model & Persistence
    final_engine = FastBacktest(ohlcv, best_robust_params, config=config)
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
        "results_manager": rm,
    }
