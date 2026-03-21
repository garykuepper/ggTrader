"""Centralized orchestration logic for backtesting, sensitivity analysis, and WFO."""

import gc
import itertools
import time
import traceback
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from tabulate import tabulate

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers


def _eta_str(seconds: float) -> str:
    """Format seconds as HH:MM:SS for ETA display."""
    return str(timedelta(seconds=int(max(0, seconds))))


def _wall_clock_eta(seconds: float) -> str:
    """Return wall-clock estimate (e.g., '1:30 PM')."""
    from datetime import datetime, timedelta
    finish = datetime.now() + timedelta(seconds=max(0, seconds))
    return finish.strftime("%I:%M %p")


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
    # Sensitivity analysis uses parallel lists (not grid), so must disable vectorized path
    # The vectorized strategy path expects grids and treats parallel lists as cross-product ranges
    chunk_config = {**config, "PARAM_PRODUCT": False, "USE_VECTORIZED": False}

    engine = FastBacktest(ohlcv, chunk_params, config=chunk_config)
    pf = engine.run(show_progress=show_progress)

    sharpe_series = pf.sharpe_ratio()
    
    # Handle case where sharpe_series is a scalar (single combo)
    if not isinstance(sharpe_series, pd.Series):
        sharpe_series = pd.Series([sharpe_series])

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
    logger: Any = None,
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
    t0 = time.time()

    for i in range(0, total_total, chunk_size):
        chunk_idx = i // chunk_size + 1
        chunk = combinations[i : i + chunk_size]
        chunk_end = min(i + chunk_size, total_total)
        print(
            f"  > Processing chunk {chunk_idx}/{total_chunks} "
            f"(combos {i}-{chunk_end})..."
        )
        chunk_start = time.time()

        sharpe_series = _process_sensitivity_chunk(chunk, keys, config, ohlcv, show_progress)
        all_sharpe_series.append(sharpe_series)

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
    logger: Any = None,
) -> Dict[str, Any]:
    """Orchestrate a vectorized sensitivity analysis (grid search)."""
    rm = ResultsManager("run_sensitivity") if save_results else None

    ohlcv, _ = load_data_with_movers(config)
    param_names = list(param_grid.keys())

    # Execute grid search
    sharpe_series = _execute_sensitivity_grid(ohlcv, config, param_grid, show_progress, logger)

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

    # Train: WFO grid search must use non-vectorized path because it produces grid results
    # The vectorized strategy path produces MultiIndex columns incompatible with sharpe_ratio() indexing
    wfo_config = {**config, "USE_VECTORIZED": False}
    train_engine = FastBacktest(train_ohlcv, param_grid, config=wfo_config, mover_mask=train_mask)
    pf_train = train_engine.run(show_progress=show_progress)
    train_metrics = pf_train.sharpe_ratio()
    if not isinstance(train_metrics, pd.Series):
        train_metrics = pd.Series([train_metrics])

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

    # Test: use the same config as training (with vectorized disabled)
    test_engine = FastBacktest(test_ohlcv, fold_best_params, config=wfo_config, mover_mask=test_mask)
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

        robust_top_5, best_robust_params = _calculate_robustness(is_metrics_by_fold, param_names, param_grid)

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


def analyze_sensitivity_results(
    results_df: pd.DataFrame,
    param_grid: Dict[str, Any],
    top_percentile: int = 20,
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

        # Keep top N% of values by mean Sharpe
        threshold_idx = max(1, int(len(grouped) * (top_percentile / 100)))
        top_values = grouped.head(threshold_idx).index.tolist()

        narrowed[param_name] = top_values

        # Print summary
        print(
            f"  {param_name}: {len(grouped)} unique values -> {len(top_values)} "
            f"(top {top_percentile}%): {top_values}"
        )

    return narrowed


def run_multi_strategy_per_coin_wfo(
    config: Dict[str, Any],
    strategy_param_grids: Dict[str, Dict],
    save_results: bool = True,
    show_progress: bool = True,
    logger: Any = None,
) -> Dict[str, Any]:
    """Run WFO for each symbol across multiple strategies, select best per coin.

    Then run final validation backtest on full 3-year range using the best
    strategy + params for each coin, combining them into one portfolio.

    Args:
        config: Portfolio and data configuration
        strategy_param_grids: Dict mapping strategy name to narrowed param grid
        save_results: Whether to save results
        show_progress: Whether to show progress bars

    Returns:
        Dict with per_coin_results, final_portfolio, final_stats, results_manager
    """
    from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY

    rm = ResultsManager("run_wfo_per_coin_multi_strategy") if save_results else None
    ohlcv, mover_mask = load_data_with_movers(config)

    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config.get("TEST_RATIO", 3.0)

    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    n_symbols = len(symbols)
    n_strategies = len(strategy_param_grids)
    print(f"\nStarting Multi-Strategy Per-Coin WFO ({n_symbols} symbols, {n_splits} splits)...")

    # Dictionary to store best strategy + params per symbol
    per_coin_results: Dict[str, Dict[str, Any]] = {}
    t0_wfo_coins = time.time()

    # Phase 2: Per-coin WFO loop - test all strategies for each coin
    for coin_idx, symbol in enumerate(symbols):
        coin_start = time.time()
        print(f"\n--- Optimizing {symbol} ({coin_idx+1}/{n_symbols}) ---")

        # Extract single-symbol data
        symbol_ohlcv = ohlcv[[symbol]]
        symbol_mover_mask = mover_mask[[symbol]] if mover_mask is not None else None

        best_strategy = None
        best_robust_score = -np.inf
        best_params_for_coin = {}
        best_wfo_stats = []
        best_robust_top_5 = []

        # Try each strategy
        for strategy_name, param_grid in strategy_param_grids.items():
            print(f"  Testing strategy: {strategy_name}")

            # Set strategy in config
            config_with_strategy = {**config, "ENTRY_STRATEGY": strategy_name}

            # Run WFO for this coin+strategy combination
            wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
                symbol_ohlcv,
                symbol_mover_mask,
                param_grid,
                config_with_strategy,
                list(param_grid.keys()),
                n_splits,
                test_ratio,
                show_progress,
                logger,
            )

            robust_top_5, best_robust_params = _calculate_robustness(
                is_metrics_by_fold, list(param_grid.keys()), param_grid
            )

            # Extract robustness score
            robustness_score = robust_top_5[0]["robustness_score"] if robust_top_5 else -np.inf

            print(f"    {strategy_name} robustness: {robustness_score:.4f}")

            # Track best strategy for this coin
            if robustness_score > best_robust_score:
                best_robust_score = robustness_score
                best_strategy = strategy_name
                best_params_for_coin = best_robust_params
                best_wfo_stats = wfo_stats
                best_robust_top_5 = robust_top_5

        per_coin_results[symbol] = {
            "best_strategy": best_strategy,
            "best_params": best_params_for_coin,
            "robustness_score": best_robust_score,
            "wfo_stats": best_wfo_stats,
            "robust_top_5": best_robust_top_5,
        }

        coin_elapsed = time.time() - coin_start
        total_elapsed = time.time() - t0_wfo_coins
        avg_per_coin = total_elapsed / (coin_idx + 1)
        eta = (n_symbols - coin_idx - 1) * avg_per_coin
        status_msg = (
            f"  > {symbol} ({coin_idx+1}/{n_symbols}): Best={best_strategy} "
            f"Robustness={best_robust_score:.4f} | {coin_elapsed:.0f}s | ETA {_eta_str(eta)} (est. {_wall_clock_eta(eta)})"
        )
        print(status_msg)
        if logger:
            logger.update(status_msg)

    # Phase 3: Final Validation Backtest on Full 3-Year Range
    print("\n" + "=" * 100)
    print("PHASE 3: FINAL VALIDATION BACKTEST (FULL 3-YEAR RANGE)")
    print("=" * 100)

    combined_entries_list = []
    combined_exits_list = []
    combined_close_list = []
    per_coin_final_stats = {}

    for symbol in symbols:
        strategy_name = per_coin_results[symbol]["best_strategy"]
        best_params = per_coin_results[symbol]["best_params"]

        print(f"\nGenerating signals for {symbol} with {strategy_name}...")

        symbol_ohlcv = ohlcv[[symbol]]

        # Set strategy and run backtest on full range
        config_for_final = {**config, "ENTRY_STRATEGY": strategy_name}
        engine = FastBacktest(symbol_ohlcv, best_params, config=config_for_final)
        pf = engine.run(show_progress=False)

        # Extract signals from engine (they were computed in FastBacktest.run)
        # We need to regenerate them since the portfolio object doesn't store the raw signals
        close = symbol_ohlcv.xs("close", axis=1, level=1, drop_level=True)
        high = symbol_ohlcv.xs("high", axis=1, level=1, drop_level=True)
        low = symbol_ohlcv.xs("low", axis=1, level=1, drop_level=True)
        
        # Regenerate signals for final backtest
        use_vectorized = config_for_final.get("USE_VECTORIZED", False)
        if use_vectorized:
            entries, exits, _ = engine._generate_signals_vectorized(show_progress=False)
        else:
            entries, exits, _ = engine._generate_signals(show_progress=False)

        combined_entries_list.append(entries)
        combined_exits_list.append(exits)
        combined_close_list.append(close)

        # Extract per-coin stats
        stats = engine.get_stats()
        per_coin_final_stats[symbol] = {
            "strategy": strategy_name,
            "params": _to_native(best_params),
            **stats,
        }

    # Concatenate all symbols into combined signals
    combined_entries = pd.concat(combined_entries_list, axis=1)
    combined_exits = pd.concat(combined_exits_list, axis=1)
    combined_close = pd.concat(combined_close_list, axis=1)

    # Create combined final portfolio
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

    # Extract final combined stats
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

    print(f"\nFinal Combined Portfolio (Full 3-Year Range):")
    print(f"  Total Return: {final_stats['profit_pct']:.2f}%")
    print(f"  Sharpe Ratio: {final_stats['sharpe']:.4f}")
    print(f"  Max Drawdown: {final_stats['max_drawdown']:.2f}%")
    print(f"  Total Trades: {final_stats['total_trades']}")

    if save_results and rm:
        # Save per-coin strategy selection
        strategy_summary = []
        for symbol, results in per_coin_results.items():
            strategy_summary.append({
                "symbol": symbol,
                "strategy": results["best_strategy"],
                "robustness_score": results["robustness_score"],
            })

        strategy_df = pd.DataFrame(strategy_summary)
        rm.save_metrics(strategy_df, "per_coin_strategy_selection.csv")

        # Save per-coin final stats
        final_stats_df = pd.DataFrame([
            {"symbol": sym, **stats} for sym, stats in per_coin_final_stats.items()
        ])
        rm.save_metrics(final_stats_df, "per_coin_final_stats.csv")

        # Save overall results
        metadata = {
            **config,
            "per_coin_results": _to_native(per_coin_results),
            "per_coin_final_stats": _to_native(per_coin_final_stats),
        }

        rm.save_run_results(params={"per_coin": _to_native(per_coin_results)}, metrics=final_stats, metadata=metadata)
        rm.save_vbt_dashboard(final_pf, "combined_portfolio_final_dashboard")
        print(f"\nMulti-Strategy WFO Results saved to: {rm.run_dir}")

    return {
        "final_portfolio": final_pf,
        "per_coin_results": per_coin_results,
        "per_coin_final_stats": per_coin_final_stats,
        "final_stats": final_stats,
        "results_manager": rm,
    }
