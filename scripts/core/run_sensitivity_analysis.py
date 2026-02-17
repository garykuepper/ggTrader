"""Run a vectorized sensitivity analysis (grid search) using FastBacktest."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is in path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_and_setup

# Handle import of visualizer depending on execution context
try:
    from scripts.core.sensitivity_visualizer import plot_optimization_landscape
except ImportError:
    try:
        from sensitivity_visualizer import plot_optimization_landscape
    except ImportError:
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        from sensitivity_visualizer import plot_optimization_landscape


# --- Configuration ---
CONSTANTS = {
    "SYMBOLS": [
        "BTC/USD",
        "ETH/USD",
    ],
    "SYMBOLS_FILE": "data/top_10_consistent_movers.json",
    "START_DATE": "2024-01-01",
    "END_DATE": "2024-06-01",
    "INTERVAL": "4h",
    "START_CASH": 10000,
    "PORTFOLIO_SHARE": 0.20,
    "FEES": 0.001,
}


def main() -> None:
    """Run vectorized sensitivity analysis across a parameter grid."""
    parser = argparse.ArgumentParser(description="Run Vectorized Sensitivity Analysis")
    parser.add_argument("--params", type=str, help="Path to params.json (optional)")
    args = parser.parse_args()

    rm = ResultsManager("run_sensitivity")

    # Parameter grid — VectorBT creates the Cartesian product
    params = {
        "adx_threshold": list(range(15, 45, 5)),
        "adx_length": 14,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": list(np.arange(2.0, 5.0, 0.5)),
        "atr_length": 14,
        "use_dmp_cross": True,
    }

    print("Loading data...")
    try:
        ohlcv = load_data_and_setup(CONSTANTS)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    param_combinations = len(params["adx_threshold"]) * len(params["atr_multiplier"])
    print("Running Vectorized Sensitivity Analysis...")
    print(
        f"Testing {param_combinations} parameter combinations "
        f"across {len(ohlcv.columns.levels[0])} symbols..."
    )

    engine = FastBacktest(ohlcv, params, config=CONSTANTS)
    pf = engine.run()

    print("Backtest complete. Calculating metrics...")

    # Calculate Sharpe for all combinations
    # aggregated over all symbols (or mean across symbols? usually vbt returns Series per param combo if grouped)
    # FastBacktest usually returns Portfolio grouped by Params if passed list.

    # If using FastBacktest with lists, it creates a wrapper.
    # The portfolio `pf` is likely multi-indexed by params.

    try:
        sharpe_series = pf.sharpe_ratio()

        # 1. Identify Best Params
        best_idx = sharpe_series.idxmax()
        best_sharpe = sharpe_series.max()

        print(f"Global Best Sharpe: {best_sharpe:.4f}")
        print(f"Best Param Index: {best_idx}")

        # Reconstruct best param dict
        # The index names should match param names
        best_params = {}
        if isinstance(best_idx, tuple):
            # MultiIndex
            for name, val in zip(sharpe_series.index.names, best_idx):
                # Clean up name if it has vbt prefixes
                clean_name = name.replace("sf_", "")  # example
                best_params[clean_name] = val
        else:
            # Single Index (1 param varying)
            best_params[sharpe_series.index.name] = best_idx

        # 2. Process Results for CSV/Plotting
        results_df = sharpe_series.reset_index()
        results_df.rename(
            columns={results_df.columns[-1]: "Sharpe Ratio"}, inplace=True
        )

        rm.save_metrics(results_df, "sensitivity_results.csv", save_csv=True)

        print("\nTop 5 Parameter Combinations:")
        print(results_df.sort_values("Sharpe Ratio", ascending=False).head(5))

        # 3. Visualizations
        print("Generating visualizations...")
        param_names = [c for c in results_df.columns if c != "Sharpe Ratio"]
        plot_optimization_landscape(
            results_df,
            params_to_plot=param_names,
            metric_name="Sharpe Ratio",
            results_manager=rm,
        )

        # 4. Best Case Dashboard (Consistency)
        print(f"\nRunning Best Case Backtest with params: {best_params}...")

        # We need to run a NEW backtest with scalar params to get a single portfolio dashboard
        # Merge best_params with original params to ensure all required keys exist?
        # FastBacktest expects certain keys.
        # We start with a copy of the input params, but force the best values.
        final_params = params.copy()
        for k, v in best_params.items():
            final_params[k] = v
        # Ensure no lists remain (for strict single run)
        # Any key in params that was a list but NOT in best_params (?)
        # (Should be impossible if grid covered all lists)
        for k, v in final_params.items():
            if isinstance(v, list):
                # This would be an error or un-optimized param. Pick first?
                print(f"Warning: Param {k} is still a list {v}. Using first value.")
                final_params[k] = v[0]

        best_engine = FastBacktest(ohlcv, final_params, config=CONSTANTS)
        best_pf = best_engine.run()
        best_stats = best_engine.get_stats()

        rm.save_run_results(
            params=final_params,
            metrics=best_stats,
            metadata={**CONSTANTS, "NOTE": "Best Case from Sensitivity Analysis"},
        )

        rm.save_vbt_dashboard(best_pf, "best_case_dashboard")
        print(f"Best Case Results saved to: {rm.run_dir}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error processing sensitivity results: {e}")

    print(f"\nAnalysis complete. Results saved to: {rm.run_dir}")


if __name__ == "__main__":
    main()
