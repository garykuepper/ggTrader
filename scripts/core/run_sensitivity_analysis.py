import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.results_manager import ResultsManager
from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.setup import load_data_and_setup

# Handle import of visualizer depending on execution context
try:
    from scripts.core.sensitivity_visualizer import plot_optimization_landscape
except ImportError:
    try:
        from sensitivity_visualizer import plot_optimization_landscape
    except ImportError:
        # Fallback: append current directory to path
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        from sensitivity_visualizer import plot_optimization_landscape


# --- Configuration ---
CONSTANTS = {
    "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
    "START_DATE": "2024-01-01",
    "END_DATE": "2024-06-01",
    "INTERVAL": "4h",
    "START_CASH": 10000,
}


def main():
    """
    Run a vectorized sensitivity analysis (grid search) across multiple parameters.
    Uses VectorBT's broadcasting to simulate all parameter combinations simultaneously.
    """
    rm = ResultsManager("run_sensitivity")

    # Define Parameter Grid
    # VectorBT will create a Cartesian product of these lists (Broadcasting)
    params = {
        "adx_threshold": list(range(15, 45, 5)),  # [15, 20, ..., 40]
        "adx_length": 14,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": list(np.arange(2.0, 5.0, 0.5)),  # [2.0, 2.5, ..., 4.5]
        "atr_length": 14,
        "use_dmp_cross": True,
    }

    print("Loading data...")
    try:
        ohlcv = load_data_and_setup(CONSTANTS)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Metadata for logging
    param_combinations = len(params["adx_threshold"]) * len(params["atr_multiplier"])
    print("Running Vectorized Sensitivity Analysis...")
    print(
        f"Testing {param_combinations} parameter combinations across {len(ohlcv.columns.levels[0])} symbols..."
    )

    # Initialize and run backtest engine
    # FastBacktest handles SignalFactory.run() which broadcasts parameters
    engine = FastBacktest(ohlcv, params)
    pf = engine.run()

    print("Backtest complete. Calculating metrics...")

    # Calculate Sharpe Ratio for all combinations
    # Result is a Series with MultiIndex (param_val1, param_val2, ..., symbol)
    sharpe = pf.sharpe_ratio()

    # Convert Series to proper DataFrame for analysis
    if isinstance(sharpe.index, pd.MultiIndex):
        results_df = sharpe.reset_index()
        # Rename the value column (last column) to "Sharpe Ratio"
        results_df.rename(
            columns={results_df.columns[-1]: "Sharpe Ratio"}, inplace=True
        )
    else:
        results_df = pd.DataFrame({"Sharpe Ratio": sharpe})

    # Save raw CSV results using ResultsManager
    rm.save_metrics(results_df, "sensitivity_results.csv")

    # --- Visualization ---
    print("Generating heatmaps...")
    plot_optimization_landscape(
        results_df,
        params_to_plot=["adx_threshold", "atr_multiplier"],
        metric_name="Sharpe Ratio",
        results_manager=rm,
    )

    print(f"\nAnalysis complete. Results saved to: {rm.run_dir}")


if __name__ == "__main__":
    main()
