import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.results_manager import ResultsManager
from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.setup import load_data_and_setup

try:
    from scripts.core.sensitivity_visualizer import plot_optimization_landscape
except ImportError:
    try:
        from sensitivity_visualizer import plot_optimization_landscape
    except ImportError:
        # Fallback if running from root
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
        from sensitivity_visualizer import plot_optimization_landscape


def main():
    rm = ResultsManager("run_sensitivity")

    # --- USER CONFIGURATION ---
    CONSTANTS = {
        "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
        "START_DATE": "2024-01-01",
        "END_DATE": "2024-06-01",
        "INTERVAL": "4h",
    }

    # Define Parameter Grid (Vectorized)
    # Using lists creates a Cartesian product in FastBacktest via SignalFactory
    params = {
        "adx_threshold": list(range(15, 45, 5)),  # 15, 20, 25, 30, 35, 40
        "adx_length": 14,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": list(
            np.arange(2.0, 5.0, 0.5)
        ),  # 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
        "atr_length": 14,
        "use_dmp_cross": True,
    }

    print("Loading data...")
    try:
        ohlcv = load_data_and_setup(CONSTANTS)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Running Vectorized Sensitivity Analysis...")
    print(
        f"Testing {len(params['adx_threshold']) * len(params['atr_multiplier'])} parameter combinations across symbols..."
    )

    engine = FastBacktest(ohlcv, params)
    pf = engine.run()

    print("Backtest complete. Calculating metrics...")
    # Calculate Sharpe Ratio for all combinations
    # Returns Series with MultiIndex (param1, param2, ..., symbol)
    sharpe = pf.sharpe_ratio()

    # Convert to DataFrame
    if isinstance(sharpe.index, pd.MultiIndex):
        results_df = sharpe.reset_index()
        # The Series value doesn't have a name by default, usually 0
        results_df.rename(
            columns={results_df.columns[-1]: "Sharpe Ratio"}, inplace=True
        )
    else:
        results_df = pd.DataFrame({"Sharpe Ratio": sharpe})

    # Save raw results
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
