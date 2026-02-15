import argparse
import sys
import os
import pandas as pd
import numpy as np
import vectorbt as vbt
from tabulate import tabulate

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.results_manager import ResultsManager
from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.setup import load_data_and_setup
from ggTrader.utils.utils import make_end_anchored_tscv, plot_cv_indices

# --- Configuration ---
CONSTANTS = {
    "SYMBOLS": None,  # Set to a list like ["BTC/USD", "ETH/USD"] to override JSON
    "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-06-01",
    "INTERVAL": "4h",
    "START_CASH": 10000,
    "N_SPLITS": 3,
    "TEST_RATIO": 0.334,
}


def main():
    """
    Run Walk-Forward Optimization (WFO) using VectorBT.
    Iterates through time using a rolling window (Train -> Test).
    """
    parser = argparse.ArgumentParser(description="Run Walk-Forward Optimization (WFO)")
    parser.add_argument("--params", type=str, help="Path to params.json (optional)")
    args = parser.parse_args()

    rm = ResultsManager("run_wfo")

    # Vectorized Parameter Grid
    # In a real scenario, you might load these from a file
    params = {
        "adx_threshold": list(range(15, 35, 5)),
        "adx_length": [14],
        "sar_acceleration": [0.02],
        "sar_maximum": [0.2],
        "atr_multiplier": list(np.arange(2.0, 4.5, 0.5)),
        "atr_length": [14],
        "use_dmp_cross": [True, False],
    }

    print("Loading data...")
    try:
        ohlcv = load_data_and_setup(CONSTANTS)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. Setup WFO Splitter
    import matplotlib.pyplot as plt

    tscv, test_size, max_train_size = make_end_anchored_tscv(
        n_samples=len(ohlcv),
        n_splits=CONSTANTS["N_SPLITS"],
        test_ratio=CONSTANTS["TEST_RATIO"],
    )

    # 2. Visualize Splits and Save
    print("Generating split visualization...")
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_cv_indices(tscv, ohlcv.index, ax, CONSTANTS["N_SPLITS"])
    rm.save_plot(fig, "wfo_splits.png")
    plt.close(fig)

    # 3. Run Vectorized Backtest on ENTIRE Period
    print("Running Vectorized Backtest on full dataset to prep WFO...")
    engine = FastBacktest(ohlcv, params)
    pf_full = engine.run()

    wfo_stats = []
    print(f"Starting WFO Loop ({CONSTANTS['N_SPLITS']} splits)...")

    # 4. Iterate over splits
    param_names = list(params.keys())

    for i, (train_idx, test_idx) in enumerate(tscv.split(ohlcv.index), 1):
        try:
            # A. IN-SAMPLE: Find Best Params
            pf_train = pf_full.iloc[train_idx]
            train_metrics = pf_train.sharpe_ratio(group_by=param_names)
            best_param_idx = train_metrics.idxmax()

            # B. OUT-OF-SAMPLE: Test Best Params
            pf_test = pf_full.iloc[test_idx]
            best_pf_test = pf_test.select(best_param_idx, group_by=param_names)

            # Metrics
            step_profit = best_pf_test.total_profit().sum()
            step_return = best_pf_test.total_return().mean()
            step_sharpe = best_pf_test.sharpe_ratio().mean()

            wfo_stats.append(
                {
                    "fold": i,
                    "train_start": str(ohlcv.index[train_idx[0]]),
                    "test_start": str(ohlcv.index[test_idx[0]]),
                    "test_end": str(ohlcv.index[test_idx[-1]]),
                    "best_params": str(best_param_idx),
                    "is_sharpe": train_metrics.max(),
                    "oos_sharpe": step_sharpe,
                    "profit": step_profit,
                    "return": step_return,
                }
            )
            print(f"Fold {i} complete. OOS Sharpe: {step_sharpe:.4f}")

        except Exception as e:
            print(f"Error in WFO step {i}: {e}")

    # 5. Results
    df_results = pd.DataFrame(wfo_stats)
    print("\nWFO Results Summary:")
    print(tabulate(df_results, headers="keys", tablefmt="github"))

    # Save
    rm.save_metrics(df_results, "wfo_results.csv")
    print(f"Saved results to {rm.run_dir}")


if __name__ == "__main__":
    main()
