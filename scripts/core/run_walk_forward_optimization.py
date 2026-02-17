"""Run Walk-Forward Optimization (WFO) using VectorBT time-series CV."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from tabulate import tabulate

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_and_setup
from ggTrader.utils.Utils import make_end_anchored_tscv, plot_cv_indices

# --- Configuration ---
CONSTANTS = {
    "SYMBOLS": None,
    "SYMBOLS_FILE": "data/top_20_USD_1095_movers.json",
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    "START_CASH": 10000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.001,
    "N_SPLITS": 5,
    "TEST_RATIO": 0.334,
}


def main() -> None:
    """Run Walk-Forward Optimization using vectorbt time-series CV."""
    parser = argparse.ArgumentParser(description="Run Walk-Forward Optimization (WFO)")
    parser.add_argument("--params", type=str, help="Path to params.json (optional)")
    args = parser.parse_args()

    rm = ResultsManager("run_wfo")

    # Vectorized Parameter Grid
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
    tscv, test_size, max_train_size = make_end_anchored_tscv(
        n_samples=len(ohlcv),
        n_splits=CONSTANTS["N_SPLITS"],
        test_ratio=CONSTANTS["TEST_RATIO"],
    )

    # 2. Visualize Splits
    print("Generating split visualization...")
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_cv_indices(tscv, ohlcv.index, ax, CONSTANTS["N_SPLITS"])
    rm.save_plot(fig, "wfo_splits.png")
    plt.close(fig)

    # 3. Run Vectorized Backtest on ENTIRE Period
    print("Running Vectorized Backtest on full dataset to prep WFO...")
    engine = FastBacktest(ohlcv, params, config=CONSTANTS)
    pf_full = engine.run()

    # Debug Shapes
    print(f"DEBUG: OHLCV Shape: {ohlcv.shape}")
    print(f"DEBUG: Portfolio Wrapper Shape: {pf_full.wrapper.shape}")
    print(f"DEBUG: Portfolio Index Start: {pf_full.wrapper.index[0]}")
    print(f"DEBUG: Portfolio Index End: {pf_full.wrapper.index[-1]}")

    # We need to access the underlying vbt Portfolio to slice it
    # FastBacktest.run() returns vbt.Portfolio

    wfo_stats = []
    oos_returns_list = []

    print(f"Starting WFO Loop ({CONSTANTS['N_SPLITS']} splits)...")

    # 4. Iterate over splits
    param_names = list(params.keys())

    best_params_last_fold = None

    for i, (train_idx, test_idx) in enumerate(tscv.split(ohlcv.index), 1):
        try:
            # A. IN-SAMPLE: Find Best Params
            # Use iloc to slice the portfolio object
            pf_train = pf_full.iloc[train_idx]

            # Calculate Sharpe for all param combos
            train_metrics = pf_train.sharpe_ratio()

            # If we have multiple params, train_metrics is MultiIndex.
            # idxmax() returns the tuple of best params.
            best_param_idx = train_metrics.idxmax()

            # Store for final run
            best_params_last_fold = best_param_idx

            # B. OUT-OF-SAMPLE: Test Best Params
            pf_test = pf_full.iloc[test_idx]

            # Select the single best param combo for this fold
            # vectorbt select() handles scalar or tuple indices
            best_pf_test = pf_test.select(best_param_idx)

            # Collect OOS returns for stitching
            oos_returns_list.append(best_pf_test.returns())

            step_profit = best_pf_test.total_profit()
            step_return = best_pf_test.total_return()
            step_sharpe = best_pf_test.sharpe_ratio()

            # Convert param tuple back to dict for logging
            # (Assumes engine.params keys usage order matches MultiIndex levels)
            # SignalFactory/VBT usually preserves order.
            current_best_params = {}
            if isinstance(best_param_idx, tuple):
                for k, v in zip(param_names, best_param_idx):
                    current_best_params[k] = v
            else:
                current_best_params[param_names[0]] = best_param_idx

            wfo_stats.append(
                {
                    "fold": i,
                    "train_start": str(ohlcv.index[train_idx[0]]),
                    "test_start": str(ohlcv.index[test_idx[0]]),
                    "test_end": str(ohlcv.index[test_idx[-1]]),
                    "best_params": current_best_params,
                    "is_sharpe": float(train_metrics.max()),
                    "oos_sharpe": float(step_sharpe),
                    "profit": float(step_profit),
                    "return": float(step_return),
                }
            )
            print(f"Fold {i} complete. OOS Sharpe: {step_sharpe:.4f}")

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error in WFO step {i}: {e}")

    # 5. Results
    df_results = pd.DataFrame(wfo_stats)
    print("\nWFO Results Summary:")
    print(tabulate(df_results, headers="keys", tablefmt="github"))
    rm.save_metrics(df_results, "wfo_results.csv")

    # 6. Stitched OOS Dashboard (True WFO Performance)
    if oos_returns_list:
        print("\nConstructing Stitched OOS Portfolio...")
        try:
            # Concatenate returns (they are Series with DatetimeIndex)
            stitched_returns = pd.concat(oos_returns_list)

            # Remove duplicates if any overlap (should not ideally, but safety first)
            stitched_returns = stitched_returns[
                ~stitched_returns.index.duplicated(keep="first")
            ]

            # Create a Portfolio from these returns
            # Note: We approximate using from_returns.
            # Ideally we'd stitch orders, but returns is sufficient for equity curve & stats.
            pf_dynamic = vbt.Portfolio.from_returns(
                stitched_returns,
                init_cash=CONSTANTS["START_CASH"],
                freq=CONSTANTS["INTERVAL"],
            )

            rm.save_vbt_dashboard(pf_dynamic, "wfo_dynamic_strategy")
            print("Saved Stitched OOS Dashboard.")

        except Exception as e:
            print(f"Error constructing stitched portfolio: {e}")

    # 7. Final Model Backtest (Consistency Requirement)
    # Run a full backtest with the LAST fold's best parameters
    # This represents the "production candidate" model.
    if best_params_last_fold:
        print("\nRunning Final Best Model Backtest (Full History)...")

        # Convert tuple back to dict structure for FastBacktest
        final_params = {}
        if isinstance(best_params_last_fold, tuple):
            for k, v in zip(param_names, best_params_last_fold):
                final_params[k] = v
        else:
            final_params[param_names[0]] = best_params_last_fold

        # Add non-optimized params from original params dict if they were singletons/lists?
        # FastBacktest expects `params` to match SignalFactory inputs.
        # The `params` dict passed originally had lists. We need to pass SCALARS now.

        # Re-merge with any constant params that weren't in the grid?
        # The `params` dict in main() *only* had the grid keys.
        # But FastBacktest expects what SignalFactory expects.
        # So we just pass this `final_params` dict.
        # Check: Does SignalFactory need lists or scalars? It handles scalars fine.

        try:
            # Create a new engine for the single run
            final_engine = FastBacktest(ohlcv, final_params, config=CONSTANTS)
            final_pf = final_engine.run()

            final_stats = final_engine.get_stats()

            # IMPORTANT: Save exactly like run_backtest.py
            # 1. JSON
            rm.save_run_results(
                params=final_params,
                metrics=final_stats,
                metadata={**CONSTANTS, "NOTE": "Final Best Model from WFO"},
            )
            # 2. Dashboard
            rm.save_vbt_dashboard(final_pf, "final_best_model_dashboard")

            print(f"Final Model Results saved to: {rm.run_dir}")

        except Exception as e:
            print(f"Error running Final Model backtest: {e}")

    print(f"WFO Analysis complete.")


if __name__ == "__main__":
    main()
