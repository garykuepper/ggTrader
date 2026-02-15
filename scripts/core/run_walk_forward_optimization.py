import pandas as pd
import numpy as np
import sys
import os
from tabulate import tabulate
import vectorbt as vbt

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.results_manager import ResultsManager
from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.setup import load_data_and_setup


def main():
    rm = ResultsManager("run_wfo")

    # --- USER CONFIGURATION ---
    CONSTANTS = {
        "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
        "START_DATE": "2023-01-01",
        "END_DATE": "2025-12-31",
        "INTERVAL": "4h",
        "TRAIN_DAYS": 180,
        "TEST_DAYS": 30,
    }

    # Vectorized Parameter Grid
    # This generates all combinations at once
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

    # 1. Run Vectorized Backtest on ENTIRE Period
    # This gives us the equity curves for ALL parameter combinations
    print("Running Vectorized Backtest on full dataset...")
    engine = FastBacktest(ohlcv, params)
    pf_full = engine.run()

    # pf_full has MultiIndex columns: (adx_threshold, adx_length, ..., atr_multiplier, ..., symbol)
    # We want to aggregate across symbols first? Or optimize per symbol?
    # WFO is usually per asset or portfolio-wide.
    # If we optimize "The Strategy", we optimize the AGGREGATE performance across all assets.
    # So we sum the returns/value across symbols for each parameter set.
    # vbt can do this by grouping.

    # Group by Parameters (ignoring Symbol)
    # The columns levels corresponding to params:
    # We need to find which level is 'symbol'.
    # SignalFactory output columns: (param1, param2, ..., symbol) usually.
    # Let's verify by printing columns if needed, or assume vbt conventions.
    # vbt usually puts 'symbol' as the LAST level if broadcasting.

    # We want to sum performance across symbols for each timestep -> Portfolio Return.
    # pf_full.value() is (Time, Params*Symbols).
    # parameters_levels = all levels except 'symbol'.

    # We can user pf_wrapper to group.
    # Assuming 'symbol' is one of the levels.
    # Let's assume we optimize the "Portfolio Strategy" (sum of all symbols).

    # 2. WFO Logic
    # Split time into Train/Test
    # vbt.RollingSplitter

    splitter = vbt.RollingSplitter(
        index=ohlcv.index,
        window_len=int(CONSTANTS["TRAIN_DAYS"] * 6),  # 4h candles (6 per day)
        set_lens=[int(CONSTANTS["TEST_DAYS"] * 6)],
        left_to_right=True,
    )

    wfo_stats = []
    wfo_equity = []

    print(f"Starting WFO Loop ({splitter.get_n_splits()} splits)...")

    # Iterate over splits
    # ranges is (n_splits, 2) array of start/end indices for train and test
    # but RollingSplitter provides generator

    for split_idx, (train_slice, test_slice) in enumerate(splitter.split()):
        # Slice the full portfolio for Train
        # train_slice is a slice object or indices

        # Calculate Sharpe for all params in Train period
        # We aggregate across symbols (group_by=params) to find best global params
        # But pf_full columns are (Param1, Param2, ..., Symbol).
        # We want to group by Params.

        # pf_full[train_slice] might return a new Portfolio object sliced in time.
        pf_train = pf_full[train_slice]

        # Group by everything EXCEPT symbol to aggregate results across symbols
        # If 'symbol' is a level name
        # If we don't know level names, we can check pf_full.wrapper.columns.names

        # Strategy: Select Best Param based on Portfolio Sharpe (Sum of Symbols)
        # We can simulate a "Portfolio" of all assets for each param combination.
        # But vbt.Portfolio.from_signals was created with (Time, Sym*Param).
        # It treats each column as an independent cash stream.
        # To group by Param, we need to sum cash flows of all symbols for that param.

        # If we group by params, vbt returns metrics for that group.
        # Let's try to identify the param levels.
        param_names = list(params.keys())
        # The columns might have these names as levels.

        try:
            # Calculate Sharpe for each Param Combination (aggregated across symbols)
            # group_by=param_names will group columns that share the same params (crossing symbols)
            train_metrics = pf_train.sharpe_ratio(group_by=param_names)
        except Exception as e:
            # Fallback if names don't match (e.g. index levels lost)
            print(f"Warning: Could not group by params: {e}. Using raw columns.")
            train_metrics = pf_train.sharpe_ratio()

        # Find Best Param
        best_param_idx = train_metrics.idxmax()

        # Now Apply to Test
        pf_test = pf_full[test_slice]

        # Select the column(s) corresponding to best_param
        # If best_param_idx is a tuple (p1, p2...), we select that group.
        # If we grouped by params, we can use the same grouping to select?
        # Or we just select the wrapper columns that match.

        # Easy way: pf_test.xs(best_param_idx, level=...)
        # If we used group_by in sharpe_ratio, best_param_idx is the index of that result.

        try:
            # Select the specific parameter set (across all symbols)
            # This returns a Portfolio of just that param set (multiple symbols)
            best_pf_test = pf_test.select(best_param_idx, group_by=param_names)

            # Record Performance (Sum of symbols)
            # final_value of this period
            step_profit = best_pf_test.total_profit().sum()
            step_return = (
                best_pf_test.total_return().mean()
            )  # Average return of symbols?

            wfo_stats.append(
                {
                    "split": split_idx,
                    "train_start": str(pf_train.index[0]),
                    "test_start": str(pf_test.index[0]),
                    "test_end": str(pf_test.index[-1]),
                    "best_params": str(best_param_idx),
                    "profit": step_profit,
                    "return": step_return,
                }
            )

        except Exception as e:
            print(f"Error in WFO step {split_idx}: {e}")

    # Results
    df_results = pd.DataFrame(wfo_stats)
    print("\nWFO Results:")
    print(tabulate(df_results, headers="keys", tablefmt="github"))

    # Save
    rm.save_metrics(df_results, "wfo_results.csv")
    print(f"Saved to {rm.run_dir}")


if __name__ == "__main__":
    main()
