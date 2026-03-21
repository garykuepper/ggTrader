"""Run Walk-Forward Optimization (WFO) using VectorBT time-series CV."""

import argparse
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from tabulate import tabulate

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ggTrader.core.orchestrator import run_backtest_orchestrator, run_wfo_orchestrator, run_wfo_per_coin_orchestrator

# =====================================================================
# USER CONFIGURATION — edit these values to customize the backtest
# =====================================================================
CONSTANTS = {
    # Symbol pool (set SYMBOLS to None to use SYMBOLS_FILE instead)
    "SYMBOLS": None,
    "SYMBOLS_FILE": "data/top_25_USD_2023-01-01_2025-12-31.json",
    # Date range
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    # Portfolio
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.004,
    "SLIPPAGE": 0.003,
    # Dynamic movers: set to 0 to disable, or e.g. 20 for top-20 daily
    "USE_MOVERS": 0,
    # WFO-specific configuration
    "N_SPLITS": 4,
    "TEST_RATIO": 2,
    # Minimum trades to accept a result in optimization
    "MIN_TRADES": 2,
    # Memory optimization: process in chunks of N parameter combinations
    "CHUNK_SIZE": 500,
    # Strategy selection
    "ENTRY_STRATEGY": "psar_adx",  # "psar_adx", "ema_cross", "rsi_reversal"
    "EXIT_STRATEGY": "atr_trailing",  # "atr_trailing", "fixed_sl_tp"
    # WFO mode: "universal" (all symbols same params) or "per_coin" (optimize each symbol independently)
    "WFO_MODE": "universal",
    # Use vectorized signal generation (experimental)
    "USE_VECTORIZED": False,
    # Strategy parameters
    "DEFAULT_PARAMS": {
        "adx_threshold": 25,
        "adx_length": 14,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": 3.0,
        "atr_length": 14,
        "use_dmp_cross": False,
    },
}
# =====================================================================


def main() -> None:
    """Run Walk-Forward Optimization using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run WFO")
    parser.add_argument("--progress", action="store_true", default=True, help="Show progress bar")
    parser.add_argument("--mode", choices=["universal", "per_coin"], default="universal", help="WFO mode")
    args = parser.parse_args()

    # Override WFO_MODE if provided
    config = {**CONSTANTS}
    if args.mode:
        config["WFO_MODE"] = args.mode

    # Vectorized Parameter Grid
    params = {
        # entry
        "sar_acceleration": [0.02],
        "sar_maximum": [0.2],
        "use_dmp_cross": [False],
        "adx_threshold": list(range(20, 46, 5)),
        "adx_length": list(range(7, 22, 7)),
        # exit
        "atr_length": list(range(7, 22, 7)),
        "atr_multiplier": list(np.arange(1.0, 4.1, 0.5)),
    }

    # Run appropriate WFO mode
    wfo_mode = config.get("WFO_MODE", "universal")
    if wfo_mode == "per_coin":
        results = run_wfo_per_coin_orchestrator(
            config=config,
            param_grid=params,
            save_results=True,
            show_progress=args.progress,
        )
        per_coin_results = results["per_coin_results"]
        final_pf = results["final_portfolio"]
        final_stats = results["final_stats"]

        print("\n" + "=" * 80)
        print("PER-COIN OPTIMIZATION RESULTS")
        print("=" * 80)

        per_coin_df = pd.DataFrame(
            [{"symbol": s, **r["best_params"]} for s, r in per_coin_results.items()]
        )
        print("\nBest Parameters per Symbol:")
        print(tabulate(per_coin_df, headers="keys", tablefmt="github", showindex=False))

        print("\nFINAL COMBINED PORTFOLIO PERFORMANCE:")
        stats_df = pd.DataFrame(final_stats.items(), columns=["Metric", "Value"])
        print(tabulate(stats_df, headers="keys", tablefmt="simple", showindex=False))

    else:
        results = run_wfo_orchestrator(
            config=config,
            param_grid=params,
            save_results=True,
            show_progress=args.progress,
        )

        if not results:
            print("WFO optimization returned no results.")
            return

        wfo_stats = results["wfo_stats"]
        robust_top_5 = results["robust_top_5"]
        final_pf = results["final_portfolio"]

        # --- Analysis & Visualization ---

        print("\nPARAMETER ROBUSTNESS REPORT (Top 5 Overall):")
        robust_report_df = pd.DataFrame(
            [{"Score": r["robustness_score"], **r["params"]} for r in robust_top_5]
        )
        print(tabulate(robust_report_df.round(2), headers="keys", tablefmt="github", showindex=False))

        print("\nWFO FOLD RESULTS SUMMARY:")
        df_results = pd.DataFrame(wfo_stats)

        # Create a concise string for the best params
        def simplify_params(d):
            # Only show the params that usually change (e.g., adx and atr)
            return f"ADX:{d.get('adx_threshold')}/{d.get('adx_length')} | ATR:{d.get('atr_multiplier'):.2f}"

        df_clean = df_results.copy()
        df_clean["params"] = df_clean["params"].apply(simplify_params)
        df_clean["train"] = pd.to_datetime(df_clean["train_start"]).dt.strftime("%Y-%m-%d")
        df_clean["test"] = pd.to_datetime(df_clean["test_start"]).dt.strftime("%Y-%m-%d")

        # Select only the readable columns
        cols = ["fold", "train", "test", "params", "is_sharpe", "oos_sharpe", "profit"]
        # Handle potential discrepancies in naming
        existing_cols = [c for c in cols if c in df_clean.columns]
        print(
            tabulate(
                df_clean[existing_cols].round(2),
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )

        print("\nFINAL ROBUST MODEL PERFORMANCE (Across Entire Period):")
        # Convert stats to a DataFrame for a cleaner table view
        stats_df = final_pf.stats().to_frame(name="Value").reset_index()

        # 2. Format only the numbers to 2 decimals (ignoring dates and durations)
        def format_values(x):
            if isinstance(x, (float, np.floating)):
                return f"{x:.2f}"
            return str(x)

        stats_df["Value"] = stats_df["Value"].apply(format_values)

        # 3. Print with tabulate (no floatfmt needed now because we formatted them in step 2)
        print(
            tabulate(
                stats_df,
                headers=["Metric", "Value"],
                tablefmt="simple",
                numalign="right",
                showindex=False,
            )
        )

        print(tabulate(CONSTANTS.items()))

        # 1. Run backtest for just BTC
        btc_pf = run_backtest_orchestrator(
            config={
                **CONSTANTS,
                "SYMBOLS": ["BTC"],
                "PORTFOLIO_SHARE": 1,
                "USE_CASH_SHARING": False,
                "group_by": False,
            },
            params=results["best_robust_params"],
            save_results=False,
            show_progress=True,
        )["portfolio"]

        # --- Visualization ---
        print("Global Portfolio Stats:")
        # Convert stats to a DataFrame for a cleaner table view
        stats_df = btc_pf.stats().to_frame(name="Value").reset_index()

        # 2. Format only the numbers to 2 decimals (ignoring dates and durations)
        def format_values(x):
            if isinstance(x, (float, np.floating)):
                return f"{x:.2f}"
            return str(x)

        stats_df["Value"] = stats_df["Value"].apply(format_values)

        # 3. Print with tabulate (no floatfmt needed now because we formatted them in step 2)
        print(
            tabulate(
                stats_df,
                headers=["Metric", "Value"],
                tablefmt="simple",
                numalign="right",
                showindex=False,
            )
        )


if __name__ == "__main__":
    main()
