"""Run Walk-Forward Optimization (WFO) using VectorBT time-series CV."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

from ggTrader.core.orchestrator import (
    run_backtest_orchestrator,
    run_wfo_orchestrator,
    run_wfo_per_coin_orchestrator,
)
from ggTrader.utils.run_config import merge_run_config, wfo_script_config


def main() -> None:
    """Run Walk-Forward Optimization using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run WFO")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--mode",
        choices=["universal", "per_coin"],
        default="universal",
        help="WFO mode",
    )
    args = parser.parse_args()

    config = merge_run_config(wfo_script_config(), WFO_MODE=args.mode)
    show_progress = not args.no_progress and sys.stdout.isatty()

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
            show_progress=show_progress,
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
            show_progress=show_progress,
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

        # 1. Run backtest for just BTC
        btc_pf = run_backtest_orchestrator(
            config={
                **config,
                "SYMBOLS": ["BTC"],
                "PORTFOLIO_SHARE": 1,
                "USE_CASH_SHARING": False,
                "group_by": False,
            },
            params=results["best_robust_params"],
            save_results=False,
            show_progress=show_progress,
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
