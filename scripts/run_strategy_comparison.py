"""Run multiple strategies and compare their performance side-by-side."""

import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ggTrader.core.orchestrator import run_backtest_orchestrator
from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY

# =====================================================================
# USER CONFIGURATION
# =====================================================================
CONSTANTS = {
    "SYMBOLS": None,
    "SYMBOLS_FILE": "data/top_10_USD_2023-01-01_2025-12-31.json",
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.004,
    "SLIPPAGE": 0.001,
    "USE_MOVERS": 0,
}

# Default parameter sets for each strategy
STRATEGY_PARAMS = {
    "psar_adx": {
        "adx_threshold": 25,
        "adx_length": 14,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": 3.0,
        "atr_length": 14,
        "use_dmp_cross": False,
    },
    "ema_cross": {
        "ema_fast": 9,
        "ema_slow": 21,
        "atr_multiplier": 3.0,
        "atr_length": 14,
    },
    "rsi_reversal": {
        "rsi_length": 14,
        "rsi_oversold": 30,
        "atr_multiplier": 3.0,
        "atr_length": 14,
    },
}
# =====================================================================


def main() -> None:
    """Compare multiple entry strategies."""
    parser = argparse.ArgumentParser(description="Compare entry strategies")
    parser.add_argument("--strategies", nargs="+", default=list(ENTRY_REGISTRY.keys()),
                        help="Strategies to compare")
    parser.add_argument("--progress", action="store_true", default=True, help="Show progress")
    args = parser.parse_args()

    results_summary = []

    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)

    for strategy_name in args.strategies:
        if strategy_name not in ENTRY_REGISTRY:
            print(f"  ⚠ Unknown strategy: {strategy_name}. Skipping.")
            continue

        print(f"\nTesting strategy: {strategy_name}")

        params = STRATEGY_PARAMS.get(strategy_name, {})
        config = {**CONSTANTS, "ENTRY_STRATEGY": strategy_name, "EXIT_STRATEGY": "atr_trailing"}

        try:
            result = run_backtest_orchestrator(
                config=config,
                params=params,
                save_results=False,
                show_progress=args.progress,
            )

            stats = result["stats"]
            pf = result["portfolio"]

            results_summary.append({
                "Strategy": strategy_name,
                "Total Value": f"${stats.get('total_value', 0):,.2f}",
                "Total Profit": f"${stats.get('total_profit', 0):,.2f}",
                "Profit %": f"{stats.get('profit_pct', 0):.2f}%",
                "Trades": int(stats.get('total_trades', 0)),
                "Win Rate": f"{stats.get('win_rate', 0):.2f}%",
                "Sharpe": f"{stats.get('sharpe', 0):.2f}",
                "Max DD": f"{stats.get('max_drawdown', 0):.2f}%",
            })

            print(f"  ✓ {strategy_name}: Sharpe={stats.get('sharpe', 0):.2f}, "
                  f"Total Return={stats.get('profit_pct', 0):.2f}%")

        except Exception as e:
            print(f"  ✗ {strategy_name} failed: {e}")

    # Print comparison table
    if results_summary:
        print("\n" + "=" * 100)
        print("COMPARISON RESULTS")
        print("=" * 100 + "\n")
        print(tabulate(results_summary, headers="keys", tablefmt="grid"))

        # Find best strategy
        best_strategy = max(results_summary, key=lambda x: float(x["Sharpe"].replace(" ", "")))
        print(f"\n✓ Best Strategy (by Sharpe Ratio): {best_strategy['Strategy']}")
    else:
        print("\nNo strategies completed successfully.")


if __name__ == "__main__":
    main()
