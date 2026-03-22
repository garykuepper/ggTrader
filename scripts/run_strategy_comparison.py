"""Run multiple strategies and compare their performance side-by-side."""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from tabulate import tabulate

from ggTrader.core.orchestrator import run_backtest_orchestrator
from ggTrader.indicators.strategies import ENTRY_REGISTRY
from ggTrader.utils.run_config import strategy_comparison_config

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
    "macd_cross": {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_multiplier": 3.0,
        "atr_length": 14,
    },
    "bbands_mean_reversion": {
        "bb_length": 20,
        "bb_std": 2.0,
        "atr_multiplier": 3.0,
        "atr_length": 14,
    },
    "donchian_breakout": {
        "donchian_length": 20,
        "atr_multiplier": 3.0,
        "atr_length": 14,
    },
    "supertrend_flip": {
        "st_length": 10,
        "st_multiplier": 3.0,
        "atr_multiplier": 3.0,
        "atr_length": 14,
    },
}


def main() -> None:
    """Compare multiple entry strategies."""
    parser = argparse.ArgumentParser(description="Compare entry strategies")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(ENTRY_REGISTRY.keys()),
        help="Strategies to compare",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    args = parser.parse_args()

    show_progress = not args.no_progress and sys.stdout.isatty()
    base_config = strategy_comparison_config()

    results_summary = []

    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)

    for strategy_name in args.strategies:
        if strategy_name not in ENTRY_REGISTRY:
            print(f"  Unknown strategy: {strategy_name}. Skipping.")
            continue

        print(f"\nTesting strategy: {strategy_name}")

        params = STRATEGY_PARAMS.get(strategy_name, {})
        run_config = {
            **base_config,
            "ENTRY_STRATEGY": strategy_name,
            "EXIT_STRATEGY": "atr_trailing",
        }

        try:
            result = run_backtest_orchestrator(
                config=run_config,
                params=params,
                save_results=False,
                show_progress=show_progress,
            )

            stats = result["stats"]

            results_summary.append(
                {
                    "Strategy": strategy_name,
                    "Total Value": f"${stats.get('total_value', 0):,.2f}",
                    "Total Profit": f"${stats.get('total_profit', 0):,.2f}",
                    "Profit %": f"{stats.get('profit_pct', 0):.2f}%",
                    "Trades": int(stats.get("total_trades", 0)),
                    "Win Rate": f"{stats.get('win_rate', 0):.2f}%",
                    "Sharpe": f"{stats.get('sharpe', 0):.2f}",
                    "Max DD": f"{stats.get('max_drawdown', 0):.2f}%",
                }
            )

            print(
                f"  OK {strategy_name}: Sharpe={stats.get('sharpe', 0):.2f}, "
                f"Total Return={stats.get('profit_pct', 0):.2f}%"
            )

        except Exception as e:
            print(f"  FAIL {strategy_name}: {e}")

    if results_summary:
        print("\n" + "=" * 100)
        print("COMPARISON RESULTS")
        print("=" * 100 + "\n")
        print(tabulate(results_summary, headers="keys", tablefmt="grid"))

        best_strategy = max(results_summary, key=lambda x: float(x["Sharpe"].replace(" ", "")))
        print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy['Strategy']}")
    else:
        print("\nNo strategies completed successfully.")


if __name__ == "__main__":
    main()
