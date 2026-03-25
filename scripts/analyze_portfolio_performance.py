"""
Analyze portfolio performance and asset correlation from the latest WFO run.
Identifies alpha drivers, laggards, and diversification opportunities.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))


def main():
    # 1. Locate latest result
    results_dir = Path("results")
    latest_dir = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_wfo")],
        key=os.path.getmtime,
        reverse=True,
    )[0]

    results_file = latest_dir / "run_results.json"
    print(f"Analyzing: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    # 2. Extract per-coin final stats
    # Structure from previous view_file: data['per_coin_final_stats'] ? No, I'll check first.
    # From Step 1245: 'results' has 'total_value', etc.
    # and Step 1356 showed 'strategy_parameters' -> 'per_coin'

    # Actually, let's look at the per-coin performance printed in stdout
    # In run_results.json, it should be under 'results' -> 'per_coin_stats' or similar.
    # I'll check the keys again to be safe.

    per_coin_stats = data.get("per_coin_final_stats", {})
    if not per_coin_stats:
        # Try another common key
        per_coin_stats = data.get("results", {}).get("per_symbol_metrics", {})

    if not per_coin_stats:
        print("Error: Could not find per-coin stats in JSON.")
        return

    df = pd.DataFrame.from_dict(per_coin_stats, orient="index")

    # Cleanup dataframe
    # Ensure columns exist: profit_pct, sharpe, max_drawdown, total_trades
    required_cols = ["profit_pct", "sharpe", "max_drawdown", "total_trades", "win_rate"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_values("profit_pct", ascending=False)

    # 3. Print Alpha Drivers
    print("\n--- TOP ALPHA DRIVERS (Profit %) ---")
    print(df[["strategy", "profit_pct", "sharpe", "max_drawdown", "total_trades"]].head(10))

    print("\n--- BOTTOM LAGGARDS (Profit %) ---")
    print(df[["strategy", "profit_pct", "sharpe", "max_drawdown", "total_trades"]].tail(5))

    # 4. Filter out zero-trade assets
    active_df = df[df["total_trades"] > 0]
    print(f"\nActive Assets: {len(active_df)} / {len(df)}")

    # 5. Portfolio Synergy Recommendation
    profitable = active_df[active_df["profit_pct"] > 0]
    print(f"Profitable Assets: {len(profitable)}")

    # If we only traded the profitable ones:
    hypothetical_return = profitable["profit_pct"].mean()
    print(f"Hypothetical Mean Return (Profitable Only): {hypothetical_return:.2f}%")

    # Save a CSV for further analysis
    csv_out = latest_dir / "portfolio_analysis.csv"
    df.to_csv(csv_out)
    print(f"\nAnalysis saved to: {csv_out}")


if __name__ == "__main__":
    main()
