"""
VectorBT Grouping and Cash Sharing Demonstration.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt


def run_demo():
    """Run the VectorBT grouping demonstration."""

    # 1. Create dummy OHLCV data for 6 'assets'
    # We'll use a date range with a frequency to avoid VectorBT errors
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    price_data = pd.DataFrame(
        np.random.randn(100, 6).cumsum(axis=0) + 100,
        index=dates,
        columns=["A1", "A2", "A3", "B1", "B2", "B3"],
    )

    # 2. Define simple signals (random for demo)
    entries = price_data > price_data.shift(1)
    exits = price_data < price_data.shift(1)

    # 3. Define Groups
    # We'll group the 6 assets into two groups: 'Growth' and 'Value'
    group_by = pd.Index(
        ["Growth", "Growth", "Growth", "Value", "Value", "Value"], name="Strategy"
    )

    print("--- Creating Portfolio with Grouping and Cash Sharing ---")
    # init_cash='autoalign' with cash_sharing=True means each GROUP gets the initial cash
    pf = vbt.Portfolio.from_signals(
        price_data,
        entries,
        exits,
        init_cash=10000,
        cash_sharing=True,
        group_by=group_by,
        fees=0.001,
        freq="D",  # Explicitly set frequency
    )

    # 4. Analyze Results
    print("\nTotal Profit per Group (Default behavior when grouped):")
    print(pf.total_profit())

    print("\nTotal Profit per Column (Disabling grouping for analysis):")
    print(pf.total_profit(group_by=False))

    print("\nSharpe Ratio per Group:")
    print(pf.sharpe_ratio(group_by=False))

    print("\nPortfolio Summary by Group:")
    # Using a subset of columns to keep output readable
    print(pf.stats().loc[["Start", "End", "Total Return [%]", "Sharpe Ratio"]])


if __name__ == "__main__":
    run_demo()
