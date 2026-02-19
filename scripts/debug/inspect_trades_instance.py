import vectorbt as vbt
import pandas as pd
import numpy as np

# Create dummy portfolio
dates = pd.date_range(start="2023-01-01", periods=10, freq="1d")
price = pd.Series([10, 11, 12, 11, 10, 9, 8, 9, 10, 11], index=dates)
entries = pd.Series(
    [True, False, False, False, False, False, False, False, False, False], index=dates
)
exits = pd.Series(
    [False, False, False, False, True, False, False, False, False, False], index=dates
)

pf = vbt.Portfolio.from_signals(price, entries, exits)
print("Items in pf.trades dir:")
print(dir(pf.trades))
print("\nChecking total_profit:")
try:
    print(pf.trades.total_profit)
except Exception as e:
    print(f"Error accessing total_profit: {e}")

print("\nChecking pnl:")
try:
    print(pf.trades.pnl)
except Exception as e:
    print(f"Error accessing pnl: {e}")
