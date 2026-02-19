import vectorbt as vbt
from vectorbt.portfolio.trades import Trades
import inspect

print("Checking Trades.total_profit...")
if hasattr(Trades, "total_profit"):
    print("Trades has total_profit")
else:
    print("Trades DOES NOT have total_profit")

print("Checking ancestors...")
for cls in inspect.getmro(Trades):
    if hasattr(cls, "total_profit"):
        print(f"Found total_profit in {cls}")
