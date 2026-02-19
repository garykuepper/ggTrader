import vectorbt as vbt
from vectorbt.portfolio.trades import Trades, ExitTrades, EntryTrades
import inspect

print(f"Trades: {Trades}")
print(f"ExitTrades: {ExitTrades}")
print(f"EntryTrades: {EntryTrades}")

print("\n--- Method Resolution Order (ExitTrades) ---")
print(ExitTrades.mro())

print("\n--- Inspecting profit_factor on Trades ---")
print(inspect.getsource(Trades.profit_factor) if hasattr(Trades, "profit_factor") else "Missing")

print("\n--- Inspecting profit_factor on ExitTrades ---")
print("Does ExitTrades have its own profit_factor?", "profit_factor" in ExitTrades.__dict__)
print(
    inspect.getsource(ExitTrades.profit_factor)
    if hasattr(ExitTrades, "profit_factor")
    else "Missing"
)

# Check if patch works
print("\n--- Patching Trades and checking ExitTrades ---")


def patched(self):
    pass


Trades.profit_factor = patched
print("After patch, ExitTrades.profit_factor is patched?", ExitTrades.profit_factor == patched)
