import vectorbt as vbt
from vectorbt.portfolio.trades import Trades
import inspect

try:
    print(inspect.getsource(Trades.profit_factor))
except Exception as e:
    print(f"Could not get source: {e}")
