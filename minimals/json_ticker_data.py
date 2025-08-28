import json
from dataclasses import dataclass
from tabulate import tabulate


@dataclass
class TickerParameters:
    symbol: str
    interval: str = "1d"
    cooldown_period: int = 2
    hold_min_periods: int = 2
    ema_fast_window: int = 20
    ema_slow_window: int = 50
    trailing_stop_pct: int = 3
    position_share_pct: float = 1.0


# Load JSON file containing a list of entries
with open('tickers.json', 'r') as file:
    data_list = json.load(file)

# Create a list of TickerParameters instances
tickers = [TickerParameters(**entry) for entry in data_list]

for ticker in tickers:
    print(ticker)

# Convert each dataclass instance to a dict
data_dicts = [ticker.__dict__ for ticker in tickers]

# Print table
print("\nTicker Parameters")
print(tabulate(data_dicts, headers="keys", tablefmt="github"))