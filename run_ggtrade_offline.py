import os
import pandas as pd

daily_movers = pd.read_csv("data/kraken_historical_volume_movers.csv")

dates = pd.date_range(start="2023-01-01", end="2025-06-30", freq="D")
dates_4h = pd.date_range(start="2023-01-01", end="2025-06-30", freq="4h")


for date in dates_4h:
    print(date)