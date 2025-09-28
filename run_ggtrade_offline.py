import os
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta, timezone
daily_movers = pd.read_csv("data/kraken_historical_volume_movers.csv",
                           parse_dates=["date"],
                           index_col=0)

dates = pd.date_range(start="2023-01-01", end="2025-06-30", freq="D")
dates_4h = pd.date_range(start="2023-01-01", end="2025-06-30", freq="4h")

# get top n movers per day

#top_by_volume.groupby('date').head(N).reset_index(drop=True)

single_date = datetime(year=2025, month=3, day=1, tzinfo=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

single_date_movers = daily_movers[daily_movers['date'] == single_date]

print(tabulate(single_date_movers.head(20), headers="keys", tablefmt="github"))
print(single_date_movers.iloc[0])

