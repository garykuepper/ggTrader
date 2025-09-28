import os
import pandas as pd
import numpy as np
import mplfinance as mpf
from tabulate import tabulate
from datetime import datetime, timedelta, timezone
from ggTrader.Position import Position
from ggTrader.Portfolio import Portfolio
from ggTrader.Signals import Signals

def load_daily_movers():
    daily_movers = pd.read_csv("data/kraken_historical_volume_movers.csv",
                               parse_dates=["date"],
                               index_col=0)
    return daily_movers


def get_daily_movers_by_date(daily_movers: pd.DataFrame, date: datetime, top_n: int = 20):
    top_df = daily_movers[daily_movers['date'] == date].head(top_n)
    return top_df['ticker'].to_list()


def get_ohlcv_csv(path: str, tickers: list = None):
    with os.scandir(path) as it:
        files = [entry.name for entry in it if entry.is_file()]
    num_files = len(files)
    ohlcv_dict = {}
    datetime_index = pd.date_range(start="2024-01-01",
                                   end="2025-06-30",
                                   freq="4h",
                                   tz="UTC")

    for f in files:
        ticker = f.split("_")[0]
        if tickers is not None and ticker not in tickers:
            continue
        print(f"({files.index(f) + 1}/{num_files}) Processing {f} ")
        file_path = os.path.join(path, f)
        df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
        # Reindex to full timeline, but interpolate only after data starts
        tmp = df.reindex(datetime_index)
        first_valid = tmp.first_valid_index()
        if first_valid is not None:
            tmp.loc[first_valid:] = tmp.loc[first_valid:].interpolate()
        ohlcv_dict[ticker] = tmp
    print(f'Loaded OHLCV data for {len(ohlcv_dict)} tickers.')
    return ohlcv_dict


dates = pd.date_range(start="2023-01-01", end="2025-06-30", freq="D")
dates_4h = pd.date_range(start="2023-01-01", end="2025-06-30", freq="4h")

top_n = 20
single_date = datetime(year=2025, month=3, day=1, tzinfo=timezone.utc).replace(hour=0, minute=0, second=0,
                                                                               microsecond=0)

daily_movers = load_daily_movers()
daily_movers_list = get_daily_movers_by_date(daily_movers, single_date, top_n=top_n)

print(daily_movers_list)

ohlcv_dict = get_ohlcv_csv("data/kraken_hist_4h_latest", daily_movers_list)

ohlcv = ohlcv_dict['TRUMP']

signals = Signals().compute(ohlcv)

print(tabulate(signals.head(20), headers="keys", tablefmt="github"))
print(tabulate(signals.tail(20), headers="keys", tablefmt="github"))