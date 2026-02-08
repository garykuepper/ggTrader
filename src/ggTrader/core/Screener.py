from ggTrader.data.kraken.data_manager import KrakenData
from tabulate import tabulate
import pandas as pd
import os
from ggTrader.data.kraken.historical_data import KrakenHistoricalData

class Screener:

    def __init__(self):
        self.k_data = KrakenData()
        self.k_hdata = KrakenHistoricalData()

    def get_daily_top_kraken_by_volume(self, top_n=25):
        top = self.k_data.top_kraken_by_volume(limit=top_n, only_usd=True, exclude_stables=True, verbose=False)

        return top

    def print_top_kraken_by_volume(self, top_n=25):
        top = self.get_daily_top_kraken_by_volume(top_n=top_n)
        print(tabulate(top, headers="keys", tablefmt="github"))

    def get_historical_daily_kraken_by_volume(self, date, top_n=25):
        top_daily = self.k_hdata.get_historical_movers_by_day(date,top_n=top_n)
        return top_daily

    def print_historical_daily_kraken_by_volume(self, date, top_n=25):
        top_daily = self.get_historical_daily_kraken_by_volume(date,top_n=top_n)
        print(tabulate(top_daily, headers="keys", tablefmt="github"))

if __name__ == "__main__":
    s = Screener()
    print("\nDaily top kraken by volume:")
    s.print_top_kraken_by_volume()
    date = pd.Timestamp("2024-01-01").tz_localize('UTC')
    print(f"\nHistorical top kraken by volume: {date}")
    s.print_historical_daily_kraken_by_volume(date)