from utils.KrakenData import KrakenData
from tabulate import tabulate
import pandas as pd
import os


class Screener:

    def __init__(self):
        self.k_data = KrakenData()

    def get_daily_top_kraken_by_volume(self, top_n=25):
        top = self.k_data.top_kraken_by_volume(limit=top_n, only_usd=True, exclude_stables=True, verbose=False)

        return top

    def print_top_kraken_by_volume(self, top_n=25):
        top = self.get_daily_top_kraken_by_volume(top_n=top_n)
        print(tabulate(top, headers="keys", tablefmt="github"))



if __name__ == "__main__":
    s = Screener()
    s.print_top_kraken_by_volume()
    date = pd.Timestamp("2024-01-01").tz_localize('UTC')
