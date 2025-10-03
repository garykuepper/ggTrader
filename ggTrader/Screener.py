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

    def get_historical_daily_kraken_by_volume(self, date: pd.Timestamp, top_n=25):
        all_historical_movers = self.load_historical_daily_kraken_by_volume()
        if date in all_historical_movers.index:
            historical_movers = all_historical_movers.loc[date]
            return historical_movers.head(top_n)
        else:
            print(f"Date {date} not found in historical data.")
            return  pd.DataFrame()


    @staticmethod
    def load_historical_daily_kraken_by_volume():
        current_file = os.path.abspath(__file__)
        one_level_up = os.path.dirname(os.path.dirname(current_file))
        path = os.path.join(one_level_up, "data")
        # Ensure the directory exists (no crash if it doesn't yet)
        os.makedirs(path, exist_ok=True)
        all_historical_movers = pd.read_csv(os.path.join(path, f"kraken_historical_volume_movers.csv"),
                                            index_col="date",
                                            parse_dates=["date"]
                                            )

        return all_historical_movers

    def print_historical_daily_kraken_by_volume(self, date: pd.Timestamp, top_n=25):
        historical_movers = self.get_historical_daily_kraken_by_volume(date, top_n=top_n)
        print(tabulate(historical_movers, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    s = Screener()
    s.print_top_kraken_by_volume()
    date = pd.Timestamp("2024-01-01").tz_localize('UTC')
    s.print_historical_daily_kraken_by_volume(date)
