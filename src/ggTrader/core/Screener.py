"""Asset screener for daily and historical volume-based ranking."""

import pandas as pd
from tabulate import tabulate

from ggTrader.data.kraken.data_manager import KrakenData
from ggTrader.data.kraken.historical_data import KrakenHistoricalData


class Screener:

    def __init__(self) -> None:
        self.k_data = KrakenData()
        self.k_hdata = KrakenHistoricalData()

    def get_daily_top_kraken_by_volume(self, top_n: int = 25) -> pd.DataFrame:
        """Fetch the current daily top movers by USD volume from ccxt."""
        return self.k_data.top_kraken_by_volume(
            limit=top_n, only_usd=True, exclude_stables=True, verbose=False
        )

    def print_top_kraken_by_volume(self, top_n: int = 25) -> None:
        """Print the current daily top movers table."""
        top = self.get_daily_top_kraken_by_volume(top_n=top_n)
        print(tabulate(top, headers="keys", tablefmt="github"))

    def get_historical_daily_kraken_by_volume(
        self, date: pd.Timestamp, top_n: int = 25
    ) -> pd.DataFrame:
        """Fetch historical daily movers for a specific date."""
        return self.k_hdata.get_historical_movers_by_day(date, top_n=top_n)

    def print_historical_daily_kraken_by_volume(
        self, date: pd.Timestamp, top_n: int = 25
    ) -> None:
        """Print historical daily movers table for a specific date."""
        top_daily = self.get_historical_daily_kraken_by_volume(date, top_n=top_n)
        print(tabulate(top_daily, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    s = Screener()
    print("\nDaily top kraken by volume:")
    s.print_top_kraken_by_volume()
    date = pd.Timestamp("2024-01-01").tz_localize("UTC")
    print(f"\nHistorical top kraken by volume: {date}")
    s.print_historical_daily_kraken_by_volume(date)
