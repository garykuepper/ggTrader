"""Asset screener for daily and historical volume-based ranking."""

from __future__ import annotations

import pandas as pd
from tabulate import tabulate

from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader
from ggTrader.data.live.exchange_loader import LiveExchangeLoader


class Screener:
    """Provides daily and historical mover symbol lists."""

    def __init__(self) -> None:
        self.live_loader = LiveExchangeLoader(exchange_id="kraken")
        self.hist_loader = TimescaleDBLoader()

    def get_daily_top_kraken_by_volume(self, top_n: int = 25) -> pd.DataFrame:
        """Fetch the current daily top movers by USD volume from ccxt."""
        return self.live_loader.get_top_by_volume(
            limit=top_n, quote="USD", exclude_stables=True, verbose=False
        )

    def print_top_kraken_by_volume(self, top_n: int = 25) -> None:
        """Print the current daily top movers table."""
        top = self.get_daily_top_kraken_by_volume(top_n=top_n)
        print(tabulate(top, headers="keys", tablefmt="github"))

    def get_historical_daily_kraken_by_volume(
        self, date: pd.Timestamp, top_n: int = 25
    ) -> pd.DataFrame:
        """Fetch historical daily movers for a specific date."""
        mask = self.hist_loader.get_daily_mover_mask(start=date, end=date, top_n=top_n)
        if mask.empty:
            return pd.DataFrame()

        if date in mask.index:
            symbols = mask.columns[mask.loc[date]].tolist()
            return pd.DataFrame({"symbol": symbols})
        return pd.DataFrame()

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

