# yahoo_data_manager.py
import yfinance as yf
from ggTrader.data_manager.base_data_manager import DataManager
from ggTrader.utils.config import PROVIDER_CAPABILITIES
from ggTrader.utils.rate_limiter import RateLimiter
from ggTrader.utils.metadata_helper import MetadataTracker
import pandas as pd

class YahooDataManager(DataManager):
    def __init__(self, symbol, interval='1d', mongo_uri="mongodb://localhost:27017/"):
        super().__init__(symbol, interval, market='stock', provider='yahoo', mongo_uri=mongo_uri)

        if self.interval not in PROVIDER_CAPABILITIES['yahoo']['intervals']:
            raise ValueError(f"Interval '{self.interval}' not supported by Yahoo Finance")

        self.metadata_tracker = MetadataTracker(self.db, 'stock')

    def fetch(self, start_date, end_date):
        df = yf.download(self.symbol, start=start_date, end=end_date, interval=self.interval, auto_adjust=True, progress=False)

        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.columns = df.columns.str.lower()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index.name = 'datetime'
        self.metadata_tracker.update_metadata(self.symbol, self.interval, self.provider)
        return df
