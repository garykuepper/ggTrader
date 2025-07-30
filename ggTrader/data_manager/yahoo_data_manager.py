# yahoo_data_manager.py
import yfinance as yf
from base_data_manager import DataManager
from provider_config import PROVIDER_CAPABILITIES
from metadata_helper import MetadataTracker

class YahooDataManager(DataManager):
    def __init__(self, symbol, interval='1d', mongo_uri="mongodb://localhost:27017/"):
        super().__init__(symbol, interval, market='stock', provider='yahoo', mongo_uri=mongo_uri)

        if self.interval not in PROVIDER_CAPABILITIES['yahoo']['intervals']:
            raise ValueError(f"Interval '{self.interval}' not supported by Yahoo Finance")

        self.metadata_tracker = MetadataTracker(self.db, 'stock')

    def fetch(self, start_date, end_date):
        df = yf.download(self.symbol, start=start_date, end=end_date, interval=self.interval)
        if df.empty:
            return df
        df.rename(columns=str.lower, inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index.name = 'datetime'
        self.metadata_tracker.update_metadata(self.symbol, self.interval, self.provider)
        return df
