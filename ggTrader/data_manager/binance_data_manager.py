# binance_data_manager.py
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from ggTrader.data_manager.base_data_manager import DataManager
from ggTrader.utils.config import PROVIDER_CAPABILITIES
from ggTrader.utils.rate_limiter import RateLimiter
from ggTrader.utils.metadata_helper import MetadataTracker

class BinanceDataManager(DataManager):
    def __init__(self, symbol=None, interval=None, mongo_uri=None):
        super().__init__(symbol, interval, market='crypto', provider='binance', mongo_uri=mongo_uri)

        if self.interval not in PROVIDER_CAPABILITIES['binance']['intervals']:
            raise ValueError(f"Interval '{self.interval}' not supported by Binance")

        self.api_key = PROVIDER_CAPABILITIES['binance']['api_key']
        self.headers = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
        self.ratelimiter = RateLimiter(calls_per_minute=1200 / 10)  # 10 weight per call
        self.metadata_tracker = MetadataTracker(self.db, 'crypto')

    def fetch_batch(self, start_time, end_time, limit=1000):
        self.ratelimiter.wait()
        url = 'https://api.binance.us/api/v3/klines'
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000)
        }
        res = requests.get(url, params=params, headers=self.headers)
        res.raise_for_status()
        raw = res.json()
        df = pd.DataFrame(raw, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.rename(columns={'quote_asset_volume': 'volume'}, inplace=True)
        df.set_index('datetime', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    def fetch(self, start_date, end_date):
        all_data = []
        start_time = self._convert_to_datetime(start_date)
        end_time = self._convert_to_datetime(end_date)

        # Parse interval to get appropriate deltas
        interval_minutes = self._parse_interval_to_minutes(self.interval)

        # Set batch size based on interval (aim for reasonable API call sizes)
        if interval_minutes >= 1440:  # 1d or larger
            batch_size_minutes = interval_minutes * 500  # 500 intervals per batch
        elif interval_minutes >= 60:  # 1h or larger
            batch_size_minutes = interval_minutes * 1000  # 1000 intervals per batch
        else:  # smaller than 1h
            batch_size_minutes = interval_minutes * 1000  # 1000 intervals per batch

        delta = timedelta(minutes=batch_size_minutes)
        increment = timedelta(minutes=interval_minutes)

        while start_time < end_time:
            batch_end = min(start_time + delta, end_time)
            print(f"Fetching {start_time} to {batch_end}")
            df = self.fetch_batch(start_time, batch_end)
            if df.empty:
                break
            all_data.append(df)
            start_time = df.index[-1] + increment

        final_df = pd.concat(all_data) if all_data else pd.DataFrame()
        if not final_df.empty:
            self.metadata_tracker.update_metadata(self.symbol, self.interval, self.provider)
        return final_df