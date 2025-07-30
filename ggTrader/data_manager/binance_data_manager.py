# binance_data_manager.py
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from base_data_manager import DataManager
from provider_config import PROVIDER_CAPABILITIES
from rate_limiter import RateLimiter
from metadata_helper import MetadataTracker

class BinanceDataManager(DataManager):
    def __init__(self, symbol='BTCUSDT', interval='1m', mongo_uri="mongodb://localhost:27017/"):
        super().__init__(symbol, interval, market='crypto', provider='binance', mongo_uri=mongo_uri)

        if self.interval not in PROVIDER_CAPABILITIES['binance']['intervals']:
            raise ValueError(f"Interval '{self.interval}' not supported by Binance")

        self.api_key = PROVIDER_CAPABILITIES['binance']['api_key']
        self.headers = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
        self.ratelimiter = RateLimiter(calls_per_minute=1200 / 10)  # 10 weight per call
        self.metadata_tracker = MetadataTracker(self.db, 'crypto')

    def fetch_batch(self, start_time, end_time, limit=1000):
        self.ratelimiter.wait()
        url = 'https://api.binance.com/api/v3/klines'
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
        df.set_index('datetime', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    def fetch(self, start_date, end_date):
        all_data = []
        start_time = datetime.strptime(start_date, "%Y-%m-%d")
        end_time = datetime.strptime(end_date, "%Y-%m-%d")
        delta = timedelta(minutes=1000)

        while start_time < end_time:
            batch_end = min(start_time + delta, end_time)
            print(f"Fetching {start_time} to {batch_end}")
            df = self.fetch_batch(start_time, batch_end)
            if df.empty:
                break
            all_data.append(df)
            start_time = df.index[-1] + timedelta(minutes=1)
        final_df = pd.concat(all_data) if all_data else pd.DataFrame()
        if not final_df.empty:
            self.metadata_tracker.update_metadata(self.symbol, self.interval, self.provider)
        return final_df
