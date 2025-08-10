import os
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient
from tabulate import tabulate
import pandas_market_calendars as mcal
import pandas as pd
import requests


class DataManager():

    def __init__(self):

        load_dotenv()
        self.mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client['market_data']
        self.collection = None

    @staticmethod
    def find_missing_dates(request_dates, available_dates):
        """
        Takes request_dates and available_dates (either DataFrame, Series, or DatetimeIndex).
        Returns missing dates between request_dates and available_dates as a list.
        """
        # Ensure both are DatetimeIndex
        if isinstance(request_dates, (pd.DataFrame, pd.Series)):
            request_dates = request_dates.index
        if isinstance(available_dates, (pd.DataFrame, pd.Series)):
            available_dates = available_dates.index

        if len(request_dates) == 0:
            return [[]], []

        missing_dates = request_dates.difference(available_dates)
        missing_list = [[d.replace(minute=0, second=0, microsecond=0)] for d in missing_dates]
        dates_only = [d[0] for d in missing_list]
        return missing_list, dates_only

    def fetch_data_db(self, query):
        pass

    def fetch_market_data_db(self, symbol, interval, start_date, end_date):
        """
        return as df
        """

        query = {
            'symbol': symbol,
            'interval': interval,
            'date': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        results = list(self.collection.find(query).sort('date', 1))
        return self.mongodb_fmt_to_df(results)

    @staticmethod
    def mongodb_fmt_to_df(results):

        if not results:
            return pd.DataFrame().set_index(pd.DatetimeIndex([]))
        df = pd.DataFrame(results)
        # If 'date' is already the index, access it accordingly
        if 'date' in df.columns:
            if df['date'].dt.tz is None:
                df['date'] = df['date'].dt.tz_localize('UTC')
            df = df.set_index('date')
        elif isinstance(df.index, pd.DatetimeIndex):
            # If index is DatetimeIndex and naive, localize it
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
        else:
            raise KeyError("'date' column or index not found in DataFrame")

        # Drop _id if exists
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        return df

    @staticmethod
    def df_to_mdb_fmt(df):

        df['date'] = df.index
        records = df.to_dict('records')
        return records

    @staticmethod
    def remove_duplicates(combined_df):
        combined_df.index.name = 'date'  # Name the index 'date'
        if 'date' in combined_df.columns:
            combined_df = combined_df.drop(columns=['date'])
        combined_df = combined_df.reset_index()  # 'date' becomes a column
        combined_df = combined_df.drop_duplicates(subset=['symbol', 'date', 'interval'])
        combined_df = combined_df.set_index('date')
        return combined_df

    def insert_market_data_to_mdb(self, df):
        records = self.df_to_mdb_fmt(df)
        for record in records:
            self.collection.update_one(
                {'symbol': record['symbol'], 'date': record['date'], 'interval': record['interval']},
                # unique key filter
                {'$set': record},  # update with new data
                upsert=True  # insert if not exists
            )
        print(f"Upserted {len(records)} documents.")

    def get_missing_dates(self, symbol, interval, start_date, end_date, dates_only):
        pass

    @staticmethod
    def to_datetime(dt):

        if isinstance(dt, datetime):
            return dt
        else:
            return datetime.combine(dt, datetime.min.time())

    @staticmethod
    def convert_ohlcv_to_float(df):
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df

    @staticmethod
    def ensure_utc_timezone(df):
        """Ensure DataFrame index has UTC timezone."""
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df

    @staticmethod
    def interval_to_timedelta(interval):
        """Convert Binance interval string to timedelta."""
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        else:
            raise ValueError(f"Unknown interval: {interval}")
class StockDataManager(DataManager):

    def __init__(self):
        super().__init__()
        self.collection = self.db['stock_data']

    @staticmethod
    def get_market_days(start_date, end_date):
        nyse = mcal.get_calendar('NYSE')
        market_days = nyse.schedule(start_date=start_date, end_date=end_date)
        return market_days

    @staticmethod
    def yf_download_df(symbol, interval, start_date, end_date):
        df = yf.download(symbol,
                         start=start_date,
                         end=end_date,
                         interval=interval,
                         progress=False,
                         multi_level_index=False,
                         auto_adjust=True)

        df['symbol'] = symbol
        df['interval'] = interval
        df.columns = df.columns.str.lower()
        return df

    def get_stock_data(self, symbol, interval, start_date, end_date):
        """
        Fetch stock data from MongoDB or download it if not available.
        """
        # Fetch from MongoDB
        from_mongodb_df = self.fetch_market_data_db(symbol, interval, start_date, end_date)
        market_days = self.get_market_days(start_date, end_date)
        missing_list, dates_only = self.find_missing_dates(market_days, from_mongodb_df)

        # For df with timezone-naive index
        if from_mongodb_df.index.tz is None:
            from_mongodb_df.index = from_mongodb_df.index.tz_localize('UTC')

        # if none missing just return
        if len(missing_list) <= 0:
            return from_mongodb_df

        missing_df = self.get_missing_dates(symbol, interval, start_date, end_date, dates_only)

        # insert missing into db
        self.insert_market_data_to_mdb(missing_df)
        # join and sort
        combined_df = (pd.concat([from_mongodb_df, missing_df]).sort_index())

        return self.remove_duplicates(combined_df)

    def get_missing_dates(self, symbol, interval, start_date, end_date, dates_only):
        dfs = []
        for date in dates_only:
            df = self.yf_download_df(symbol, interval, date, date + timedelta(days=1))
            dfs.append(df)

        missing_df = pd.concat(dfs) if dfs else pd.DataFrame()
        # For missing_df (downloaded)
        if missing_df.index.tz is None:
            missing_df.index = missing_df.index.tz_localize('UTC')
        return missing_df


# Python
import os
import time
from datetime import timedelta, timezone, datetime
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

class CryptoDataManager(DataManager):

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.collection = self.db['crypto_data']
        self.binance_url = 'https://api.binance.us/api/v3/klines'

        # NEW: robust HTTP session with retries and backoff
        self._session = requests.Session()
        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST", "DELETE"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        if self.api_key:
            self._session.headers.update({'X-MBX-APIKEY': self.api_key})
        # Default timeout (seconds) for all HTTP calls from this manager
        self._http_timeout = 20

    def get_crypto_data(self, symbol, interval, start_date, end_date):
        """
        Fetch crypto data from MongoDB or download it if not available.
        """
        # Fetch from MongoDB
        from_mongodb_df = self.fetch_market_data_db(symbol, interval, start_date, end_date)
        date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
        missing_list, dates_only = self.find_missing_dates(date_range, from_mongodb_df)
        # For df with timezone-naive index
        if from_mongodb_df.index.tz is None:
            from_mongodb_df.index = from_mongodb_df.index.tz_localize('UTC')

        # if none missing just return
        if len(missing_list) <= 0:
            return from_mongodb_df

        missing_df = self.get_missing_dates(symbol, interval, start_date, end_date, dates_only)
        if not missing_df.empty:
            print(f"Missing dates: {missing_df}")
        # insert missing into db
        self.insert_market_data_to_mdb(missing_df)
        # Before concatenation, filter out empty DataFrames
        dfs_to_concat = [df for df in [from_mongodb_df, missing_df] if not df.empty]
        if dfs_to_concat:
            combined_df = pd.concat(dfs_to_concat).sort_index()
        else:
            combined_df = pd.DataFrame()
        # join and sort

        return self.remove_duplicates(combined_df)

    def get_binance_klines(self, symbol, interval, start_time, end_time, limit=1000) -> pd.DataFrame:
        """
        Fetch one page of klines from Binance US with proper timeout and retries.
        Returns an OHLCV DataFrame indexed by UTC datetime ('date').
        """
        url = self.binance_url
        # Normalize to UTC aware datetimes
        sdt = self.to_datetime(start_time)
        edt = self.to_datetime(end_time)
        if sdt.tzinfo is None:
            sdt = sdt.replace(tzinfo=timezone.utc)
        if edt.tzinfo is None:
            edt = edt.replace(tzinfo=timezone.utc)

        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(int(limit), 1000),
            'startTime': int(sdt.timestamp() * 1000),
            'endTime': int(edt.timestamp() * 1000),
        }

        try:
            resp = self._session.get(url, params=params, timeout=self._http_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            # Graceful failure: return empty frame so pagination logic can stop/continue
            # You can log this if you have logging configured.
            # print(f"Binance request failed: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]

        # Binance returns a list of lists; construct then coerce types
        df = pd.DataFrame(data, columns=cols)

        # Convert types safely
        numeric_cols = ['open', 'high', 'low', 'close', 'quote_asset_volume']
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Use quote_asset_volume as 'volume' to be consistent with your code
        df['volume'] = df['quote_asset_volume']

        df['symbol'] = symbol
        df['interval'] = interval
        # Force UTC timezone and floor to seconds
        df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True).dt.floor('s')
        new_df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'interval']]
        new_df = new_df.set_index('date')
        return new_df

    def get_missing_dates(self, symbol, interval, start_date, end_date, dates_only):
        dfs = []
        delta = self.interval_to_timedelta(interval)
        for date in dates_only:
            df = self.get_binance_klines_paginated(symbol, interval, date, date + delta)
            if not df.empty:
                dfs.append(df)

        missing_df = pd.concat(dfs) if dfs else pd.DataFrame()
        if isinstance(missing_df.index, pd.DatetimeIndex) and missing_df.index.tz is None:
            missing_df.index = missing_df.index.tz_localize('UTC')
        # Drop duplicates by index and symbol/interval
        if not missing_df.empty:
            missing_df = missing_df[~missing_df.index.duplicated(keep='first')]
            missing_df = missing_df.drop_duplicates(subset=['symbol', 'interval'])
        return missing_df

    def get_binance_klines_paginated(self, symbol, interval, start_time, end_time, limit=1000):
        """
        Fetches all available klines in [start_time, end_time] (handles pagination).
        Returns a single DataFrame.
        """
        all_dfs = []

        # Normalize inputs to UTC-aware datetimes
        start_dt = self.to_datetime(start_time)
        end_dt = self.to_datetime(end_time)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        # Safety controls
        max_pages = 5000  # hard cap to avoid infinite loops in worst cases
        page = 0

        # Determine natural candle step for forward progress
        step_td = self.interval_to_timedelta(interval)
        if step_td <= timedelta(0):
            step_td = timedelta(seconds=1)

        while start_dt < end_dt and page < max_pages:
            df = self.get_binance_klines(symbol, interval, start_dt, end_dt, limit=limit)
            page += 1

            if df.empty:
                # No more data available or request failed; stop pagination for this slice
                break

            all_dfs.append(df)

            # Forward progress based on the last returned candle open time
            last_idx = df.index[-1]
            if last_idx.tzinfo is None:
                last_idx = last_idx.tz_localize('UTC')

            # If we've reached or passed end, stop
            if last_idx >= end_dt:
                break

            # Move start forward to the next candle open to avoid overlap
            # Increment by the natural interval (safer than +1s)
            start_dt = last_idx + step_td

            # Extra guard: if somehow start_dt did not advance, bump by one second
            if start_dt <= last_idx:
                start_dt = last_idx + timedelta(seconds=1)

            # If the API returned fewer rows than limit, we likely reached the end
            if len(df) < limit:
                # However, we might still be within the [start_dt, end_dt) range;
                # try one more small step to ensure completeness.
                # If this fetch returns empty, the next loop iteration will break.
                continue

            # Respectful small delay to avoid hitting rate limits in tight loops
            time.sleep(0.05)

        if all_dfs:
            out = pd.concat(all_dfs).sort_index()
            # Deduplicate: same candle can appear across overlapping pages
            out = out[~out.index.duplicated(keep='first')]
            return out
        else:
            return pd.DataFrame()

    @staticmethod
    def get_24hr_top_binance(top_n=10, quote=None, min_change=.5, min_trades=50, min_volume=0):
        url = "https://api.binance.us/api/v3/ticker/24hr"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if quote:
                data = [x for x in data if x['symbol'].endswith(quote)]
            # Filter using all criteria
            data = [
                x for x in data
                if float(x['quoteVolume']) > min_volume
                   and float(x['lastPrice']) > 0
                   and int(x.get('count', 0)) > min_trades
                   and abs(float(x['priceChangePercent'])) > min_change
            ]
            sorted_pairs = sorted(data, key=lambda x: float(x['quoteVolume']), reverse=True)
            return sorted_pairs[:top_n]
        except Exception as e:
            print(f"⚠️ Error fetching Binance.US data: {e}")
            return []

    @staticmethod
    def calculate_spread_pct(pair):
        try:
            bid = float(pair.get('bidPrice', 0))
            ask = float(pair.get('askPrice', 0))
            last = float(pair.get('lastPrice', 1)) or 1
            return abs(ask - bid) / last if last else 0
        except Exception:
            return 0

    @staticmethod
    def format_pair_row(pair, spread_pct):
        return {
            'Symbol': pair['symbol'],
            'Volume': f"${float(pair['quoteVolume']):,.2f}",
            'Change': f"{float(pair['priceChangePercent']):.2f}%",
            'Price': f"${float(pair['lastPrice']):.4f}",
            'Wt. Price': f"${float(pair['weightedAvgPrice']):.4f}",
            'Trades': pair.get('count', 0),
            'Spread %': f"{spread_pct * 100:.3f}%"
        }

    def print_top_pairs(self, top_n=10, quote=None, min_volume=None):
        top_pairs = self.get_24hr_top_binance(top_n, quote=quote, min_volume=min_volume)
        if not top_pairs:
            print("No data available.")
            return

        table_data = []
        for pair in top_pairs:
            spread_pct = self.calculate_spread_pct(pair)
            table_data.append(self.format_pair_row(pair, spread_pct))

        print(tabulate(table_data, headers="keys", tablefmt="github"))
