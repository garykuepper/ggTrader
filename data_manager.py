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

        # Normalize timezones
        request_dates = request_dates.tz_convert('UTC') if request_dates.tz else request_dates.tz_localize('UTC')
        available_dates = available_dates.tz_convert('UTC') if available_dates.tz else available_dates.tz_localize(
            'UTC')

        # Normalize to date (remove times)
        request_dates = request_dates.normalize()
        available_dates = available_dates.normalize()

        missing_dates = request_dates.difference(available_dates)
        missing_list = [[d.date()] for d in missing_dates]
        dates_only = [d[0] for d in missing_list]

        return missing_list, dates_only

    def fetch_data_db(self, query):
        pass
    def get_market_day(self, symbol, date):

        return self.fetch_market_data_db(symbol, '1d', date, date)

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


class CryptoDataManager(DataManager):

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.collection = self.db['crypto_data']
        self.binance_url = 'https://api.binance.us/api/v3/klines'

    # Implement crypto-specific methods if needed
    # For example, fetching crypto data from an API or database
    # and inserting it into MongoDB.

    def get_crypto_data(self, symbol, interval, start_date, end_date):
        """
        Fetch crypto data from MongoDB or download it if not available.
        """
        # Fetch from MongoDB
        from_mongodb_df = self.fetch_market_data_db(symbol, interval, start_date, end_date)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        missing_list, dates_only = self.find_missing_dates(date_range, from_mongodb_df)

        # For df with timezone-naive index
        if from_mongodb_df.index.tz is None:
            from_mongodb_df.index = from_mongodb_df.index.tz_localize('UTC')

        # if none missing just return
        if len(missing_list) <= 0:
            return from_mongodb_df

        missing_df = self.get_missing_dates(symbol, interval, start_date, end_date, dates_only)
        print(f"Missing dates: {missing_df}")
        # insert missing into db
        self.insert_market_data_to_mdb(missing_df)
        # join and sort
        combined_df = (pd.concat([from_mongodb_df, missing_df]).sort_index())

        return self.remove_duplicates(combined_df)

    def get_binance_klines(self, symbol, interval, start_time, end_time, limit=1000):
        url = self.binance_url
        params = {'symbol': symbol, 'interval': interval, 'limit': limit,
                  'startTime': int(self.to_datetime(start_time).timestamp() * 1000),
                  'endTime': int(self.to_datetime(end_time).timestamp() * 1000)}

        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key  # Add your API key to the headers

        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]

        df = pd.DataFrame(data, columns=cols)
        df['volume'] = df['quote_asset_volume'].astype(float)  # Ensure volume is float
        df = self.convert_ohlcv_to_float(df)
        df['symbol'] = symbol
        df['interval'] = interval
        df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.floor('s')
        new_df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume','interval']]
        new_df = new_df.set_index('date')
        return new_df

    def get_missing_dates(self, symbol, interval, start_date, end_date, dates_only):
        dfs = []
        for date in dates_only:
            df = self.get_binance_klines_paginated(symbol, interval, date, date + timedelta(days=1))
            dfs.append(df)

        missing_df = pd.concat(dfs) if dfs else pd.DataFrame()
        # For missing_df (downloaded)
        if isinstance(missing_df.index, pd.DatetimeIndex) and missing_df.index.tz is None:
            missing_df.index = missing_df.index.tz_localize('UTC')

        return missing_df

    def get_binance_klines_paginated(self, symbol, interval, start_time, end_time, limit=1000):
        """
        Fetches all available klines in [start_time, end_time] (handles pagination).
        Returns a single DataFrame.
        """
        all_dfs = []
        start_dt = self.to_datetime(start_time)
        end_dt = self.to_datetime(end_time)

        while True:
            df = self.get_binance_klines(symbol, interval, start_dt, end_dt, limit=limit)
            if df.empty:
                break
            all_dfs.append(df)
            last_idx = df.index[-1]
            # Add one interval to avoid overlap, but not more than end_dt
            if last_idx >= end_dt:
                break
            # Increment start_dt to the last returned candle's open + 1 second to prevent overlap
            start_dt = last_idx + timedelta(seconds=1)
            # Safety check to avoid infinite loop
            if start_dt >= end_dt:
                break
            # Optional: If less than limit rows were returned, we've reached the end
            if len(df) < limit:
                break

        if all_dfs:
            return pd.concat(all_dfs).sort_index()
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
            'Spread %': f"{spread_pct*100:.3f}%"
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
