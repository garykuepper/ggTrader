import os
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient
from tabulate import tabulate
import pandas_market_calendars as mcal
import pandas as pd


class DataManager():

    def __init__(self):

        load_dotenv()
        self.mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
        self.client = MongoClient(self.mongo.uri)
        self.db = self.client['market_data']
        self.collection = None

    def find_missing_dates(self, request_dates, available_dates):

        if request_dates.empty:
            missing_list = [[]]
            dates_only = []
            return missing_list, dates_only

        # normalize timezone
        available_dates = available_dates.index.tz_convert(
            'UTC') if available_dates.index.tz else available_dates.index.tz_localize('UTC')
        request_dates = request_dates.index.tz_convert(
            'UTC') if request_dates.index.tz else request_dates.index.tz_localize('UTC')
        # normalize
        available_dates = available_dates.normalize()
        request_dates = request_dates.normalize()

        missing_dates = request_dates.difference(available_dates)
        missing_list = [[d.date()] for d in missing_dates]
        dates_only = [d[0] for d in missing_list]

        return missing_list, dates_only

    def fetch_from_mongodb(symbol, interval, start_date, end_date):
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

        # missing_df = pd.concat(dfs) if dfs else pd.DataFrame()
        return self.mongodb_to_df(results)

    def mongodb_to_df(results):

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

class StockDataManager(DataManager):

    def __init__(self):
        super().__init__()
        self.collection = self.db['stock_data']

    @staticmethod
    def get_market_days(start_date, end_date):
        nyse = mcal.get_calendar('NYSE')
        market_days = nyse.schedule(start_date=start_date, end_date=end_date)
        return market_days