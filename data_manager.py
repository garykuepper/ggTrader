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


class StockDataManager(DataManager):

    def __init__(self):
        super().__init__()
        self.collection = self.db['stock_data']

    @staticmethod
    def get_market_days(start_date, end_date):
        nyse = mcal.get_calendar('NYSE')
        market_days = nyse.schedule(start_date=start_date, end_date=end_date)
        return market_days