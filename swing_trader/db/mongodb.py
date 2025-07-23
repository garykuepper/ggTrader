# swing_trader/db/mongodb.py

import os
from pymongo import MongoClient
from dotenv import load_dotenv


class MongoDBClient:

    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
        mongo_uri = os.getenv('MONGO_URI')
        self.client = MongoClient(mongo_uri)
        self.db = self.client['ggTrader']

    def insert_stock_data(self, symbol, data):
        collection = self.db['stock_data']

    def find_stock_data(self, symbol, start=None, end=None):
        collection = self.db['stock_data']
        query = {'symbol': symbol}
        if start:
            query['start'] = start
        if end:
            query['end'] = end
        return list(collection.find(query))

    def insert_trade(self, trade):
        collection = self.db['trades']
        collection.insert_one(trade)

    def find_trades(self, symbol=None):
        collection = self.db['trades']
        query = {'symbol': symbol} if symbol else {}
        return list(collection.find(query))

    def insert_performance(self, perf):
        collection = self.db['performance']
        collection.insert_one(perf)

    def find_performance(self, symbol=None):
        collection = self.db['performance']
        query = {'symbol': symbol} if symbol else {}
        return list(collection.find(query))

    def get_collection(self, name):
        return self.db[name]

    def find_signals(self, symbol, short_window=None, long_window=None):
        collection = self.db['signals']
        query = {'symbol': symbol}
        if short_window is not None:
            query['short_window'] = short_window
        if long_window is not None:
            query['long_window'] = long_window
        return list(collection.find(query))
