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
        normalized_data = []
        for row in data:
            normalized_row = {}
            for k, v in row.items():
                key = k.lower()
                if key.endswith(f"_{symbol.lower()}"):
                    key = key.replace(f"_{symbol.lower()}", "")
                if key == "date" or key == "datetime":
                    normalized_row["date"] = v
                elif key in ["close", "high", "low", "open", "volume"]:
                    normalized_row[key] = v
            normalized_row["symbol"] = symbol  # Add symbol to each row
            # Only add if all required fields are present
            if all(field in normalized_row for field in ["date", "close", "high", "low", "open", "volume", "symbol"]):
                normalized_data.append(normalized_row)
        if normalized_data:
            collection.insert_many(normalized_data)

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
