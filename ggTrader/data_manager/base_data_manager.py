# base_data_manager.py
import pandas as pd
from pymongo import MongoClient, ASCENDING
from datetime import datetime
from abc import ABC, abstractmethod

class DataManager(ABC):
    def __init__(self, symbol, interval, market, provider, mongo_uri="mongodb://localhost:27017/"):
        self.symbol = symbol.upper()
        self.interval = interval
        self.market = market
        self.provider = provider
        self.client = MongoClient(mongo_uri)
        self.db = self.client["market_data"]
        self.collection = self.db[f"{self.market}_market_data"]

        self.collection.create_index(
            [("symbol", ASCENDING), ("datetime", ASCENDING), ("interval", ASCENDING)],
            unique=True
        )

    def save_to_db(self, df):
        if df.empty:
            return

        df = df.copy()
        df["symbol"] = self.symbol
        df["interval"] = self.interval
        df["provider"] = self.provider
        df.reset_index(inplace=True)

        records = df.to_dict("records")
        for record in records:
            record["datetime"] = pd.to_datetime(record["datetime"])

        try:
            self.collection.insert_many(records, ordered=False)
        except Exception as e:
            if "duplicate key error" not in str(e):
                raise

    def load_from_db(self, start_date, end_date):
        query = {
            "symbol": self.symbol,
            "interval": self.interval,
            "datetime": {
                "$gte": pd.to_datetime(start_date),
                "$lte": pd.to_datetime(end_date)
            }
        }
        cursor = self.collection.find(query)
        df = pd.DataFrame(cursor)
        if df.empty:
            return df
        df.set_index("datetime", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        return df.sort_index()

    def get_missing_ranges(self, start_date, end_date):
        existing = self.load_from_db(start_date, end_date)
        if existing.empty:
            return [(start_date, end_date)]

        expected = pd.date_range(start=start_date, end=end_date, freq=self.interval)
        missing = expected.difference(existing.index)
        if missing.empty:
            return []

        ranges = []
        current_start = missing[0]
        for i in range(1, len(missing)):
            if (missing[i] - missing[i - 1]) != pd.Timedelta(self.interval):
                ranges.append((current_start, missing[i - 1]))
                current_start = missing[i]
        ranges.append((current_start, missing[-1]))
        return [(d[0].strftime("%Y-%m-%d %H:%M:%S"), d[1].strftime("%Y-%m-%d %H:%M:%S")) for d in ranges]

    def load_or_fetch(self, start_date, end_date):
        df = self.load_from_db(start_date, end_date)
        missing_ranges = self.get_missing_ranges(start_date, end_date)
        for start, end in missing_ranges:
            new_data = self.fetch(start, end)
            self.save_to_db(new_data)
            df = pd.concat([df, new_data])
        return df.sort_index()

    def force_update(self, start_date, end_date):
        fresh = self.fetch(start_date, end_date)
        self.save_to_db(fresh)
        return fresh

    @abstractmethod
    def fetch(self, start_date, end_date):
        pass
