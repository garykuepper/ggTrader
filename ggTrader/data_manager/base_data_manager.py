# base_data_manager.py
import pandas as pd
from pymongo import MongoClient, ASCENDING
from datetime import datetime
from abc import ABC, abstractmethod
import pandas_market_calendars as mcal


class DataManager(ABC):
    def __init__(self, symbol, interval, market, provider, mongo_uri=None):
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

        # Flatten MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        records = df.reset_index().to_dict("records")
        for record in records:
            # Handle both 'datetime' column and index-based datetime
            if "datetime" in record:
                record["datetime"] = pd.to_datetime(record["datetime"])
            elif "Date" in record:  # Yahoo Finance uses 'Date' as column name after reset_index
                record["datetime"] = pd.to_datetime(record["Date"])
                del record["Date"]  # Remove the original Date column
            else:
                # If no datetime column found, skip this record or raise error
                continue

            record["symbol"] = self.symbol
            record["interval"] = self.interval
            record["provider"] = self.provider

        if records:
            try:
                self.collection.insert_many(records, ordered=False)
            except Exception as e:
                # Handle duplicate key errors gracefully
                if "duplicate key error" not in str(e).lower():
                    print(f"Error inserting records: {e}")

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

    def _get_trading_days(self, start_date, end_date):
        """Get valid trading days for the given date range."""
        if self.market == 'stock':
            # Use NYSE calendar for stock market
            nyse = mcal.get_calendar('NYSE')
            schedule = nyse.schedule(start_date=start_date, end_date=end_date)
            return schedule.index.date
        else:
            # For crypto, all days are trading days
            return pd.date_range(start=start_date, end=end_date, freq='D').date

    def get_missing_ranges(self, start_date, end_date):
        existing = self.load_from_db(start_date, end_date)
        print(f"Existing data shape: {existing.shape}")

        if existing.empty:
            # Check if the entire range is during non-trading days
            if self.market == 'stock' and self.interval == '1d':
                trading_days = self._get_trading_days(start_date, end_date)
                if len(trading_days) == 0:
                    return []  # No trading days in range
            return [(pd.to_datetime(start_date), pd.to_datetime(end_date))]

        # For daily stock data, only consider trading days
        if self.market == 'stock' and self.interval == '1d':
            trading_days = self._get_trading_days(start_date, end_date)
            expected = pd.to_datetime(trading_days)
        else:
            # Handle deprecated frequency aliases
            freq_mapping = {
                'm': 'ME',  # month end frequency
                'M': 'ME',  # month end frequency
            }

            # Check if interval is a time-based frequency (like 5m for 5 minutes)
            if self.interval.endswith('m') and len(self.interval) > 1 and self.interval[:-1].isdigit():
                # This is a minute-based interval like "5m", "15m", etc.
                mapped_freq = self.interval  # Keep as is for minute intervals
            else:
                # This is a period-based frequency that might need mapping
                mapped_freq = freq_mapping.get(self.interval, self.interval)

            expected = pd.date_range(start=start_date, end=end_date, freq=mapped_freq)

        missing = expected.difference(existing.index)
        # ... rest of the method remains the same

        if missing.empty:
            return []

        # Group consecutive missing timestamps into ranges
        ranges = []
        if len(missing) > 0:
            start_gap = missing[0]
            end_gap = missing[0]

            for i in range(1, len(missing)):
                # For daily intervals, check if dates are consecutive trading days
                if self.interval == '1d':
                    if missing[i] <= missing[i - 1] + pd.Timedelta(days=7):  # Allow for weekends/holidays
                        end_gap = missing[i]
                    else:
                        ranges.append((start_gap, end_gap))
                        start_gap = missing[i]
                        end_gap = missing[i]
                else:
                    # Check if current timestamp is consecutive to previous
                    if missing[i] == missing[i - 1] + pd.Timedelta(self.interval):
                        end_gap = missing[i]
                    else:
                        ranges.append((start_gap, end_gap))
                        start_gap = missing[i]
                        end_gap = missing[i]

            # Add the last range
            ranges.append((start_gap, end_gap))

        print(f"Missing ranges: {ranges}")
        return ranges

    def load_or_fetch(self, start_date, end_date):
        df = self.load_from_db(start_date, end_date)
        missing_ranges = self.get_missing_ranges(start_date, end_date)

        for start, end in missing_ranges:
            new_data = self.fetch(start, end)
            if not new_data.empty:  # Only process non-empty data
                # Remove duplicate columns before saving and concatenating
                new_data = new_data.loc[:, ~new_data.columns.duplicated()]
                self.save_to_db(new_data)

                # Also ensure existing df has no duplicate columns
                if not df.empty:
                    df = df.loc[:, ~df.columns.duplicated()]

                df = pd.concat([df, new_data])

        return df.sort_index()
    def force_update(self, start_date, end_date):
        fresh = self.fetch(start_date, end_date)
        self.save_to_db(fresh)
        return fresh

    @abstractmethod
    def fetch(self, start_date, end_date):
        pass

    @staticmethod
    def _convert_to_datetime(date_input):
        """Convert string date or datetime object to datetime object."""
        if isinstance(date_input, str):
            # Try parsing with time first, then date-only format
            try:
                return datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return datetime.strptime(date_input, "%Y-%m-%d")
        elif isinstance(date_input, datetime):
            return date_input
        else:
            raise ValueError(f"Invalid date format: {date_input}. Expected string 'YYYY-MM-DD' or datetime object.")

    @staticmethod
    def _parse_interval_to_minutes(interval):
        """Convert interval string to minutes."""
        try:
            if interval.endswith('m'):
                return int(interval[:-1])
            elif interval.endswith('h'):
                return int(interval[:-1]) * 60
            elif interval.endswith('d'):
                return int(interval[:-1]) * 1440
            elif interval.endswith('w'):
                return int(interval[:-1]) * 10080
            else:
                raise ValueError(f"Unsupported interval format: {interval}")
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Unsupported interval format: {interval}")
            raise
