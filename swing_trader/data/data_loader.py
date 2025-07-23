import os
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from swing_trader.db.mongodb import MongoDBClient

class DataLoader:
    def __init__(self):
        # Load environment variables for MongoDB connection if needed
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
        self.mongo_client = MongoDBClient()

    def get_stock_data(self, symbol, start, end):
        # Fetch existing data from MongoDB
        docs = self.mongo_client.find_stock_data(symbol, start, end)
        stored_dates = set(doc['date'] for doc in docs if 'date' in doc)


        # Generate all business dates between start and end
        all_dates = set(pd.date_range(start=start, end=end, freq='B').strftime('%Y-%m-%d'))
        missing_dates = sorted(list(all_dates - stored_dates))

        df = yf.download(symbol, start=min(missing_dates), end=max(missing_dates), auto_adjust=True)
        return df