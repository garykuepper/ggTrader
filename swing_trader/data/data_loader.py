python
import os
import datetime
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from swing_trader.db.mongodb import MongoDBClient

class DataLoader:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
        self.mongo_client = MongoDBClient()

    def get_stock_data(self, symbol, start, end):
        docs = self.mongo_client.find_stock_data(symbol)
        stored_dates = set()
        for doc in docs:
            for row in doc['data']:
                stored_dates.add(row.get('Date') or row.get('Datetime'))
        all_dates = pd.date_range(start=start, end=end, freq='B').strftime('%Y-%m-%d')
        missing_dates = [d for d in all_dates if d not in stored_dates]
        if missing_dates:
            df = yf.download(symbol, start=start, end=end)
            df = df.loc[missing_dates]
            if not df.empty:
                data = df.reset_index().to_dict('records')
                self.mongo_client.insert_stock_data(
                    symbol=symbol,
                    data=data,
                    start=missing_dates[0],
                    end=missing_dates[-1]
                )
                print(f"Fetched and cached missing data for {symbol}: {missing_dates[0]} to {missing_dates[-1]}")
        docs = self.mongo_client.find_stock_data(symbol)
        all_data = []
        for doc in docs:
            all_data.extend(doc['data'])
        df_all = pd.DataFrame(all_data).drop_duplicates(subset=['Date']).sort_values('Date')
        return df_all.to_dict('records')