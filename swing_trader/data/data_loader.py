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

        if missing_dates:
            # Download missing data from Yahoo Finance
            df = yf.download(symbol, start=min(missing_dates), end=max(missing_dates), auto_adjust=True)
            if not df.empty:
                # Flatten MultiIndex columns if necessary
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [str(col[0]).lower() for col in df.columns]
                else:
                    df.columns = [str(col).lower() for col in df.columns]

                df = df.reset_index()

                # Find the date column (case-insensitive), rename to 'date'
                date_col = next((col for col in df.columns if str(col).lower() in ['date', 'datetime']), None)
                if date_col is None:
                    print("No date column found! Columns:", df.columns)
                    raise KeyError("No date column found in DataFrame.")
                df = df.rename(columns={date_col: 'date'})
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                # Only keep expected columns, plus symbol
                expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: missing columns {missing_cols} in downloaded data.")
                df = df[[col for col in expected_cols if col in df.columns]]
                df['symbol'] = symbol

                # Only insert rows for missing dates
                df = df[df['date'].isin(missing_dates)]

                if not df.empty:
                    self.mongo_client.insert_stock_data(symbol, df.to_dict('records'))
                    print(f"Fetched and cached {len(df)} missing rows for {symbol}: {df['date'].min()} to {df['date'].max()}")

        # Get all requested data from DB and return as sorted records
        docs = self.mongo_client.find_stock_data(symbol, start, end)
        if not docs:
            print(f"No data found for {symbol} in {start} to {end}")
            return []
        df_all = pd.DataFrame(docs).sort_values('date')
        return df_all.to_dict('records')
