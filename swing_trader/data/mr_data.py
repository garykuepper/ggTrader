import os
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas_market_calendars as mcal
import ta


class MrData:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
        mongo_uri = os.getenv('MONGO_URI')
        self.client = MongoClient(mongo_uri)
        self.db = self.client['ggTrader']

    @staticmethod
    def format_yf_download(df, ticker):
        df.columns = df.columns.droplevel('Ticker')
        df.columns.name = None
        df = df.reset_index()
        df['Date'] = df['Date'].astype(str)
        df['Ticker'] = ticker
        return df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    @staticmethod
    def get_existing_docs(collection, ticker, start, end):
        query = {"Ticker": ticker, "Date": {"$gte": start, "$lte": end}}
        return list(collection.find(query))

    @staticmethod
    def get_missing_dates(existing_docs, start, end):
        db_dates = set(doc['Date'] for doc in existing_docs)
        wanted_dates = pd.date_range(start, end, freq='B').strftime('%Y-%m-%d').tolist()

        # Get NYSE trading days (removes holidays)
        nyse = mcal.get_calendar('NYSE')
        trading_days = nyse.schedule(start_date=start, end_date=end).index.strftime('%Y-%m-%d').tolist()

        # Only consider missing dates that are actual trading days
        missing_dates = [date for date in trading_days if date not in db_dates]
        return trading_days, missing_dates

    def fetch_yf_data(self, ticker, min_date, max_date):
        return yf.download(
            ticker,
            start=min_date,
            end=(pd.to_datetime(max_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            auto_adjust=True
        )

    def insert_new_docs(self, collection, df_new, missing_dates, ticker):
        df_new = self.format_yf_download(df_new, ticker)
        df_new = df_new[df_new['Date'].isin(missing_dates)]
        new_docs = df_new.to_dict(orient='records')
        if new_docs:
            collection.insert_many(new_docs)
            print(f"Inserted {len(new_docs)} new records for {ticker} into MongoDB.")
        return new_docs

    def download_and_insert_missing(self, collection, ticker, missing_dates):
        if not missing_dates:
            return []
        min_date, max_date = min(missing_dates), max(missing_dates)
        raw_df = yf.download(
            ticker,
            start=min_date,
            end=(pd.to_datetime(max_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            auto_adjust=True
        )
        df_new = self.format_yf_download(raw_df, ticker)
        df_new = df_new[df_new['Date'].isin(missing_dates)]
        new_docs = df_new.to_dict(orient='records')
        if new_docs:
            collection.insert_many(new_docs)
            print(f"Inserted {len(new_docs)} new records for {ticker} into MongoDB.")
        return new_docs

    def get_stock_data(self, ticker, start, end):

        # TODO: Validate start and end dates
        # TODO: Apply indicators to the data and save to the database
        collection = self.db['stock_data']
        existing_docs = self.get_existing_docs(collection, ticker, start, end)
        wanted_dates, missing_dates = self.get_missing_dates(existing_docs, start, end)
        new_docs = self.download_and_insert_missing(collection, ticker, missing_dates)
        docs = existing_docs + new_docs
        if docs:
            df = pd.DataFrame(docs)
            df = df.drop(columns=['_id'], errors='ignore')
            df = df[df['Date'].isin(wanted_dates)].sort_values('Date')
            return df.reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    def enrich_ticker_with_indicators(self, ticker):
        collection = self.db['stock_data']
        docs = list(collection.find({'Ticker': ticker}))
        if not docs:
            print(f"No data found for ticker {ticker}.")
            return
        df = pd.DataFrame(docs)
        df = df.sort_values('Date').reset_index(drop=True)
        df = ta.add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True
        )
        indicator_cols = [col for col in df.columns if
                          col not in ['_id', 'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        for _, row in df.iterrows():
            update_fields = {col: row[col] for col in indicator_cols}
            collection.update_one(
                {"_id": row["_id"]},
                {"$set": update_fields}
            )
        print(f"Updated {len(df)} documents for {ticker} with technical indicators.")
