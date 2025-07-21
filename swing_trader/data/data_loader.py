import os
import datetime
import yfinance as yf
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
MONGO_URI = os.getenv('MONGO_URI')

client = MongoClient(MONGO_URI)
db = client['ggTrader']

def get_stock_data(symbol, start, end):
    collection = db['stock_data']
    # Find all records for this symbol
    docs = list(collection.find({'symbol': symbol}))
    # Collect all stored dates
    stored_dates = set()
    for doc in docs:
        for row in doc['data']:
            stored_dates.add(row['Date'] if 'Date' in row else row['Datetime'])
    # Build full date range
    all_dates = pd.date_range(start=start, end=end, freq='B').strftime('%Y-%m-%d')
    missing_dates = [d for d in all_dates if d not in stored_dates]
    if missing_dates:
        # Download missing data
        df = yf.download(symbol, start=start, end=end)
        df = df.loc[missing_dates]
        if not df.empty:
            data = df.reset_index().to_dict('records')
            collection.insert_one({
                'symbol': symbol,
                'start': missing_dates[0],
                'end': missing_dates[-1],
                'data': data,
                'fetched_at': datetime.datetime.utcnow()
            })
            print(f"Fetched and cached missing data for {symbol}: {missing_dates[0]} to {missing_dates[-1]}")
    # Merge all data
    docs = list(collection.find({'symbol': symbol}))
    all_data = []
    for doc in docs:
        all_data.extend(doc['data'])
    # Remove duplicates and sort
    df_all = pd.DataFrame(all_data).drop_duplicates(subset=['Date']).sort_values('Date')
    return df_all.to_dict('records')