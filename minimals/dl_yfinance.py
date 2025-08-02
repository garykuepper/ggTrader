import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient


def yf_download(symbol, interval, start_date, end_date):
    df = yf.download(symbol,
                     start=start_date,
                     end=end_date,
                     interval=interval,
                     progress=False,
                     multi_level_index=False,
                     auto_adjust=True)

    df['symbol'] = symbol
    df['interval'] = interval
    df['date'] = df.index
    df.columns = df.columns.str.lower()

    return df


# Setup mongodb
client = MongoClient('mongodb://localhost:27017/')
db = client['learning']
collection = db['stock_market_data']
# collection.create_index([('symbol', 1), ('date', 1),('interval', 1)], unique=True)

# Stock Data
symbol = "SPY"
interval = "1d"
end_date = datetime(2025, 8, 1)
start_date = end_date - timedelta(days=1)

df = yf_download(symbol,
                 interval,
                 start_date.strftime("%Y-%m-%d"),
                 end_date.strftime("%Y-%m-%d"))

print(df.head())
print(df.columns)
data = df.to_dict('records')

# result = collection.insert_many(data)

# print(F"Num of ids: {len(result.inserted_ids)}")
