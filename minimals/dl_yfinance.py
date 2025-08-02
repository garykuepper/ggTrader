import yfinance as yf
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')

df = yf.download("SPY", period="1y", interval="1d", progress=False, multi_level_index=False, auto_adjust=True)
df.columns = df.columns.str.lower()

print(df.head())
print(df.columns)

db = client['learning']
collection = db['stock_market_data']

data = df.to_dict('records')

result = collection.insert_many(data)

print(F"")
