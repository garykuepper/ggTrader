import requests
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import os
from dotenv import load_dotenv

def get_binance_klines(symbol, interval, start_time=None, end_time=None, limit=1000, api_key=None):
    url = 'https://api.binance.us/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)

    headers = {}
    if api_key:
        headers['X-MBX-APIKEY'] = api_key  # Add your API key to the headers

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]

    df = pd.DataFrame(data, columns=cols)
    df['volume'] = df['quote_asset_volume']
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.floor('S')
    new_df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
    new_df = new_df.set_index('date')
    return new_df


load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
# Example usage:
symbol = 'BTCUSDT'
interval = '4h'
start_date = datetime(2023, 8, 1)
end_date = datetime(2023, 8, 2)

df = get_binance_klines(symbol, interval, start_date, end_date, api_key=api_key)
columns = ['symbol','close', 'high', 'low', 'open', 'volume']

for idx, row in df.iterrows():
    print(idx, row['close'])


print(tabulate(df[columns], headers='keys', tablefmt='github'))