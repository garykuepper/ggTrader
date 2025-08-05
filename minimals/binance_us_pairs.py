import requests
import pandas as pd

def fetch_binance_us_pairs():
    url = "https://api.binance.us/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()

    # Filter for Binance.US trading pairs
    binance_us_pairs = [
        {
            'symbol': symbol['symbol'],
            'baseAsset': symbol['baseAsset'],
            'quoteAsset': symbol['quoteAsset'],
            'status': symbol['status']
        }
        for symbol in data['symbols']
        if 'binanceus' in symbol['exchange'] and symbol['status'] == 'TRADING'
    ]

    # Convert to DataFrame for easy viewing and manipulation
    df = pd.DataFrame(binance_us_pairs)
    return df

# Fetch and display the trading pairs
binance_us_pairs_df = fetch_binance_us_pairs()
print(binance_us_pairs_df.head())
