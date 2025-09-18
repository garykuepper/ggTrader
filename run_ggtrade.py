from utils.kraken_yfinance_cmc import get_top_kraken_usd_pairs
from utils.top_crypto import get_top_cmc
from tabulate import tabulate
from utils.kraken_ohlcv import fetch_ohlcv_df
import ccxt


def get_top_crypto_ohlcv(top_n=20):
    top_crypto = get_top_cmc(limit=25, print_table=True)
    n = 0
    kraken = ccxt.kraken()
    kraken.load_markets()
    df = {}
    for _, row in top_crypto.iterrows():

        symbol = row.get("Symbol")
        print(f"Fetching {symbol} OHLC data...")
        try:
            df[symbol] = fetch_ohlcv_df(kraken, symbol + '/USD', timeframe='4h', limit=30)
            n += 1
            if n >= top_n:
                break
        except Exception as e:
            print(f"Error fetching {symbol} OHLC data: {e}")
    return df


df = get_top_crypto_ohlcv(top_n=5)
first_ohlcv = next(iter(df.values()), None)
print(df.keys())
print(tabulate(first_ohlcv, headers="keys", tablefmt="github"))


