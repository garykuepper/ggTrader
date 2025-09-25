from utils.kraken_yfinance_cmc import get_top_kraken_by_volume, get_kraken_asset_pairs_usd
from utils.top_crypto import get_top_cmc
from tabulate import tabulate
# df = get_top_kraken_by_volume(top_n=20)
#
# print(tabulate(df, headers="keys", tablefmt="github"))
import pandas as pd
import pickle

col_names = ["date", "open", "high", "low", "close", "volume", "trades"]

# python
symbols = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "TRX", "ADA", "LINK", "AVAX",
    "SUI", "XLM", "BCH",  "LTC", "TON", "SHIB", "CRO", "DOT", "MNT", "XMR"
]


kraken_altnames = {
    "BTC": "XBT",
    "DOGE": "XDG"
}

def get_kraken_top_crypto(top_n=25):
    top_crypto = get_top_cmc(limit=top_n + 5, print_table=False)
    # top_crypto = get_top_kraken_by_volume(top_n=top_n)
    # filter out non-Kraken pairs
    kraken_usd_pairs = pd.DataFrame(get_kraken_asset_pairs_usd())
    top_crypto = top_crypto[top_crypto["Symbol"].isin(kraken_usd_pairs["base_common"])]
    top_crypto = top_crypto.reset_index(drop=True)
    return top_crypto.head(top_n)

def save_ohlcv_dict(ohlcv, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(ohlcv, f)


def build_ohlcv_dict():
    ohlcv_dict = {}
    start = pd.to_datetime("2024-01-01", utc=True)
    end = pd.to_datetime("2025-03-31", utc=True)
    datetime_index = pd.date_range(start=start, end=end,  freq='4h')

    for symbol in symbols:
        if symbol in kraken_altnames:
            symbol = kraken_altnames[symbol]

        try:
            ohlcv_dict[symbol]  = pd.read_csv(
                f"data/kraken_historical_4h/{symbol}USD_240.csv",
                header=None,
                names=col_names,
                converters={"date": lambda x: pd.to_datetime(int(x), unit="s", utc=True)},
                index_col="date"
            )
            ohlcv_dict[symbol] = ohlcv_dict[symbol].reindex(datetime_index)
        except Exception as e:
            print(f"Error fetching {symbol} OHLC data: {e}")


    return ohlcv_dict


df = pd.read_csv(
    f"data/kraken_historical_4h/XBTUSD_240.csv",
    header=None,
    names=col_names,
    converters={"date": lambda x: pd.to_datetime(int(x), unit="s", utc=True)},
    index_col="date"
)
print(df.info())
top_crypto = get_kraken_top_crypto()
print(tabulate(top_crypto, headers="keys", tablefmt="github"))
print(top_crypto.info())
ohlcv_dict = build_ohlcv_dict()
save_ohlcv_dict(ohlcv_dict, "datakraken_historical_4h/ohlcv_dict.pkl")
print(ohlcv_dict.keys())
print(len(ohlcv_dict.keys()))

for key in ohlcv_dict.keys():
    print(f"{key}: {ohlcv_dict[key].shape}")
    print(ohlcv_dict[key].info())
    print(ohlcv_dict[key].describe())