import pandas as pd
import os

from utils.kraken_yfinance_cmc import get_kraken_asset_pairs_usd


def get_common_ticker(pairs: list, name: str):
    for p in pairs:
        if p["altname"] == name:
            return p["base_common"]
    return None


def get_kraken_csv_data(path: str):
    ohlcv_dict = {}
    col_names = ["date", "open", "high", "low", "close", "volume", "trades"]
    pairs = get_kraken_asset_pairs_usd()

    with os.scandir(path) as it:
        files = [entry.name for entry in it if entry.is_file()]
    num_files = len(files)

    for f in files:
        print(f"({files.index(f) + 1}/{num_files}) Processing {f} ")
        file_path = os.path.join(path, f)
        if f.split(".")[1] != "csv":
            continue
        df = pd.read_csv(file_path,
                         header=None,
                         names=col_names,
                         converters={"date": lambda x: pd.to_datetime(int(x), unit="s", utc=True)},
                         index_col="date"
                         )
        ticker = get_common_ticker(pairs, f.split("_")[0])
        if ticker:
            ohlcv_dict[ticker] = df
    return ohlcv_dict


def join_and_write_ohlcv_dict(out_path, ohlcv_dict, ohlcv_dict_new, interval="4h"):
    ohlcv_dict = {}
    for ticker in ohlcv_dict_quarterly.keys():
        if ticker in ohlcv_dict_hist.keys():
            if ohlcv_dict_hist[ticker].empty:
                ohlcv_dict[ticker] = ohlcv_dict_quarterly[ticker]
            else:
                ohlcv_dict[ticker] = pd.concat([ohlcv_dict_hist[ticker], ohlcv_dict_quarterly[ticker]])
        else:
            ohlcv_dict[ticker] = ohlcv_dict_quarterly[ticker]
        # adjust volume to be in USD
        ohlcv_dict[ticker]['volume'] = ohlcv_dict[ticker]['volume'] * ohlcv_dict[ticker]['close']
    num_files = len(ohlcv_dict.keys())

    for i, ticker in enumerate(ohlcv_dict.keys()):
        filename = ticker + "_" + interval + ".csv"
        print(f"{i+1}/{num_files} Writing {filename}")
        ohlcv_dict[ticker].to_csv(os.path.join(out, filename))


path = "data/kraken_hist_1d/"
path_quarterly = "data/kraken_hist_2025_q2_1d/"
out = "data/kraken_hist_1d_latest/"
interval = "1d"

ohlcv_dict_quarterly = get_kraken_csv_data(path_quarterly)
ohlcv_dict_hist = get_kraken_csv_data(path)

join_and_write_ohlcv_dict(out, ohlcv_dict_quarterly, ohlcv_dict_hist, interval)
