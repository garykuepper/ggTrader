import pandas as pd
import os
import json
import requests  # new: used for optional local fetch of Kraken AssetPairs
from utils.KrakenData import KrakenData
# --- Removed external dependency import ---
# from utils.kraken_yfinance_cmc import get_kraken_asset_pairs_usd


def get_common_ticker(pairs: list, name: str):
    for p in pairs:
        if p["altname"] == name:
            return p["base_common"]
    return None

# New local helper: fetch Kraken USD asset pairs without relying on kraken_yfinance_cmc
def _fetch_kraken_asset_pairs_usd_local():
    # Try Kraken API first (public endpoint)
    try:
        url = "https://api.kraken.com/0/public/AssetPairs"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload and payload["error"]:
            # Fall back to empty list if Kraken returns errors
            return []
        # Normalize payload to a list of dicts similar to what get_kraken_asset_pairs_usd returns
        # Kraken API returns a dict keyed by pair; we convert to a list with fields we used previously
        assets = []
        for alt, val in payload.items():
            base = val.get("base")  # e.g., "BTC"
            quote = val.get("quote")  # e.g., "USD"
            if base and quote == "USD":
                assets.append({"altname": alt, "base_common": base})
        return assets
    except Exception:
        # On any failure, return an empty list to avoid crashing pre-processing
        return []

def _load_asset_pairs():
    # Try to fetch via local fetch; otherwise, fall back to an empty list.
    pairs = _fetch_kraken_asset_pairs_usd_local()
    if not pairs:
        # Optional: load from a small cache file if you maintain one
        cache_path = os.path.join(os.path.dirname(__file__), "kraken_asset_pairs_usd_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    pairs = json.load(f)
            except Exception:
                pairs = []
    return pairs

def get_raw_kraken_csv_data(path: str):
    ohlcv_dict = {}
    col_names = ["date", "open", "high", "low", "close", "volume", "trades"]

    # Use local loader instead of external helper
    pairs = _load_asset_pairs()
    # If still empty, we’ll proceed with an empty mapping (no ticker mapping will be found)

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


def join_and_write_ohlcv_dict(out_path, ohlcv_dict_hist, ohlcv_dict_quarterly, interval="4h"):
    ohlcv_dict = {}
    # Note: original function referenced undefined names ohlcv_dict_quarterly and ohlcv_dict_hist.
    # If these globals exist in your actual code, keep their usage; otherwise adapt accordingly.
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

interval = "1d"
path = f"data/kraken_hist_{interval}/"
path_quarterly = f"data/kraken_hist_2025_q2_{interval}/"
out = f"data/kraken_hist_{interval}_latest/"

kData = KrakenData()



# ohlcv_dict_quarterly = get_raw_kraken_csv_data(path_quarterly)
# ohlcv_dict_hist = get_raw_kraken_csv_data(path)

pairs = _fetch_kraken_asset_pairs_usd_local()
print(pairs)

# join_and_write_ohlcv_dict(out, ohlcv_dict_quarterly, ohlcv_dict_hist, interval)
