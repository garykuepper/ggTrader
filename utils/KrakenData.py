import json

import ccxt
import os
import pandas as pd
import requests

from tabulate import tabulate

kraken_map = {
    "XETC": "ETC",
    "XETH": "ETH",
    "XLTC": "LTC",
    "XMLN": "MLN",
    "XREP": "REP",
    "XXBT": "XBT",
    "XXDG": "XDG",
    "XXLM": "XLM",
    "XXMR": "XMR",
    "XXRP": "XRP",
    "XZEC": "ZEC",
    "ZAUD": "AUD",
    "ZCAD": "CAD",
    "ZEUR": "EUR",
    "ZGBP": "GBP",
    "ZJPY": "JPY",
    "ZUSD": "USD",
    "XBT": "BTC",
    "XDG": "DOGE"
}

STABLE_BASES = {"USDT", "USDC", "DAI", "USDP", "TUSD", "EUR", "GBP", "AUD", "USDG"}


class KrakenData:
    def __init__(self):
        self.kraken = ccxt.kraken()

    @staticmethod
    def fetch_ohlcv_df(exchange, symbol, timeframe='4h', limit=100, since=None):
        """
        Fetch OHLCV data via ccxt and return as a pandas DataFrame.

        Args:
            exchange: ccxt exchange instance (e.g., ccxt.kraken())
            symbol: string, e.g., 'BTC/USD'
            timeframe: string, e.g., '1h', '15m', '1d'
            limit: maximum number of candles
            since: fetch since timestamp (ms), or None

        Returns:
            Pandas DataFrame with columns: open, high, low, close, volume and datetime index
        """
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv or len(ohlcv[0]) < 6:
            raise ValueError("Returned OHLCV data is empty or unexpected format.")

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["datetime"] = df["datetime"].dt.tz_localize('UTC')
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def get_kraken_usd_ccxt():
        kraken = ccxt.kraken()
        kraken.load_markets()
        all_markets = list(kraken.markets.keys())
        usd_markets = [s for s in all_markets if s.endswith("/USD")]
        return usd_markets

    @staticmethod
    def top_kraken_by_volume(limit=20, only_usd=True, exclude_stables: bool = True, verbose=False):
        kraken = ccxt.kraken()
        kraken.load_markets()

        # Step 1: get candidate symbols
        markets = kraken.markets  # dict: symbol -> market info
        symbols = [s for s in markets.keys() if (s.endswith('/USD') if only_usd else True)]
        # Optional: exclude stablecoins by base symbol
        if exclude_stables:
            filtered = []
            for s in symbols:
                # s is like 'BASE/USD'; take the base part before '/'
                base = s.split('/', 1)[0]
                if base.upper() not in STABLE_BASES:
                    filtered.append(s)
            symbols = filtered

        # Step 2: bulk fetch tickers
        tickers = {}
        try:
            bulk = kraken.fetch_tickers(symbols or list(markets.keys()))
            if isinstance(bulk, dict):
                tickers.update(bulk)
        except Exception as e:
            if verbose:
                print(f"Bulk fetch_tickers failed: {e}. Falling back to per-symbol fetch.")
            for s in symbols:
                try:
                    tickers[s] = kraken.fetch_ticker(s)
                except Exception as ex:
                    if verbose:
                        print(f"Failed to fetch ticker for {s}: {ex}")

        # Step 3: compute a volume metric per symbol
        rows = []
        for s, t in tickers.items():
            # last price
            # print(f"{s}, {t}")
            last = t.get('last') or t.get('close')
            low = t.get('low')
            high = t.get('high')
            open = t.get('open')
            percentage = t.get('percentage')
            # notional volume: prefer Kraken-provided fields if present
            vol_usd = 0.0
            # 3a) try quoteVolume (if available and already in USD terms)
            qv = t.get('quoteVolume')
            if qv is not None:
                vol_usd = float(qv)
            else:
                # 3b) Kraken-specific v/p notations
                v = t.get('v', [0, 0])[1] if isinstance(t.get('v'), (list, tuple)) else 0.0
                p = t.get('p', [0, 0])[1] if isinstance(t.get('p'), (list, tuple)) else 0.0

                if v is not None and p is not None:
                    vol_usd = float(v) * float(p)
                elif last is not None:
                    # fallback: approximate with baseVolume * last
                    base_vol = t.get('baseVolume') or 0.0
                    vol_usd = float(base_vol) * float(last)
            rows.append({
                'symbol': s,
                'volume_usd': vol_usd,
                'percentage': round(percentage, 2) if percentage is not None else None,
                'last': float(last) if last is not None else None,
                'open': open,
                'high': high,
                'low': low,

            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values(by='volume_usd', ascending=False).reset_index(drop=True)
        if limit and limit > 0:
            df = df.head(limit)

        return df

    def print_top_kraken_by_volume(self, limit=20, only_usd=True, exclude_stables: bool = True, verbose=False):
        top_volume = self.top_kraken_by_volume(limit=limit, only_usd=only_usd, exclude_stables=exclude_stables)
        print(tabulate(top_volume, headers="keys", tablefmt="github"))

    @staticmethod
    def get_kraken_historical_csv(symbol: str, interval: str = "4h"):
        current_file = os.path.abspath(__file__)
        one_level_up = os.path.dirname(os.path.dirname(current_file))
        path = os.path.join(one_level_up, "data", f"kraken_hist_{interval}_latest")
        # Ensure the directory exists (no crash if it doesn't yet)
        os.makedirs(path, exist_ok=True)

        filename = f"{symbol}_{interval}.csv"
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath, index_col="date", parse_dates=["date"])
        return df

    @staticmethod
    def get_all_kraken_historical_csv(interval: str = "4h"):
        current_file = os.path.abspath(__file__)
        one_level_up = os.path.dirname(os.path.dirname(current_file))
        path = os.path.join(one_level_up, "data", f"kraken_hist_{interval}_latest")
        # Ensure the directory exists (no crash if it doesn't yet)
        os.makedirs(path, exist_ok=True)
        with os.scandir(path) as it:
            files = [entry.name for entry in it if entry.is_file()]
        num_files = len(files)

        ohlcv_dict = {}
        for i, f in enumerate(files):
            print(f"({i + 1}/{num_files}) Processing {f} ")
            file_path = os.path.join(path, f)
            if f.split(".")[1] != "csv":
                continue
            df = pd.read_csv(file_path, index_col="date", parse_dates=["date"])
            symbol = f.split("_")[0]
            ohlcv_dict[symbol] = df
        return ohlcv_dict

    def get_raw_kraken_csv_data(self, path: str):
        ohlcv_dict = {}
        col_names = ["date", "open", "high", "low", "close", "volume", "trades"]

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
            ticker = f.split("_")[0][:-3]
            if self.special_kraken_map(ticker) is not None:
                ticker = self.special_kraken_map(ticker)

            if ticker:
                ohlcv_dict[ticker] = df
        return ohlcv_dict

    @staticmethod
    def get_kraken_asset_pairs():
        url = "https://api.kraken.com/0/public/AssetPairs"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            raise RuntimeError(f"Kraken AssetPairs error: {payload['error']}")
        return payload['result']

    @staticmethod
    def special_kraken_map(symbol: str):
        return kraken_map.get(symbol)

    @staticmethod
    def join_ohlcv_dict(ohlcv_dict_hist, ohlcv_dict_new, interval="4h"):
        # TODO: decide whether to join raw data or data already in OHLCV format?
        ohlcv_dict = {}
        for ticker in ohlcv_dict_new.keys():
            if ticker in ohlcv_dict_hist.keys():
                if ohlcv_dict_hist[ticker].empty:
                    ohlcv_dict[ticker] = ohlcv_dict_new[ticker]
                else:
                    ohlcv_dict[ticker] = pd.concat([ohlcv_dict_hist[ticker], ohlcv_dict_new[ticker]])
            else:
                ohlcv_dict[ticker] = ohlcv_dict_new[ticker]
            # adjust volume to be in USD
            ohlcv_dict[ticker]['volume'] = ohlcv_dict[ticker]['volume'] * ohlcv_dict[ticker]['close']
        return ohlcv_dict

    @staticmethod
    def write_ohlcv_dict(out, ohlcv_dict, interval="4h"):
        num_files = len(ohlcv_dict.keys())
        for i, ticker in enumerate(ohlcv_dict.keys()):
            filename = ticker + "_" + interval + ".csv"
            print(f"{i + 1}/{num_files} Writing {filename}")
            ohlcv_dict[ticker].to_csv(os.path.join(out, filename))

    @staticmethod
    def write_all_ohlcv_dict(ohlcv_dict, interval="4h"):
        current_file = os.path.abspath(__file__)
        one_level_up = os.path.dirname(os.path.dirname(current_file))
        filename = f"kraken_hist_{interval}.pkl"
        path = os.path.join(one_level_up, "data", "kraken_dict",filename)
        # Ensure the directory exists (no crash if it doesn't yet)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.to_pickle(ohlcv_dict, path)

    @staticmethod
    def get_all_ohlcv_dict(interval="4h"):
        current_file = os.path.abspath(__file__)
        one_level_up = os.path.dirname(os.path.dirname(current_file))
        filename = f"kraken_hist_{interval}.pkl"
        path = os.path.join(one_level_up, "data", "kraken_dict",filename)
        return pd.read_pickle(path)


if __name__ == "__main__":
    kData = KrakenData()
    pairs = kData.get_kraken_usd_ccxt()
    print(f"Found {len(pairs)} USD pairs on Kraken.")

    print(f"\nTop by Volume")
    kData.print_top_kraken_by_volume(limit=20, only_usd=True, exclude_stables=True, verbose=True)

    symbol = 'BNB'
    df = kData.get_kraken_historical_csv(symbol, interval="4h")
    print(f"\nHistorical Data for {symbol}:")
    out = pd.concat([df.head(10), df.tail(10)])
    print(tabulate(out, headers="keys", tablefmt="github"))
    nan_count = df.isna().sum().sum()
    print(f"\nNaN count: {nan_count}")

    hist_data = kData.get_all_kraken_historical_csv(interval="4h")
    kData.write_all_ohlcv_dict(hist_data, interval="4h")
    # print(f"\nAll Historical Data:")
    # print(hist_data.keys())
    pairs = kData.get_kraken_asset_pairs()
    kraken_symbols = list(pairs.keys())
    print(f"\nKraken Asset Pairs ({len(kraken_symbols)}):")

    ## Raw Data
    current_file = os.path.abspath(__file__)
    one_level_up = os.path.dirname(os.path.dirname(current_file))
    path = os.path.join(one_level_up, "data", f"kraken_hist_4h")
    # Ensure the directory exists (no crash if it doesn't yet)
    os.makedirs(path, exist_ok=True)

    woot = kData.get_raw_kraken_csv_data(path)
    print(f"\nKraken Raw Data ({len(woot)}):")
    woot_list = list(woot.keys())
    woot_list.sort()
    for key in woot_list:
        print(key)
