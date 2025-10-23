import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

kraken_map = {
    "XETC": "ETC", "XETH": "ETH", "XLTC": "LTC", "XMLN": "MLN", "XREP": "REP",
    "XXBT": "XBT", "XXDG": "XDG", "XXLM": "XLM", "XXMR": "XMR", "XXRP": "XRP",
    "XZEC": "ZEC", "ZAUD": "AUD", "ZCAD": "CAD", "ZEUR": "EUR", "ZGBP": "GBP",
    "ZJPY": "JPY", "ZUSD": "USD", "XBT": "BTC", "XDG": "DOGE"
}

STABLE_BASES = {"USDT", "USDC", "DAI", "USDP",  "EUR", "GBP", "AUD", "USD", "JPY", "CAD"}


def clean_ccy(ccy: str) -> str:
    # Map Kraken prefixes like XXBT -> XBT -> BTC, and ZUSD -> USD
    return kraken_map.get(ccy, ccy)


def split_pair(raw_pair: str):
    """
    Split Kraken pair (e.g., 'XXBTZUSD', 'XETHZUSD', 'BTCUSDT') into base/quote robustly.
    Strategy:
      1) Try 4-letter quote (USDT/USDC/DAI/…)
      2) Else try 3-letter quote
      3) Fallback: last 3 chars
    Then clean both sides via kraken_map.
    """
    p = raw_pair.upper()

    # Try to find a known quote suffix (4 first)
    for qlen in (4, 3):
        q = p[-qlen:]
        if q in STABLE_BASES:
            base = p[:-qlen]
            quote = q
            break
    else:
        base, quote = p[:-3], p[-3:]

    base = clean_ccy(base)
    quote = clean_ccy(quote)

    # Normalize special codes (XBT -> BTC)
    base = "BTC" if base == "XBT" else base
    quote = "BTC" if quote == "XBT" else quote

    pair_std = f"{base}-{quote}"
    return base, quote, pair_std


class KrakenHistoricalData:
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.raw_path = os.path.join(self.root_dir, 'data', 'raw')
        self.paths = [self.raw_path]
        self.parquet_root = os.path.join(self.root_dir, 'data', 'parquet')  # dataset dir
        os.makedirs(self.parquet_root, exist_ok=True)

    def get_folder_paths(self):
        valid_dirs = [p for p in self.paths if os.path.isdir(p)]
        subdirs = []
        for p in valid_dirs:
            for name in os.listdir(p):
                full = os.path.join(p, name)
                if os.path.isdir(full):
                    subdirs.append(full)
        return subdirs

    def get_file_names(self, path, quote_only="USD"):
        """
        Return a list of CSV file names in the given directory,
        optionally filtered by quote currency.
        """
        if not os.path.isdir(path):
            return []

        all_files = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".csv")
        ]

        if quote_only:
            all_files = self.filter_files_by_quote(all_files, quote_only)

        return all_files

    def filter_files_by_quote(self, files, quote_only="USD"):
        """
        Filter file names by quote currency (e.g. keep only USD-quoted pairs).

        Parameters
        ----------
        files : list[str]
            List of file names like ['XXBTZUSD_1m.csv', 'ETHUSDT_1h.csv']
        quote_only : str
            e.g. "USD" to only include those quoted in USD

        Returns
        -------
        list[str] : filtered file names
        """
        filtered = []
        for f in files:
            name = f.split(".")[0]  # remove .csv
            parts = name.split("_")
            if len(parts) < 2:
                continue

            pair = parts[0].upper()  # e.g. XXBTZUSD or BTCUSDT

            if pair.endswith(quote_only):
                filtered.append(f)

        return filtered

    def load_csv(self, dir_, file_):
        col_names = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
        p = file_.split(".")
        if len(p) < 2 or p[1].lower() != "csv":
            return None

        pair_interval = p[0].split("_")
        if len(pair_interval) < 2:
            return None

        raw_pair, interval = pair_interval[0], pair_interval[1]
        base, quote, pair_std = split_pair(raw_pair)

        file_path = os.path.join(dir_, file_)
        try:
            df = pd.read_csv(
                file_path,
                header=None,
                names=col_names,
                converters={"timestamp": lambda x: pd.to_datetime(int(x), unit="s", utc=True)},
                index_col='timestamp'
            )

            # Basic cleaning / typing
            df["open"] = df["open"].astype("float32")
            df["high"] = df["high"].astype("float32")
            df["low"] = df["low"].astype("float32")
            df["close"] = df["close"].astype("float32")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

            # Add metadata columns
            df["interval"] = interval
            df["base"] = base
            df["quote"] = quote
            df["pair"] = pair_std

            return df

        except pd.errors.EmptyDataError:
            print(f"{p[0]} CSV is empty.")
            return None

    def csvs_to_parquet_dataset(self, input_dir):
        files = self.get_file_names(input_dir)
        n = len(files)
        print(f"Processing {n} files from: {input_dir}")

        for i, f in enumerate(files, 1):
            df = self.load_csv(input_dir, f)
            if df is None or df.empty:
                print(f"Skipped {i}/{n}: {f}")
                continue

            # Write incrementally to a hive-partitioned Parquet dataset
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=self.parquet_root,
                partition_cols=["pair", "interval"],  # directories: pair=BTC-USD/interval=1m/...
                compression="zstd",
            )
            if i % 50 == 0 or i == n:
                print(f"Wrote {i}/{n}: {f}")

    def read_parquet(self, pair=None, interval=None, columns=None, filters=None):
        """
        Read Parquet data selectively.

        Parameters
        ----------
        pair : str, optional
            e.g. "BTC-USD"
        interval : str, optional
            e.g. "1m", "1h", "1d"
        columns : list[str], optional
            Columns to read (saves memory)
        filters : list[tuple], optional
            pyarrow-style filters, e.g. [("pair", "=", "BTC-USD")]

        Returns
        -------
        pandas.DataFrame
        """
        base_path = self.parquet_root

        # Narrow the path if pair and/or interval specified
        if pair and interval:
            path = os.path.join(base_path, f"pair={pair}", f"interval={interval}")
        elif pair:
            path = os.path.join(base_path, f"pair={pair}")
        else:
            path = base_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"No parquet data found at {path}")

        # Load the data
        df = pd.read_parquet(path, columns=columns, filters=filters)

        return df


if __name__ == "__main__":
    kData = KrakenHistoricalData()
    dirs = kData.get_folder_paths()
    if not dirs:
        raise SystemExit("No subdirectories found under data/raw")
    src = dirs[0]
    files = kData.get_file_names(src)
    df = kData.load_csv(src, files[0])
    print(df.head())
    print(f"Reading from: {src}")
    kData.csvs_to_parquet_dataset(src)
    print(f"Done. Parquet dataset at: {kData.parquet_root}")

    # read everything
    df_all = kData.read_parquet()
    print(df_all.head())

    # read just BTC-USD 1m candles
    # df_btc = kData.read_parquet(pair="BTC-USD", interval="1m")

    # read only the timestamp and close columns
    # df_subset = kData.read_parquet(pair="BTC-USD", interval="1m", columns=["timestamp", "close"])
