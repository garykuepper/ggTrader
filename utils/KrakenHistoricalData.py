import os
import random
from tqdm.auto import tqdm
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tabulate import tabulate

kraken_map = {
    "XETC": "ETC", "XETH": "ETH", "XLTC": "LTC", "XMLN": "MLN", "XREP": "REP",
    "XXBT": "XBT", "XXDG": "XDG", "XXLM": "XLM", "XXMR": "XMR", "XXRP": "XRP",
    "XZEC": "ZEC", "ZAUD": "AUD", "ZCAD": "CAD", "ZEUR": "EUR", "ZGBP": "GBP",
    "ZJPY": "JPY", "ZUSD": "USD", "XBT": "BTC", "XDG": "DOGE"
}

STABLE_BASES = {"USDT", "USDC", "DAI", "USDP", "EUR", "GBP", "AUD", "USD", "JPY", "CAD"}


def clean_ccy(ccy: str) -> str:
    # Map Kraken prefixes like XXBT -> XBT -> BTC, and ZUSD -> USD
    return kraken_map.get(ccy, ccy)


def split_pair(raw_pair: str, quote_only="USD"):
    """
    Split Kraken pair (e.g., 'XXBTZUSD', 'XETHZUSD', 'BTCUSDT') into base/quote robustly.
    Strategy:
      1) Try 4-letter quote (USDT/USDC/DAI/…)
      2) Else try 3-letter quote
      3) Fallback: last 3 chars
    Then clean both sides via kraken_map.
    """
    p = raw_pair.upper()

    if p.endswith(quote_only):
        base = clean_ccy(p[:-3])
        quote = quote_only
        pair_std = f"{base}-{quote}"
        return clean_ccy(p[:-3]), clean_ccy(quote_only), pair_std

    return None, None, None


class KrakenHistoricalData:
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.raw_path = os.path.join(self.root_dir, 'data', 'raw')
        self.parquet_root = os.path.join(self.root_dir, 'data', 'parquet')  # dataset dir
        os.makedirs(self.parquet_root, exist_ok=True)

    # ---------- directory helpers ----------
    def list_quarter_dirs(self, prefix="Kraken_OHLCVT_"):
        """Find quarterly Kraken folders under data/raw."""
        out = []
        if not os.path.isdir(self.raw_path):
            return out
        for name in os.listdir(self.raw_path):
            full = os.path.join(self.raw_path, name)
            if os.path.isdir(full) and name.startswith(prefix):
                out.append(full)
        out.sort()
        return out

    def get_file_names(self, path, quote_only="USD"):
        if not os.path.isdir(path):
            return []
        files = [f for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".csv")]
        return self.filter_files_by_quote(files, quote_only)

    def filter_files_by_quote(self, files, quote_only="USD"):
        """Keep files where the pair part ends with quote and has non-empty base."""
        kept = []
        for f in files:
            stem = f.rsplit(".", 1)[0]
            pair = stem.split("_", 1)[0].upper()
            if pair.endswith(quote_only) and len(pair) > len(quote_only):
                kept.append(f)
        return kept

    # ---------- IO ----------
    def load_csv(self, dir_, file_):
        col_names = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
        name_noext = file_.rsplit(".", 1)[0]
        parts = name_noext.split("_")
        if len(parts) < 2:
            return None

        raw_pair, interval = parts[0], parts[1]
        base, quote, pair_std = split_pair(raw_pair, quote_only="USD")
        if base is None:  # skip non-USD or malformed
            return None

        # normalize interval (example: Kraken numeric minutes -> human)
        interval_map = {"1": "1m", "5": "5m", "15": "15m", "30": "30m", "60": "1h", "240": "4h",
                        "1440": "1d", "10080": "1w"}
        interval = interval_map.get(interval.lower(), interval)

        file_path = os.path.join(dir_, file_)
        try:
            df = pd.read_csv(
                file_path,
                header=None,
                names=col_names,
                converters={"timestamp": lambda x: pd.to_datetime(int(x), unit="s", utc=True)},
                index_col='timestamp'  # KEEP INDEX
            )
        except pd.errors.EmptyDataError:
            return None

        if df.empty:
            return None

        # dtypes & cleaning
        df["open"] = pd.to_numeric(df["open"], errors="coerce").astype("float32")
        df["high"] = pd.to_numeric(df["high"], errors="coerce").astype("float32")
        df["low"] = pd.to_numeric(df["low"], errors="coerce").astype("float32")
        df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

        # dedup by index & sort
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # metadata columns
        df["interval"] = interval
        df["base"] = base
        df["quote"] = quote
        df["pair"] = pair_std
        return df

    def csvs_dir_to_parquet(self, input_dir, sample=None, position=1):
        files = self.get_file_names(input_dir, quote_only="USD")
        if not files:
            tqdm.write(f"[skip] no USD files in {input_dir}")  # <- tqdm-safe
            return

        if sample:
            files = random.sample(files, k=min(sample, len(files)))

        # tqdm-safe message
        tqdm.write(f"[dir] {input_dir} -> {len(files)} files")

        # single-folder progress bar
        for f in tqdm(
                files,
                desc=f"Processing {os.path.basename(input_dir)}",
                ncols=0,  # let tqdm auto-size (prevents wrapping)
                dynamic_ncols=True,
                position=position,  # <- important for nested bars
                leave=False,  # <- collapse when done (outer bar stays)
                miniters=1,
                smoothing=0.1,
        ):
            df = self.load_csv(input_dir, f)
            if df is None or df.empty:
                # if you want to see skips:
                # tqdm.write(f"[skip] {f}")
                continue

            table = pa.Table.from_pandas(df, preserve_index=True)
            pq.write_to_dataset(
                table,
                root_path=self.parquet_root,
                partition_cols=["pair", "interval"],
                compression="zstd",
            )

    def csvs_many_dirs_to_parquet(self, dirs=None, sample_per_dir=None):
        if dirs is None:
            dirs = self.list_quarter_dirs()

        # outer bar for the list of quarterly folders
        for d in tqdm(
                dirs,
                desc="Quarterly Folders",
                ncols=0,
                dynamic_ncols=True,
                position=0,  # outer bar on line 0
                leave=True,
        ):
            # inner bar uses position=1 so it always reuses the same line
            self.csvs_dir_to_parquet(d, sample=sample_per_dir, position=1)

    # ---------- readers ----------
    def read_parquet(self, pair=None, interval=None, columns=None, filters=None, sort=True):
        base_path = self.parquet_root
        if pair and interval:
            path = os.path.join(base_path, f"pair={pair}", f"interval={interval}")
        elif pair:
            path = os.path.join(base_path, f"pair={pair}")
        else:
            path = base_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"No parquet data found at {path}")

        df = pd.read_parquet(path, columns=columns, filters=filters)
        # If index was preserved, pandas restores it automatically
        if sort and "timestamp" in df.columns:  # in case index wasn't restored
            df = df.sort_values("timestamp")
        return df


if __name__ == "__main__":
    k = KrakenHistoricalData()

    # 1) Process ALL quarterly folders (USD pairs only), keeping timestamp as index
    quarter_dirs = k.list_quarter_dirs()  # e.g. Kraken_OHLCVT_Q1_2023, ...
    k.csvs_many_dirs_to_parquet(quarter_dirs)  # or sample_per_dir=200

    # 2) Read back one slice
    df_btc_1h = k.read_parquet(pair="BTC-USD", interval="1h")
    print(df_btc_1h.head())
