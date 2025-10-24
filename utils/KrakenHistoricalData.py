import os
import random
from tqdm.auto import tqdm
import sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tabulate import tabulate
import shutil

kraken_map = {
    "XETC": "ETC", "XETH": "ETH", "XLTC": "LTC", "XMLN": "MLN", "XREP": "REP",
    "XXBT": "XBT", "XXDG": "XDG", "XXLM": "XLM", "XXMR": "XMR", "XXRP": "XRP",
    "XZEC": "ZEC", "ZAUD": "AUD", "ZCAD": "CAD", "ZEUR": "EUR", "ZGBP": "GBP",
    "ZJPY": "JPY", "ZUSD": "USD", "XBT": "BTC", "XDG": "DOGE"
}

STABLE_BASES = ["USDT", "USDC", "DAI", "USDP", "EUR", "GBP", "AUD", "USD", "JPY", "CAD","MKR"]

interval_map = {"1": "1m", "5": "5m", "15": "15m",
                "30": "30m", "60": "1h", "240": "4h",
                "720": "12h", "1440": "1d", "10080": "1w"}
# --- at top level (same module) ---
from concurrent.futures import ProcessPoolExecutor, as_completed


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


# Shared CSV loading helper (new)
def _load_csv_common(file_path: str, col_names: list, interval: str = None):
    """
    Shared CSV loader used by both _process_one_file and KrakenHistoricalData.load_csv.
    Keeps the same parsing logic (header=None, names=col_names, converters for timestamp,
    index_col='timestamp'). Returns (df, interval) tuple or (None, None) on skip/empty.
    """
    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=col_names,
            converters={"timestamp": lambda x: pd.to_datetime(int(x), unit="s", utc=True)},
            index_col="timestamp",
        )
    except pd.errors.EmptyDataError:
        return None, None
    if df is None or df.empty:
        return None, None
    # Normalize volume to USD by multiplying by the close price
    if "volume" in df.columns and "close" in df.columns:
        df["volume"] = df["volume"] * df["close"]
    # normalize/standardize interval if provided
    if interval is not None:
        interval_map = {"1": "1m", "5": "5m", "15": "15m", "30": "30m", "60": "1h", "240": "4h",
                        "1440": "1d", "10080": "1w"}
    interval = interval_map.get(interval.lower(), interval)
    return df, interval


def _process_one_file(args):
    """Top-level worker so it can be pickled on Windows."""
    input_dir, file_, parquet_root = args
    # Re-import minimal bits to avoid heavy pickling (optional micro-opt)
    import os, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

    # ---- replicate the essentials from your class methods ----
    # You can also refactor your load_csv to @staticmethod and import it here.
    col_names = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    name_noext = file_.rsplit(".", 1)[0]
    parts = name_noext.split("_")
    if len(parts) < 2:
        return (file_, 0)

    raw_pair, interval = parts[0], parts[1]
    # use your split_pair from module scope
    base, quote, pair_std = split_pair(raw_pair, quote_only="USD")
    if base is None:
        return (file_, 0)

    interval = interval_map.get(interval.lower(), interval)

    file_path = os.path.join(input_dir, file_)
    df, interval = _load_csv_common(file_path, col_names, interval)
    if df is None:
        return (file_, 0)
    if df.empty:
        return (file_, 0)

    # dtypes & clean
    df["open"] = pd.to_numeric(df["open"], errors="coerce").astype("float32")
    df["high"] = pd.to_numeric(df["high"], errors="coerce").astype("float32")
    df["low"] = pd.to_numeric(df["low"], errors="coerce").astype("float32")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["interval"] = interval
    df["base"] = base
    df["quote"] = quote
    df["pair"] = pair_std
    # adjust volume to be in usd
    df['volume'] = df['volume'] * df['close']
    # write directly from the worker (each creates its own part file)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_to_dataset(
        table,
        root_path=parquet_root,
        partition_cols=["pair", "interval"],
        compression="zstd",
        # use the default file_visitor to generate unique part names
    )
    return (file_, 1)


class KrakenHistoricalData:
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.raw_path = os.path.join(self.root_dir, 'data', 'raw')
        self.parquet_root = os.path.join(self.root_dir, 'data', 'parquet')  # dataset dir
        self.historical_mover_path = os.path.join(self.root_dir, 'data', 'historical_movers')
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

        # adjust volume to be in usd
        df['volume'] = df['volume'] * df['close']
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

        return df.sort_index()

    def csvs_dir_to_parquet_parallel(self, input_dir, sample=None, max_workers=None):
        files = self.get_file_names(input_dir, quote_only="USD")
        if not files:
            tqdm.write(f"[skip] no USD files in {input_dir}")
            return

        if sample:
            files = np.random.choice(files, size=sample, replace=False)
            # files = random.sample(files, k=min(sample, len(files)))

        tqdm.write(f"[dir] {input_dir} -> {len(files)} files (parallel)")

        tasks = [(input_dir, f, self.parquet_root) for f in files]
        done = 0
        # default: one process per CPU
        max_workers = max_workers or os.cpu_count() or 4

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for _file, ok in tqdm(
                    ex.map(_process_one_file, tasks, chunksize=8),
                    total=len(tasks),
                    desc=f"Processing {os.path.basename(input_dir)} (mp x{max_workers})",
                    dynamic_ncols=True,
                    leave=False,
            ):
                done += ok

        tqdm.write(f"[done] {os.path.basename(input_dir)}: wrote {done}/{len(files)} files")

    def csvs_many_dirs_to_parquet_parallel(self, dirs=None, sample_per_dir=None, max_workers=None):
        if dirs is None:
            dirs = self.list_quarter_dirs()
        for d in tqdm(dirs, desc="Quarterly Folders", dynamic_ncols=True, position=0, leave=True):
            self.csvs_dir_to_parquet_parallel(d, sample=sample_per_dir, max_workers=max_workers)

    def get_ohlcv_df(self, symbols: list, interval="1d", quote="USD"):
        dfs = []
        for symbol in symbols:
            pair = f"{symbol}-{quote}"
            df = k.read_parquet(pair=pair, interval=interval)
            dfs.append(df)

        # deduplicate indices
        dfs = [d[~d.index.duplicated(keep='last')] for d in dfs]

        # build a common index (union of all per-symbol indices)
        common_idx = dfs[0].index
        for d in dfs[1:]:
            common_idx = common_idx.union(d.index)

        # 3) reindex all dfs to the common index
        dfs = [d.reindex(common_idx) for d in dfs]

        # join into a single dataframe
        ohlcv_df = pd.concat(dfs, axis=1, keys=symbols)

        # clean up
        ohlcv_df = ohlcv_df.sort_index()
        ohlcv_df = self.align_to_datetime_index(ohlcv_df)
        ohlcv_df = self.fill_after_first_non_nan_multilevel_safe(ohlcv_df, symbols=symbols)
        ohlcv_df = self.fill_symbol_metadata(ohlcv_df, symbols)

        return ohlcv_df

    def list_parquet_pairs(self, ) -> list[str]:
        pairs = set()
        if not os.path.isdir(self.parquet_root):
            return []
        for name in os.listdir(self.parquet_root):
            full = os.path.join(self.parquet_root, name)
            if not os.path.isdir(full):
                continue
            # Expect folder name like pair=BTC-USD
            if name.startswith("pair="):
                pair = name.split("=", 1)[-1]
                pairs.add(pair)
            else:
                # If there are deeper structures, peek inside
                for sub in os.listdir(full):
                    subfull = os.path.join(full, sub)
                    if os.path.isdir(subfull) and sub.startswith("interval="):
                        # Try to derive pair from the parent folder
                        parent = os.path.basename(full)
                        if parent.startswith("pair="):
                            pairs.add(parent.split("=", 1)[-1])
                        break
        return sorted(pairs)

    # ohlcv_df: multi-index columns: (symbol, 'base'), (symbol, 'quote'), etc.
    @staticmethod
    def fill_symbol_metadata(ohlcv_df, symbols):
        df_out = ohlcv_df.copy()
        for sym in symbols:
            # base column for this symbol
            base_col = (sym, 'base')
            quote_col = (sym, 'quote')

            if base_col in df_out.columns:
                df_out[base_col] = df_out[base_col].ffill().bfill()
            if quote_col in df_out.columns:
                df_out[quote_col] = df_out[quote_col].ffill().bfill()
        return df_out

    @staticmethod
    def fill_after_first_non_nan_single(df: pd.DataFrame) -> pd.DataFrame:
        """
        For a flat (single-level columns) DataFrame:
        - For each numeric column, fill NaNs that occur after the first non-NaN value
          using forward-fill.
        - Leading NaNs (before the first non-NaN) are preserved.
        Returns a new DataFrame (original is not mutated).
        """
        df_out = df.copy()
        for col in df_out.columns:
            # operate only on numeric dtype columns
            if pd.api.types.is_numeric_dtype(df_out[col]):
                first_valid = df_out[col].first_valid_index()
                if first_valid is not None:
                    # forward-fill from the first valid index onward
                    tail = df_out[col].loc[first_valid:]
                    tail_filled = tail.ffill()
                    df_out.loc[first_valid:, col] = tail_filled
        return df_out

    @staticmethod
    def align_to_datetime_index(ohlcv_df: pd.DataFrame):
        first_date = ohlcv_df.index[0]
        last_date = ohlcv_df.index[-1]
        date_range = pd.date_range(start=first_date, end=last_date, freq=interval)
        ohlcv_df = ohlcv_df.reindex(date_range)
        return ohlcv_df

    @staticmethod
    def fill_after_first_non_nan_multilevel_safe(ohlcv_df: pd.DataFrame, symbols: list) -> pd.DataFrame:
        """
        For a DataFrame with multilevel columns where the top level is a symbol
        (e.g., (BTC, open), (BTC, high), ..., (ETH, open), ...):
        - For each symbol, apply the per-column fill-after-first-non-nan logic
          to that symbol's sub-DataFrame.
        - Update the original DataFrame column-by-column to avoid
          MultiIndex assignment issues.
        - Returns a new DataFrame; original is not mutated.
        """
        df_out = ohlcv_df.copy()
        for sym in symbols:
            # Work on the per-symbol subframe
            df_sym = ohlcv_df.xs(sym, axis=1, level=0).copy()

            # Fill each numeric column after the first non-NaN
            for col in df_sym.columns:
                if pd.api.types.is_numeric_dtype(df_sym[col]):
                    first_valid = df_sym[col].first_valid_index()
                    if first_valid is not None:
                        tail = df_sym[col].loc[first_valid:]
                        tail_filled = tail.ffill()
                        df_sym.loc[first_valid:, col] = tail_filled

            # Write back column-by-column to avoid slice assignment issues
            for col in df_sym.columns:
                df_out[(sym, col)] = df_sym[col]

        return df_out

    def get_random_symbols(self, n=10):
        symbols = self.list_parquet_symbols()
        return np.random.choice(symbols, size=n, replace=False)

    def list_parquet_symbols(self, quote='USD'):
        pairs = self.list_parquet_pairs()
        symbols = [p[:-4] if p.endswith(quote) else p for p in pairs]
        return symbols

    def get_daily_historical_movers(self, top_n=20,trades_threshold=500, sample=None, stables=False):
        interval = "1d"

        if sample:
            symbols = np.random.choice(self.list_parquet_symbols(), size=sample, replace=False)
        else:
            symbols = self.list_parquet_symbols()
        if not stables:
            symbols = self.filter_out_stables(symbols)

        # symbols = ["AVAX", "AAVE","AUDIO"]
        ohlcv_df = self.get_ohlcv_df(symbols=symbols, interval=interval, quote="USD")

        # go long!
        ohlcv_df_long = ohlcv_df.stack(level=0, future_stack=True).reset_index().rename(
            columns={'level_0': 'date', 'level_1': 'symbol'})
        # filter out low trades
        # print(ohlcv_df_long.describe())
        ohlcv_df_long = ohlcv_df_long[ohlcv_df_long['trades'] > trades_threshold]
        top_by_volume = ohlcv_df_long.sort_values(['date', 'volume'], ascending=[True, False])
        top_per_day = top_by_volume.groupby('date').head(top_n).reset_index(drop=True)
        return top_per_day

    @staticmethod
    def filter_out_stables(symbols: list):
        to_remove = set(STABLE_BASES)
        filtered = [x for x in symbols if x not in to_remove]
        return filtered

    def save_historical_movers_to_parquet(self):
        top_per_day = self.get_daily_historical_movers(top_n=100, sample=None, stables=False)
        top_per_day.to_parquet(os.path.join(self.historical_mover_path, "historical_movers.parquet"))

    def load_historical_movers_from_parquet(self):
        return pd.read_parquet(os.path.join(self.historical_mover_path, "historical_movers.parquet"))

    def get_historical_movers_by_day(self, date: pd.Timestamp):
        date = self.ensure_utc_timestamp(date)
        top_per_day = self.load_historical_movers_from_parquet()
        top = top_per_day[top_per_day.date == date]
        return top.reset_index(drop=True)

    @staticmethod
    def ensure_utc_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tz is None:
            return ts.tz_localize("UTC")
        else:
            return ts.tz_convert("UTC")

if __name__ == "__main__":
    # k = KrakenHistoricalData()
    #
    # # 1) Process ALL quarterly folders (USD pairs only), keeping timestamp as index
    # quarter_dirs = k.list_quarter_dirs()  # e.g. Kraken_OHLCVT_Q1_2023, ...
    # k.csvs_many_dirs_to_parquet(quarter_dirs)  # or sample_per_dir=200
    #
    # # 2) Read back one slice
    # df_btc_1h = k.read_parquet(pair="BTC-USD", interval="1h")
    # print(df_btc_1h.head())

    # Important on Windows for multiprocessing to work reliably
    import multiprocessing as mp

    mp.freeze_support()

    k = KrakenHistoricalData()
    quarter_dirs = k.list_quarter_dirs()

    # Per-folder parallelism (good balance), sampling optional
    # k.csvs_many_dirs_to_parquet_parallel(quarter_dirs,
    #                                      max_workers=os.cpu_count(),
    #                                      sample_per_dir=None)

    # one folder only
    # k.csvs_dir_to_parquet_parallel(quarter_dirs[0], max_workers=8)

    # # symbols = ["BTC", "ETH", "BNB", "PEPE", "DOGE"]
    symbols = k.get_random_symbols(n=3)
    # symbols = ["AVAX", "AAVE"]
    print(f"Symbols: {symbols}")
    quote = "USD"
    interval = "1d"
    ohlcv_df = k.get_ohlcv_df(symbols, interval=interval, quote=quote)

    # print(f"\nStart of OHLCV Data")
    # print(ohlcv_df.head())
    # print(f"\nEnd of OHLCV Data")
    # print(ohlcv_df.tail())
    #
    # # date range check
    first_date = ohlcv_df.index[0]
    last_date = ohlcv_df.index[-1]
    date_range = pd.date_range(start=first_date, end=last_date, freq=interval)

    print(f"\nDate Range: {date_range[0]} --> {date_range[-1]}")
    print(f"Date Range Length: {len(date_range)}")
    print(f"Ohlcv Dataframe Length: {ohlcv_df.shape[0]}")
    #
    print("\n", ohlcv_df.info())

    # Get daily historical movers and save to parquet
    # k.save_historical_movers_to_parquet()
    historical_movers = k.load_historical_movers_from_parquet()
    # print(historical_movers.head(20))

    random_date = np.random.choice(date_range)
    date = pd.Timestamp("2025-06-26")
    date = random_date
    print("\n", f"Historical Movers for {date}")
    historical_movers_by_day = k.get_historical_movers_by_day(date)
    print(historical_movers_by_day.head(20))
