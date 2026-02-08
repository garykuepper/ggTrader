import os
import random
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from .constants import interval_map
from .utils import split_pair, _load_csv_common, get_file_names

def _process_one_file(args):
    """Top-level worker so it can be pickled on Windows."""
    input_dir, file_, parquet_root = args
    # Import minimal bits inside worker
    import os, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

    col_names = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    name_noext = file_.rsplit(".", 1)[0]
    parts = name_noext.split("_")
    if len(parts) < 2:
        return (file_, 0)

    raw_pair, interval = parts[0], parts[1]
    base, quote, pair_std = split_pair(raw_pair, quote_only="USD")
    if base is None:
        return (file_, 0)

    # Use interval_map from KrakenConstants (available via import in outer scope)
    # But for pickling safety on some systems, maybe re-import or pass in?
    # Actually, KrakenConstants.interval_map should be available if imported at module level.
    # However, _load_csv_common already handles interval mapping.
    
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
    
    # write directly from the worker (each creates its own part file)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_to_dataset(
        table,
        root_path=parquet_root,
        partition_cols=["pair", "interval"],
        compression="zstd",
    )
    return (file_, 1)

class KrakenConverter:
    def __init__(self, parquet_root):
        self.parquet_root = parquet_root

    def csvs_dir_to_parquet(self, input_dir, sample=None, position=1):
        """Sequential conversion from CSV to Parquet."""
        files = get_file_names(input_dir, quote_only="USD")
        if not files:
            tqdm.write(f"[skip] no USD files in {input_dir}")
            return

        if sample:
            files = random.sample(files, k=min(sample, len(files)))

        tqdm.write(f"[dir] {input_dir} -> {len(files)} files")

        for f in tqdm(
                files,
                desc=f"Processing {os.path.basename(input_dir)}",
                dynamic_ncols=True,
                position=position,
                leave=False,
        ):
            df = self._load_csv_into_df(input_dir, f)
            if df is None or df.empty:
                continue

            table = pa.Table.from_pandas(df, preserve_index=True)
            pq.write_to_dataset(
                table,
                root_path=self.parquet_root,
                partition_cols=["pair", "interval"],
                compression="zstd",
            )

    def _load_csv_into_df(self, dir_, file_):
        """Helper for sequential loading."""
        col_names = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
        name_noext = file_.rsplit(".", 1)[0]
        parts = name_noext.split("_")
        if len(parts) < 2:
            return None

        raw_pair, interval = parts[0], parts[1]
        base, quote, pair_std = split_pair(raw_pair, quote_only="USD")
        if base is None:
            return None

        file_path = os.path.join(dir_, file_)
        df, interval = _load_csv_common(file_path, col_names, interval)
        if df is None or df.empty:
            return None

        # dtypes & cleaning
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
        return df

    def csvs_dir_to_parquet_parallel(self, input_dir, sample=None, max_workers=None):
        """Parallel conversion from CSV to Parquet."""
        files = get_file_names(input_dir, quote_only="USD")
        if not files:
            tqdm.write(f"[skip] no USD files in {input_dir}")
            return

        if sample:
            files = np.random.choice(files, size=sample, replace=False)

        tqdm.write(f"[dir] {input_dir} -> {len(files)} files (parallel)")

        tasks = [(input_dir, f, self.parquet_root) for f in files]
        done = 0
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
