import os
import fsspec
import pandas as pd
import pyarrow.dataset as ds
from .utils import (
    align_to_datetime_index,
    fill_after_first_non_nan_multilevel_safe,
    fill_symbol_metadata
)

class KrakenRemoteReader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.remote_base_url = None
        self.remote_fs = None

    def use_remote(self, base_url: str, username: str = None, password: str = None, headers: dict = None):
        """Configure remote filesystem access."""
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self.remote_base_url = base_url

        target_opts = {}
        if username and password:
            target_opts["client_kwargs"] = {"auth": (username, password)}
        if headers:
            ck = target_opts.setdefault("client_kwargs", {})
            hk = ck.setdefault("headers", {})
            hk.update(headers)

        cache_dir = os.path.join(self.root_dir, ".fsspec-cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.remote_fs = fsspec.filesystem(
            "simplecache",
            target_protocol="https",
            target_options=target_opts,
            cache_storage=cache_dir,
            same_names=True,
        )

    def _remote_url_for(self, pair: str = None, interval: str = None) -> str:
        """Build the URL for a specific pair and interval."""
        if self.remote_base_url is None:
            raise ValueError("Remote base URL is not set. Call use_remote(...) first.")
        parts = [self.remote_base_url]
        if pair:
            parts.append(f"pair={pair}")
        if interval:
            parts.append(f"interval={interval}")
        return "/".join(parts) + "/"

    def read_parquet_remote(self, pair: str = None, interval: str = None,
                            columns=None, filters=None, sort=True) -> pd.DataFrame:
        """Read Parquet data from remote HTTPS source."""
        if self.remote_fs is None or self.remote_base_url is None:
            raise ValueError("Remote not configured. Call use_remote('https://...') first.")

        url = self._remote_url_for(pair=pair, interval=interval)
        entries = self.remote_fs.ls(url)
        files = []
        for e in entries:
            name = e["name"] if isinstance(e, dict) else str(e)
            if name.lower().endswith(".parquet"):
                files.append(name if name.startswith("http") else url + name.rsplit("/", 1)[-1])

        if not files:
            raise FileNotFoundError(f"No parquet files under {url}")

        dataset = ds.dataset(files, format="parquet", filesystem=self.remote_fs)
        table = dataset.to_table(columns=columns)
        df = table.to_pandas()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp", drop=True)

        if filters is not None:
            for col, op, val in filters:
                if op == "==": df = df[df[col] == val]
                elif op == "!=": df = df[df[col] != val]
                elif op == ">": df = df[df[col] > val]
                elif op == ">=": df = df[df[col] >= val]
                elif op == "<": df = df[df[col] < val]
                elif op == "<=": df = df[df[col] <= val]

        if sort:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
            elif "timestamp" in df.columns:
                df = df.sort_values("timestamp")
        return df

    def get_ohlcv_df_remote(self, symbols: list, interval: str = "1d", quote: str = "USD") -> pd.DataFrame:
        """Retrieve aligned OHLCV data from remote source."""
        dfs = []
        for symbol in symbols:
            pair = f"{symbol}-{quote}"
            dfi = self.read_parquet_remote(pair=pair, interval=interval)
            dfs.append(dfi)

        dfs = [d[~d.index.duplicated(keep='last')] for d in dfs]
        common_idx = dfs[0].index
        for d in dfs[1:]:
            common_idx = common_idx.union(d.index)

        dfs = [d.reindex(common_idx) for d in dfs]
        ohlcv_df = pd.concat(dfs, axis=1, keys=symbols).sort_index()
        ohlcv_df = align_to_datetime_index(ohlcv_df, interval=interval)
        ohlcv_df = fill_after_first_non_nan_multilevel_safe(ohlcv_df, symbols=symbols)
        ohlcv_df = fill_symbol_metadata(ohlcv_df, symbols)
        return ohlcv_df
