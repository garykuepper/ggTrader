import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .KrakenUtils import (
    align_to_datetime_index,
    fill_after_first_non_nan_multilevel_safe,
    fill_symbol_metadata,
    ensure_utc_timestamp,
    filter_out_stables
)

class KrakenParquetReader:
    def __init__(self, parquet_root, historical_mover_path):
        self.parquet_root = parquet_root
        self.historical_mover_path = historical_mover_path
        self.historical_mover_df = None

    def read_parquet(self, pair=None, interval=None, columns=None, filters=None, sort=True):
        """Read a local Parquet partition into a DataFrame."""
        base_path = self.parquet_root
        if pair and interval:
            path = os.path.join(base_path, f"pair={pair}", f"interval={interval}")
        elif pair:
            path = os.path.join(base_path, f"pair={pair}")
        else:
            path = base_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"No parquet data for {pair} found at {path}")

        df = pd.read_parquet(path, columns=columns, filters=filters)
        if sort and "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        return df.sort_index()

    def get_ohlcv_df(self, symbols: list, interval="1d", quote="USD", start: pd.Timestamp = None, end: pd.Timestamp = None):
        """Retrieve aligned OHLCV data for multiple symbols."""
        dfs = []
        for symbol in symbols:
            pair = f"{symbol}-{quote}"
            try:
                df = self.read_parquet(pair=pair, interval=interval)
                dfs.append(df)
            except FileNotFoundError:
                continue

        if not dfs:
            return pd.DataFrame()

        dfs = [d[~d.index.duplicated(keep='last')] for d in dfs]
        common_idx = dfs[0].index
        for d in dfs[1:]:
            common_idx = common_idx.union(d.index)

        dfs = [d.reindex(common_idx) for d in dfs]
        ohlcv_df = pd.concat(dfs, axis=1, keys=symbols)
        ohlcv_df = ohlcv_df.sort_index()
        ohlcv_df = align_to_datetime_index(ohlcv_df, interval=interval)
        ohlcv_df = fill_after_first_non_nan_multilevel_safe(ohlcv_df, symbols=symbols)
        ohlcv_df = fill_symbol_metadata(ohlcv_df, symbols)

        return ohlcv_df.loc[start:end] if start is not None and end is not None else ohlcv_df

    def list_parquet_pairs(self) -> list[str]:
        """List all trading pairs available in the Parquet dataset."""
        pairs = set()
        if not os.path.isdir(self.parquet_root):
            return []
        for name in os.listdir(self.parquet_root):
            full = os.path.join(self.parquet_root, name)
            if not os.path.isdir(full):
                continue
            if name.startswith("pair="):
                pair = name.split("=", 1)[-1]
                pairs.add(pair)
        return sorted(pairs)

    def list_parquet_symbols(self, quote='USD'):
        """List all base symbols available in the Parquet dataset for a specific quote."""
        pairs = self.list_parquet_pairs()
        symbols = [p[:-4] if p.endswith(quote) else p for p in pairs]
        return symbols

    def get_random_symbols(self, n=10):
        """Get a random selection of available symbols."""
        symbols = self.list_parquet_symbols()
        return np.random.choice(symbols, size=n, replace=False).tolist()

    def get_daily_historical_movers(self, top_n=20, trades_threshold=500, sample=None, stables=False):
        """Identify top movers based on historical volume."""
        interval = "1d"
        symbols = self.list_parquet_symbols()
        if sample:
            symbols = np.random.choice(symbols, size=sample, replace=False).tolist()
        if not stables:
            symbols = filter_out_stables(symbols)

        ohlcv_df = self.get_ohlcv_df(symbols=symbols, interval=interval, quote="USD")
        if ohlcv_df.empty:
            return pd.DataFrame()

        stacked = []
        for sym in symbols:
            if sym in ohlcv_df.columns.levels[0]:
                df_sym = ohlcv_df[sym].copy()
                df_sym['symbol'] = sym
                stacked.append(df_sym.reset_index())
        
        if not stacked:
            return pd.DataFrame()
            
        ohlcv_df_long = pd.concat(stacked, ignore_index=True)
        rename_map = {col: 'date' for col in ohlcv_df_long.columns if str(col).lower() in ['datetime', 'timestamp', 'index']}
        ohlcv_df_long = ohlcv_df_long.rename(columns=rename_map)
        ohlcv_df_long = ohlcv_df_long.dropna(subset=["trades"])
        ohlcv_df_long = ohlcv_df_long[ohlcv_df_long['trades'] > trades_threshold]
        
        top_by_volume = ohlcv_df_long.sort_values(['date', 'volume'], ascending=[True, False])
        top_per_day = top_by_volume.groupby('date').head(top_n).reset_index(drop=True)
        return top_per_day

    def save_historical_movers_to_parquet(self):
        """Pre-calculate and save historical top movers."""
        top_per_day = self.get_daily_historical_movers(top_n=100, sample=None, stables=False)
        top_per_day.to_parquet(os.path.join(self.historical_mover_path, "historical_movers.parquet"))

    def load_historical_movers_from_parquet(self):
        """Load pre-calculated historical movers."""
        path = os.path.join(self.historical_mover_path, "historical_movers.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def get_historical_movers_by_day(self, date: pd.Timestamp, top_n=100):
        """Get top movers for a specific historical date."""
        date = ensure_utc_timestamp(date)
        if self.historical_mover_df is None:
            self.historical_mover_df = self.load_historical_movers_from_parquet()
        
        if self.historical_mover_df.empty:
            return pd.DataFrame()
            
        top = self.historical_mover_df[self.historical_mover_df.date == date]
        return top.reset_index(drop=True).head(top_n)

    def build_4h_from_1h_and_merge(self, pair: str, year: int = 2023) -> pd.DataFrame:
        """Resample 1h data into 4h and merge with existing 4h files."""
        try:
            df_1h = self.read_parquet(pair=pair, interval="1h")
        except FileNotFoundError:
            return pd.DataFrame()

        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h.index = pd.to_datetime(df_1h.index, utc=True, errors="coerce")
        elif df_1h.index.tz is None:
            df_1h.index = df_1h.index.tz_localize("UTC")
        
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        df_1h_y = df_1h[(df_1h.index >= start) & (df_1h.index < end)]

        try:
            df_4h_real = self.read_parquet(pair=pair, interval="4h")
        except FileNotFoundError:
            df_4h_real = pd.DataFrame()

        if df_1h_y.empty:
            return df_4h_real

        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "trades": "sum"}
        df_4h_from_1h = df_1h_y.resample("4h", label="right", closed="right").agg(agg).dropna(subset=["open", "close"])

        for meta in ("base", "quote", "pair"):
            if meta in df_1h_y.columns:
                df_4h_from_1h[meta] = df_1h_y[meta].iloc[0]
        df_4h_from_1h["interval"] = "4h"

        combined = pd.concat([df_4h_from_1h, df_4h_real]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined
