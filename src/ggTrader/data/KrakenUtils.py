import pandas as pd
import numpy as np
from .KrakenConstants import kraken_map, interval_map, STABLE_BASES

def clean_ccy(ccy: str) -> str:
    """Map Kraken prefixes like XXBT -> XBT -> BTC, and ZUSD -> USD."""
    return kraken_map.get(ccy, ccy)

def split_pair(raw_pair: str, quote_only="USD"):
    """
    Split Kraken pair (e.g., 'XXBTZUSD', 'XETHZUSD', 'BTCUSDT') into base/quote robustly.
    """
    p = raw_pair.upper()
    if p.endswith(quote_only):
        base = clean_ccy(p[:-3])
        quote = quote_only
        pair_std = f"{base}-{quote}"
        return clean_ccy(p[:-3]), clean_ccy(quote_only), pair_std
    return None, None, None

def _load_csv_common(file_path: str, col_names: list, interval: str = None):
    """
    Shared CSV loader with standard parsing logic.
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
    
    if "volume" in df.columns and "close" in df.columns:
        df["volume"] = df["volume"] * df["close"]
        
    if interval is not None:
        interval = interval_map.get(interval.lower(), interval)
    return df, interval

def fill_symbol_metadata(ohlcv_df: pd.DataFrame, symbols: list):
    """Fill metadata columns and standardize MultiIndex."""
    df_out = ohlcv_df.copy()
    for sym in symbols:
        base_col = (sym, 'base')
        quote_col = (sym, 'quote')
        if base_col in df_out.columns:
            df_out[base_col] = df_out[base_col].ffill().bfill()
        if quote_col in df_out.columns:
            df_out[quote_col] = df_out[quote_col].ffill().bfill()
            
    if isinstance(df_out.columns, pd.MultiIndex):
        df_out.columns = pd.MultiIndex.from_tuples(df_out.columns.values, names=["symbol", "ohlcv"])
    else:
        df_out.columns.name = "symbol"
    df_out.index.name = "Datetime"
    return df_out

def fill_after_first_non_nan_single(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill NaNs after the first valid entry for single-level DataFrame."""
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            first_valid = df_out[col].first_valid_index()
            if first_valid is not None:
                tail = df_out[col].loc[first_valid:]
                tail_filled = tail.ffill()
                df_out.loc[first_valid:, col] = tail_filled
    return df_out

def align_to_datetime_index(ohlcv_df: pd.DataFrame, interval: str = "1d"):
    """Reindex to a clean DatetimeIndex matching the frequency."""
    if ohlcv_df.empty:
        return ohlcv_df
    first_date = ohlcv_df.index[0]
    last_date = ohlcv_df.index[-1]
    date_range = pd.date_range(start=first_date, end=last_date, freq=interval)
    ohlcv_df = ohlcv_df.reindex(date_range)
    ohlcv_df.index.name = "Datetime"
    return ohlcv_df

def fill_after_first_non_nan_multilevel_safe(ohlcv_df: pd.DataFrame, symbols: list) -> pd.DataFrame:
    """Forward-fill NaNs after the first valid entry for MultiIndex DataFrame."""
    df_out = ohlcv_df.copy()
    for sym in symbols:
        df_sym = ohlcv_df.xs(sym, axis=1, level=0).copy()
        for col in df_sym.columns:
            if pd.api.types.is_numeric_dtype(df_sym[col]):
                first_valid = df_sym[col].first_valid_index()
                if first_valid is not None:
                    tail = df_sym[col].loc[first_valid:]
                    tail_filled = tail.ffill()
                    df_sym.loc[first_valid:, col] = tail_filled
        for col in df_sym.columns:
            df_out[(sym, col)] = df_sym[col]
    return df_out

def ensure_utc_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """Ensure a timestamp is in UTC."""
    if ts.tz is None:
        return ts.tz_localize("UTC")
    else:
        return ts.tz_convert("UTC")

def filter_out_stables(symbols: list):
    """Remove known stablecoins from a list of symbols."""
    stables = set(STABLE_BASES)
    return [x for x in symbols if x not in stables]

def get_file_names(path: str, quote_only: str = "USD") -> list:
    """List CSV files in a directory that match the quote."""
    import os
    if not os.path.isdir(path):
        return []
    files = [f for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".csv")]
    return filter_files_by_quote(files, quote_only)

def filter_files_by_quote(files: list, quote_only: str = "USD") -> list:
    """Keep files where the pair part ends with quote and has non-empty base."""
    kept = []
    for f in files:
        stem = f.rsplit(".", 1)[0]
        pair = stem.split("_", 1)[0].upper()
        if pair.endswith(quote_only) and len(pair) > len(quote_only):
            kept.append(f)
    return kept
