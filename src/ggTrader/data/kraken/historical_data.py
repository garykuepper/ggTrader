import os
import sys
import pandas as pd
import json
from pathlib import Path
from .utils import get_file_names
from .converter import KrakenConverter
from .parquet_reader import KrakenParquetReader
from .remote_reader import KrakenRemoteReader

class KrakenHistoricalData:
    """
    Facade for Kraken historical data operations.
    Delegates to specialized modules for conversion, reading, and remote access.
    """
    def __init__(self):
        self.root_dir = self._find_project_root()
        self.raw_path = os.path.join(self.root_dir, 'data', 'raw')
        self.parquet_root = os.path.join(self.root_dir, 'data', 'parquet')
        self.historical_mover_path = os.path.join(self.root_dir, 'data', 'historical_movers')
        
        os.makedirs(self.parquet_root, exist_ok=True)
        os.makedirs(self.historical_mover_path, exist_ok=True)

        self.converter = KrakenConverter(self.parquet_root)
        self.reader = KrakenParquetReader(self.parquet_root, self.historical_mover_path)
        self.remote_reader = KrakenRemoteReader(self.root_dir)

    def _find_project_root(self):
        """Finds the project root by looking for pyproject.toml."""
        # Check current directory and its parents
        current = Path(os.getcwd()).absolute()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return str(parent)
        
        # Fallback to file-based relative path if not found (e.g. running from IDE)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        return base_dir

    # ---------- Directory Helpers ----------
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
        return get_file_names(path, quote_only)

    # ---------- Conversion (Delegated to KrakenConverter) ----------
    def csvs_dir_to_parquet(self, *args, **kwargs):
        return self.converter.csvs_dir_to_parquet(*args, **kwargs)

    def csvs_many_dirs_to_parquet(self, dirs=None, **kwargs):
        if dirs is None:
            dirs = self.list_quarter_dirs()
        for d in dirs:
            self.converter.csvs_dir_to_parquet(d, **kwargs)

    def csvs_dir_to_parquet_parallel(self, *args, **kwargs):
        return self.converter.csvs_dir_to_parquet_parallel(*args, **kwargs)

    def csvs_many_dirs_to_parquet_parallel(self, dirs=None, **kwargs):
        if dirs is None:
            dirs = self.list_quarter_dirs()
        for d in dirs:
            self.converter.csvs_dir_to_parquet_parallel(d, **kwargs)

    def sync_local_data(self, **kwargs):
        """
        Intelligently sync new Kraken raw CSV directories into the Parquet dataset.
        Uses a manifest to avoid re-processing directories.
        """
        manifest_path = os.path.join(self.parquet_root, ".processed_dirs.json")
        processed_dirs = set()
        
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    processed_dirs = set(json.load(f))
            except Exception as e:
                print(f"Warning: could not load sync manifest: {e}")

        all_dirs = self.list_quarter_dirs()
        new_dirs = [d for d in all_dirs if os.path.basename(d) not in processed_dirs]

        if not new_dirs:
            print("No new Kraken data directories found to sync.")
            return

        print(f"Found {len(new_dirs)} new directories to sync: {[os.path.basename(d) for d in new_dirs]}")
        
        for d in new_dirs:
            self.csvs_dir_to_parquet_parallel(d, **kwargs)
            processed_dirs.add(os.path.basename(d))
            
            # Update manifest after each directory is successfully processed
            try:
                with open(manifest_path, 'w') as f:
                    json.dump(sorted(list(processed_dirs)), f, indent=4)
            except Exception as e:
                print(f"Warning: could not update sync manifest: {e}")
        
        print("Data synchronization complete.")

    # ---------- Local Data Reading (Delegated to KrakenParquetReader) ----------
    def read_parquet(self, *args, **kwargs):
        return self.reader.read_parquet(*args, **kwargs)

    def get_ohlcv_df(self, *args, **kwargs):
        return self.reader.get_ohlcv_df(*args, **kwargs)

    def list_parquet_pairs(self):
        return self.reader.list_parquet_pairs()

    def list_parquet_symbols(self, **kwargs):
        return self.reader.list_parquet_symbols(**kwargs)

    def get_random_symbols(self, **kwargs):
        return self.reader.get_random_symbols(**kwargs)

    def get_daily_historical_movers(self, **kwargs):
        return self.reader.get_daily_historical_movers(**kwargs)

    def save_historical_movers_to_parquet(self):
        return self.reader.save_historical_movers_to_parquet()

    def load_historical_movers_from_parquet(self):
        return self.reader.load_historical_movers_from_parquet()

    def get_historical_movers_by_day(self, *args, **kwargs):
        return self.reader.get_historical_movers_by_day(*args, **kwargs)

    def build_4h_from_1h_and_merge(self, *args, **kwargs):
        return self.reader.build_4h_from_1h_and_merge(*args, **kwargs)

    # ---------- Remote Data Access (Delegated to KrakenRemoteReader) ----------
    def use_remote(self, *args, **kwargs):
        return self.remote_reader.use_remote(*args, **kwargs)

    def read_parquet_remote(self, *args, **kwargs):
        return self.remote_reader.read_parquet_remote(*args, **kwargs)

    def get_ohlcv_df_remote(self, *args, **kwargs):
        return self.remote_reader.get_ohlcv_df_remote(*args, **kwargs)

    # ---------- Static Utility Methods (Backward Compatibility) ----------
    @staticmethod
    def align_to_datetime_index(*args, **kwargs):
        from . import KrakenUtils
        return KrakenUtils.align_to_datetime_index(*args, **kwargs)

    @staticmethod
    def fill_after_first_non_nan_single(*args, **kwargs):
        from . import KrakenUtils
        return KrakenUtils.fill_after_first_non_nan_single(*args, **kwargs)

    @staticmethod
    def fill_after_first_non_nan_multilevel_safe(*args, **kwargs):
        from . import KrakenUtils
        return KrakenUtils.fill_after_first_non_nan_multilevel_safe(*args, **kwargs)

    @staticmethod
    def fill_symbol_metadata(*args, **kwargs):
        from . import KrakenUtils
        return KrakenUtils.fill_symbol_metadata(*args, **kwargs)

    @staticmethod
    def ensure_utc_timestamp(*args, **kwargs):
        from . import KrakenUtils
        return KrakenUtils.ensure_utc_timestamp(*args, **kwargs)

    @staticmethod
    def filter_out_stables(*args, **kwargs):
        from . import KrakenUtils
        return KrakenUtils.filter_out_stables(*args, **kwargs)

if __name__ == "__main__":
    # Internal test/demo
    from tabulate import tabulate
    k = KrakenHistoricalData()
    
    # Simple test: list pairs and get random OHLCV
    pairs = k.list_parquet_pairs()
    print(f"Total pairs in Parquet: {len(pairs)}")
    
    symbols = k.get_random_symbols(n=3)
    print(f"Testing random symbols: {symbols}")
    
    df = k.get_ohlcv_df(symbols, interval="1d")
    print(df.head())
