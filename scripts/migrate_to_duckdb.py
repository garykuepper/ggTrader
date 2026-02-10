import duckdb
import os
import sys
from tqdm import tqdm


def migrate_parquet_to_duckdb(parquet_dir="data/parquet", db_path="data/ggtrader.db"):
    """
    Migrates Parquet data to DuckDB incrementally by pair.
    """
    if not os.path.exists(parquet_dir):
        print(f"Error: Parquet directory {parquet_dir} not found.")
        return

    # List pairs from folder names
    pairs = [
        d.replace("pair=", "") for d in os.listdir(parquet_dir) if d.startswith("pair=")
    ]
    if not pairs:
        print("No pairs found in parquet directory.")
        return

    conn = duckdb.connect(db_path)
    print(f"Migrating {len(pairs)} pairs from {parquet_dir} to {db_path}...")

    for pair in tqdm(pairs, desc="Migrating pairs"):
        parquet_pattern = os.path.join(
            parquet_dir, f"pair={pair}", "**/*.parquet"
        ).replace("\\", "/")

        try:
            sql = f"""
                INSERT INTO ohlcv 
                SELECT 
                    timestamp, open, high, low, close, volume, trades, base, quote, pair, interval 
                FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
                ON CONFLICT (pair, interval, timestamp) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    trades = excluded.trades;
            """
            conn.execute(sql)
            # DuckDB autocommits by default in this mode, but let's be sure
        except Exception as e:
            print(f"Error during migration of {pair}: {e}")

    new_count = conn.execute("SELECT count(*) FROM ohlcv").fetchone()[0]
    print(f"Migration complete. Total rows in DuckDB: {new_count}")
    conn.close()


if __name__ == "__main__":
    migrate_parquet_to_duckdb()
