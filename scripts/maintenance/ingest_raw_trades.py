import duckdb
import os
import pandas as pd
import json
from tqdm import tqdm
import sys

# Add src to sys.path to reuse KrakenConstants
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)
try:
    from ggTrader.data.kraken.constants import kraken_map
except ImportError:
    # Fallback if the path structure is slightly different
    from src.ggTrader.data.kraken.constants import kraken_map


def clean_ccy(ccy):
    return kraken_map.get(ccy.upper(), ccy.upper())


def split_pair(filename_stem, quote_only="USD"):
    p = filename_stem.upper()
    if p.endswith(quote_only):
        raw_base = p[: -len(quote_only)]
        base = clean_ccy(raw_base)
        base = clean_ccy(base)
        quote = quote_only
        pair_std = f"{base}-{quote}"
        return base, quote, pair_std
    return None, None, None


def ingest_raw_csvs(raw_dir="data/raw", db_path="data/ggtrader.db", intervals=None):
    if intervals is None:
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]

    conn = duckdb.connect(db_path)

    # Manifest to track processed files
    manifest_path = os.path.join(os.path.dirname(db_path), ".processed_csvs.json")
    processed_files = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            processed_files = json.load(f)
    processed_set = set(processed_files)

    all_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".csv"):
                full_path = os.path.abspath(os.path.join(root, f))
                if full_path not in processed_set:
                    all_files.append(full_path)

    if not all_files:
        print("No new CSV files to process.")
        return

    print(f"Found {len(all_files)} new CSV files to process.")

    for file_path in tqdm(all_files, desc="Ingesting CSVs"):
        filename = os.path.basename(file_path)
        stem = filename.split(".")[0]
        base, quote, pair_std = split_pair(stem)

        if not base:
            continue

        try:
            # DuckDB is very fast at reading CSVs and aggregating
            # We use a temporary table to load the trades
            temp_table = f"temp_trades_{stem}"
            conn.execute(
                f"CREATE TEMP TABLE {temp_table} (timestamp DOUBLE, price DOUBLE, volume DOUBLE)"
            )

            # Load CSV into temp table
            conn.execute(
                f"COPY {temp_table} FROM '{file_path.replace('\\', '/')}' (DELIMITER ',', HEADER FALSE)"
            )

            for interval in intervals:
                # interval conversion like '1m' to DuckDB interval
                # DuckDB doesn't have a direct '1m' alias in the same way pandas resample does for strings
                # but we can use time_bucket

                bucket_sql = ""
                if interval.endswith("m"):
                    bucket_sql = f"INTERVAL {interval[:-1]} MINUTE"
                elif interval.endswith("h"):
                    bucket_sql = f"INTERVAL {interval[:-1]} HOUR"
                elif interval.endswith("d"):
                    bucket_sql = f"INTERVAL {interval[:-1]} DAY"

                sql = f"""
                    INSERT INTO ohlcv
                    SELECT 
                        time_bucket({bucket_sql}, to_timestamp(timestamp)) as timestamp,
                        first(price) as open,
                        max(price) as high,
                        min(price) as low,
                        last(price) as close,
                        sum(volume * price) as volume,
                        count(*) as trades,
                        '{base}' as base,
                        '{quote}' as quote,
                        '{pair_std}' as pair,
                        '{interval}' as interval
                    FROM {temp_table}
                    GROUP BY 1
                    ON CONFLICT (pair, interval, timestamp) DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume,
                        trades = excluded.trades;
                """
                conn.execute(sql)

            conn.execute(f"DROP TABLE {temp_table}")
            processed_files.append(file_path)

            # Save manifest periodically
            with open(manifest_path, "w") as f:
                json.dump(processed_files, f, indent=4)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    conn.close()


if __name__ == "__main__":
    ingest_raw_csvs()
