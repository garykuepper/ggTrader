import os
import sys
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Add src to sys.path to reuse KrakenConstants and KrakenUtils
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.data.kraken.constants import kraken_map


def clean_ccy(ccy):
    """Simplified clean_ccy from KrakenUtils."""
    return kraken_map.get(ccy.upper(), ccy.upper())


def split_pair(filename_stem, quote_only="USD"):
    """
    Splits pair from filename stem (e.g. 'XBTUSD' -> 'BTC', 'USD', 'BTC-USD').
    Handles both 'XBTUSD' and 'BTCUSD' styles.
    """
    p = filename_stem.upper()
    if p.endswith(quote_only):
        raw_base = p[: -len(quote_only)]
        base = clean_ccy(raw_base)
        # Special case: clean_ccy mapping might return 'XBT', let's ensure it maps to 'BTC' if needed
        # but kraken_map already handles XXBT -> XBT and XBT -> BTC.
        # Wait, kraken_map in KrakenConstants.py:
        # "XBT": "BTC", "XXBT": "XBT" ... wait, XXBT -> XBT then XBT -> BTC?
        # Actually it's a flat dict.

        # Let's re-run clean_ccy if it's still in the map
        base = clean_ccy(base)

        quote = quote_only
        pair_std = f"{base}-{quote}"
        return base, quote, pair_std
    return None, None, None


def process_file(args):
    file_path, parquet_root, intervals = args
    filename = os.path.basename(file_path)
    stem = filename.split(".")[0]

    base, quote, pair_std = split_pair(stem)
    if not base:
        return f"Skipped: {filename} (Not a USD pair)"

    try:
        # Load raw trade data: timestamp, price, volume
        # Format: 1735689600,93370.80000,0.00011607
        df = pd.read_csv(
            file_path,
            header=None,
            names=["timestamp", "price", "volume"],
            usecols=[0, 1, 2],
            dtype={"price": "float32", "volume": "float64"},
        )

        if df.empty:
            return f"Skipped: {filename} (Empty file)"

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.set_index("timestamp", inplace=True)

        # Aggregation logic
        agg_map = {"price": ["first", "max", "min", "last"], "volume": "sum"}

        for interval in intervals:
            # Resample trades to OHLCV
            # label='left' and closed='left' is standard for crypto OHLC
            resampled = df.resample(interval, label="left", closed="left").agg(agg_map)

            # Flatten columns
            resampled.columns = ["open", "high", "low", "close", "volume_base"]

            # Drop rows with no trades
            resampled.dropna(subset=["open"], inplace=True)

            if resampled.empty:
                continue

            # Calculate volume in quote currency (USD) to match existing schema
            # Existing schema has 'volume' column which is volume in quote currency.
            # KrakenUtils.py line 39: df["volume"] = df["volume"] * df["close"]
            resampled["volume"] = resampled["volume_base"] * resampled["close"]

            # Add trades count
            resampled["trades"] = df.resample(
                interval, label="left", closed="left"
            ).size()
            resampled["trades"] = resampled["trades"].astype("Int64")

            # Metadata columns
            resampled["base"] = base
            resampled["quote"] = quote
            resampled["pair"] = pair_std
            resampled["interval"] = interval

            # Drop volume_base to match original schema
            resampled.drop(columns=["volume_base"], inplace=True)

            # Convert to float32 for OHLC
            for col in ["open", "high", "low", "close"]:
                resampled[col] = resampled[col].astype("float32")

            # Write to dataset
            table = pa.Table.from_pandas(resampled, preserve_index=True)
            pq.write_to_dataset(
                table,
                root_path=parquet_root,
                partition_cols=["pair", "interval"],
                compression="zstd",
            )

        return f"Processed: {filename} -> {pair_std}"
    except Exception as e:
        return f"Error: {filename} -> {str(e)}"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process raw Kraken trades to OHLC Parquet."
    )
    parser.add_argument(
        "--raw_dir", default="data/raw", help="Directory containing raw CSVs"
    )
    parser.add_argument(
        "--parquet_root",
        default="data/parquet",
        help="Root directory for Parquet dataset",
    )
    parser.add_argument(
        "--sample", action="store_true", help="Process only a few files for testing"
    )
    parser.add_argument("--file", help="Process a specific file")
    args = parser.parse_args()

    intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"]
    manifest_path = os.path.join(args.parquet_root, ".processed_raw_files.json")

    processed_files = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            processed_files = json.load(f)

    processed_set = set(processed_files)

    all_files = []
    if args.file:
        all_files = [os.path.abspath(args.file)]
    else:
        for root, _, files in os.walk(args.raw_dir):
            for f in files:
                if f.endswith(".csv"):
                    full_path = os.path.abspath(os.path.join(root, f))
                    if full_path not in processed_set:
                        # Exclude files already in the "OHLCVT" format (they have more columns usually)
                        # or just rely on the manifest and the fact that we're adding new data.
                        all_files.append(full_path)

    if not all_files:
        print("No new files to process.")
        return

    if args.sample and not args.file:
        all_files = all_files[:5]
        print(f"Sampling {len(all_files)} files.")

    print(f"Found {len(all_files)} new files to process.")

    # Process tasks
    tasks = [(f, args.parquet_root, intervals) for f in all_files]

    results = []
    # Sequential for now to ensure stability and better progress monitoring
    for task in tqdm(tasks, desc="Processing CSVs"):
        res = process_file(task)
        results.append(res)
        if "Processed" in res:
            processed_files.append(task[0])
            # Intermediary manifest save
            with open(manifest_path, "w") as f:
                json.dump(processed_files, f, indent=4)

    # Print summary
    errors = [r for r in results if r.startswith("Error")]
    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(err)

    print("\nSummary:")
    print(f"Successfully processed: {len(results) - len(errors)}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
