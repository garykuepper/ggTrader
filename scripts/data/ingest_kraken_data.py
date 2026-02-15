import sys
import os
import argparse

# Ensure project root is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.data.kraken.historical_data import KrakenHistoricalData


def main():
    parser = argparse.ArgumentParser(description="Ingest Kraken Data into PostgreSQL")
    parser.add_argument(
        "--sync", action="store_true", help="Sync new directories from data/raw"
    )
    parser.add_argument("--dir", type=str, help="Specific directory to ingest")

    args = parser.parse_args()

    print(f"Connecting to DB via KrakenHistoricalData...")
    data_manager = KrakenHistoricalData()

    if args.dir:
        print(f"Ingesting specific directory: {args.dir}")
        data_manager.ingestor.ingest_dir(args.dir)
    elif args.sync:
        print("Syncing new data directories...")
        data_manager.sync_local_data()
    else:
        print("Usage: python ingest_kraken_data.py --sync OR --dir <path>")
        print("Defaulting to --sync behavior...")
        data_manager.sync_local_data()


if __name__ == "__main__":
    main()
