import argparse
import os
import sys

# Ensure project root is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.data.kraken.historical_data import KrakenHistoricalData


def main() -> None:
    """
    Main entry point for ingesting Kraken historical data into PostgreSQL.
    """
    parser = argparse.ArgumentParser(description="Ingest Kraken Data into PostgreSQL")
    parser.add_argument(
        "--sync", action="store_true", help="Sync new directories from data/raw"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync all directories (ignore manifest)",
    )
    parser.add_argument("--dir", type=str, help="Specific directory to ingest")

    args = parser.parse_args()

    print("Connecting to database...")
    try:
        data_manager = KrakenHistoricalData()
    except Exception as e:
        print(f"Failed to initialize data manager: {e}")
        return

    if args.dir:
        print(f"Ingesting specific directory: {args.dir}")
        try:
            data_manager.ingestor.ingest_dir(args.dir)
        except Exception as e:
            print(f"Error during ingestion of {args.dir}: {e}")
    elif args.sync:
        print("Syncing new data directories...")
        try:
            data_manager.sync_local_data(force=args.force)
        except Exception as e:
            print(f"Error during sync: {e}")
    else:
        print("Usage: python ingest_kraken_data.py --sync [--force] OR --dir <path>")
        print("Defaulting to --sync behavior...")
        try:
            data_manager.sync_local_data(force=args.force)
        except Exception as e:
            print(f"Error during sync: {e}")


if __name__ == "__main__":
    main()
