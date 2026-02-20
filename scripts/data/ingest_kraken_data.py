import argparse
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from ggTrader.data.historical.postgres_ingestor import PostgresIngestor
from ggTrader.utils.config import get_db_connection_string


def main() -> None:
    """
    Main entry point for ingesting Kraken historical data into PostgreSQL.
    """
    parser = argparse.ArgumentParser(description="Ingest Kraken Data into PostgreSQL")
    parser.add_argument("--sync", action="store_true", help="Sync new directories from data/raw")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync all directories (ignore manifest)",
    )
    parser.add_argument("--dir", type=str, help="Specific directory to ingest")

    args = parser.parse_args()

    print("Connecting to database...")
    try:
        connection_string = get_db_connection_string()
        ingestor = PostgresIngestor(connection_string)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    except Exception as e:
        print(f"Failed to initialize ingestor: {e}")
        return

    if args.dir:
        print(f"Ingesting specific directory: {args.dir}")
        try:
            ingestor.ingest_dir(args.dir)
        except Exception as e:
            print(f"Error during ingestion of {args.dir}: {e}")
    elif args.sync:
        print("Syncing new data directories...")
        try:
            ingestor.sync_local_data(root_dir=project_root, force=args.force)
        except Exception as e:
            print(f"Error during sync: {e}")
    else:
        print("Usage: python ingest_kraken_data.py --sync [--force] OR --dir <path>")
        print("Defaulting to --sync behavior...")
        try:
            ingestor.sync_local_data(root_dir=project_root, force=args.force)
        except Exception as e:
            print(f"Error during sync: {e}")


if __name__ == "__main__":
    main()
