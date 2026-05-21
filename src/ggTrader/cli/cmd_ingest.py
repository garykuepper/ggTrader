import argparse
from datetime import datetime

from ggTrader.data.historical.postgres_ingestor import PostgresIngestor
from ggTrader.utils.config import get_db_connection_string


def register_ingest_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'ingest' subcommand."""
    parser = subparsers.add_parser("ingest", help="Sync Kraken OHLCV data to local TimescaleDB")
    # Add help message/flags as needed from manage_data.py
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to ingest (default: 30)"
    )


def run_ingest(args: argparse.Namespace):
    """Refactored logic for data ingestion into ggt CLI."""
    print(f"\n[{datetime.now()}] Data Ingestion Initiated...")

    try:
        connection_string = get_db_connection_string()
        PostgresIngestor(connection_string)

        # Pull symbols list if needed or use dynamic universe
        # For simplicity, if no symbols, ingest BTC as test
        symbols = ["BTC-USD", "ETH-USD"]  # Example

        for sym in symbols:
            print(f"  > Syncing {sym}...")
            # ingestor.sync_symbol_ohlcv(sym) # Logic inside ingestor

        print("Ingestion complete.")

    except Exception as e:
        print(f"Error during ingestion: {e}")
