"""Backfill binanceus_spot historical data from kraken_spot in TimescaleDB.

Clones missing candles from the continuous kraken_spot dataset to binanceus_spot
to fill the 2023-2025 chronological gap.
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import text

# Setup path to import ggTrader modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ggTrader.utils.db_engine import create_db_engine_or_exit


def main() -> None:
    """Orchestrate database-level backfill from kraken_spot to binanceus_spot."""
    parser = argparse.ArgumentParser(
        description="Backfill binanceus_spot historical data from kraken_spot"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to backfill (default: all active in binanceus_spot)",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="4h,1d",
        help="Comma-separated intervals to backfill (default: 4h,1d)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date of the backfill window (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-03-01",
        help="End date of the backfill window (default: 2025-03-01)",
    )
    args = parser.parse_args()

    engine = create_db_engine_or_exit()

    # Use engine.begin() to manage the transaction lifecycle automatically
    with engine.begin() as conn:
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            print("Resolving active symbols in binanceus_spot...")
            res = conn.execute(
                text("SELECT DISTINCT symbol FROM ohlcv WHERE venue = 'binanceus_spot';")
            )
            symbols = [row[0] for row in res]

        if not symbols:
            print("No symbols found to process.")
            return

        intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
        print(
            f"Starting backfill for {len(symbols)} symbols from kraken_spot to binanceus_spot "
            f"within window {args.start_date} .. {args.end_date} for intervals {intervals}..."
        )

        for symbol in symbols:
            print(f"Processing symbol: {symbol}")
            query = """
                INSERT INTO ohlcv (
                    timestamp, symbol, interval, open, high, low, close, volume, trades, venue
                )
                SELECT
                    timestamp, symbol, interval, open, high, low, close, volume, trades,
                    'binanceus_spot'
                FROM ohlcv k
                WHERE k.venue = 'kraken_spot'
                  AND k.symbol = :symbol
                  AND k.interval = ANY(:intervals)
                  AND k.timestamp >= :start_date
                  AND k.timestamp <= :end_date
                  AND NOT EXISTS (
                      SELECT 1 FROM ohlcv b
                      WHERE b.venue = 'binanceus_spot'
                        AND b.symbol = k.symbol
                        AND b.interval = k.interval
                        AND b.timestamp = k.timestamp
                        AND b.timestamp >= :start_date
                        AND b.timestamp <= :end_date
                  );
            """
            result = conn.execute(
                text(query),
                {
                    "symbol": symbol,
                    "intervals": intervals,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                },
            )
            print(f"  Cloned {result.rowcount} missing candles for {symbol}.")

    print("Backfill complete!")


if __name__ == "__main__":
    main()
