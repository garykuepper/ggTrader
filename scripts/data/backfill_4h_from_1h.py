"""One-time migration: build 4h candles from 1h data for missing date ranges."""

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from sqlalchemy import create_engine, text

from ggTrader.utils.config import get_db_connection_string

# SQL to aggregate 1h candles into 4h candles using time_bucket
# Only inserts for date ranges where 4h data is missing
AGGREGATE_4H_SQL = """
INSERT INTO ohlcv (timestamp, symbol, interval, open, high, low, close, volume, trades)
SELECT
    time_bucket('4 hours', timestamp) AS ts,
    symbol,
    '4h' AS interval,
    (ARRAY_AGG(open ORDER BY timestamp ASC))[1] AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    (ARRAY_AGG(close ORDER BY timestamp DESC))[1] AS close,
    SUM(volume) AS volume,
    SUM(trades) AS trades
FROM ohlcv
WHERE interval = '1h'
  AND timestamp < :cutoff
GROUP BY ts, symbol
HAVING COUNT(*) = 4
ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    trades = EXCLUDED.trades;
"""

# Same pattern for 30m from 15m (bonus — also missing before Q2 2024)
AGGREGATE_30M_SQL = """
INSERT INTO ohlcv (timestamp, symbol, interval, open, high, low, close, volume, trades)
SELECT
    time_bucket('30 minutes', timestamp) AS ts,
    symbol,
    '30m' AS interval,
    (ARRAY_AGG(open ORDER BY timestamp ASC))[1] AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    (ARRAY_AGG(close ORDER BY timestamp DESC))[1] AS close,
    SUM(volume) AS volume,
    SUM(trades) AS trades
FROM ohlcv
WHERE interval = '15m'
  AND timestamp < :cutoff
GROUP BY ts, symbol
HAVING COUNT(*) = 2
ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    trades = EXCLUDED.trades;
"""

# Cutoff date: 4h data already exists from this point forward
CUTOFF_4H = "2024-04-01"
CUTOFF_30M = "2024-04-01"


def main() -> None:
    """Run the migration."""
    engine = create_engine(get_db_connection_string())

    with engine.begin() as conn:
        # Build 4h from 1h
        print(f"Building 4h candles from 1h data (before {CUTOFF_4H})...")
        result = conn.execute(text(AGGREGATE_4H_SQL), {"cutoff": CUTOFF_4H})
        print(f"  Inserted/updated {result.rowcount:,} 4h rows.")

        # Build 30m from 15m
        print(f"Building 30m candles from 15m data (before {CUTOFF_30M})...")
        result = conn.execute(text(AGGREGATE_30M_SQL), {"cutoff": CUTOFF_30M})
        print(f"  Inserted/updated {result.rowcount:,} 30m rows.")

    engine.dispose()

    # Verify
    engine = create_engine(get_db_connection_string())
    with engine.connect() as conn:
        for iv in ["4h", "30m"]:
            row = conn.execute(
                text(
                    "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) "
                    "FROM ohlcv WHERE interval = :iv AND symbol = 'BTC-USD'"
                ),
                {"iv": iv},
            ).fetchone()
            print(
                f"  BTC {iv}: {str(row[0])[:10]} to {str(row[1])[:10]} ({row[2]:,} rows)"
            )

    engine.dispose()
    print("Done!")


if __name__ == "__main__":
    main()
