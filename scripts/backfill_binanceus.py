"""Backfill BTC-USD, ETH-USD, and DOGE-USD historical OHLCV data on Binance.US.

Fetches 4h and 1d candles from 2023-01-01 to present, and inserts them into
TimescaleDB with venue = 'binanceus_spot'.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import ccxt
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

SLEEP_SEC = 0.5
LIMIT = 1000

SYMBOLS = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "DOGE-USD": "DOGE/USD",
    "TRX-USD": "TRX/USD",
}

INTERVALS = {
    "4h": 4 * 3600 * 1000,
    "1d": 24 * 3600 * 1000,
}


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "ggtrader"),
        password=os.getenv("DB_PASS", "ggtrader"),
        dbname=os.getenv("DB_NAME", "ggtrader"),
    )


def backfill_symbol_interval(
    conn: psycopg2.extensions.connection,
    ex: ccxt.binanceus,
    symbol: str,
    ccxt_symbol: str,
    interval: str,
    start_dt: datetime,
) -> int:
    cursor_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    step_ms = INTERVALS[interval]
    inserted = 0

    print(f"Backfilling {symbol} ({interval}) starting from {start_dt}...")

    while cursor_ms < end_ms:
        try:
            bars = ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, since=cursor_ms, limit=LIMIT)
        except Exception as e:
            print(f"Error fetching {ccxt_symbol} {interval} since {cursor_ms}: {e}")
            time.sleep(2.0)
            continue

        if not bars:
            print(f"No bars returned for {symbol} {interval} at {cursor_ms}. Moving forward.")
            cursor_ms += LIMIT * step_ms
            time.sleep(SLEEP_SEC)
            continue

        rows = [
            (
                datetime.fromtimestamp(b[0] / 1000, tz=timezone.utc).replace(tzinfo=None),
                symbol,
                interval,
                float(b[1]) if b[1] is not None else None,
                float(b[2]) if b[2] is not None else None,
                float(b[3]) if b[3] is not None else None,
                float(b[4]) if b[4] is not None else None,
                float(b[5]) if b[5] is not None else None,
                0,  # trades placeholder
                "binanceus_spot",
            )
            for b in bars
        ]

        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO ohlcv (
                    "timestamp", symbol, "interval", open, high, low, close, volume, trades, venue
                )
                VALUES %s
                ON CONFLICT ("timestamp", symbol, "interval", venue) DO NOTHING
                """,
                rows,
            )
        conn.commit()
        inserted += len(rows)
        last_ms = bars[-1][0]
        cursor_ms = last_ms + step_ms
        time.sleep(SLEEP_SEC)
        dt_str = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc)
        print(f"  ...{symbol} {interval} through {dt_str} (+{len(bars)})")

    return inserted


if __name__ == "__main__":
    ex = ccxt.binanceus(
        {
            "apiKey": os.getenv("BINANCE_API_LIVE_KEY"),
            "secret": os.getenv("BINANCE_SECRET_LIVE_KEY"),
            "enableRateLimit": True,
        }
    )
    conn = _connect()
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)

    total_inserted = 0
    for symbol, ccxt_symbol in SYMBOLS.items():
        for interval in INTERVALS.keys():
            n = backfill_symbol_interval(conn, ex, symbol, ccxt_symbol, interval, start_date)
            total_inserted += n
            print(f"Completed {symbol} {interval}: inserted {n} records.")

    conn.close()
    print(f"Backfill complete. Total inserted records: {total_inserted}")
