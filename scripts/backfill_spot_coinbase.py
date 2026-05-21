"""Fill BTC-USD 1h spot gap (2026-01 → 2026-04) from Coinbase.

Kraken's spot OHLC endpoint only serves the most recent ~720 candles. Coinbase
honors `since` for deeper history. One-off gap-fill so the funding-arb
backtest has continuous data across the 2025-05 → 2026-05 funding window.
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


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "ggtrader"),
        password=os.getenv("DB_PASS", "ggtrader"),
        dbname=os.getenv("DB_NAME", "ggtrader"),
    )


def main() -> None:
    conn = _connect()
    cb = ccxt.coinbase({"enableRateLimit": True})
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(tz=timezone.utc)
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bar_ms = 3600 * 1000
    inserted = 0
    while cursor_ms < end_ms:
        bars = cb.fetch_ohlcv("BTC/USD", timeframe="1h", since=cursor_ms, limit=300)
        if not bars:
            cursor_ms += 300 * bar_ms
            time.sleep(0.5)
            continue
        rows = [
            (
                datetime.fromtimestamp(b[0] / 1000, tz=timezone.utc).replace(tzinfo=None),
                "BTC-USD",
                "1h",
                float(b[1]),
                float(b[2]),
                float(b[3]),
                float(b[4]),
                float(b[5]),
                0,
            )
            for b in bars
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO ohlcv ("timestamp", symbol, "interval", open, high, low, close, volume, trades)
                VALUES %s
                ON CONFLICT ("timestamp", symbol, "interval") DO NOTHING
                """,
                rows,
            )
        conn.commit()
        inserted += len(rows)
        cursor_ms = bars[-1][0] + bar_ms
        print(
            f"  through {datetime.fromtimestamp(bars[-1][0] / 1000, tz=timezone.utc)} (+{len(bars)})"
        )
        time.sleep(0.4)
    print(f"Total inserted (deduped): {inserted}")
    conn.close()


if __name__ == "__main__":
    main()
