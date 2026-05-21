"""Fill BTC-USD 1h spot gaps in TimescaleDB via ccxt Kraken.

Existing coverage: 2023-01-01 → 2026-02-22.
Gaps to fill for the funding-arb backtest:
  - 2022-03-23 → 2022-12-31 (pre-existing, needed for 3-year basis-proxy run)
  - 2026-02-23 → today        (recent, needed for the 1-year real-funding run)

One-off script. Not part of the runtime data path.
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

KRAKEN_CCXT_LIMIT = 720
SLEEP_SEC = 0.5


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "ggtrader"),
        password=os.getenv("DB_PASS", "ggtrader"),
        dbname=os.getenv("DB_NAME", "ggtrader"),
    )


def fill_gap(
    conn: psycopg2.extensions.connection,
    start: datetime,
    end: datetime,
    symbol: str = "BTC-USD",
    ccxt_symbol: str = "BTC/USD",
    interval: str = "1h",
) -> int:
    k = ccxt.kraken({"enableRateLimit": True})
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bar_ms = 3600 * 1000
    inserted = 0
    while cursor_ms < end_ms:
        bars = k.fetch_ohlcv(
            ccxt_symbol, timeframe=interval, since=cursor_ms, limit=KRAKEN_CCXT_LIMIT
        )
        if not bars:
            cursor_ms += KRAKEN_CCXT_LIMIT * bar_ms
            time.sleep(SLEEP_SEC)
            continue
        rows = [
            (
                datetime.fromtimestamp(b[0] / 1000, tz=timezone.utc).replace(tzinfo=None),
                symbol,
                interval,
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
        last_ms = bars[-1][0]
        cursor_ms = last_ms + bar_ms
        time.sleep(SLEEP_SEC)
        print(
            f"  ...{symbol} {interval} through {datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc)} (+{len(bars)})"
        )
    return inserted


if __name__ == "__main__":
    conn = _connect()
    print("Gap 1: 2022-03-23 → 2022-12-31")
    n1 = fill_gap(
        conn, datetime(2022, 3, 23, tzinfo=timezone.utc), datetime(2023, 1, 1, tzinfo=timezone.utc)
    )
    print(f"  inserted (may be deduped): {n1}")
    print("Gap 2: 2026-02-23 → today")
    n2 = fill_gap(conn, datetime(2026, 2, 23, tzinfo=timezone.utc), datetime.now(tz=timezone.utc))
    print(f"  inserted (may be deduped): {n2}")
    conn.close()
