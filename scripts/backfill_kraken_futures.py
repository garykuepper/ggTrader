"""Run the Phase 4 Kraken Futures backfill.

Pulls PF_XBTUSD hourly OHLCV from listing (2022-03-23) through today, and the
funding-rate window the API currently exposes (~1 year). Logs the actual
window pinned for the run.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

from ggTrader.data.sources.kraken_futures import (
    backfill_funding,
    backfill_perp,
    refresh_views,
)

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
    snapshot_at = datetime.now(tz=timezone.utc)
    print(f"Backfill snapshot pinned at: {snapshot_at.isoformat()}")

    print()
    print("=== PF_XBTUSD perp OHLCV (1h, 2022-03-23 → today) ===")
    t0 = time.time()
    inserted = backfill_perp(
        conn,
        symbol="PF_XBTUSD",
        start=datetime(2022, 3, 23, tzinfo=timezone.utc),
        end=snapshot_at,
        resolution="1h",
    )
    print(f"  rows inserted (deduped): {inserted}  elapsed: {time.time() - t0:.1f}s")

    print()
    print("=== PF_XBTUSD funding rates (fixed ~1y window) ===")
    t0 = time.time()
    n, earliest, latest = backfill_funding(conn, "PF_XBTUSD")
    print(f"  rows inserted (deduped): {n}  elapsed: {time.time() - t0:.1f}s")
    print(f"  pinned window: {earliest.isoformat()} → {latest.isoformat()}")

    print()
    print("=== Refresh materialized views ===")
    t0 = time.time()
    refresh_views(conn)
    print(f"  done in {time.time() - t0:.1f}s")

    conn.close()


if __name__ == "__main__":
    main()
