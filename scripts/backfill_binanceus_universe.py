"""Backfill Binance.US OHLCV for the latest top-N crypto universe.

Loads the most recent universe_cache entry, iterates all symbols, and calls
``backfill_symbol_interval`` from ``backfill_binanceus``. Symbols not listed on
Binance.US (ccxt raises) are logged and skipped, not fatal.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

load_dotenv()

from backfill_binanceus import INTERVALS, _connect, backfill_symbol_interval  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from ggTrader.utils.config import get_db_connection_string  # noqa: E402


def _latest_top_universe() -> list[str]:
    engine = create_engine(get_db_connection_string())
    with engine.connect() as c:
        row = c.execute(
            text(
                "SELECT symbols FROM universe_cache "
                "WHERE asset_class='crypto' ORDER BY snapshot_date DESC LIMIT 1"
            )
        ).fetchone()
    if row is None:
        return []
    payload = row[0]
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    return [e["symbol"] for e in entries if "symbol" in e]


if __name__ == "__main__":
    syms = _latest_top_universe()
    print(f"Universe size: {len(syms)} symbols")
    print(f"Symbols: {syms}")

    ex = ccxt.binanceus(
        {
            "apiKey": os.getenv("BINANCE_API_LIVE_KEY"),
            "secret": os.getenv("BINANCE_SECRET_LIVE_KEY"),
            "enableRateLimit": True,
        }
    )
    conn = _connect()
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)

    skipped, ok, total_inserted = [], [], 0
    for i, sym in enumerate(syms, 1):
        ccxt_sym = f"{sym}/USD"
        for interval in INTERVALS.keys():
            try:
                n = backfill_symbol_interval(conn, ex, f"{sym}-USD", ccxt_sym, interval, start_date)
                total_inserted += n
                ok.append((sym, interval, n))
                print(f"[{i}/{len(syms)}] {sym}-USD {interval}: +{n}")
            except Exception as e:
                skipped.append((sym, interval, type(e).__name__, str(e)[:80]))
                print(f"[{i}/{len(syms)}] {sym}-USD {interval}: SKIP ({type(e).__name__})")

    conn.close()
    print("\n=== summary ===")
    print(f"ok: {len(ok)} (symbol, interval) tasks, +{total_inserted} rows")
    print(f"skipped: {len(skipped)}")
    for s, i, et, msg in skipped[:20]:
        print(f"  {s}-USD {i}: {et} — {msg}")
