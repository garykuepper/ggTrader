#!/usr/bin/env python
"""One-time S&P 500 equity OHLCV backfill into TimescaleDB.

Run once (takes ~5-10 min for ~600 symbols). After this, equity lab runs
hit the DB cache instead of downloading live from yfinance.

Usage:
    source .venv/bin/activate
    python scripts/equity_backfill.py [--start 2000-01-01] [--batch 50]
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

import pandas as pd

from ggTrader.data.core.index_constituents import all_members_between, normalize_yf_ticker
from ggTrader.lab.data import fetch_stock_ohlcv


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill S&P 500 OHLCV into TimescaleDB.")
    p.add_argument("--start", default="2000-01-01", help="History start date (default: 2000-01-01)")
    p.add_argument("--batch", type=int, default=50, help="Symbols per yfinance batch (default: 50)")
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp.now(tz="UTC").normalize()

    # All S&P 500 members from start to today (PIT union).
    members = sorted({normalize_yf_ticker(t) for t in all_members_between(start_ts, end_ts)})
    # Always include SPY (benchmark).
    if "SPY" not in members:
        members = ["SPY"] + members

    print(f"Backfilling {len(members)} symbols from {args.start} to {end_ts.date()}")
    print(f"Batch size: {args.batch}")

    total_batches = (len(members) + args.batch - 1) // args.batch
    for i in range(0, len(members), args.batch):
        batch = members[i : i + args.batch]
        batch_num = i // args.batch + 1
        print(f"\n[{batch_num}/{total_batches}] {batch[0]}..{batch[-1]} ({len(batch)} symbols)")
        t0 = time.time()
        try:
            fetch_stock_ohlcv(batch, start=args.start, use_db_cache=True)
            print(f"  done in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"  ERROR: {exc!r} — skipping batch")

    print(f"\nBackfill complete: {len(members)} symbols processed.")


if __name__ == "__main__":
    main()
