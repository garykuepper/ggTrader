#!/usr/bin/env python
"""Equity OHLCV backfill into TimescaleDB for any universe.

Run once per universe (takes ~5-10 min). After this, equity lab runs
hit the DB cache instead of downloading live from yfinance.

Usage:
    source .venv/bin/activate
    python scripts/equity_backfill.py [--universe sp500] [--start 2000-01-01] [--batch 50]
    python scripts/equity_backfill.py --universe midcap400 --start 2018-01-01
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

import pandas as pd

from ggTrader.data.core.index_constituents import (
    normalize_yf_ticker,
    universe_all_between,
)
from ggTrader.lab.data import fetch_stock_ohlcv


def resolve_symbols(
    universe: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, benchmark: str
) -> list[str]:
    """All members of `universe` across [start, end] plus the benchmark, normalized + sorted."""
    members = {normalize_yf_ticker(t) for t in universe_all_between(universe, start_ts, end_ts)}
    members.add(normalize_yf_ticker(benchmark))
    return sorted(members)


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill equity OHLCV into TimescaleDB.")
    p.add_argument("--start", default="2000-01-01", help="History start date (default: 2000-01-01)")
    p.add_argument("--batch", type=int, default=50, help="Symbols per yfinance batch (default: 50)")
    p.add_argument("--universe", default="sp500", help="Universe to backfill (default: sp500)")
    p.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark ticker to include (default: SPY for sp500, MDY for midcap400)",
    )
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp.now(tz="UTC").normalize()

    benchmark = args.benchmark or ("MDY" if args.universe == "midcap400" else "SPY")
    members = resolve_symbols(args.universe, start_ts, end_ts, benchmark)

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
