#!/usr/bin/env python3
"""Equity WFO research CLI — per-stock strategy tournament over an index universe.

Replaces stock_wfo_research.py / nasdaq100_wfo_research.py /
russell2000_wfo_research.py / blended_wfo_research.py / stock_research_quick.py.

Usage:
    source .venv/bin/activate
    python -u scripts/equity_wfo_research.py --universe sp500 --quick
    python -u scripts/equity_wfo_research.py --universe sp500 \
        --entries psar_adx,ema_cross --exits atr_trailing --jobs 8

The combined top-N validation printed at the end uses IN-SAMPLE selection
(top-N by scores from the same period) — smoke test only. The honest
out-of-sample estimate comes from scripts/sp500_monthly_walkforward.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY  # noqa: E402
from ggTrader.research.equity_wfo import (  # noqa: E402
    STOCK_BASE_CONFIG,
    fetch_stock_ohlcv,
    grid_books,
    normalize_yf_ticker,
    print_tournament_summary,
    run_combined_validation,
    run_wfo_per_stock,
)

UNIVERSE_SNAPSHOTS = {
    "nasdaq100": "data/universe/nasdaq100_tickers_snapshot_2026-06-09.txt",
    "russell2000": "data/universe/russell2000_tickers_snapshot_2026-06-09.txt",
}

QUICK_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "LLY",
    "AVGO",
    "COST",
    "WMT",
    "MPC",
    "VLO",
    "GE",
    "CAT",
    "SCHW",
    "NFLX",
    "ISRG",
    "LIN",
    "PM",
    "HD",
]


def load_universe(args: argparse.Namespace) -> list[str]:
    if args.tickers_file:
        return [
            line.strip()
            for line in Path(args.tickers_file).read_text().splitlines()
            if line.strip()
        ]
    if args.quick:
        return QUICK_TICKERS
    if args.universe == "sp500":
        import pandas as pd

        from ggTrader.data.core.index_constituents import all_members_between

        end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
        return all_members_between(pd.Timestamp(args.start, tz="UTC"), end)
    snapshot = UNIVERSE_SNAPSHOTS.get(args.universe)
    if snapshot is None:
        raise SystemExit(f"Unknown universe: {args.universe}")
    print(
        f"NOTE: {args.universe} uses a static current-constituents snapshot "
        f"(survivorship-biased); sp500 uses point-in-time membership."
    )
    return [
        line.strip() for line in (PROJECT_ROOT / snapshot).read_text().splitlines() if line.strip()
    ]


def parse_strategies(value: str, registry: dict, kind: str) -> list[str]:
    names = [v.strip() for v in value.split(",") if v.strip()]
    if names == ["all"]:
        return list(registry.keys())
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"Unknown {kind} strategies: {unknown}. Available: {list(registry)}")
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", choices=["sp500", "nasdaq100", "russell2000"], default="sp500")
    p.add_argument("--tickers-file", default=None, help="Explicit ticker list (overrides universe)")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--entries", default="all", help="Comma list of entry strategies, or 'all'")
    p.add_argument("--exits", default="all", help="Comma list of exit strategies, or 'all'")
    p.add_argument("--grid", choices=["coarse", "detailed"], default="detailed")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-ratio", type=float, default=3.0)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--max-position-pct", type=float, default=0.02)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--no-db-cache", action="store_true")
    p.add_argument(
        "--quick",
        action="store_true",
        help="20-ticker subset, 2 splits, psar_adx+ema_cross x atr_trailing",
    )
    args = p.parse_args()

    entries = parse_strategies(args.entries, ENTRY_REGISTRY, "entry")
    exits = parse_strategies(args.exits, EXIT_REGISTRY, "exit")
    n_splits = args.n_splits
    if args.quick:
        n_splits = 2
        if args.entries == "all":
            entries = ["psar_adx", "ema_cross"]
        if args.exits == "all":
            exits = ["atr_trailing"]

    tickers = load_universe(args)
    print("=" * 78)
    print(
        f"Equity WFO research — universe={args.universe} ({len(tickers)} tickers) "
        f"entries={entries} exits={exits} grid={args.grid} splits={n_splits}"
    )
    print("=" * 78)

    ohlcv = fetch_stock_ohlcv(
        [normalize_yf_ticker(t) for t in tickers],
        start=args.start,
        end=args.end,
        use_db_cache=not args.no_db_cache,
        min_coverage=0.5,
    )

    entry_book, exit_book = grid_books(args.grid)
    per_stock = run_wfo_per_stock(
        ohlcv,
        STOCK_BASE_CONFIG,
        entries,
        exits,
        entry_book,
        exit_book,
        n_splits=n_splits,
        test_ratio=args.test_ratio,
        n_jobs=args.jobs,
    )

    print_tournament_summary(per_stock)

    print("\n" + "=" * 78)
    print("COMBINED PORTFOLIO — IN-SAMPLE TOP-N SELECTION (smoke test, NOT an OOS estimate)")
    print("=" * 78)
    stats = run_combined_validation(
        ohlcv,
        per_stock,
        STOCK_BASE_CONFIG,
        max_position_pct=args.max_position_pct,
        top_n=args.top_n,
    )
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
