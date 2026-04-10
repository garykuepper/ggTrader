"""CLI Command: ``ggt repair`` — reconcile local trade CSVs against Kraken.

This is the manual recovery counterpart to the auto-sync that runs at the
start of ``ggt pnl-daily``. It pulls the full trade history from Kraken via
CCXT, dedupes by ``order_id`` against ``trade_log.csv``, and rebuilds
``position_closes.csv`` from scratch via FIFO buy/sell pairing.

Use cases:

- One-off cleanup after a known incident (e.g. the bot crashed mid-sell).
- Backfilling history when starting tracking on an account that already has
  trades on Kraken.
- Diagnosing whether the local CSVs and Kraken history actually agree —
  ``--dry-run`` reports counts without writing anything.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone


def register_repair_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "repair",
        help="Sync trade history from Kraken and rebuild position_closes.csv",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/live",
        help="TradeTracker data directory (default: data/live)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help=(
            "Only pull trades newer than this date (YYYY-MM-DD UTC). "
            "Default: pull the entire history (sync_from_kraken is incremental "
            "via order_id deduplication, so this is safe)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report what would change without writing anything. Compares "
            "Kraken trade count against local CSV count and prints the gap."
        ),
    )


def _parse_since(s: str | None) -> int | None:
    if not s:
        return None
    try:
        d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise SystemExit(f"Invalid --since '{s}' — expected YYYY-MM-DD") from e
    return int(d.timestamp() * 1000)


def run_repair(args: argparse.Namespace) -> None:
    from ggTrader.core.trade_tracker import TradeTracker
    from ggTrader.data.live.cached_loader import CachedExchangeLoader

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    print("Connecting to Kraken...")
    loader = CachedExchangeLoader(exchange_id="kraken")
    exchange = loader.exchange
    if not exchange.apiKey:
        exchange.apiKey = os.getenv("KRAKEN_KEY")
        exchange.secret = os.getenv("KRAKEN_SECRET")
    if not exchange.apiKey or not exchange.secret:
        raise SystemExit(
            "ERROR: Kraken API keys missing. Set KRAKEN_KEY and KRAKEN_SECRET in .env."
        )

    tracker = TradeTracker(data_dir=args.data_dir)
    since_ms = _parse_since(args.since)

    if args.dry_run:
        print("[dry-run] Counting trades on Kraken since "
              f"{args.since or 'epoch'}...")
        local_log = tracker.get_trade_log()
        local_count = len(local_log)
        local_ids: set[str] = set()
        if not local_log.empty and "order_id" in local_log.columns:
            local_ids = set(local_log["order_id"].astype(str))

        # Pull a sample window to estimate the gap. We don't loop all 500-page
        # batches in dry-run mode because that defeats the purpose of "fast
        # diagnostic"; the user can run without --dry-run if they want a full
        # reconciliation.
        try:
            kraken_trades = exchange.fetch_my_trades(
                symbol=None, since=since_ms or 0, limit=500
            )
        except Exception as e:
            raise SystemExit(f"Kraken fetch_my_trades failed: {e!r}") from e

        kraken_ids = {
            str(t.get("order", t.get("id", ""))) for t in kraken_trades
        }
        new_ids = kraken_ids - local_ids
        print(f"[dry-run] Local trade_log.csv rows: {local_count}")
        print(f"[dry-run] Kraken returned: {len(kraken_trades)} trades "
              f"(may be capped at 500 — run without --dry-run for full sync)")
        print(f"[dry-run] Trades on Kraken not in local log: {len(new_ids)}")
        if new_ids:
            print(f"[dry-run] First 10 missing order IDs: "
                  f"{sorted(new_ids)[:10]}")
        return

    print(f"Syncing trades from Kraken since {args.since or 'epoch'}...")
    new_count = tracker.sync_from_kraken(exchange, since_timestamp=since_ms)
    print(f"[Sync] {new_count} new trade(s) added to trade_log.csv")
    if new_count:
        print("[Sync] position_closes.csv rebuilt via FIFO matching")
    else:
        print("[Sync] No new trades — local CSVs already in sync with Kraken")

    summary = tracker.compute_summary_stats()
    print()
    print("Post-repair summary:")
    print(f"  Total closed trades: {summary['total_trades']}")
    print(f"  Wins / Losses:       {summary['wins']} / {summary['losses']}")
    print(f"  Total net PnL:       ${summary['total_net_pnl']:+.2f}")
    print(f"  Total fees:          ${summary['total_fees']:.2f}")
