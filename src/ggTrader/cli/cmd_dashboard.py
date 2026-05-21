"""CLI Command: ggt dashboard — view live trading performance."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from ggTrader.core.trade_tracker import TradeTracker


def register_dashboard_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("dashboard", help="View live trading performance")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync trade history from Kraken before displaying",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Sync trades since date (YYYY-MM-DD). Implies --sync.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for HTML charts (default: data/live/dashboard)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Print summary only, skip chart generation",
    )


def run_dashboard(args: argparse.Namespace) -> None:
    from ggTrader.core.dashboard_charts import (
        create_cumulative_fees,
        create_cumulative_pnl,
        create_equity_curve,
        create_pnl_by_symbol,
        create_pnl_per_trade,
        create_summary_gauges,
        save_chart,
    )
    from ggTrader.core.trade_tracker import TradeTracker

    tracker = TradeTracker(data_dir="data/live")

    # --sync / --since: pull trades from broker
    if args.sync or args.since:
        _sync_from_exchange(tracker, args.since)

    # Load data
    closes = tracker.get_closed_positions()
    balances = tracker.get_balance_history()
    stats = tracker.compute_summary_stats()

    # Print console summary
    _print_summary(stats, closes, balances)

    # Generate charts
    if not args.no_plots:
        output_dir = Path(args.output or "data/live/dashboard")
        output_dir.mkdir(parents=True, exist_ok=True)

        charts = {
            "equity_curve": create_equity_curve(balances),
            "pnl_per_trade": create_pnl_per_trade(closes),
            "cumulative_pnl": create_cumulative_pnl(closes),
            "cumulative_fees": create_cumulative_fees(closes),
            "summary_gauges": create_summary_gauges(stats),
            "pnl_by_symbol": create_pnl_by_symbol(closes),
        }

        saved = 0
        for name, fig in charts.items():
            if fig is not None:
                save_chart(fig, str(output_dir / name))
                saved += 1

        if saved:
            print(f"\n  Charts saved to: {output_dir}/")
            print(f"  ({saved} chart(s) — open .html files for interactive view)")
        else:
            print("\n  No charts generated (no data yet).")

    print(f"{'=' * 52}\n")


def _sync_from_exchange(tracker: TradeTracker, since_date: str | None) -> None:
    """Initialize CCXT exchange and sync trades."""
    from dotenv import load_dotenv

    load_dotenv()

    from ggTrader.data.live.cached_loader import CachedExchangeLoader

    exchange_id = os.getenv("EXCHANGE", "kraken").lower()
    loader = CachedExchangeLoader(exchange_id=exchange_id)
    exchange = loader.exchange
    if not exchange.apiKey:
        if exchange_id == "binanceus":
            exchange.apiKey = os.getenv("BINANCE_API_LIVE_KEY")
            exchange.secret = os.getenv("BINANCE_SECRET_LIVE_KEY")
        else:
            exchange.apiKey = os.getenv("KRAKEN_KEY")
            exchange.secret = os.getenv("KRAKEN_SECRET")

    since_ts = None
    if since_date:
        try:
            dt = datetime.strptime(since_date, "%Y-%m-%d")
            since_ts = int(dt.timestamp() * 1000)
        except ValueError:
            print(f"  Invalid date format: {since_date} (expected YYYY-MM-DD)")
            return

    print(f"  Syncing trades from {exchange_id.upper()}...")
    new_count = tracker.sync_from_kraken(exchange, since_timestamp=since_ts)
    print(f"  Synced {new_count} new trade(s).\n")


def _print_summary(stats: dict, closes, balances) -> None:
    """Print formatted performance summary to console."""
    print(f"\n{'=' * 52}")
    print("  ggTrader Live Performance Dashboard")
    print(f"{'=' * 52}")

    if stats.get("first_snapshot") and stats.get("latest_snapshot"):
        print(
            f"  Period:          {stats['first_snapshot'][:10]} -> {stats['latest_snapshot'][:10]}"
        )

    if stats.get("current_balance") is not None:
        print(f"  Account Value:   ${stats['current_balance']:,.2f}")

    print("\n  --- P&L ---")
    print(f"  Gross P&L:       ${stats['total_gross_pnl']:+,.2f}")
    print(f"  Total Fees:      ${stats['total_fees']:,.2f}")
    print(f"  Net P&L:         ${stats['total_net_pnl']:+,.2f}")

    print("\n  --- Trades ---")
    print(f"  Total Trades:    {stats['total_trades']}")
    if stats["total_trades"] > 0:
        print(
            f"  Win Rate:        {stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L)"
        )
        print(f"  Avg Win:         ${stats['avg_win']:+,.2f}")
        print(f"  Avg Loss:        ${stats['avg_loss']:+,.2f}")
        pf = stats["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  Profit Factor:   {pf_str}")
        print(f"  Best Trade:      ${stats['best_trade_pnl']:+,.2f} ({stats['best_trade_symbol']})")
        print(
            f"  Worst Trade:     ${stats['worst_trade_pnl']:+,.2f} ({stats['worst_trade_symbol']})"
        )
    else:
        print("  (no closed positions yet)")
