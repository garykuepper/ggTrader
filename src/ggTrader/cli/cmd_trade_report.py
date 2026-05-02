"""CLI Command: ggt trade-report — summarise live trade performance."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def register_trade_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "trade-report",
        help="Summarise closed live trades from data/live/position_closes.csv",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Window start date YYYY-MM-DD (default: 30 days before now)",
    )
    parser.add_argument(
        "--group-by",
        choices=["exit_reason", "symbol", "week"],
        default="exit_reason",
        help="Grouping dimension (default: exit_reason)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to position_closes.csv (default: data/live/position_closes.csv)",
    )


def run_trade_report(args: argparse.Namespace) -> None:
    import pandas as pd

    from ggTrader.utils.paths import find_project_root

    csv_path = Path(args.csv) if args.csv else find_project_root() / "data" / "live" / "position_closes.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No closed trades recorded.")
        return

    df["close_timestamp"] = pd.to_datetime(
        df["close_timestamp"], utc=True, format="mixed", errors="coerce"
    )
    if args.since:
        since = pd.Timestamp(args.since, tz="UTC")
    else:
        since = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=30))
    df = df[df["close_timestamp"] >= since].copy()
    if df.empty:
        print(f"No closed trades since {since.date()}.")
        return

    if args.group_by == "week":
        df["_group"] = df["close_timestamp"].dt.tz_convert(None).dt.to_period("W").astype(str)
    else:
        df["_group"] = df[args.group_by].fillna("(unknown)")

    def _agg(g: "pd.DataFrame") -> "pd.Series":
        wins = (g["net_pnl"] > 0).sum()
        losses = (g["net_pnl"] <= 0).sum()
        n = len(g)
        loss_pcts = g.loc[g["net_pnl"] <= 0, "pnl_pct"]
        return pd.Series({
            "n": n,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / n * 100.0) if n else 0.0,
            "sum_net_pnl": g["net_pnl"].sum(),
            "avg_pnl_pct": g["pnl_pct"].mean(),
            "median_hold_h": g["hold_duration_hours"].median(),
            "p90_loss_pct": loss_pcts.quantile(0.10) if not loss_pcts.empty else 0.0,
        })

    summary = df.groupby("_group", sort=True).apply(_agg, include_groups=False)
    totals = _agg(df).rename("TOTAL")
    summary = pd.concat([summary, totals.to_frame().T])

    print(f"\nWindow: {since.date()} → now   ({len(df)} closed trades)\n")
    print(summary.to_string(
        formatters={
            "n": "{:>4.0f}".format,
            "wins": "{:>4.0f}".format,
            "losses": "{:>4.0f}".format,
            "win_rate": "{:>6.1f}%".format,
            "sum_net_pnl": "${:>+8.2f}".format,
            "avg_pnl_pct": "{:>+6.2f}%".format,
            "median_hold_h": "{:>7.1f}h".format,
            "p90_loss_pct": "{:>+6.2f}%".format,
        }
    ))
    print()
