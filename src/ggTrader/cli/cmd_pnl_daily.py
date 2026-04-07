"""CLI Command: ggt pnl-daily — generate daily PnL report and push to notifiers."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path


def register_pnl_daily_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pnl-daily",
        help="Generate a daily PnL report and push to Telegram/Discord",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Window start date (YYYY-MM-DD UTC). Default: 24h before --until.",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Window end date (YYYY-MM-DD UTC). Default: now.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/live",
        help="TradeTracker data directory (default: data/live)",
    )
    parser.add_argument(
        "--positions-file",
        type=str,
        default="data/active_positions.json",
        help="Path to active_positions.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path (default: results/reports/daily_<date>.md)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip sending to Telegram/Discord (write local file only)",
    )
    parser.add_argument(
        "--print",
        dest="print_to_stdout",
        action="store_true",
        help="Also print the markdown report to stdout",
    )


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        return d.replace(tzinfo=timezone.utc)
    except ValueError:
        raise SystemExit(f"Invalid date '{s}' — expected YYYY-MM-DD")


def run_pnl_daily(args: argparse.Namespace) -> None:
    from ggTrader.utils.notifier import TelegramNotifier, build_notifiers_from_env
    from ggTrader.utils.pnl_report_builder import (
        build_daily_pnl_report,
        build_daily_pnl_summary_text,
    )

    # Load .env if present so notifier env vars are visible
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    until = _parse_date(args.until) or datetime.now(timezone.utc)
    since = _parse_date(args.since) or (until - timedelta(days=1))

    print(f"Building PnL report for {since.date()} → {until.date()}...")

    # Full markdown report (file output, Discord embed-friendly)
    report_md = build_daily_pnl_report(
        data_dir=args.data_dir,
        active_positions_path=args.positions_file,
        since=since,
        until=until,
    )

    # Plain-text summary (Telegram-friendly — no tables, no markdown syntax)
    summary_text = build_daily_pnl_summary_text(
        data_dir=args.data_dir,
        active_positions_path=args.positions_file,
        since=since,
        until=until,
    )

    # Save markdown file to disk
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("results/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"daily_{until.strftime('%Y%m%d')}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md)
    print(f"Report saved to: {out_path}")

    if args.print_to_stdout:
        print()
        print(summary_text)
        print()
        print("--- Full markdown ---")
        print(report_md)

    # Push to notifiers
    if not args.no_notify:
        notifiers = build_notifiers_from_env()
        if not notifiers:
            print("[Notify] No notification channels configured — local file only.")
            return
        for n in notifiers:
            # Telegram: send plain-text summary inline (markdown tables don't render).
            # Other channels (Discord) get the full markdown.
            if isinstance(n, TelegramNotifier):
                # Disable Telegram's markdown parser so emojis/symbols pass through cleanly
                original_mode = n.parse_mode
                n.parse_mode = ""
                ok = n.send_text(summary_text)
                n.parse_mode = original_mode
                # Also attach the full markdown file for the detailed view
                if ok and out_path.exists():
                    _send_telegram_document(n, out_path, caption="Full report")
            else:
                ok = n.send_text(report_md)
            print(f"[Notify] {n.name}: {'sent' if ok else 'FAILED'}")
    else:
        print("[Notify] --no-notify set — skipping push.")


def _send_telegram_document(notifier, file_path: Path, caption: str = "") -> bool:
    """Send a file as a Telegram document attachment.

    Inline helper since the notifier abstraction's send_photo is for images;
    sendDocument is the right endpoint for arbitrary files like .md reports.
    """
    import requests

    try:
        url = f"https://api.telegram.org/bot{notifier.bot_token}/sendDocument"
        with open(file_path, "rb") as f:
            r = requests.post(
                url,
                data={"chat_id": notifier.chat_id, "caption": caption[:1024]},
                files={"document": (file_path.name, f, "text/markdown")},
                timeout=30,
            )
        return r.ok
    except Exception:
        return False
