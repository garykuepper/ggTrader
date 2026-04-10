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
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help=(
            "Skip the Kraken auto-sync that runs at the start of the report. "
            "By default, the report syncs trade history from Kraken first so "
            "any trades the live trader skipped (e.g. emergency rollbacks, "
            "fill-price extraction failures) get healed before the metrics "
            "are computed. Use --no-sync for fast offline runs only."
        ),
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
        build_daily_pnl_summary_html,
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

    # Auto-sync from Kraken before reading any local CSV. This is the safety
    # net for the entire trade-recording layer: any sells that the live trader
    # skipped (Bug A emergency rollback, Bug B reconcile-skip, Bug C untracked
    # exit) are pulled from Kraken's authoritative trade history and the
    # position_closes.csv is rebuilt via FIFO matching. Failures degrade
    # gracefully: a STALE banner is added to the report and the existing CSVs
    # are used as-is.
    stale_warning: str | None = None
    if not args.no_sync:
        stale_warning = _autosync_from_kraken(args.data_dir)
    else:
        print("[Sync] --no-sync set — skipping Kraken sync.")

    print(f"Building PnL report for {since.date()} → {until.date()}...")

    # Full markdown report (file output, Discord embed-friendly)
    report_md = build_daily_pnl_report(
        data_dir=args.data_dir,
        active_positions_path=args.positions_file,
        since=since,
        until=until,
        stale_warning=stale_warning,
    )

    # Plain-text summary (kept for stdout printing and as a fallback if HTML
    # rendering ever fails on Telegram's side).
    summary_text = build_daily_pnl_summary_text(
        data_dir=args.data_dir,
        active_positions_path=args.positions_file,
        since=since,
        until=until,
        stale_warning=stale_warning,
    )

    # HTML summary with monospace tables — what we actually push to Telegram.
    summary_html = build_daily_pnl_summary_html(
        data_dir=args.data_dir,
        active_positions_path=args.positions_file,
        since=since,
        until=until,
        stale_warning=stale_warning,
    )

    # Save markdown file to disk. The default filename uses YYYYMMDD so the 8am
    # cron run produces a clean ``daily_<date>.md`` archive entry. If a file with
    # the same date already exists (e.g. a manual mid-day re-run after the cron
    # already wrote one), append a HHMMSS suffix so we never clobber history.
    # Also maintain a ``latest.md`` symlink for quick access.
    if args.output:
        out_path = Path(args.output)
    else:
        from zoneinfo import ZoneInfo

        out_dir = Path("results/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use Pacific time for the filename so it matches the date shown
        # inside the report (which also uses America/Los_Angeles).
        local_until = until.astimezone(ZoneInfo("America/Los_Angeles"))
        date_str = local_until.strftime("%Y%m%d")
        out_path = out_dir / f"daily_{date_str}.md"
        if out_path.exists():
            time_str = local_until.strftime("%H%M%S")
            out_path = out_dir / f"daily_{date_str}_{time_str}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md)
    print(f"Report saved to: {out_path}")

    # Update ``latest.md`` symlink (best-effort — falls back to a copy if the
    # filesystem doesn't support symlinks, e.g. some Docker volume drivers).
    if not args.output:
        latest = out_path.parent / "latest.md"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(out_path.name)
        except OSError:
            try:
                latest.write_text(report_md)
            except OSError:
                pass

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
            # Telegram: send the HTML summary inline (monospace tables via <pre>).
            # Other channels (Discord) get the full markdown report.
            if isinstance(n, TelegramNotifier):
                original_mode = n.parse_mode
                n.parse_mode = "HTML"
                ok = n.send_text(summary_html)
                n.parse_mode = original_mode
                # Also attach the full markdown file for the detailed view
                if ok and out_path.exists():
                    _send_telegram_document(n, out_path, caption="Full report")
            else:
                ok = n.send_text(report_md)
            print(f"[Notify] {n.name}: {'sent' if ok else 'FAILED'}")
    else:
        print("[Notify] --no-notify set — skipping push.")


def _autosync_from_kraken(data_dir: str) -> str | None:
    """Pull recent trades from Kraken and rebuild position_closes.csv.

    Returns ``None`` on success, or a short human-readable error string on
    failure (e.g. ``"Kraken unreachable: ..."``). The caller passes the
    returned string into the report builders as ``stale_warning`` so the
    user sees a clear marker at the top of the daily report when the data
    may be behind reality.

    The sync is idempotent: ``TradeTracker.sync_from_kraken`` dedupes by
    order_id and rebuilds the position_closes table via FIFO matching, so
    repeated calls are safe.
    """
    import os

    from ggTrader.core.trade_tracker import TradeTracker
    from ggTrader.data.live.cached_loader import CachedExchangeLoader

    try:
        loader = CachedExchangeLoader(exchange_id="kraken")
        exchange = loader.exchange
        if not exchange.apiKey:
            exchange.apiKey = os.getenv("KRAKEN_KEY")
            exchange.secret = os.getenv("KRAKEN_SECRET")
        if not exchange.apiKey or not exchange.secret:
            msg = "KRAKEN_KEY/SECRET missing — cannot auto-sync"
            print(f"[Sync] WARNING: {msg}")
            return msg

        tracker = TradeTracker(data_dir=data_dir)
        new_count = tracker.sync_from_kraken(exchange)
        print(f"[Sync] {new_count} new trade(s) pulled from Kraken")
        return None
    except Exception as e:
        msg = f"Kraken sync failed ({e!r}) — report uses local CSV as-is"
        print(f"[Sync] WARNING: {msg}")
        return msg


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
