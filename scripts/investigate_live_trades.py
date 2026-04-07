#!/usr/bin/env python3
"""Forensic investigation of live trading: Kraken history vs local CSV.

Pulls actual order/trade history from Kraken via CCXT and cross-references
against the local TradeTracker CSVs to identify:
- Whether trailing-stop / OCO orders were actually placed
- Whether they triggered (and at what price)
- Discrepancies between Kraken's realized PnL and local CSV PnL
- Untracked orders (in Kraken history but missing from local CSV)
- Bogus exit prices in the local CSV (e.g. $0.00 fill)

Read-only: makes no orders or modifications. Safe to run with the live
trader paused or active.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running directly: python scripts/investigate_live_trades.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ggTrader.core.trade_tracker import TradeTracker  # noqa: E402
from ggTrader.data.live.cached_loader import CachedExchangeLoader  # noqa: E402


def _build_exchange() -> Any:
    """Construct a CCXT Kraken instance with API keys from .env."""
    from dotenv import load_dotenv

    load_dotenv()
    loader = CachedExchangeLoader(exchange_id="kraken")
    exchange = loader.exchange
    if not exchange.apiKey:
        exchange.apiKey = os.getenv("KRAKEN_KEY")
        exchange.secret = os.getenv("KRAKEN_SECRET")
    if not exchange.apiKey or not exchange.secret:
        raise RuntimeError(
            "Kraken API keys not found. Set KRAKEN_KEY and KRAKEN_SECRET in .env."
        )
    return exchange


def fetch_kraken_orders(exchange: Any, since_ms: int) -> list[dict]:
    """Fetch all closed orders from Kraken since the given timestamp (ms)."""
    all_orders: list[dict] = []
    since = since_ms
    while True:
        try:
            orders = exchange.fetch_closed_orders(symbol=None, since=since, limit=500)
        except Exception as e:
            print(f"  [warn] fetch_closed_orders failed: {e!r}")
            break
        if not orders:
            break
        all_orders.extend(orders)
        if len(orders) < 500:
            break
        since = orders[-1]["timestamp"] + 1
        time.sleep(1)
    return all_orders


def fetch_kraken_trades(exchange: Any, since_ms: int) -> list[dict]:
    """Fetch all my trades (fills) from Kraken since the given timestamp (ms)."""
    all_trades: list[dict] = []
    since = since_ms
    while True:
        try:
            trades = exchange.fetch_my_trades(symbol=None, since=since, limit=500)
        except Exception as e:
            print(f"  [warn] fetch_my_trades failed: {e!r}")
            break
        if not trades:
            break
        all_trades.extend(trades)
        if len(trades) < 500:
            break
        since = trades[-1]["timestamp"] + 1
        time.sleep(1)
    return all_trades


def compute_kraken_realized_pnl(trades: list[dict]) -> dict[str, dict]:
    """Compute realized PnL per symbol from Kraken trade fills.

    Uses FIFO matching: each sell consumes the oldest open buy.
    Returns: {symbol: {"realized_pnl": float, "trades": int, "wins": int, "losses": int}}
    """
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        sym = t.get("symbol", "").replace("/", "-")
        by_symbol[sym].append(t)

    results: dict[str, dict] = {}
    for sym, ts in by_symbol.items():
        ts_sorted = sorted(ts, key=lambda x: x.get("timestamp", 0))
        open_buys: list[dict] = []
        realized = 0.0
        wins = 0
        losses = 0
        n_round_trips = 0
        for t in ts_sorted:
            side = t.get("side")
            amt = float(t.get("amount", 0) or 0)
            price = float(t.get("price", 0) or 0)
            fee = float((t.get("fee") or {}).get("cost", 0) or 0)
            if side == "buy":
                open_buys.append({"amount": amt, "price": price, "fee": fee})
            elif side == "sell":
                remaining = amt
                trip_pnl = -fee
                while remaining > 0 and open_buys:
                    b = open_buys[0]
                    consume = min(remaining, b["amount"])
                    trip_pnl += (price - b["price"]) * consume
                    # Pro-rated entry fee
                    trip_pnl -= b["fee"] * (consume / b["amount"]) if b["amount"] else 0
                    b["amount"] -= consume
                    b["fee"] *= (b["amount"] / (b["amount"] + consume)) if (b["amount"] + consume) else 0
                    remaining -= consume
                    if b["amount"] <= 1e-12:
                        open_buys.pop(0)
                if remaining > 1e-12:
                    # Sell with no matching buy (pre-existing position)
                    trip_pnl += price * remaining
                realized += trip_pnl
                n_round_trips += 1
                if trip_pnl > 0:
                    wins += 1
                else:
                    losses += 1
        results[sym] = {
            "realized_pnl": round(realized, 4),
            "round_trips": n_round_trips,
            "wins": wins,
            "losses": losses,
            "buys": sum(1 for t in ts_sorted if t.get("side") == "buy"),
            "sells": sum(1 for t in ts_sorted if t.get("side") == "sell"),
        }
    return results


def categorize_orders(orders: list[dict]) -> dict[str, list[dict]]:
    """Group orders by type for inspection."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for o in orders:
        otype = o.get("type") or o.get("info", {}).get("descr", {}).get("ordertype", "unknown")
        groups[otype].append(o)
    return groups


def _safe_get_avg(order: dict) -> float | None:
    """Try multiple fallbacks to extract a meaningful fill price from an order."""
    avg = order.get("average")
    if avg:
        return float(avg)
    price = order.get("price")
    if price and order.get("status") == "closed":
        return float(price)
    trades = order.get("trades") or []
    if trades:
        total_amt = sum(float(t.get("amount", 0) or 0) for t in trades)
        total_val = sum(
            float(t.get("amount", 0) or 0) * float(t.get("price", 0) or 0) for t in trades
        )
        if total_amt > 0:
            return total_val / total_amt
    return None


def build_report(
    days: int,
    kraken_orders: list[dict],
    kraken_trades: list[dict],
    csv_closes: pd.DataFrame,
    csv_log: pd.DataFrame,
) -> str:
    lines: list[str] = []
    now = datetime.now(timezone.utc)
    lines.append(f"# Live Trade Investigation — {now.date()}")
    lines.append("")
    lines.append(f"**Window**: last {days} days (since {(now - timedelta(days=days)).date()})")
    lines.append("")

    # ── Kraken summary ────────────────────────────────────────────────────
    lines.append("## Kraken Activity (source of truth)")
    lines.append("")
    lines.append(f"- Total closed orders: **{len(kraken_orders)}**")
    lines.append(f"- Total trade fills: **{len(kraken_trades)}**")

    by_type = categorize_orders(kraken_orders)
    lines.append(f"- Order types: {dict({k: len(v) for k, v in by_type.items()})}")
    lines.append("")

    # Realized PnL by symbol from Kraken trades (ground truth)
    pnl_by_sym = compute_kraken_realized_pnl(kraken_trades)
    total_realized = sum(v["realized_pnl"] for v in pnl_by_sym.values())
    total_trips = sum(v["round_trips"] for v in pnl_by_sym.values())
    total_wins = sum(v["wins"] for v in pnl_by_sym.values())
    total_losses = sum(v["losses"] for v in pnl_by_sym.values())
    lines.append("### Kraken realized PnL (FIFO from trade fills)")
    lines.append("")
    lines.append(f"- **Total realized PnL: ${total_realized:+.2f}**")
    lines.append(f"- Round trips: {total_trips} ({total_wins}W / {total_losses}L)")
    win_rate = (total_wins / total_trips * 100.0) if total_trips else 0.0
    lines.append(f"- Win rate: {win_rate:.1f}%")
    lines.append("")
    lines.append("| Symbol | Round Trips | W | L | Realized PnL |")
    lines.append("|---|---|---|---|---|")
    for sym, v in sorted(pnl_by_sym.items(), key=lambda x: x[1]["realized_pnl"]):
        if v["round_trips"] == 0:
            continue
        lines.append(
            f"| {sym} | {v['round_trips']} | {v['wins']} | {v['losses']} | ${v['realized_pnl']:+.2f} |"
        )
    lines.append("")

    # ── Stop / OCO orders ─────────────────────────────────────────────────
    lines.append("## Stop / OCO Order Audit")
    lines.append("")
    stop_types = ["trailing-stop", "stop-loss", "stop-loss-limit", "take-profit"]
    stop_orders = [
        o for o in kraken_orders
        if any(s in str(o.get("type", "")).lower() for s in stop_types)
        or any(
            s in str((o.get("info") or {}).get("descr", {}).get("ordertype", "")).lower()
            for s in stop_types
        )
    ]
    lines.append(f"- Stop-style orders found in Kraken history: **{len(stop_orders)}**")
    if not stop_orders:
        lines.append("")
        lines.append(
            "⚠️ **WARNING**: No stop or trailing-stop orders found in the Kraken history "
            "for this period. This means stops were either never placed or all expired/cancelled. "
            "If positions still closed at large losses, the bot was relying on strategy_signal "
            "exits at 4h candles — which can produce huge gap losses on volatile coins."
        )
    else:
        lines.append("")
        lines.append("| Time | Symbol | Type | Status | Avg Fill | Amount |")
        lines.append("|---|---|---|---|---|---|")
        for o in sorted(stop_orders, key=lambda x: x.get("timestamp") or 0):
            ts = datetime.fromtimestamp(
                (o.get("timestamp") or 0) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
            sym = o.get("symbol", "").replace("/", "-")
            typ = o.get("type") or "?"
            status = o.get("status") or "?"
            avg = _safe_get_avg(o)
            avg_str = f"${avg:.6g}" if avg else "—"
            amt = o.get("amount") or 0
            lines.append(f"| {ts} | {sym} | {typ} | {status} | {avg_str} | {amt} |")
    lines.append("")

    # ── CSV vs Kraken comparison ──────────────────────────────────────────
    lines.append("## Local CSV vs Kraken Cross-Reference")
    lines.append("")
    if csv_closes.empty:
        lines.append("- Local `position_closes.csv` is empty.")
    else:
        # Filter to investigation window
        cutoff = pd.Timestamp(now - timedelta(days=days))
        csv_closes["close_timestamp"] = pd.to_datetime(csv_closes["close_timestamp"], utc=True)
        recent = csv_closes[csv_closes["close_timestamp"] >= cutoff].copy()
        lines.append(f"- Local closes in window: **{len(recent)}**")
        if not recent.empty:
            csv_total = float(recent["net_pnl"].sum())
            csv_wins = int((recent["net_pnl"] > 0).sum())
            csv_losses = int((recent["net_pnl"] <= 0).sum())
            lines.append(f"- CSV total net PnL: **${csv_total:+.2f}**")
            lines.append(f"- CSV W/L: {csv_wins}W / {csv_losses}L")
            lines.append("")

            # Highlight discrepancy
            diff = total_realized - csv_total
            lines.append(f"### Reconciliation: Kraken vs CSV")
            lines.append("")
            lines.append("| Source | Total Realized PnL |")
            lines.append("|---|---|")
            lines.append(f"| Kraken (FIFO) | ${total_realized:+.2f} |")
            lines.append(f"| Local CSV     | ${csv_total:+.2f} |")
            lines.append(f"| **Difference** | **${diff:+.2f}** |")
            lines.append("")
            if abs(diff) > 5.0:
                lines.append(
                    f"⚠️ **Discrepancy of ${abs(diff):.2f} between Kraken and local CSV.** "
                    "The local tracker is recording inaccurate PnL — likely due to the "
                    "exit-price-falls-back-to-zero bug in `_reconcile_positions`."
                )
            lines.append("")

            # Look for poisoned exit prices
            zero_exits = recent[recent["exit_price"] == 0]
            if not zero_exits.empty:
                lines.append(
                    f"### ⚠️ {len(zero_exits)} CSV records with exit_price = $0.00 (DATA BUG)"
                )
                lines.append("")
                lines.append("| Symbol | Entry | Exit | Net PnL | Reason |")
                lines.append("|---|---|---|---|---|")
                for _, r in zero_exits.iterrows():
                    lines.append(
                        f"| {r['symbol']} | ${r['entry_price']:.6g} | $0.00 | "
                        f"${r['net_pnl']:+.2f} | {r['exit_reason']} |"
                    )
                lines.append("")

            # Per-trade detail
            lines.append("### Closed Positions in Window (CSV)")
            lines.append("")
            lines.append("| Time | Symbol | Entry | Exit | Net PnL | % | Reason |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in recent.sort_values("close_timestamp").iterrows():
                ts = r["close_timestamp"].strftime("%Y-%m-%d %H:%M")
                lines.append(
                    f"| {ts} | {r['symbol']} | ${r['entry_price']:.6g} | "
                    f"${r['exit_price']:.6g} | ${r['net_pnl']:+.2f} | "
                    f"{r['pnl_pct']:+.2f}% | {r['exit_reason']} |"
                )
    lines.append("")

    # ── Findings summary ──────────────────────────────────────────────────
    lines.append("## Key Findings")
    lines.append("")
    findings = []
    if not stop_orders:
        findings.append(
            "**No stop/trailing-stop orders found on Kraken** — bot may not be placing them, "
            "or they're being cancelled before reaching the closed-orders endpoint."
        )
    if csv_closes is not None and not csv_closes.empty:
        cutoff = pd.Timestamp(now - timedelta(days=days))
        csv_closes["close_timestamp"] = pd.to_datetime(csv_closes["close_timestamp"], utc=True)
        recent = csv_closes[csv_closes["close_timestamp"] >= cutoff]
        if not recent.empty:
            zero_count = int((recent["exit_price"] == 0).sum())
            if zero_count:
                findings.append(
                    f"**{zero_count} CSV records have exit_price=$0.00** — confirms the "
                    "`closed_order.get('average', 0)` fallback bug."
                )
            csv_total = float(recent["net_pnl"].sum())
            if abs(total_realized - csv_total) > 5.0:
                findings.append(
                    f"**${abs(total_realized - csv_total):.2f} PnL discrepancy** between "
                    "Kraken (truth) and local CSV — proves CSV PnL is unreliable."
                )
    if not findings:
        findings.append("No major issues detected. Local CSV appears consistent with Kraken.")
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="Days of history to investigate")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path (default: results/investigation/live_trades_<date>.md)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/live",
        help="TradeTracker data directory (default: data/live)",
    )
    args = parser.parse_args()

    print(f"Investigating last {args.days} days of live trading...")
    print()

    print("Connecting to Kraken...")
    exchange = _build_exchange()
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    print(f"Fetching Kraken closed orders since {datetime.fromtimestamp(since_ms/1000, tz=timezone.utc)}...")
    kraken_orders = fetch_kraken_orders(exchange, since_ms)
    print(f"  -> {len(kraken_orders)} orders")

    print("Fetching Kraken trade fills...")
    kraken_trades = fetch_kraken_trades(exchange, since_ms)
    print(f"  -> {len(kraken_trades)} trades")

    print("Loading local CSV tracker data...")
    tracker = TradeTracker(data_dir=args.data_dir)
    csv_closes = tracker.get_closed_positions()
    csv_log = tracker.get_trade_log()
    print(f"  -> {len(csv_closes)} closed positions, {len(csv_log)} log entries")
    print()

    print("Building report...")
    report_md = build_report(args.days, kraken_orders, kraken_trades, csv_closes, csv_log)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("results/investigation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"live_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md)
    print(f"\nReport written to: {out_path}")
    print()
    print("=" * 70)
    print(report_md)


if __name__ == "__main__":
    main()
