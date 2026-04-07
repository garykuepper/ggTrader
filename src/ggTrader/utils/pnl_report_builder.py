"""Daily live-trading PnL report builder.

Reads the local TradeTracker CSVs (closed positions, balance snapshots,
trade log) and produces a markdown summary suitable for both file output
and Telegram/Discord push notifications.

Key sections:
  - Executive snapshot (current balance, 24h change, alerts)
  - 24h window stats (closed trades, win rate, realised PnL)
  - All-time stats (Sharpe, Sortino, max DD, profit factor)
  - Open positions (with stored entry prices — relies on the post-fix
    execution_engine writing real fill prices to active_positions.json)
  - Recent closed trades table

Used by ``ggt pnl-daily`` CLI command and the cron wrapper script.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ggTrader.core.trade_tracker import TradeTracker
from ggTrader.utils.live_metrics import (
    compute_consecutive_losses,
    equity_curve_from_balance,
    summarise_window,
)

# Default thresholds for ⚠️ alert callouts. Override via run_config or CLI args.
# balance_floor is the critical intervention level. balance_warning is set equal
# to the floor so any balance dip ≥ floor triggers the critical alert directly.
DEFAULT_ALERT_THRESHOLDS = {
    "balance_floor": 200.0,
    "balance_warning": 200.0,
    "consecutive_loss_threshold": 5,
    "max_drawdown_pct": -25.0,
}


def _fmt_money(x: Optional[float], sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    fmt = "{:+,.2f}" if sign else "{:,.2f}"
    return f"${fmt.format(x)}"


def _fmt_pct(x: Optional[float], sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    fmt = "{:+.2f}%" if sign else "{:.2f}%"
    return fmt.format(x)


def _fmt_float(x: Optional[float], decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{decimals}f}"


def _load_active_positions(path: Path) -> dict[str, dict]:
    """Load active_positions.json. Returns empty dict on missing/unreadable file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _build_alerts(
    snapshot_balance: Optional[float],
    consecutive_losses: int,
    max_dd_pct: float,
    thresholds: dict[str, float],
) -> list[str]:
    alerts: list[str] = []
    if snapshot_balance is not None:
        if snapshot_balance < thresholds["balance_floor"]:
            alerts.append(
                f"🚨 **Balance ${snapshot_balance:.2f} is BELOW the "
                f"${thresholds['balance_floor']:.0f} intervention floor**"
            )
        elif snapshot_balance < thresholds["balance_warning"]:
            alerts.append(
                f"⚠️ Balance ${snapshot_balance:.2f} is below the "
                f"${thresholds['balance_warning']:.0f} warning level"
            )
    if consecutive_losses >= thresholds["consecutive_loss_threshold"]:
        alerts.append(
            f"⚠️ {consecutive_losses} consecutive losing trades "
            f"(threshold: {thresholds['consecutive_loss_threshold']})"
        )
    if max_dd_pct < thresholds["max_drawdown_pct"]:
        alerts.append(
            f"⚠️ Max drawdown {max_dd_pct:.1f}% exceeds threshold "
            f"({thresholds['max_drawdown_pct']:.0f}%)"
        )
    return alerts


def _gather_report_data(
    data_dir: str,
    active_positions_path: str,
    since: datetime,
    until: datetime,
    thresholds: dict[str, float],
) -> dict:
    """Collect all data needed for both markdown and plain-text report builders.

    Returns a dict with: window, alltime, alerts, snapshot_balance, all_consec,
    closes, active, summary_all, since, until.
    """
    tracker = TradeTracker(data_dir=data_dir)
    closes = tracker.get_closed_positions()
    balances = tracker.get_balance_history()
    summary_all = tracker.compute_summary_stats()
    active = _load_active_positions(Path(active_positions_path))

    window = summarise_window(
        closes,
        balances,
        pd.Timestamp(since),
        pd.Timestamp(until),
    )

    alltime: dict[str, Any] = {}
    if balances is not None and not balances.empty:
        equity = equity_curve_from_balance(balances)
        if not equity.empty:
            from ggTrader.utils.live_metrics import (
                compute_calmar_ratio,
                compute_max_drawdown,
                compute_sharpe_ratio,
                compute_sortino_ratio,
                daily_returns_from_equity,
            )

            returns = daily_returns_from_equity(equity)
            if len(returns) >= 2:
                alltime["sharpe"] = compute_sharpe_ratio(returns)
                alltime["sortino"] = compute_sortino_ratio(returns)
            else:
                alltime["sharpe"] = float("nan")
                alltime["sortino"] = float("nan")
            alltime["max_dd"] = compute_max_drawdown(equity)[0] * 100.0
            alltime["calmar"] = compute_calmar_ratio(equity)
            alltime["balance_first"] = float(equity.iloc[0])
            alltime["balance_last"] = float(equity.iloc[-1])
            alltime["balance_change_pct"] = (
                (alltime["balance_last"] / alltime["balance_first"]) - 1.0
            ) * 100.0

    if closes is not None and not closes.empty:
        sorted_closes = closes.copy()
        sorted_closes["close_timestamp"] = pd.to_datetime(
            sorted_closes["close_timestamp"], utc=True, errors="coerce"
        )
        sorted_closes = sorted_closes.sort_values("close_timestamp")
        all_consec = compute_consecutive_losses(sorted_closes["net_pnl"])
    else:
        all_consec = 0

    snapshot_balance = (
        alltime.get("balance_last")
        or window.get("balance_end")
        or summary_all.get("current_balance")
    )
    alerts = _build_alerts(
        snapshot_balance=snapshot_balance,
        consecutive_losses=all_consec,
        max_dd_pct=alltime.get("max_dd", 0.0),
        thresholds=thresholds,
    )

    return {
        "window": window,
        "alltime": alltime,
        "alerts": alerts,
        "snapshot_balance": snapshot_balance,
        "all_consec": all_consec,
        "closes": closes,
        "active": active,
        "summary_all": summary_all,
        "since": since,
        "until": until,
    }


def _normalise_window(
    since: Optional[datetime], until: Optional[datetime]
) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    until = until or now
    since = since or (until - timedelta(days=1))
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return since, until


def build_daily_pnl_summary_text(
    data_dir: str = "data/live",
    active_positions_path: str = "data/active_positions.json",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> str:
    """Plain-text Telegram/Discord-friendly summary (no markdown, no tables).

    Designed for messaging apps that handle markdown poorly. Keep it concise —
    the full markdown report should be sent as a file attachment alongside.
    """
    since, until = _normalise_window(since, until)
    thresholds = {**DEFAULT_ALERT_THRESHOLDS, **(thresholds or {})}
    data = _gather_report_data(data_dir, active_positions_path, since, until, thresholds)
    window = data["window"]
    alltime = data["alltime"]
    alerts = data["alerts"]
    active = data["active"]
    summary_all = data["summary_all"]
    all_consec = data["all_consec"]
    snapshot_balance = data["snapshot_balance"]

    lines: list[str] = []
    lines.append(f"📊 ggTrader Daily PnL — {until.date()}")
    lines.append("")

    # Snapshot
    bal_str = _fmt_money(snapshot_balance)
    bal_change = window.get("balance_change_pct")
    bal_change_usd = (
        (window.get("balance_end") or 0) - (window.get("balance_start") or 0)
        if window.get("balance_start") is not None
        else None
    )
    if bal_change is not None:
        arrow = "🟢" if bal_change >= 0 else "🔴"
        lines.append(
            f"💰 Balance: {bal_str}  {arrow} "
            f"{_fmt_money(bal_change_usd, sign=True)} "
            f"({_fmt_pct(bal_change, sign=True)})"
        )
    else:
        lines.append(f"💰 Balance: {bal_str}")
    if active:
        lines.append(f"📂 Open positions: {len(active)}")
    lines.append("")

    # 24h activity
    pnl_emoji = "🟢" if window["net_pnl"] > 0 else ("⚪" if window["net_pnl"] == 0 else "🔴")
    lines.append("— Last 24h —")
    lines.append(
        f"  Trades: {window['trades']} ({window['wins']}W / {window['losses']}L), "
        f"win rate {_fmt_pct(window['win_rate'])}"
    )
    lines.append(
        f"  {pnl_emoji} Realised PnL: {_fmt_money(window['net_pnl'], sign=True)} "
        f"(fees {_fmt_money(window['fees'])})"
    )
    if window["best"] is not None:
        lines.append(
            f"  Best: {_fmt_money(window['best'], sign=True)} ({window['best_symbol']})"
        )
    if window["worst"] is not None and window["worst"] != window["best"]:
        lines.append(
            f"  Worst: {_fmt_money(window['worst'], sign=True)} ({window['worst_symbol']})"
        )
    lines.append("")

    # All-time health
    lines.append("— All-time —")
    if alltime.get("balance_first") is not None:
        lines.append(
            f"  Total return: {_fmt_pct(alltime['balance_change_pct'], sign=True)} "
            f"({_fmt_money(alltime['balance_first'])} → {_fmt_money(alltime['balance_last'])})"
        )
    lines.append(
        f"  Sharpe: {_fmt_float(alltime.get('sharpe'))}  "
        f"Sortino: {_fmt_float(alltime.get('sortino'))}"
    )
    lines.append(
        f"  Max DD: {_fmt_pct(alltime.get('max_dd'))}  "
        f"Calmar: {_fmt_float(alltime.get('calmar'))}"
    )
    lines.append(
        f"  Win rate: {_fmt_pct(summary_all.get('win_rate'))}  "
        f"Profit factor: {_fmt_float(summary_all.get('profit_factor'))}"
    )
    lines.append(f"  Consecutive losses: {all_consec}")
    lines.append("")

    # Alerts
    if alerts:
        lines.append("⚠️ Alerts:")
        for a in alerts:
            # Strip markdown bold for plain text
            clean = a.replace("**", "")
            lines.append(f"  • {clean}")
        lines.append("")

    # Open positions (compact)
    if active:
        lines.append(f"Open ({len(active)}):")
        for sym, p in sorted(active.items()):
            entry = p.get("entry_price")
            entry_str = f"${entry:.6g}" if entry else "—"
            exit_name = p.get("exit_name", "?")
            lines.append(f"  {sym}: {entry_str} ({exit_name})")

    return "\n".join(lines)


def build_daily_pnl_report(
    data_dir: str = "data/live",
    active_positions_path: str = "data/active_positions.json",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> str:
    """Build a markdown daily PnL report.

    Args:
        data_dir: TradeTracker CSV directory (default: data/live)
        active_positions_path: Path to active_positions.json
        since: Window start (default: 24h ago, UTC)
        until: Window end (default: now, UTC)
        thresholds: Alert thresholds (default: DEFAULT_ALERT_THRESHOLDS)

    Returns:
        Markdown string ready to write to file.
    """
    since, until = _normalise_window(since, until)
    thresholds = {**DEFAULT_ALERT_THRESHOLDS, **(thresholds or {})}
    data = _gather_report_data(data_dir, active_positions_path, since, until, thresholds)
    window = data["window"]
    alltime = data["alltime"]
    alerts = data["alerts"]
    active = data["active"]
    closes = data["closes"]
    summary_all = data["summary_all"]
    all_consec = data["all_consec"]
    snapshot_balance = data["snapshot_balance"]
    now = datetime.now(timezone.utc)

    # ── Render markdown ─────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# Daily PnL Report — {until.date()}")
    lines.append("")
    window_str = (
        f"{since.strftime('%Y-%m-%d %H:%M')} → "
        f"{until.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    lines.append(f"*Window: {window_str}*")
    lines.append("")

    # Snapshot
    bal = snapshot_balance
    bal_change = window.get("balance_change_pct")
    bal_change_usd = (
        (window.get("balance_end") or 0) - (window.get("balance_start") or 0)
        if window.get("balance_start") is not None
        else None
    )
    arrow = ""
    if bal_change is not None:
        arrow = "▲" if bal_change >= 0 else "▼"
    lines.append("## 💰 Account Snapshot")
    lines.append("")
    lines.append(f"- **Balance**: {_fmt_money(bal)}")
    if bal_change is not None:
        lines.append(
            f"- **24h change**: {arrow} {_fmt_money(bal_change_usd, sign=True)} "
            f"({_fmt_pct(bal_change, sign=True)})"
        )
    if active:
        lines.append(f"- **Open positions**: {len(active)}")
    lines.append("")

    # Alerts
    if alerts:
        lines.append("## ⚠️ Alerts")
        lines.append("")
        for a in alerts:
            lines.append(f"- {a}")
        lines.append("")

    # 24h trade summary
    lines.append("## 📊 24h Trading Activity")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    trades_cell = f"{window['trades']} ({window['wins']}W / {window['losses']}L)"
    lines.append(f"| Closed trades | {trades_cell} |")
    lines.append(f"| Win rate | {_fmt_pct(window['win_rate'])} |")
    lines.append(f"| Realised PnL | {_fmt_money(window['net_pnl'], sign=True)} |")
    lines.append(f"| Fees | {_fmt_money(window['fees'])} |")
    if window["best"] is not None:
        lines.append(
            f"| Best trade | {_fmt_money(window['best'], sign=True)} ({window['best_symbol']}) |"
        )
    if window["worst"] is not None:
        lines.append(
            f"| Worst trade | {_fmt_money(window['worst'], sign=True)} ({window['worst_symbol']}) |"
        )
    lines.append("")

    # All-time health
    lines.append("## 📈 All-Time Account Health")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    if alltime.get("balance_first") is not None:
        lines.append(
            f"| Total return | {_fmt_pct(alltime['balance_change_pct'], sign=True)} "
            f"(${alltime['balance_first']:.2f} → ${alltime['balance_last']:.2f}) |"
        )
    lines.append(f"| Sharpe (daily) | {_fmt_float(alltime.get('sharpe'))} |")
    lines.append(f"| Sortino (daily) | {_fmt_float(alltime.get('sortino'))} |")
    lines.append(f"| Max drawdown | {_fmt_pct(alltime.get('max_dd'))} |")
    lines.append(f"| Calmar | {_fmt_float(alltime.get('calmar'))} |")
    lines.append(f"| Profit factor | {_fmt_float(summary_all.get('profit_factor'))} |")
    lines.append(f"| All-time win rate | {_fmt_pct(summary_all.get('win_rate'))} |")
    lines.append(f"| Consecutive losses | {all_consec} |")
    lines.append("")

    # Open positions
    if active:
        lines.append(f"## 📂 Open Positions ({len(active)})")
        lines.append("")
        lines.append("| Symbol | Entry | Amount | Stop % | Exit |")
        lines.append("|---|---|---|---|---|")
        for sym, p in sorted(active.items()):
            entry = p.get("entry_price")
            entry_str = f"${entry:.6g}" if entry else "—"
            amt = p.get("amount", 0)
            stop = p.get("stop_pct")
            stop_str = f"{stop:.2f}%" if stop is not None else "—"
            exit_name = p.get("exit_name", "?")
            lines.append(f"| {sym} | {entry_str} | {amt:.6g} | {stop_str} | {exit_name} |")
        lines.append("")

    # Recent trades (last 10)
    if closes is not None and not closes.empty:
        recent = closes.copy()
        recent["close_timestamp"] = pd.to_datetime(
            recent["close_timestamp"], utc=True, errors="coerce"
        )
        recent = recent.sort_values("close_timestamp", ascending=False).head(10)
        lines.append("## 📋 Recent Closed Trades (last 10)")
        lines.append("")
        lines.append("| Time | Symbol | Net PnL | % | Reason |")
        lines.append("|---|---|---|---|---|")
        for _, r in recent.iterrows():
            ts = r["close_timestamp"].strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"| {ts} | {r['symbol']} | {_fmt_money(r['net_pnl'], sign=True)} | "
                f"{_fmt_pct(r['pnl_pct'], sign=True)} | {r['exit_reason']} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Generated {now.strftime('%Y-%m-%d %H:%M:%S UTC')} by ggTrader pnl-daily*")

    return "\n".join(lines)
