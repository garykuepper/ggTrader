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

import html as _html
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from ggTrader.core.trade_tracker import TradeTracker
from ggTrader.utils.kraken_ledger import (
    cumulative_net_deposits_usd,
    fetch_kraken_ledger_cached,
)
from ggTrader.utils.live_metrics import (
    apply_deposit_adjustment,
    compute_consecutive_losses,
    equity_curve_from_balance,
    summarise_window,
)

# All user-facing dates/times in reports use this timezone.
_DISPLAY_TZ = ZoneInfo("America/Los_Angeles")


def _h(s: Any) -> str:
    """HTML-escape a value for Telegram parse_mode=HTML.

    Telegram HTML mode requires escaping only ``<``, ``>`` and ``&``. We pass
    ``quote=False`` so quotes pass through (they're valid inside text and we
    don't emit them inside tag attributes).
    """
    return _html.escape(str(s), quote=False)


def _render_kv_table(rows: list[tuple[str, str]]) -> str:
    """Render a 2-column key/value block aligned for monospace display.

    Designed to live inside a Telegram ``<pre>`` block. Right-pads keys to
    the longest key width so values line up.
    """
    if not rows:
        return ""
    key_w = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(key_w)}  {v}" for k, v in rows)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an aligned ASCII table for use inside a ``<pre>`` block.

    All cells are stringified and left-justified to the max width of the
    column. A dashed separator row sits between header and body.
    """
    if not rows:
        return ""
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    sep = "  "
    lines = [sep.join(str(h).ljust(w) for h, w in zip(headers, widths))]
    lines.append(sep.join("-" * w for w in widths))
    for r in rows:
        lines.append(sep.join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(lines)


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


def _fmt_dt(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime in the display timezone (America/Los_Angeles)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_DISPLAY_TZ).strftime(fmt)


def _fmt_ts(ts: Any, fmt: str = "%m-%d %H:%M") -> str:
    """Format a pandas Timestamp / string in the display timezone."""
    try:
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        if hasattr(ts, "tz_localize") and ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(_DISPLAY_TZ).strftime(fmt)
    except Exception:
        return str(ts)


def _fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current last-trade prices for the given dashed symbols (e.g. BTC-USD).

    Uses Kraken's public ticker endpoint via ccxt — no API key needed.
    Returns ``{symbol: price}`` for symbols that resolved successfully; symbols
    that error out are silently omitted (the caller renders "—" for missing).
    Network or import failures degrade to an empty dict so the report still
    builds offline.
    """
    if not symbols:
        return {}
    try:
        import ccxt  # type: ignore

        ex = ccxt.kraken({"enableRateLimit": True})
        ex.load_markets()
    except Exception:
        return {}

    out: dict[str, float] = {}
    for sym in symbols:
        ccxt_sym = sym.replace("-", "/")
        try:
            t = ex.fetch_ticker(ccxt_sym)
            price = t.get("last") or t.get("close")
            if price:
                out[sym] = float(price)
        except Exception:
            continue
    return out


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
    circuit_breaker_triggered: bool = False,
) -> list[str]:
    alerts: list[str] = []
    if circuit_breaker_triggered:
        alerts.append("🛑 **Daily Loss Circuit Breaker is ACTIVE (Entries Halted)**")

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


def _fetch_and_persist_fear_greed() -> Optional[dict]:
    """Fetch the latest Fear & Greed Index, persist to DB, return formatted row.

    Returns ``{value, classification, emoji, history}`` (history includes last
    8 days for delta context) or ``None`` on fetch failure.
    """
    from ggTrader.utils.fear_greed import fetch_fear_greed

    latest = fetch_fear_greed(limit=8)  # latest + 7 days for delta
    if latest is None:
        return None
    try:
        from ggTrader.utils.result_db_manager import ResultDBManager

        db = ResultDBManager()
        for entry in latest.get("history") or [latest]:
            db.upsert_fear_greed(entry["date"], entry["value"], entry["classification"])
    except Exception as e:
        print(f"  [F&G] WARNING: persist failed (continuing) — {e!r}")
    return latest


def _fg_delta_suffix(fg_row: dict, days: int = 7) -> str:
    """Return ``" (+5 vs 7d)"`` style suffix when history is available."""
    hist = fg_row.get("history") or []
    if len(hist) >= days + 1:
        delta = fg_row["value"] - hist[days]["value"]
        sign = "+" if delta >= 0 else ""
        return f" ({sign}{delta} vs {days}d)"
    return ""


def _fetch_regime_status() -> dict[str, Any]:
    """Compute current crypto regime status (BTC vs altcoin index, F&G)."""
    return _fetch_crypto_regime_status()


def _fetch_crypto_regime_status() -> dict[str, Any]:
    """Fetch recent data for BTC and top coins to compute current regime status.

    Returns a dict with:
        btc_bull: bool
        alt_bull: bool
        btc_price: float
        alt_index: float (normalized)
        error: str | None
    """
    try:
        import ccxt

        from ggTrader.core.regime_filtering import _compute_btc_regime_mask
        from ggTrader.utils.run_config import full_pipeline_config

        config = full_pipeline_config()
        # Ensure we have enough bars for EMA warmup
        n_warmup = int(config.get("EMA_WARMUP_BARS", 100))
        limit = n_warmup + 50

        ex = ccxt.kraken({"enableRateLimit": True})
        # We need BTC and a handful of top coins for the alt index.
        # Using a fixed set of top coins is faster than fetching the whole universe.
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD"]
        data = {}
        for s in symbols:
            try:
                ohlcv = ex.fetch_ohlcv(s, timeframe="4h", limit=limit)
                df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                # Map to the format regime_filtering expects (MultiIndex: symbol, field)
                sym_dashed = s.replace("/", "-")
                for col in ["open", "high", "low", "close", "volume"]:
                    data[(sym_dashed, col)] = df[col]
            except Exception:
                continue

        if not data:
            return {"error": "Could not fetch regime data"}

        ohlcv_df = pd.DataFrame(data)

        try:
            btc_mask = _compute_btc_regime_mask(ohlcv_df, config)
        except Exception:
            btc_mask = None

        btc_bull = bool(btc_mask.iloc[-1]) if btc_mask is not None else False

        btc_price = 0.0
        if ("BTC-USD", "close") in ohlcv_df.columns:
            btc_price = float(ohlcv_df[("BTC-USD", "close")].iloc[-1])
        eth_price = 0.0
        if ("ETH-USD", "close") in ohlcv_df.columns:
            eth_price = float(ohlcv_df[("ETH-USD", "close")].iloc[-1])

        btc_status = "BULL 🟢" if btc_bull else "BEAR 🔴"

        fg_row = _fetch_and_persist_fear_greed()
        return {
            "btc_bull": btc_bull,
            "btc_price": btc_price,
            "eth_price": eth_price,
            "fear_greed": fg_row,
            "error": None,
            "rows": [
                ("BTC Regime", btc_status),
                ("BTC Price", f"${btc_price:,.2f}"),
                ("ETH Price", f"${eth_price:,.2f}"),
            ]
            + (
                [
                    (
                        "Fear & Greed",
                        f"{fg_row['value']} {fg_row['classification']} {fg_row['emoji']}",
                    )
                ]
                if fg_row
                else []
            ),
            "lines": [
                f"🌐 BTC Regime: {btc_status} (${btc_price:,.0f})",
                f"📈 ETH Price: ${eth_price:,.0f}",
            ]
            + (
                [
                    f"{fg_row['emoji']} Fear & Greed: {fg_row['value']} {fg_row['classification']}{_fg_delta_suffix(fg_row)}"
                ]
                if fg_row
                else []
            ),
            "md_lines": [
                f"- **BTC Regime**: {btc_status}",
                f"- **BTC Price**: ${btc_price:,.2f}",
                f"- **ETH Price**: ${eth_price:,.2f}",
            ]
            + (
                [
                    f"- **Fear & Greed**: {fg_row['value']} {fg_row['classification']} {fg_row['emoji']}{_fg_delta_suffix(fg_row)}"
                ]
                if fg_row
                else []
            ),
        }
    except Exception as e:
        return {"error": f"Regime compute failed: {e!r}"}


def _gather_report_data(
    data_dir: str,
    active_positions_path: str,
    since: datetime,
    until: datetime,
    thresholds: dict[str, float],
) -> dict:
    """Collect all data needed for both markdown and plain-text report builders.

    Returns a dict with: window, alltime, alerts, snapshot_balance, all_consec,
    closes, active, summary_all, since, until, deposits, regime.
    """
    tracker = TradeTracker(data_dir=data_dir, run_id="LIVE")
    closes = tracker.get_closed_positions()
    balances = tracker.get_balance_history()
    summary_all = tracker.compute_summary_stats()
    full_state = _load_active_positions(Path(active_positions_path))
    if isinstance(full_state, dict) and "positions" in full_state:
        active = full_state["positions"]
        cb_triggered = full_state.get("circuit_breaker_triggered", False)
    else:
        active = full_state
        cb_triggered = False

    # Filter out dust positions — sub-$1 leftovers from partial exchange fills.
    active = {
        sym: pos
        for sym, pos in active.items()
        if (pos.get("entry_price") or 0) * float(pos.get("amount", 0) or 0) >= 1.00
    }

    # Deposit/withdrawal history from Kraken.
    cum_deposits = pd.Series(dtype=float)
    try:
        ledger_df = fetch_kraken_ledger_cached()
        cum_deposits = cumulative_net_deposits_usd(ledger_df)
    except Exception as e:
        import logging

        logging.getLogger("ggTraderLive").warning(
            f"  [Report] Could not fetch Kraken ledger ({e!r}) — "
            f"reporting raw balance (deposits not subtracted)"
        )
        ledger_df = pd.DataFrame()

    deposits_info = {
        "total_deposits": 0.0,
        "total_withdrawals": 0.0,
        "net_deposits": 0.0,
        "deposit_count": 0,
    }
    if ledger_df is not None and not ledger_df.empty:
        usd_only = ledger_df[ledger_df["currency"].isin(["USD", "ZUSD"])]
        deposits_info["total_deposits"] = float(
            usd_only[usd_only["type"] == "deposit"]["amount"].sum()
        )
        deposits_info["total_withdrawals"] = float(
            usd_only[usd_only["type"] == "withdrawal"]["amount"].sum()
        )
        deposits_info["net_deposits"] = (
            deposits_info["total_deposits"] - deposits_info["total_withdrawals"]
        )
        deposits_info["deposit_count"] = int(len(usd_only[usd_only["type"] == "deposit"]))

    window = summarise_window(
        closes,
        balances,
        pd.Timestamp(since),
        pd.Timestamp(until),
    )

    alltime: dict[str, Any] = {}
    if balances is not None and not balances.empty:
        raw_equity = equity_curve_from_balance(balances)
        if not raw_equity.empty:
            from ggTrader.utils.live_metrics import (
                compute_calmar_ratio,
                compute_max_drawdown,
                compute_sharpe_ratio,
                compute_sortino_ratio,
                daily_returns_from_equity,
            )

            # Apply deposit adjustment so subsequent metrics reflect TRADING
            # PnL only, not capital flows.
            equity = apply_deposit_adjustment(raw_equity, cum_deposits)

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
            alltime["trading_pnl"] = alltime["balance_last"] - alltime["balance_first"]
            # Also expose the raw (non-adjusted) current balance for the snapshot
            alltime["raw_balance_last"] = float(raw_equity.iloc[-1])
            # Net deposits added between the first balance snapshot and now —
            # used so the report can show why current balance ≠ start + PnL.
            alltime["net_deposits_since_start"] = (
                alltime["raw_balance_last"] - alltime["balance_last"]
            )

    if closes is not None and not closes.empty:
        sorted_closes = closes.copy()
        sorted_closes["close_timestamp"] = pd.to_datetime(
            sorted_closes["close_timestamp"],
            utc=True,
            errors="coerce",
            format="ISO8601",
        )
        sorted_closes = sorted_closes.sort_values("close_timestamp")
        all_consec = compute_consecutive_losses(sorted_closes["net_pnl"])
    else:
        all_consec = 0

    # Snapshot should show ACTUAL Kraken balance (not the deposit-adjusted
    # trading equity which is what we use for return %).
    snapshot_balance = (
        alltime.get("raw_balance_last")
        or window.get("balance_end")
        or summary_all.get("current_balance")
    )
    alerts = _build_alerts(
        snapshot_balance=snapshot_balance,
        consecutive_losses=all_consec,
        max_dd_pct=alltime.get("max_dd", 0.0),
        thresholds=thresholds,
        circuit_breaker_triggered=cb_triggered,
    )

    regime = _fetch_regime_status()

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
        "deposits": deposits_info,
        "regime": regime,
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
    stale_warning: Optional[str] = None,
) -> str:
    """Plain-text Telegram/Discord-friendly summary (no markdown, no tables).

    Designed for messaging apps that handle markdown poorly. Keep it concise —
    the full markdown report should be sent as a file attachment alongside.

    Args:
        stale_warning: Optional banner shown at the very top of the report.
            Used when the auto-sync from Kraken at the start of ``ggt pnl-daily``
            failed and the data may be behind reality.
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

    label = "Crypto"
    lines: list[str] = []
    if stale_warning:
        lines.append(f"⚠️ STALE: {stale_warning}")
        lines.append("")
    lines.append(f"📊 ggTrader {label} Daily PnL — {_fmt_dt(until, '%Y-%m-%d')}")
    lines.append("")

    # Regime
    regime = data.get("regime", {})
    if not regime.get("error"):
        for line in regime.get("lines", []):
            lines.append(line)
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
        lines.append(f"  Best: {_fmt_money(window['best'], sign=True)} ({window['best_symbol']})")
    if window["worst"] is not None and window["worst"] != window["best"]:
        lines.append(
            f"  Worst: {_fmt_money(window['worst'], sign=True)} ({window['worst_symbol']})"
        )
    lines.append("")

    # All-time health
    lines.append("— All-time —")
    if alltime.get("balance_first") is not None:
        lines.append(
            f"  Trading return: {_fmt_pct(alltime['balance_change_pct'], sign=True)} "
            f"({_fmt_money(alltime.get('trading_pnl'), sign=True)} PnL)"
        )
        lines.append(
            f"  Current balance: {_fmt_money(alltime.get('raw_balance_last'))} "
            f"(incl. {_fmt_money(alltime.get('net_deposits_since_start'), sign=True)} "
            f"net deposits since start)"
        )
    lines.append(
        f"  Sharpe: {_fmt_float(alltime.get('sharpe'))}  "
        f"Sortino: {_fmt_float(alltime.get('sortino'))}"
    )
    lines.append(
        f"  Max DD: {_fmt_pct(alltime.get('max_dd'))}  Calmar: {_fmt_float(alltime.get('calmar'))}"
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


def build_daily_pnl_summary_html(
    data_dir: str = "data/live",
    active_positions_path: str = "data/active_positions.json",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    thresholds: Optional[dict[str, float]] = None,
    max_open_rows: int = 10,
    max_recent_rows: int = 8,
    stale_warning: Optional[str] = None,
) -> str:
    """Build a Telegram HTML-mode daily summary with monospace tables.

    Uses ``<pre>`` blocks (rendered monospace by Telegram) for aligned tables
    and ``<b>`` for section headers. All dynamic content is HTML-escaped.
    Designed to fit inside Telegram's 4096-char per-message limit even with
    a full position book.

    Args:
        max_open_rows: cap the open-positions table to keep the message
            under the Telegram limit. Excess rows are summarised as "+N more".
        max_recent_rows: same cap for the recent closed trades table.
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
    deposits = data.get("deposits", {})

    label = "Crypto"
    parts: list[str] = []

    # ── Stale warning banner (sync failure) ─────────────────────────────
    if stale_warning:
        parts.append(f"<b>⚠️ STALE DATA</b>\n<i>{_h(stale_warning)}</i>")
        parts.append("")

    # ── Header ──────────────────────────────────────────────────────────
    parts.append(f"<b>📊 ggTrader {label} Daily PnL — {_h(_fmt_dt(until, '%Y-%m-%d'))}</b>")
    parts.append("")

    # ── Regime Status ───────────────────────────────────────────────────
    regime = data.get("regime", {})
    if not regime.get("error"):
        reg_rows = regime.get("rows", [])
        if reg_rows:
            parts.append("<b>🌐 Market Regime</b>")
            parts.append("<pre>" + _h(_render_kv_table(reg_rows)) + "</pre>")
            parts.append("")

    # ── Account snapshot ────────────────────────────────────────────────
    bal_change = window.get("balance_change_pct")
    bal_change_usd = (
        (window.get("balance_end") or 0) - (window.get("balance_start") or 0)
        if window.get("balance_start") is not None
        else None
    )
    snap_rows: list[tuple[str, str]] = [("Balance", _fmt_money(snapshot_balance))]
    if deposits.get("net_deposits"):
        snap_rows.append(("Net deposits", _fmt_money(deposits["net_deposits"])))
    if bal_change is not None:
        arrow = "▲" if bal_change >= 0 else "▼"
        snap_rows.append(
            (
                "24h change",
                f"{arrow} {_fmt_money(bal_change_usd, sign=True)} "
                f"({_fmt_pct(bal_change, sign=True)})",
            )
        )
    if active:
        snap_rows.append(("Open positions", str(len(active))))
    parts.append("<b>💰 Account Snapshot</b>")
    parts.append("<pre>" + _h(_render_kv_table(snap_rows)) + "</pre>")
    parts.append("")

    # ── Alerts (if any) ─────────────────────────────────────────────────
    if alerts:
        parts.append("<b>⚠️ Alerts</b>")
        # Strip legacy markdown bold (**...**) so it doesn't leak as literal
        clean_alerts = "\n".join("• " + a.replace("**", "") for a in alerts)
        parts.append("<blockquote>" + _h(clean_alerts) + "</blockquote>")
        parts.append("")

    # ── 24h activity ────────────────────────────────────────────────────
    pnl_emoji = "🟢" if window["net_pnl"] > 0 else ("⚪" if window["net_pnl"] == 0 else "🔴")
    win_rows: list[tuple[str, str]] = [
        ("Trades", f"{window['trades']} ({window['wins']}W / {window['losses']}L)"),
        ("Win rate", _fmt_pct(window["win_rate"])),
        ("Realised PnL", f"{pnl_emoji} {_fmt_money(window['net_pnl'], sign=True)}"),
        ("Fees", _fmt_money(window["fees"])),
    ]
    if window["best"] is not None:
        win_rows.append(
            ("Best", f"{_fmt_money(window['best'], sign=True)} ({window['best_symbol']})")
        )
    if window["worst"] is not None and window["worst"] != window["best"]:
        win_rows.append(
            ("Worst", f"{_fmt_money(window['worst'], sign=True)} ({window['worst_symbol']})")
        )
    parts.append("<b>📊 Last 24h</b>")
    parts.append("<pre>" + _h(_render_kv_table(win_rows)) + "</pre>")
    parts.append("")

    # ── All-time health ─────────────────────────────────────────────────
    at_rows: list[tuple[str, str]] = []
    if alltime.get("balance_first") is not None:
        # "Trading return" is deposit-adjusted: starts from the first balance
        # snapshot, then subtracts any deposits made AFTER that point.
        at_rows.append(
            (
                "Trading return",
                f"{_fmt_pct(alltime['balance_change_pct'], sign=True)} "
                f"({_fmt_money(alltime.get('trading_pnl'), sign=True)} PnL)",
            )
        )
        at_rows.append(
            (
                "Current balance",
                f"{_fmt_money(alltime.get('raw_balance_last'))} "
                f"({_fmt_money(alltime.get('net_deposits_since_start'), sign=True)} deposits)",
            )
        )
    at_rows.extend(
        [
            ("Sharpe", _fmt_float(alltime.get("sharpe"))),
            ("Sortino", _fmt_float(alltime.get("sortino"))),
            ("Max DD", _fmt_pct(alltime.get("max_dd"))),
            ("Calmar", _fmt_float(alltime.get("calmar"))),
            ("Win rate", _fmt_pct(summary_all.get("win_rate"))),
            ("Profit factor", _fmt_float(summary_all.get("profit_factor"))),
            ("Consec losses", str(all_consec)),
        ]
    )
    parts.append("<b>📈 All-Time Health</b>")
    parts.append("<pre>" + _h(_render_kv_table(at_rows)) + "</pre>")
    parts.append("")

    # ── Open positions with unrealized PnL ──────────────────────────────
    if active:
        items = sorted(active.items())
        truncated = len(items) - max_open_rows
        items = items[:max_open_rows]
        # Fetch current prices for symbols with a known entry. Symbols with
        # null entry_price (reconciled-from-exchange ghosts like SHIB-USD) are
        # excluded from the fetch since we can't compute PnL anyway.
        price_syms = [s for s, p in items if p.get("entry_price")]
        prices = _fetch_current_prices(price_syms)

        rows: list[list[str]] = []
        total_cost = 0.0
        total_value = 0.0
        for sym, p in items:
            entry = p.get("entry_price")
            amt = float(p.get("amount", 0) or 0)
            cur = prices.get(sym)
            if entry and amt and cur:
                cost = entry * amt
                value = cur * amt
                pnl = value - cost
                pnl_pct = (cur / entry - 1.0) * 100.0
                total_cost += cost
                total_value += value
                arrow = "▲" if pnl >= 0 else "▼"
                rows.append(
                    [
                        sym,
                        f"${cost:,.2f}",
                        f"${value:,.2f}",
                        f"{arrow}{_fmt_money(pnl, sign=True)}",
                        _fmt_pct(pnl_pct, sign=True),
                    ]
                )
            else:
                # Missing entry or current price — show what we know.
                cost_str = f"${entry * amt:,.2f}" if (entry and amt) else "—"
                rows.append([sym, cost_str, "—", "—", "—"])

        parts.append(f"<b>📂 Open Positions ({len(active)})</b>")
        parts.append(
            "<pre>" + _h(_render_table(["Symbol", "Cost", "Value", "PnL", "%"], rows)) + "</pre>"
        )
        if total_cost > 0:
            total_pnl = total_value - total_cost
            total_pct = (total_value / total_cost - 1.0) * 100.0
            tot_emoji = "🟢" if total_pnl >= 0 else "🔴"
            parts.append(
                f"{tot_emoji} <b>Unrealized:</b> "
                f"{_h(_fmt_money(total_pnl, sign=True))} "
                f"({_h(_fmt_pct(total_pct, sign=True))}) "
                f"on {_h(_fmt_money(total_cost))} cost basis"
            )
        if truncated > 0:
            parts.append(f"<i>+{truncated} more not shown</i>")
        parts.append("")

    # ── Recent closed trades ────────────────────────────────────────────
    if closes is not None and not closes.empty:
        recent = closes.copy()
        recent["close_timestamp"] = pd.to_datetime(
            recent["close_timestamp"],
            utc=True,
            errors="coerce",
            format="ISO8601",
        )
        recent = recent.sort_values("close_timestamp", ascending=False).head(max_recent_rows)
        rows = []
        for _, r in recent.iterrows():
            ts = _fmt_ts(r["close_timestamp"], "%m-%d %H:%M")
            rows.append(
                [
                    ts,
                    str(r["symbol"]),
                    _fmt_money(r["net_pnl"], sign=True),
                    _fmt_pct(r["pnl_pct"], sign=True),
                    str(r["exit_reason"]),
                ]
            )
        parts.append(f"<b>📋 Recent Trades (last {len(rows)}) PT</b>")
        parts.append(
            "<pre>" + _h(_render_table(["Time", "Symbol", "PnL", "%", "Reason"], rows)) + "</pre>"
        )

    return "\n".join(parts)


def build_daily_pnl_report(
    data_dir: str = "data/live",
    active_positions_path: str = "data/active_positions.json",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    thresholds: Optional[dict[str, float]] = None,
    stale_warning: Optional[str] = None,
) -> str:
    """Build a markdown daily PnL report.

    Args:
        data_dir: TradeTracker CSV directory (default: data/live)
        active_positions_path: Path to active_positions.json
        since: Window start (default: 24h ago, UTC)
        until: Window end (default: now, UTC)
        thresholds: Alert thresholds (default: DEFAULT_ALERT_THRESHOLDS)
        stale_warning: Optional banner shown at the very top of the report
            when the auto-sync from Kraken failed.

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

    label = "Crypto"

    # ── Render markdown ─────────────────────────────────────────────────
    lines: list[str] = []
    if stale_warning:
        lines.append(f"> ⚠️ **STALE DATA**: {stale_warning}")
        lines.append("")
    lines.append(f"# {label} Daily PnL Report — {_fmt_dt(until, '%Y-%m-%d')}")
    lines.append("")
    window_str = f"{_fmt_dt(since)} → {_fmt_dt(until)} PT"
    lines.append(f"*Window: {window_str}*")
    lines.append("")

    # Regime
    regime = data.get("regime", {})
    if not regime.get("error"):
        md_lines = regime.get("md_lines", [])
        if md_lines:
            lines.append("## 🌐 Market Regime")
            lines.append("")
            lines.extend(md_lines)
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
            f"| Trading return (deposit-adjusted) | "
            f"{_fmt_pct(alltime['balance_change_pct'], sign=True)} "
            f"({_fmt_money(alltime.get('trading_pnl'), sign=True)} PnL) |"
        )
        lines.append(
            f"| Current balance | "
            f"{_fmt_money(alltime.get('raw_balance_last'))} "
            f"({_fmt_money(alltime.get('net_deposits_since_start'), sign=True)} net deposits since start) |"
        )
    lines.append(f"| Sharpe (annualised) | {_fmt_float(alltime.get('sharpe'))} |")
    lines.append(f"| Sortino (annualised) | {_fmt_float(alltime.get('sortino'))} |")
    lines.append(f"| Max drawdown | {_fmt_pct(alltime.get('max_dd'))} |")
    lines.append(f"| Calmar | {_fmt_float(alltime.get('calmar'))} |")
    lines.append(f"| Profit factor | {_fmt_float(summary_all.get('profit_factor'))} |")
    lines.append(f"| All-time win rate | {_fmt_pct(summary_all.get('win_rate'))} |")
    lines.append(f"| Consecutive losses | {all_consec} |")
    lines.append("")

    # Open positions with cost basis, current value, and unrealized PnL
    if active:
        price_syms = [s for s, p in sorted(active.items()) if p.get("entry_price")]
        prices = _fetch_current_prices(price_syms)

        lines.append(f"## 📂 Open Positions ({len(active)})")
        lines.append("")
        lines.append("| Symbol | Cost | Value | Unrealised | % | Exit |")
        lines.append("|---|---|---|---|---|---|")
        total_cost = 0.0
        total_value = 0.0
        for sym, p in sorted(active.items()):
            entry = p.get("entry_price")
            amt = float(p.get("amount", 0) or 0)
            cur = prices.get(sym)
            exit_name = p.get("exit_name", "?")
            if entry and amt and cur:
                cost = entry * amt
                value = cur * amt
                pnl = value - cost
                pnl_pct = (cur / entry - 1.0) * 100.0
                total_cost += cost
                total_value += value
                arrow = "▲" if pnl >= 0 else "▼"
                lines.append(
                    f"| {sym} | {_fmt_money(cost)} | {_fmt_money(value)} | "
                    f"{arrow}{_fmt_money(pnl, sign=True)} | "
                    f"{_fmt_pct(pnl_pct, sign=True)} | {exit_name} |"
                )
            else:
                cost_str = _fmt_money(entry * amt) if (entry and amt) else "—"
                lines.append(f"| {sym} | {cost_str} | — | — | — | {exit_name} |")
        if total_cost > 0:
            total_pnl = total_value - total_cost
            total_pct = (total_value / total_cost - 1.0) * 100.0
            arrow = "▲" if total_pnl >= 0 else "▼"
            lines.append(
                f"| **Total** | **{_fmt_money(total_cost)}** | "
                f"**{_fmt_money(total_value)}** | "
                f"**{arrow}{_fmt_money(total_pnl, sign=True)}** | "
                f"**{_fmt_pct(total_pct, sign=True)}** | |"
            )
        lines.append("")

    # Recent trades (last 10)
    if closes is not None and not closes.empty:
        recent = closes.copy()
        recent["close_timestamp"] = pd.to_datetime(
            recent["close_timestamp"],
            utc=True,
            errors="coerce",
            format="ISO8601",
        )
        recent = recent.sort_values("close_timestamp", ascending=False).head(10)
        lines.append("## 📋 Recent Closed Trades (last 10)")
        lines.append("")
        lines.append("| Time (PT) | Symbol | Net PnL | % | Reason |")
        lines.append("|---|---|---|---|---|")
        for _, r in recent.iterrows():
            ts = _fmt_ts(r["close_timestamp"], "%m-%d %H:%M")
            lines.append(
                f"| {ts} | {r['symbol']} | {_fmt_money(r['net_pnl'], sign=True)} | "
                f"{_fmt_pct(r['pnl_pct'], sign=True)} | {r['exit_reason']} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Generated {_fmt_dt(now, '%Y-%m-%d %H:%M:%S')} PT by ggTrader pnl-daily*")

    return "\n".join(lines)
