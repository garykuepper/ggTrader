"""Telegram notifications for paper trading alerts."""

from __future__ import annotations

import logging
import os

import requests

from ggTrader.utils.config import _load_env
from ggTrader.utils.json_safe import json_safe

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_log = logging.getLogger(__name__)


def _pnl_number(value: object) -> int | float | None:
    """Return a finite P&L value, or None when it cannot be reported."""
    value = json_safe(value)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class TelegramNotifier:
    """Send trade alerts and daily summaries to Telegram."""

    def __init__(self) -> None:
        _load_env()
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)

    def send(self, message: str) -> bool:
        if not self._enabled:
            return False
        try:
            resp = requests.post(
                _API_URL.format(token=self._token),
                json={"chat_id": self._chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def trade_alert(
        self,
        side: str,
        symbol: str,
        amount: float,
        order_id: str,
        qty: float | None = None,
        price: float | None = None,
        status: str | None = None,
    ) -> bool:
        status_suffix = ""
        if status and status not in ("filled", "partially_filled"):
            status_suffix = f" ({status.upper()})"

        msg = f"<b>📊 Paper Trade{status_suffix}</b>\n"
        msg += f"{side} <b>{symbol}</b>\n"
        if qty is not None and qty > 0 and price is not None and price > 0:
            msg += f"Shares: {qty:.4f} @ ${price:.2f}\nTotal Value: ${qty * price:.2f}\n"
        else:
            msg += f"Amount: ${amount:.2f}\n"
        msg += f"Order: <code>{order_id}</code>"
        return self.send(msg)

    def daily_summary(
        self,
        portfolio_value: float | None,
        daily_pnl: float | None,
        positions: dict[str, dict],
        total_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        split_corrections: dict[str, float] | None = None,
        booked_equity: float | None = None,
        dividend_total: float | None = None,
        new_dividend_accruals: list[dict] | None = None,
        errors: list[str] | None = None,
    ) -> bool:
        """`portfolio_value`/`positions`/`total_pnl`/`unrealized_pnl` are all
        expected to already be split-corrected AND dividend-accrual-adjusted
        by the caller (see `split_check.apply_corrections_to_positions` and
        `trader.PaperTrader._accrue_dividends`) -- this only adds a note
        calling out which symbols were corrected and, when the broker's own
        (uncorrected) equity figure is supplied via `booked_equity`, both
        numbers side by side so a reader can see exactly what changed and
        why. Unlike the old "exclude flagged from unrealized" behavior, no
        position's valid P&L is hidden; invalid values are shown as unavailable.

        `dividend_total` is the cumulative dividend accrual (an all-time
        reporting correction, not real cash -- see `dividend_check.py`);
        `new_dividend_accruals` lists any events newly credited *this run*
        (each `{symbol, ex_date, rate, qty, amount}`), so a reader can see
        exactly what changed and why it's not reflected in Alpaca's own
        balance.
        """
        portfolio_value = _pnl_number(portfolio_value)
        daily_pnl = _pnl_number(daily_pnl)
        total_was_supplied = total_pnl is not None
        total_pnl = _pnl_number(total_pnl)
        if portfolio_value is None:
            daily_pnl = None
        day_start = portfolio_value - daily_pnl if daily_pnl is not None else None
        if daily_pnl is None:
            daily_line = "Change since prior snapshot: unavailable"
        else:
            arrow = "🟢" if daily_pnl >= 0 else "🔴"
            pnl_pct = (daily_pnl / day_start * 100) if day_start > 0 else 0.0
            daily_line = (
                f"Change since prior snapshot: {arrow} ${daily_pnl:+,.2f} ({pnl_pct:+.2f}%)"
            )
        lines = ["<b>📈 Paper Portfolio Summary</b>"]
        if split_corrections:
            corr_str = ", ".join(f"{sym} {factor:g}x" for sym, factor in split_corrections.items())
            lines.append(
                f"<b>⚠️ Split correction applied:</b> {corr_str} — the broker "
                f"never applied this split to the account; figures below are "
                f"split-adjusted."
            )
            if booked_equity is not None:
                booked_equity = _pnl_number(booked_equity)
                booked = f"${booked_equity:,.2f}" if booked_equity is not None else "unavailable"
                adjusted = (
                    f"${portfolio_value:,.2f}" if portfolio_value is not None else "unavailable"
                )
                lines.append(
                    f"Booked equity (broker, uncorrected): {booked} | "
                    f"Split-adjusted equity: {adjusted}"
                )
        if dividend_total:
            lines.append(
                f"<b>💵 Dividend accrual:</b> ${dividend_total:,.2f} cumulative — "
                f"Alpaca's paper account never credits dividends; this is a "
                f"reporting correction, not real cash (account balance will "
                f"still differ from Alpaca's)."
            )
            if new_dividend_accruals:
                new_str = ", ".join(
                    f"{a['symbol']} ${a['amount']:.2f} (ex {a['ex_date']})"
                    for a in new_dividend_accruals
                )
                lines.append(f"New this run: {new_str}")
        value_line = (
            f"Value: <b>${portfolio_value:,.2f}</b>"
            if portfolio_value is not None
            else "Value: unavailable"
        )
        lines += [value_line, daily_line]
        if total_pnl is not None:
            total_pct = (total_pnl / day_start * 100) if day_start and day_start > 0 else 0.0
            total_pct_text = f" ({total_pct:+.2f}%)" if day_start is not None else ""
            unrealized_pnl = _pnl_number(unrealized_pnl)
            if unrealized_pnl is None:
                detail = "realized: unavailable, unrealized: unavailable"
            else:
                realized_pnl = total_pnl - unrealized_pnl
                detail = f"realized: ${realized_pnl:+,.2f}, unrealized: ${unrealized_pnl:+,.2f}"
            lines.append(f"Total P&L: ${total_pnl:+,.2f}{total_pct_text} — {detail}")
        elif total_was_supplied or portfolio_value is None:
            lines.append("Total P&L: unavailable")
        lines.append(f"Positions: {len(positions)}")
        if positions:
            lines.append("")
            ranked = sorted(
                positions.items(),
                key=lambda kv: (
                    _pnl_number(kv[1].get("unrealized_plpc"))
                    if _pnl_number(kv[1].get("unrealized_plpc")) is not None
                    else float("-inf")
                ),
                reverse=True,
            )
            for sym, info in ranked:
                qty = info.get("qty", 0.0)
                price = info.get("current_price", 0.0)
                qty_str = f"{qty:.0f}" if float(qty).is_integer() else f"{qty:.4f}"
                pl = _pnl_number(info.get("unrealized_pl"))
                plpc = _pnl_number(info.get("unrealized_plpc"))
                pnl_text = "P&L unavailable" if pl is None else f"${pl:+,.2f}"
                pct_text = "return unavailable" if plpc is None else f"{plpc * 100:+.2f}%"
                lines.append(f"  {sym}: {qty_str} @ ${price:.2f} — {pnl_text} ({pct_text})")
        if errors:
            lines.append("")
            lines.append(f"<b>❌ Errors ({len(errors)})</b>")
            for err in errors:
                lines.append(f"  {err}")
        return self.send("\n".join(lines))


class DryRunNotifier:
    """Wraps a notifier for the read-only dry run: nothing reaches Telegram unless asked.

    The summary and per-order alerts are never forwarded (they would read as real activity); the
    free-form ``send`` lines (gate alerts, "DRY RUN buy: ...") are logged and, with
    ``forward=True`` (``ggt paper --notify``), also sent with a ``[DRY RUN]`` prefix.
    """

    PREFIX = "[DRY RUN] "

    def __init__(self, inner: object, forward: bool = False) -> None:
        self._inner = inner
        self._forward = forward

    def send(self, message: str) -> bool:
        """Log the message; forward it with the dry-run prefix only when asked to."""
        _log.info("dry-run notification: %s", message)
        if not self._forward:
            return False
        return bool(self._inner.send(self.PREFIX + message))  # type: ignore[attr-defined]

    def trade_alert(
        self, side: str, symbol: str, amount: float, order_id: str, **_: object
    ) -> bool:
        """Log an order alert; never forwarded (a dry run submits no order)."""
        _log.info(
            "dry-run order alert suppressed: %s %s $%.2f (%s)", side, symbol, amount, order_id
        )
        return False

    def daily_summary(
        self, portfolio_value: float | None, daily_pnl: float | None, *_: object, **__: object
    ) -> bool:
        """Log the summary headline; never forwarded."""
        value = _pnl_number(portfolio_value)
        daily = _pnl_number(daily_pnl)
        value_text = f"{value:.2f}" if value is not None else "unavailable"
        _log.info(
            "dry-run summary suppressed: value=%s daily_pnl=%s",
            value_text,
            daily if daily is not None else "unavailable",
        )
        return False
