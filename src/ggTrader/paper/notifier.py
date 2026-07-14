"""Telegram notifications for paper trading alerts."""

from __future__ import annotations

import os

import requests

from ggTrader.utils.config import _load_env

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


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
            msg += (
                f"Shares: {qty:.4f} @ ${price:.2f}\n"
                f"Total Value: ${qty * price:.2f}\n"
            )
        else:
            msg += f"Amount: ${amount:.2f}\n"
        msg += f"Order: <code>{order_id}</code>"
        return self.send(msg)

    def daily_summary(
        self, portfolio_value: float, daily_pnl: float, positions: dict[str, dict]
    ) -> bool:
        arrow = "🟢" if daily_pnl >= 0 else "🔴"
        day_start = portfolio_value - daily_pnl
        pnl_pct = (daily_pnl / day_start * 100) if day_start > 0 else 0.0
        lines = [
            "<b>📈 Paper Portfolio Summary</b>",
            f"Value: <b>${portfolio_value:,.2f}</b>",
            f"Daily P&L: {arrow} ${daily_pnl:+,.2f} ({pnl_pct:+.2f}%)",
            f"Positions: {len(positions)}",
        ]
        if positions:
            lines.append("")
            for sym, info in sorted(positions.items()):
                pl = info.get("unrealized_pl", 0.0)
                plpc = info.get("unrealized_plpc", 0.0) * 100
                tod = info.get("change_today", 0.0) * 100
                pl_arrow = "+" if pl >= 0 else ""
                qty = info.get("qty", 0.0)
                price = info.get("current_price", 0.0)
                qty_str = f"{qty:.0f}" if float(qty).is_integer() else f"{qty:.4f}"
                lines.append(
                    f"  {sym}: {qty_str} @ ${price:.2f} — "
                    f"${pl_arrow}{pl:,.2f} ({pl_arrow}{plpc:.2f}%)"
                )
        return self.send("\n".join(lines))
