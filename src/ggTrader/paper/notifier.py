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

    def trade_alert(self, side: str, symbol: str, amount: float, order_id: str) -> bool:
        msg = (
            f"<b>📊 Paper Trade</b>\n"
            f"{side} <b>{symbol}</b>\n"
            f"Amount: ${amount:.2f}\n"
            f"Order: <code>{order_id}</code>"
        )
        return self.send(msg)

    def daily_summary(
        self, portfolio_value: float, daily_pnl: float, positions: dict[str, dict]
    ) -> bool:
        arrow = "🟢" if daily_pnl >= 0 else "🔴"
        lines = [
            "<b>📈 Paper Portfolio Summary</b>",
            f"Value: <b>${portfolio_value:,.2f}</b>",
            f"Daily P&L: {arrow} ${daily_pnl:+,.2f}",
            f"Positions: {len(positions)}",
        ]
        if positions:
            lines.append("")
            for sym, info in sorted(positions.items()):
                pl = info.get("unrealized_pl", 0.0)
                pl_arrow = "+" if pl >= 0 else ""
                lines.append(f"  {sym}: {info.get('qty', 0):.0f} sh (${pl_arrow}{pl:,.2f})")
        return self.send("\n".join(lines))
