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
            msg += f"Shares: {qty:.4f} @ ${price:.2f}\nTotal Value: ${qty * price:.2f}\n"
        else:
            msg += f"Amount: ${amount:.2f}\n"
        msg += f"Order: <code>{order_id}</code>"
        return self.send(msg)

    def daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
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
        position's P&L is ever hidden -- the corrected figures are real
        numbers to show, not an unknown to hide.

        `dividend_total` is the cumulative dividend accrual (an all-time
        reporting correction, not real cash -- see `dividend_check.py`);
        `new_dividend_accruals` lists any events newly credited *this run*
        (each `{symbol, ex_date, rate, qty, amount}`), so a reader can see
        exactly what changed and why it's not reflected in Alpaca's own
        balance.
        """
        arrow = "🟢" if daily_pnl >= 0 else "🔴"
        day_start = portfolio_value - daily_pnl
        pnl_pct = (daily_pnl / day_start * 100) if day_start > 0 else 0.0
        lines = ["<b>📈 Paper Portfolio Summary</b>"]
        if split_corrections:
            corr_str = ", ".join(f"{sym} {factor:g}x" for sym, factor in split_corrections.items())
            lines.append(
                f"<b>⚠️ Split correction applied:</b> {corr_str} — the broker "
                f"never applied this split to the account; figures below are "
                f"split-adjusted."
            )
            if booked_equity is not None:
                lines.append(
                    f"Booked equity (broker, uncorrected): ${booked_equity:,.2f} | "
                    f"Split-adjusted equity: ${portfolio_value:,.2f}"
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
        lines += [
            f"Value: <b>${portfolio_value:,.2f}</b>",
            f"Daily P&L: {arrow} ${daily_pnl:+,.2f} ({pnl_pct:+.2f}%)",
        ]
        if total_pnl is not None and unrealized_pnl is not None:
            realized_pnl = total_pnl - unrealized_pnl
            total_pct = (total_pnl / day_start * 100) if day_start > 0 else 0.0
            rlz_arrow = "+" if realized_pnl >= 0 else ""
            unrlz_arrow = "+" if unrealized_pnl >= 0 else ""
            lines.append(
                f"Total P&L: ${total_pnl:+,.2f} ({total_pct:+.2f}%) — "
                f"realized: ${rlz_arrow}{realized_pnl:,.2f}, "
                f"unrealized: ${unrlz_arrow}{unrealized_pnl:,.2f}"
            )
        lines.append(f"Positions: {len(positions)}")
        if positions:
            lines.append("")
            ranked = sorted(
                positions.items(), key=lambda kv: kv[1].get("unrealized_plpc", 0.0), reverse=True
            )
            for sym, info in ranked:
                qty = info.get("qty", 0.0)
                price = info.get("current_price", 0.0)
                qty_str = f"{qty:.0f}" if float(qty).is_integer() else f"{qty:.4f}"
                pl = info.get("unrealized_pl", 0.0)
                plpc = info.get("unrealized_plpc", 0.0) * 100
                pl_arrow = "+" if pl >= 0 else ""
                lines.append(
                    f"  {sym}: {qty_str} @ ${price:.2f} — "
                    f"${pl_arrow}{pl:,.2f} ({pl_arrow}{plpc:.2f}%)"
                )
        if errors:
            lines.append("")
            lines.append(f"<b>❌ Errors ({len(errors)})</b>")
            for err in errors:
                lines.append(f"  {err}")
        return self.send("\n".join(lines))
