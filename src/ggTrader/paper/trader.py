"""Paper trading orchestrator: signals -> orders -> risk guardrails -> notifications."""

from __future__ import annotations

import logging

from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.notifier import TelegramNotifier
from ggTrader.paper.persist import (
    get_latest_snapshot,
    init_paper_schema,
    log_snapshot,
    log_trade,
)
from ggTrader.paper.risk import RiskConfig, RiskGuard
from ggTrader.paper.signal_runner import generate_signals

_log = logging.getLogger(__name__)


class PaperTrader:
    """Orchestrates daily paper trading: generate signals, execute, notify."""

    def __init__(
        self,
        broker: AlpacaBroker,
        notifier: TelegramNotifier,
        risk_cfg: RiskConfig | None = None,
    ) -> None:
        self._broker = broker
        self._notifier = notifier
        self._risk = RiskGuard(risk_cfg)

    def run(self) -> dict:
        try:
            init_paper_schema()
        except Exception as exc:
            _log.warning("DB schema init failed (non-fatal): %s", exc)

        try:
            signals = generate_signals()
        except Exception as exc:
            self._notifier.send(f"Paper trading failed: signal generation error\n{exc}")
            raise

        gate = signals.get("gate", {})
        if gate.get("gate_enabled") and gate.get("scores"):
            scores = gate["scores"]
            kept = gate.get("kept_buys", 0)
            raw = gate.get("raw_buys", 0)
            score_strs = [f"{s}({v:.2f})" for s, v in sorted(scores.items(), key=lambda x: -x[1])]
            self._notifier.send(
                f"<b>🤖 ML Gate:</b> kept {kept}/{raw} signals\n" + ", ".join(score_strs)
            )
        elif gate.get("gate_enabled") is False:
            _log.info("ML gate disabled (no model file)")

        account = self._broker.get_account()
        positions = self._broker.get_positions()
        portfolio_value = account["portfolio_value"]

        # Track peak for drawdown calculation
        self._risk.update_peak(portfolio_value)

        # Check max drawdown halt
        halted, halt_reason = self._risk.check_drawdown_halt(portfolio_value)
        if halted:
            self._notifier.send(f"<b>🛑 HALTED:</b> {halt_reason}")
            return {"buys": [], "sells": [], "errors": [halt_reason]}

        # Check daily loss limit
        prev_snapshot_value = None
        try:
            prev_snapshot_value = get_latest_snapshot()
        except Exception:
            pass
        day_start = prev_snapshot_value or portfolio_value

        daily_halted, daily_reason = self._risk.check_daily_loss(portfolio_value, day_start)
        if daily_halted:
            self._notifier.send(f"<b>⚠️ Daily limit:</b> {daily_reason}")
            return {"buys": [], "sells": [], "errors": [daily_reason]}

        executed_sells: list[str] = []
        executed_buys: list[str] = []
        errors: list[str] = []

        # Sells first — free up position slots
        for symbol in signals["sells"]:
            if symbol not in positions:
                continue
            qty = positions[symbol]["qty"]
            try:
                oid = self._broker.submit_sell(symbol, qty)
                executed_sells.append(symbol)
                self._notifier.trade_alert("SELL", symbol, positions[symbol]["market_value"], oid)
                log_trade(signals["as_of"], "SELL", symbol, positions[symbol]["market_value"], oid)
            except Exception as exc:
                errors.append(f"SELL {symbol}: {exc}")

        # Buys — respect position limits
        slots_available = self._risk.max_new_positions(len(positions) - len(executed_sells))
        notional = self._risk.position_notional(portfolio_value)

        buys_attempted = 0
        for symbol in signals["buys"]:
            if symbol in positions:
                continue
            if buys_attempted >= slots_available:
                _log.info(
                    "Max positions reached (%d), skipping %s", self._risk.cfg.max_positions, symbol
                )
                break
            if self._risk.check_concentration(symbol, positions, portfolio_value):
                _log.info("Concentration limit for %s, skipping", symbol)
                continue
            try:
                oid = self._broker.submit_buy(symbol, notional)
                executed_buys.append(symbol)
                buys_attempted += 1
                self._notifier.trade_alert("BUY", symbol, notional, oid)
                log_trade(signals["as_of"], "BUY", symbol, notional, oid)
            except Exception as exc:
                errors.append(f"BUY {symbol}: {exc}")

        new_account = self._broker.get_account()
        new_value = new_account["portfolio_value"]
        updated_positions = self._broker.get_positions()

        daily_pnl = new_value - day_start

        risk_line = (
            f"Positions: {len(updated_positions)}/{self._risk.cfg.max_positions} | "
            f"Size: ${notional:.0f}/trade ({self._risk.cfg.position_pct:.1%})"
        )
        self._notifier.daily_summary(new_value, daily_pnl, updated_positions)
        self._notifier.send(f"<b>📊 Risk:</b> {risk_line}")

        try:
            log_snapshot(
                signals["as_of"],
                new_value,
                new_account["cash"],
                updated_positions,
            )
        except Exception as exc:
            _log.warning("DB snapshot failed (non-fatal): %s", exc)

        return {"buys": executed_buys, "sells": executed_sells, "errors": errors}


def run_paper_trading() -> dict:
    """Convenience entry point: wire up broker + notifier and run."""
    broker = AlpacaBroker()
    notifier = TelegramNotifier()
    trader = PaperTrader(broker, notifier)
    return trader.run()
