"""Paper trading orchestrator: signals -> orders -> notifications."""

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
from ggTrader.paper.signal_runner import generate_signals

_log = logging.getLogger(__name__)


class PaperTrader:
    """Orchestrates daily paper trading: generate signals, execute, notify."""

    def __init__(
        self,
        broker: AlpacaBroker,
        notifier: TelegramNotifier,
        position_size: float = 0.02,
    ) -> None:
        self._broker = broker
        self._notifier = notifier
        self._position_size = position_size

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

        account = self._broker.get_account()
        positions = self._broker.get_positions()
        portfolio_value = account["portfolio_value"]

        executed_sells: list[str] = []
        executed_buys: list[str] = []
        errors: list[str] = []

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

        notional = round(portfolio_value * self._position_size, 2)
        for symbol in signals["buys"]:
            if symbol in positions:
                continue
            try:
                oid = self._broker.submit_buy(symbol, notional)
                executed_buys.append(symbol)
                self._notifier.trade_alert("BUY", symbol, notional, oid)
                log_trade(signals["as_of"], "BUY", symbol, notional, oid)
            except Exception as exc:
                errors.append(f"BUY {symbol}: {exc}")

        new_account = self._broker.get_account()
        new_value = new_account["portfolio_value"]
        updated_positions = self._broker.get_positions()

        prev_snapshot_value = None
        try:
            prev_snapshot_value = get_latest_snapshot()
        except Exception:
            pass
        daily_pnl = new_value - (prev_snapshot_value or portfolio_value)

        self._notifier.daily_summary(new_value, daily_pnl, updated_positions)

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
