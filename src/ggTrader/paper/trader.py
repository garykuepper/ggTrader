"""Paper trading orchestrator: signals -> orders -> notifications."""

from __future__ import annotations

from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.notifier import TelegramNotifier
from ggTrader.paper.persist import init_paper_schema, log_snapshot, log_trade
from ggTrader.paper.signal_runner import generate_signals


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
        init_paper_schema()
        signals = generate_signals()
        account = self._broker.get_account()
        positions = self._broker.get_positions()
        portfolio_value = account["portfolio_value"]

        executed_sells: list[str] = []
        executed_buys: list[str] = []
        errors: list[str] = []

        # Phase 1: Sell exits (free up capital first)
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

        # Phase 2: Buy entries (skip if already holding)
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

        # Phase 3: Daily summary
        updated_positions = self._broker.get_positions()
        prev_value = portfolio_value
        new_account = self._broker.get_account()
        daily_pnl = new_account["portfolio_value"] - prev_value
        self._notifier.daily_summary(new_account["portfolio_value"], daily_pnl, updated_positions)

        log_snapshot(
            signals["as_of"],
            new_account["portfolio_value"],
            new_account["cash"],
            updated_positions,
        )

        return {"buys": executed_buys, "sells": executed_sells, "errors": errors}


def run_paper_trading() -> dict:
    """Convenience entry point: wire up broker + notifier and run."""
    broker = AlpacaBroker()
    notifier = TelegramNotifier()
    trader = PaperTrader(broker, notifier)
    return trader.run()
