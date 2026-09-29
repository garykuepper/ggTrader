"""Read-only check of the live trader's signal path against real data.

Runs today's core signal generation and the trader's pure helpers against the
real tape and the real Alpaca account with a mocked notifier. Places no orders
and sends no Telegram messages. Extend `main()` for the change being deployed.
"""

import argparse
from unittest.mock import MagicMock

from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.cash_sweep import sweep_symbol
from ggTrader.paper.signal_runner import generate_core_signals
from ggTrader.paper.trader import PaperTrader


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    sleeve = generate_core_signals()["sleeves"]["sp500"]
    broker = AlpacaBroker()  # only read methods are called below
    positions = {s: p for s, p in broker.get_positions().items() if s != sweep_symbol()}
    trader = PaperTrader(broker, MagicMock(), dry_run=True)

    print(f"as_of {sleeve['as_of']}  universe {sleeve['universe_size']}")
    print(f"buys ({len(sleeve['buys'])}): {sleeve['buys']}")
    held_sells = [s for s in sleeve["sells"] if s in positions]
    print(f"last-bar sells of held names: {held_sells}")
    if hasattr(trader, "_missed_exits"):
        missed = trader._missed_exits(positions, sleeve.get("last_exit", {}), set(held_sells))
        print(f"missed-exit catch-up sells: {missed}")
    print(f"held strategy positions: {len(positions)}  risk: {trader._risk.cfg}")


if __name__ == "__main__":
    main()
