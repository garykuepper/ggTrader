"""Binance.US smoke test: keys + tickers + spread + metadata.

Standalone diagnostic — do NOT import ggTrader. Run from project root so
load_dotenv() picks up .env:

    .venv/bin/python scripts/binanceus_smoke_test.py
"""

import os

import ccxt
from dotenv import load_dotenv

load_dotenv()

ex = ccxt.binanceus(
    {
        "apiKey": os.environ["BINANCE_API_LIVE_KEY"],
        "secret": os.environ["BINANCE_SECRET_LIVE_KEY"],
        "enableRateLimit": True,
    }
)

PAIRS = ["BTC/USD", "ETH/USD", "DOGE/USD", "TRX/USD"]

# === Step 1a: authentication + balance ===
print("=== Step 1: Authentication + balance ===")
bal = ex.fetch_balance()
nonzero = {k: v for k, v in bal["total"].items() if v and v > 0}
print(f"Nonzero balances: {nonzero}")

# === Step 1b: tickers + spreads ===
print("\n=== Step 1: Tickers + spreads ===")
for pair in PAIRS:
    t = ex.fetch_ticker(pair)
    bid, ask = t.get("bid"), t.get("ask")
    spread_bps = 10000 * (ask - bid) / bid if bid and ask else float("nan")
    print(f"  {pair}: bid={bid} ask={ask} spread_bps={spread_bps:.1f}")

# === Step 2: pair metadata ===
print("\n=== Step 2: Pair metadata ===")
ex.load_markets()
for pair in PAIRS:
    m = ex.markets[pair]
    print(f"  {pair}:")
    print(f"    min_amount: {m['limits']['amount']['min']}")
    print(f"    min_cost:   {m['limits']['cost']['min']}")
    print(f"    precision:  amount={m['precision']['amount']} price={m['precision']['price']}")
    print(f"    active:     {m.get('active')}")

# Tiny test trade (commented; uncomment after balance funded + permission audit clean)
# order = ex.create_market_buy_order("TRX/USD", 50)  # ~$10 worth
# print("Buy order:", order)
# import time; time.sleep(2)
# closed = ex.fetch_order(order["id"], "TRX/USD")
# print("Filled:", closed["filled"], "at avg", closed.get("average"))
# sell = ex.create_market_sell_order("TRX/USD", closed["filled"])
# print("Sell order:", sell)
