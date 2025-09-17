from dotenv import load_dotenv
import os
import ccxt
import sys
from tabulate import tabulate
# Load environment variables from .env
if not load_dotenv("../.env"):
    print("Warning: .env file not found or failed to load.")

API_KEY = os.getenv('KRAKEN_KEY')
API_SECRET = os.getenv('KRAKEN_SECRET')
if not API_KEY or not API_SECRET:
    print("Error: KRAKEN_KEY and/or KRAKEN_SECRET not set.")
    sys.exit(1)

kraken = ccxt.kraken({
    'apiKey': API_KEY,
    'secret': API_SECRET,
})
kraken.load_markets()

usd_to_spend = 5
symbol = 'BTC/USD'

last = kraken.fetch_ticker(symbol)['last']
amt = usd_to_spend / last

min_amt = kraken.markets[symbol]['limits']['amount']['min']
amt = max(amt, min_amt)
amt = round(amt, 8)

# order = kraken.create_order(
#     symbol=symbol,
#     type='market',
#     side='buy',
#     amount=amt,
# )
# print("Order info:")
# print(order)
from tabulate import tabulate

def print_balances_usd(exchange):
    """
    Fetch balances and print table of [Ticker, Amount, USD Value] using current prices.
    Handles USD, USDT, USDC as fiat/stablecoins.
    """
    balance = exchange.fetch_balance()
    balances = balance['total']
    usd_values = []
    for ticker, amount in balances.items():
        if amount == 0 or ticker is None:
            continue
        if ticker in ["USD", "USDT", "USDC"]:
            usd_value = amount
        else:
            try:
                symbol = f"{ticker}/USD"
                ticker_info = exchange.fetch_ticker(symbol)
                price = ticker_info['last']
                usd_value = amount * price
            except Exception:
                usd_value = None  # Price not available
        usd_values.append([ticker, amount, usd_value])
    print(tabulate(
        usd_values,
        headers=["Ticker", "Amount", "USD Value"],
        tablefmt="github",
        floatfmt=(".8f", ".8f", ".2f")
    ))
# Show your Kraken account balance (all currencies)
print_balances_usd(kraken)

# Show your open orders for the symbol
print(f"\nOpen orders for {symbol}:")
open_orders = kraken.fetch_open_orders(symbol)

print(tabulate(open_orders, headers='keys', tablefmt='github'))

# Show your closed (historical) orders for the symbol
print(f"\nOrder history (closed/canceled) for {symbol}:")
closed_orders = kraken.fetch_closed_orders(symbol)
print(tabulate(closed_orders, headers='keys', tablefmt='github'))