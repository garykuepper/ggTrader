from dotenv import load_dotenv
import os
import ccxt
import sys

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

# Show your Kraken account balance (all currencies)
print("\nYour account balances:")
balance = kraken.fetch_balance()
print(balance['total'])

# Show your open orders for the symbol
print(f"\nOpen orders for {symbol}:")
open_orders = kraken.fetch_open_orders(symbol)
print(open_orders)

# Show your closed (historical) orders for the symbol
print(f"\nOrder history (closed/canceled) for {symbol}:")
closed_orders = kraken.fetch_closed_orders(symbol)
print(closed_orders)