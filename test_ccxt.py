import os

import ccxt
from dotenv import load_dotenv

load_dotenv()
exchange = ccxt.kraken({
    'apiKey': os.getenv('KRAKEN_KEY'),
    'secret': os.getenv('KRAKEN_SECRET'),
})

print("Fetching balance...")
bal = exchange.fetch_balance()
print(f"Total Balance keys: {bal['total'].keys()}")

# How to compute total USD equivalent?
# Kraken doesn't automatically return a "Total USD Equivalent" in the balance dict standard
# But let's check bal['info'] to see if Kraken includes it natively (some exchanges do via 'total_fiat').
print("Kraken raw info:", {k: v for k,v in bal['info'].items() if isinstance(v, (int, float, str))})

# Calculate manual total portfolio value
total_usd = 0.0
free_usd = 0.0
try:
    tickers = exchange.fetch_tickers()
    for coin, amount in bal['total'].items():
        if amount > 0:
            if coin in ['USD', 'USDT', 'USDC', 'ZUSD']:
                total_usd += amount
                if coin in ['USD', 'ZUSD']:
                    free_usd = bal['free'].get(coin, 0.0)
            else:
                symbol = f"{coin}/USD"
                if symbol in tickers:
                    val = amount * tickers[symbol]['last']
                    total_usd += val
    print(f"Calculated Total Portfolio: ${total_usd:.2f} (Free USD: ${free_usd:.2f})")
except Exception as e:
    print("Error doing manual calc:", e)

# Test OCO formatting query limits
print("Kraken supports orderTypes:", exchange.has['createOrder'])
# For stop_loss_profit, Kraken actually doesn't have an explicit OCO boolean,
# it has a native order type "stop-loss-profit" where `price` is stop limit and `price2` is take profit.
