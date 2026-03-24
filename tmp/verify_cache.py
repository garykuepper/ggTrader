import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ggTrader.data.live.cached_loader import CachedExchangeLoader

def test_cached_loader():
    print("Initializing CachedExchangeLoader...")
    try:
        loader = CachedExchangeLoader(exchange_id="kraken")
    except ValueError as e:
        print(f"Skipping DB tests: {e}")
        return

    symbols = ["BTC-USD"]
    interval = "1h"
    limit = 5

    print(f"Fetching {limit} bars for {symbols} at {interval}...")
    # First fetch - should fetch from exchange and save to DB
    df1 = loader.fetch_ohlcv(symbols, interval, limit=limit)
    print("Fetch 1 complete.")
    print(df1)

    if df1.empty:
        print("Error: No data fetched.")
        return

    # Second fetch - should fetch from DB
    print("\nFetching again (should be from DB cache)...")
    df2 = loader.fetch_ohlcv(symbols, interval, limit=limit)
    print("Fetch 2 complete.")
    print(df2)

    if df1.equals(df2):
        print("\nSUCCESS: Data matches exactly!")
    else:
        print("\nNote: Data might differ slightly due to time passing or incomplete candles.")
        # Check if indices match
        if all(df1.index == df2.index):
            print("Indices match!")
        else:
            print("Indices differ.")

if __name__ == "__main__":
    test_cached_loader()
