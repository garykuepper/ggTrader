import os
import sys
import pandas as pd
import traceback

# Setup path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.data.kraken.historical_data import KrakenHistoricalData


def test_compatibility():
    print("Initializing KrakenHistoricalData (Postgres)...")
    kh = KrakenHistoricalData()

    # 1. Test get_ohlcv_df (MultiIndex structure)
    print("\n--- Testing get_ohlcv_df (MultiIndex) ---")
    symbols = ["XBT", "ETH"]  # Assuming these exist or we can find some
    # We need to find symbols that actually have data.
    # Let's list available first
    available = kh.reader.list_symbols()
    print(f"Available symbols (first 5): {available[:5]}")

    if not available:
        print("No symbols found in DB. Cannot test compatibility.")
        return

    test_syms = available[:3]
    print(f"Testing with: {test_syms}")

    try:
        df = kh.get_ohlcv_df(
            test_syms, interval="1d", start="2020-01-01", end="2025-01-01"
        )

        print(f"DataFrame Shape: {df.shape}")
        if df.empty:
            print("Warning: Returned DataFrame is empty.")
        else:
            print("Columns Level 0 (Symbols):", df.columns.levels[0].tolist())
            print("Columns Level 1 (Metrics):", df.columns.levels[1].tolist())
            print("Index Type:", type(df.index))

            # Verify compatibility requirements for Trading class
            # 1. MultiIndex with Symbol at level 0?
            is_multi = isinstance(df.columns, pd.MultiIndex)
            print(f"Is MultiIndex? {is_multi}")

            if is_multi and set(test_syms).intersection(df.columns.levels[0]):
                print(
                    "SUCCESS: DataFrame structure matches Trading class requirements."
                )
            else:
                print("FAILURE: DataFrame structure mismatch.")
                print(df.head())

    except Exception as e:
        print(f"Error in get_ohlcv_df: {e}")
        traceback.print_exc()

    # 2. Test Single Symbol fetch
    print("\n--- Testing get_ohlcv (Single Symbol) ---")
    try:
        sym = test_syms[0]
        df_single = kh.get_ohlcv(sym, interval="1d")
        print(f"Single DF Shape: {df_single.shape}")
        print("Columns:", df_single.columns.tolist())
        if "close" in df_single.columns:
            print("SUCCESS: Single symbol fetch contains 'close'.")
    except Exception as e:
        print(f"Error in get_ohlcv: {e}")


if __name__ == "__main__":
    test_compatibility()
