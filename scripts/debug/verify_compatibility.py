import argparse
import os
import sys
import traceback

import pandas as pd

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.data.kraken.historical_data import KrakenHistoricalData


def test_compatibility() -> None:
    """
    Verifies database compatibility with the Trading engine.
    """
    print("Initializing KrakenHistoricalData (Postgres)...")
    try:
        kh = KrakenHistoricalData()
    except Exception as e:
        print(f"Failed to initialize data manager: {e}")
        return

    # 1. Test get_ohlcv_df (MultiIndex structure)
    print("\n--- Testing get_ohlcv_df (MultiIndex) ---")
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
        # kh.get_ohlcv_df handles single as well but returns MultiIndex
        # kh.reader.get_ohlcv is the underlying call for single non-multi
        df_single = kh.reader.get_ohlcv(sym, interval="1d")
        print(f"Single DF Shape: {df_single.shape}")
        print("Columns:", df_single.columns.tolist())
        if "close" in df_single.columns:
            print("SUCCESS: Single symbol fetch contains 'close'.")
    except Exception as e:
        print(f"Error in single symbol fetch: {e}")


def main() -> None:
    """
    Main orchestration for compatibility test.
    """
    parser = argparse.ArgumentParser(description="Verify Engine/DB Compatibility")
    parser.parse_args()  # Allow --help
    test_compatibility()


if __name__ == "__main__":
    main()
