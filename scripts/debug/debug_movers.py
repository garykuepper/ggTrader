import argparse
import os
import sys

import pandas as pd

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.data.kraken.historical_data import KrakenHistoricalData


def test_movers() -> None:
    """
    Test retrieval of historical daily movers.
    """
    k_h = KrakenHistoricalData()
    date = pd.Timestamp("2024-01-01").tz_localize("UTC")

    print(f"Attempting to get movers for {date}...")
    try:
        movers = k_h.get_historical_movers_by_day(date, top_n=5)
        print(f"Success! Found {len(movers)} movers.")
        print(movers)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Failed to get movers: {e}")


def main() -> None:
    """
    Main orchestration for movers test.
    """
    parser = argparse.ArgumentParser(description="Test Historical Movers Retrieval")
    parser.parse_args()  # Allow --help
    test_movers()


if __name__ == "__main__":
    main()
