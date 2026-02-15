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
from ggTrader.indicators.signals import Signals


def debug() -> None:
    """
    Debug MultiIndex OHLCV data and signal calculation.
    """
    try:
        k_h = KrakenHistoricalData()
        symbols = ["BTC"]
        interval = "4h"
        start_dt = pd.to_datetime("2024-01-01").tz_localize("UTC")
        end_dt = pd.to_datetime("2024-01-10").tz_localize("UTC")

        print("Loading ohlcv_df...")
        ohlcv_df = k_h.get_ohlcv_df(
            symbols, interval=interval, start=start_dt, end=end_dt
        )

        print(f"ohlcv_df.columns: {ohlcv_df.columns}")

        print("Calculating signals for BTC...")
        signals = Signals()
        btc_ohlcv = ohlcv_df["BTC"]
        print(f"BTC OHLCV columns: {btc_ohlcv.columns}")

        # Testing the trailing stop calculation directly
        res = signals._atr_trailing_stop_long_ohlc_touch_2d(btc_ohlcv)
        print("Signals calculation successful")
        print(res.head())

    except Exception as e:
        print(f"DEBUG FAILED: {e}")
        traceback.print_exc()


def main() -> None:
    """
    Main orchestration for MultiIndex debug.
    """
    parser = argparse.ArgumentParser(description="Debug MultiIndex Signal Calculation")
    parser.parse_args()  # Allow --help
    debug()


if __name__ == "__main__":
    main()
