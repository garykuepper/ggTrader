import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
import vectorbt as vbt

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.trading import Trading
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.indicators.signals import Signals


def log_df_info(name: str, df: pd.DataFrame) -> None:
    """
    Log structural information about a DataFrame.
    """
    print(f"\n--- DEBUG: {name} ---")
    print(f"Type: {type(df)}")
    if hasattr(df, "columns"):
        print(f"Columns type: {type(df.columns)}")
        print(f"Columns nlevels: {df.columns.nlevels}")
        print(f"Columns names: {df.columns.names}")
        print(f"Columns values: {df.columns.values[:5]}...")
    if hasattr(df, "index"):
        print(f"Index names: {df.index.names}")


def main() -> None:
    """
    Main entry point for the full simulation debug test.
    """
    parser = argparse.ArgumentParser(description="Run Full Simulation Debug")
    parser.add_argument(
        "--log", action="store_true", help="Redirect output to full_traceback.txt"
    )
    args = parser.parse_args()

    if args.log:
        output_log = "full_traceback.txt"
        f = open(output_log, "w")
        sys.stdout = f
        sys.stderr = f
        print(f"Logging to {output_log}...")

    try:
        # Configuration
        symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
        interval = "4h"
        start_date = "2024-01-01"
        end_date = "2024-01-05"

        k_h = KrakenHistoricalData()
        start_dt = pd.to_datetime(start_date).tz_localize("UTC")
        end_dt = pd.to_datetime(end_date).tz_localize("UTC")

        print("Loading OHLCV data...")
        ohlcv_df = k_h.get_ohlcv_df(
            symbols, interval=interval, start=start_dt, end=end_dt
        )

        # Defensive strip for ohlcv_df names
        if isinstance(ohlcv_df.columns, pd.MultiIndex):
            ohlcv_df.columns.names = [None] * ohlcv_df.columns.nlevels
        else:
            ohlcv_df.columns.name = None
        ohlcv_df.index.name = None

        date_range = ohlcv_df.index

        engine = Trading(
            ohlcv_df=ohlcv_df,
            date_range=date_range,
            start_cash=10000,
            strategy_params={
                "adx_threshold": 25,
                "adx_length": 14,
                "sar_acceleration": 0.02,
                "sar_maximum": 0.2,
                "atr_multiplier": 3.0,
                "atr_length": 14,
                "use_dmp_cross": False,
            },
        )

        print("Running simulation (shortened)...")
        # Simulating first 2 iterations of engine.run()
        for i, current_date in enumerate(date_range[:2]):
            print(f"\n--- Day {i}: {current_date} ---")
            engine.current_date = current_date

            print("Getting movers...")
            daily_movers = engine.screener.get_historical_daily_kraken_by_volume(
                current_date, top_n=5
            )
            log_df_info("daily_movers", daily_movers)
            print(f"Daily movers dtypes:\n{daily_movers.dtypes}")

            print("Calculating signals for movers...")
            engine.calc_signals(daily_movers["symbol"].tolist())
            print("Signals calculated.")

            print("Checking transactions...")
            engine.check_sell()
            engine.check_buy()

        print("\nSimulation test successful!")

    except Exception as e:
        print(f"\nERROR CAUGHT: {e}")
        traceback.print_exc()
    finally:
        if args.log:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            f.close()
            print(f"Debug log written to full_traceback.txt")


if __name__ == "__main__":
    main()
