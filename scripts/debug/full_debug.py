import os
import sys
import traceback
import pandas as pd
import numpy as np

# Set PYTHONPATH to src to find ggTrader
sys.path.append(os.path.abspath('src'))

import vectorbt as vbt
from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData
from ggTrader.indicators.Signals import Signals
from ggTrader.core.Trading import Trading

def log_df_info(name, df):
    print(f"\n--- DEBUG: {name} ---")
    print(f"Type: {type(df)}")
    if hasattr(df, 'columns'):
        print(f"Columns type: {type(df.columns)}")
        print(f"Columns nlevels: {df.columns.nlevels}")
        print(f"Columns names: {df.columns.names}")
        print(f"Columns values: {df.columns.values[:5]}...")
    if hasattr(df, 'index'):
        print(f"Index names: {df.index.names}")

from ggTrader.core.Screener import Screener

def main():
    try:
        # Configuration
        symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
        interval = "4h"
        start_date = "2024-01-01"
        end_date = "2024-01-05"
        
        k_h = KrakenHistoricalData()
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        
        print("Loading OHLCV data...")
        ohlcv_df = k_h.get_ohlcv_df(symbols, interval=interval, start=start_dt, end=end_dt)
        
        # Defensive strip for ohlcv_df (like in run_backtest.py)
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
                'adx_threshold': 25,
                'adx_length': 14,
                'sar_acceleration': 0.02,
                'sar_maximum': 0.2,
                'atr_multiplier': 3.0,
                'atr_length': 14,
                'use_dmp_cross': False
            }
        )
        
        print("Running simulation (shortened)...")
        # Simulating first 2 iterations of engine.run()
        for i, current_date in enumerate(date_range[:2]):
            print(f"\n--- Day {i}: {current_date} ---")
            engine.current_date = current_date
            
            print("Getting movers...")
            daily_movers = engine.screener.get_historical_daily_kraken_by_volume(current_date, top_n=5)
            log_df_info("daily_movers", daily_movers)
            print(f"Daily movers dtypes:\n{daily_movers.dtypes}")
            
            print("Calculating signals for movers...")
            engine.calc_signals(daily_movers['symbol'].tolist())
            print("Signals calculated.")
            
            print("Checking buy/sell...")
            engine.check_sell()
            engine.check_buy()
            
        print("\nSimulation test successful!")
        
    except Exception as e:
        print(f"\nERROR CAUGHT: {e}")
        traceback.print_exc()
        
if __name__ == "__main__":
    with open("full_traceback.txt", "w") as f:
        sys.stdout = f
        sys.stderr = f
        main()
