import os
import pandas as pd
from tabulate import tabulate
from ggTrader.core.Trading import Trading
from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData
from datetime import datetime
def run_backtest():
    # --- 1. Configuration ---
    symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
    interval = "4h"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    start_cash = 10000
    top_n_movers = 5
    max_position = 0.2
    
    # Strategy Parameters
    strategy_params = {
        'adx_threshold': 25,
        'adx_length': 14,
        'sar_acceleration': 0.02,
        'sar_maximum': 0.2,
        'atr_multiplier': 3.0,
        'atr_length': 14,
        'use_dmp_cross': False
    }

    print(f"--- Backtest Configuration ---")
    print(f"Symbols: {symbols}")
    print(f"Range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print(f"Start Cash: {start_cash}")
    print(f"------------------------------\n")

    # --- 2. Data Loading ---
    k_h = KrakenHistoricalData()
    # Note: Using get_ohlcv_df which aligns symbols
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC')
    
    print("Loading data...")
    ohlcv_df = k_h.get_ohlcv_df(symbols, interval=interval, start=start_dt, end=end_dt)
    
    # Strip names to avoid MultiIndex length mismatch errors in external libraries
    if isinstance(ohlcv_df.columns, pd.MultiIndex):
        ohlcv_df.columns.names = [None] * ohlcv_df.columns.nlevels
    else:
        ohlcv_df.columns.name = None
    ohlcv_df.index.name = None
    
    if ohlcv_df.empty:
        print("No data found for the specified range and symbols.")
        return

    # Filter date range for the simulation
    date_range = ohlcv_df.index
    
    # --- 3. Run Simulation ---
    engine = Trading(
        ohlcv_df=ohlcv_df, 
        date_range=date_range, 
        start_cash=start_cash,
        top_n_movers=top_n_movers,
        max_position=max_position,
        strategy_params=strategy_params
    )
    
    print("Running backtest engine...")
    engine.run()
    
    # --- 4. Results ---
    print("\n--- Backtest Results ---")
    # stats = engine.portfolio.get_stats() # Assuming Portfolio.py has get_stats or similar
    # print(tabulate(stats.items(), headers=["Metric", "Value"], tablefmt="github"))
    
    print(f"Final Portfolio Value: {engine.portfolio.total_value:.2f}")
    print(f"Profit/Loss: {engine.portfolio.profit:.2f} ({engine.portfolio.profit_pct * 100:.2f}%)")
    print(f"Total Transactions: {len(engine.portfolio.trades)}")
    
    if engine.portfolio.trades:
        print("\nLast 10 trades:")
        trades_dict = [t.as_dict() for t in engine.portfolio.trades]
        history_df = pd.DataFrame(trades_dict)
        print(tabulate(history_df.tail(10), headers="keys", tablefmt="github"))

if __name__ == "__main__":
    run_backtest()
