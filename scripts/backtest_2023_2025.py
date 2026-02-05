import os
import sys
import pandas as pd
from tabulate import tabulate

# Add src to sys.path
sys.path.append(os.path.abspath('src'))

from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData
from ggTrader.core.Trading import Trading

def run_backtest_2023_2025():
    # --- 1. Configuration ---
    symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
    interval = "4h"
    start_date = "2023-01-01"
    end_date = "2025-12-31"
    start_cash = 10000
    top_n_movers = 5
    max_position = 0.2

    # Strategy Parameters (Same as run_backtest.py)
    strategy_params = {
        'adx_threshold': 25,
        'adx_length': 14,
        'sar_acceleration': 0.02,
        'sar_maximum': 0.2,
        'atr_multiplier': 3.0,
        'atr_length': 14,
        'use_dmp_cross': False
    }

    print(f"--- Backtest Configuration (Long Term) ---")
    print(f"Symbols: {symbols}")
    print(f"Range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print(f"Start Cash: {start_cash}")
    print(f"------------------------------------------\n")

    # --- 2. Data Loading ---
    k_h = KrakenHistoricalData()
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC')

    print("Loading data for 2023-2025...")
    ohlcv_df = k_h.get_ohlcv_df(symbols, interval=interval, start=start_dt, end=end_dt)

    if ohlcv_df.empty:
        print("No data found for the specified range and symbols.")
        return

    # Defensive cleaning for MultiIndex (as seen in existing scripts)
    if isinstance(ohlcv_df.columns, pd.MultiIndex):
        ohlcv_df.columns.names = [None] * ohlcv_df.columns.nlevels
    else:
        ohlcv_df.columns.name = None
    ohlcv_df.index.name = None

    print(f"Loaded data rows: {len(ohlcv_df)}")
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
    print("\n--- Backtest Results (2023-2025) ---")
    print(f"Final Portfolio Value: {engine.portfolio.total_value:.2f}")
    print(f"Profit/Loss: {engine.portfolio.profit:.2f} ({engine.portfolio.profit_pct * 100:.2f}%)")
    print(f"Total Transactions: {len(engine.portfolio.trades)}")

    if engine.portfolio.trades:
        print("\nLast 10 trades:")
        trades_dict = [t.as_dict() for t in engine.portfolio.trades]
        history_df = pd.DataFrame(trades_dict)
        print(tabulate(history_df.tail(10), headers="keys", tablefmt="github"))

    # Save results to a file for review
    results_path = "data/backtest_results_2023_2025.csv"
    if engine.portfolio.trades:
        history_df.to_csv(results_path, index=False)
        print(f"\nTrade history saved to {results_path}")

if __name__ == "__main__":
    run_backtest_2023_2025()
