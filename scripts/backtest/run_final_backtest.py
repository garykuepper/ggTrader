import pandas as pd
import sys
import os
import traceback

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

try:
    from ggTrader.core.Trading import Trading
    from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData
except ImportError:
    print("Failed to import ggTrader modules. Make sure you are running from project root or scripts folder.")
    sys.exit(1)

# --- Configuration ---
# User can modify these parameters
SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "DOT", "LINK", "LTC", "BCH"]
INTERVAL = "4h"
START_DATE = "2024-01-01"
END_DATE = "2024-06-01" 
START_CASH = 10000
TOP_N_MOVERS = 5
MAX_POSITION_PCT = 0.2

# Strategy Parameters (Example values - user should tune these based on WFO/Sensitivity)
STRATEGY_PARAMS = {
    'adx_threshold': 25,
    'adx_length': 14,
    'sar_acceleration': 0.02,
    'sar_maximum': 0.2,
    'atr_multiplier': 3.0,
    'atr_length': 14,
    'use_dmp_cross': True
}

def main():
    print(f"--- Starting Final Backtest ---")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Symbols: {len(SYMBOLS)} watched")
    print(f"Params: {STRATEGY_PARAMS}")
    print("-" * 30)

    # 1. Load Data
    print("Loading data...")
    k_h = KrakenHistoricalData()
    start_dt = pd.to_datetime(START_DATE).tz_localize('UTC')
    end_dt = pd.to_datetime(END_DATE).tz_localize('UTC')
    
    ohlcv_df = k_h.get_ohlcv_df(SYMBOLS, interval=INTERVAL, start=start_dt, end=end_dt)
    
    if ohlcv_df.empty:
        print("No data found for the specified range.")
        return

    # Strip MultiIndex
    if isinstance(ohlcv_df.columns, pd.MultiIndex):
        ohlcv_df.columns.names = [None] * ohlcv_df.columns.nlevels
    else:
        ohlcv_df.columns.name = None
    ohlcv_df.index.name = None
    date_range = ohlcv_df.index
    
    print(f"Data loaded: {len(ohlcv_df)} rows.")

    # 2. Setup Trading Engine
    engine = Trading(
        ohlcv_df=ohlcv_df,
        date_range=date_range,
        start_cash=START_CASH,
        top_n_movers=TOP_N_MOVERS,
        max_position=MAX_POSITION_PCT,
        strategy_params=STRATEGY_PARAMS
    )
    
    # 3. Run
    print("Running simulation...")
    try:
        engine.run()
    except Exception as e:
        print(f"Backtest failed: {e}")
        traceback.print_exc()
        return

    # 4. Results
    print("\n" + "="*30)
    print("       BACKTEST RESULTS       ")
    print("="*30)
    
    # Use Portfolio's built-in reporting
    engine.portfolio.print_stats()
    engine.portfolio.print_profit_per_symbol()
    engine.portfolio.print_trades()
    
    # Optional: Plot equity curve
    # engine.portfolio.plot_equity_curve(title="Final Backtest Equity Curve")
    print("\nDone.")

if __name__ == "__main__":
    main()
