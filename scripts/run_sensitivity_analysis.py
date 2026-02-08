import optuna
import pandas as pd
import sys
import os
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ggTrader.core.Trading import Trading
from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData
try:
    from scripts.sensitivity_visualizer import plot_optimization_landscape
except ImportError:
    from sensitivity_visualizer import plot_optimization_landscape

def objective(trial):
    # 1. Define Parameter Space
    params = {
        'adx_threshold': trial.suggest_int('adx_threshold', 15, 35),
        'adx_length': trial.suggest_int('adx_length', 10, 20),
        'sar_acceleration': trial.suggest_float('sar_acceleration', 0.01, 0.05, step=0.005),
        'sar_maximum': trial.suggest_float('sar_maximum', 0.1, 0.3, step=0.05),
        'atr_multiplier': trial.suggest_float('atr_multiplier', 1.5, 4.0, step=0.5),
        'atr_length': trial.suggest_int('atr_length', 10, 20),
        'use_dmp_cross': trial.suggest_categorical('use_dmp_cross', [True, False])
    }

    # 2. Configuration (Fixed for consistency)
    symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE"] # Can be expanded or dynamic
    interval = "4h"
    start_date = "2024-01-01"
    end_date = "2024-03-31" # Short range for speed, user can expand
    start_cash = 10000
    top_n_movers = 5
    max_position = 0.2

    # 3. Load Data (Load once outside if possible, but inside for now to keep self-contained)
    # Ideally, pass data in to avoid reloading every trial, but Trading logic is coupled
    # We will use a cached global variable pattern or similar if performance is an issue.
    # For now, let's load logic here. 
    # NOTE: To make this efficient, we should load data OUTSIDE the objective function.
    
    # Using the global `ohlcv_df` loaded in `main`
    global global_ohlcv_df, global_date_range

    if global_ohlcv_df.empty:
        return -float('inf')

    # 4. Run Trading Engine
    engine = Trading(
        ohlcv_df=global_ohlcv_df,
        date_range=global_date_range,
        start_cash=start_cash,
        top_n_movers=top_n_movers,
        max_position=max_position,
        strategy_params=params
    )
    
    # Check if we should suppress output
    # sys.stdout = open(os.devnull, 'w') # Optional: Quiet mode
    try:
        engine.run()
    except Exception as e:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"Trial failed: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*50 + "\n")
        print(f"Trial failed: {e}")
        return -float('inf')
    # sys.stdout = sys.__stdout__

    # 5. Objective Metric: Net Profit or Sharpe
    # Simple Net Profit for now
    retained_value = engine.portfolio.total_value
    net_profit = retained_value - start_cash
    
    return net_profit

if __name__ == "__main__":
    # --- Data Loading (Once) ---
    symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "DOT", "LINK", "LTC", "BCH"]
    interval = "4h"
    start_date = "2024-01-01"
    end_date = "2024-06-01" # 6 months
    
    print("Loading data for optimization...")
    k_h = KrakenHistoricalData()
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC')
    
    global_ohlcv_df = k_h.get_ohlcv_df(symbols, interval=interval, start=start_dt, end=end_dt)
    
    # Strip names to avoid MultiIndex length mismatch errors
    if isinstance(global_ohlcv_df.columns, pd.MultiIndex):
        global_ohlcv_df.columns.names = [None] * global_ohlcv_df.columns.nlevels
    else:
        global_ohlcv_df.columns.name = None
    global_ohlcv_df.index.name = None
    
    global_date_range = global_ohlcv_df.index
    print(f"Data loaded. Rows: {len(global_ohlcv_df)}")

    # --- Optimization ---
    study_name = "trading_sensitivity_analysis"
    storage_name = "sqlite:///{}.db".format(study_name)
    
    # Delete previous study if exists to start fresh (optional)
    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except:
        pass

    study = optuna.create_study(
        study_name=study_name, 
        direction="maximize", 
        storage=storage_name,
        load_if_exists=True
    )
    
    print("Starting optimization... (Press Ctrl+C to stop early)")
    try:
        study.optimize(objective, n_trials=20) # Start with 20 trials for testing
    except KeyboardInterrupt:
        print("Optimization stopped by user.")

    # --- Results ---
    print("\n--- Optimization Results ---")
    print(f"Best Value (Profit): ${study.best_value:.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # --- Visualization ---
    try:
        print("\nGenerating plots...")
        plot_optimization_landscape(study, params_to_plot=['adx_threshold', 'atr_multiplier'], metric_name='Net Profit')
    except Exception as e:
        print(f"Plotting failed: {e}")
