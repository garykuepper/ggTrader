import optuna
import pandas as pd
import sys
import os
import argparse
import traceback
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ggTrader.core.trading import Trading
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.utils.results_manager import ResultsManager
import vectorbt as vbt
import numpy as np
from tqdm import tqdm
import logging

# --- Configuration ---
SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "DOT", "LINK", "LTC", "BCH"]
INTERVAL = "4h"
TRAIN_WINDOW_DAYS = 30*6
TEST_WINDOW_DAYS = 30*3
N_TRIALS = 20  # Trials per window
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# Global data for optimization (loaded once)
global_ohlcv_df = pd.DataFrame()
global_date_range = None

def run_optimization(train_start, train_end, n_trials=N_TRIALS):
    # print(f"  Optimizing from {train_start} to {train_end}...")
    window_date_range = global_date_range[(global_date_range >= train_start) & (global_date_range < train_end)]
    
    if window_date_range.empty:
        print("  Warning: Empty training window.")
        return None

    def objective(trial):
        params = {
            'adx_threshold': trial.suggest_int('adx_threshold', 15, 35),
            'adx_length': trial.suggest_int('adx_length', 10, 20),
            'sar_acceleration': trial.suggest_float('sar_acceleration', 0.01, 0.05, step=0.005),
            'sar_maximum': trial.suggest_float('sar_maximum', 0.1, 0.3, step=0.05),
            'atr_multiplier': trial.suggest_float('atr_multiplier', 1.5, 4.0, step=0.5),
            'atr_length': trial.suggest_int('atr_length', 10, 20),
            'use_dmp_cross': trial.suggest_categorical('use_dmp_cross', [True, False])
        }
        
        engine = Trading(
            ohlcv_df=global_ohlcv_df,
            date_range=window_date_range,
            start_cash=10000,
            top_n_movers=5,
            max_position=0.2,
            strategy_params=params
        )
        
        try:
            engine.run()
        except Exception:
            return -10.0 # Extreme penalty for failure

        sortino = engine.portfolio.sortino_ratio()
        
        # Penalize if no trades were made (optimization shouldn't favor doing nothing)
        if len(engine.portfolio.trades) == 0:
            return -5.0
            
        return sortino if not np.isnan(sortino) else -1.0

    # Suppress Optuna logging to keep tqdm clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize")
    
    with tqdm(total=n_trials, desc="    Trials", leave=False) as pbar:
        def callback(study, trial):
            pbar.update(1)
            
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    return study.best_params

def run_backtest(test_start, test_end, params, start_cash):
    # print(f"  Testing from {test_start} to {test_end} with params: {params}")
    window_date_range = global_date_range[(global_date_range >= test_start) & (global_date_range < test_end)]
    
    if window_date_range.empty:
        print("  Warning: Empty test window.")
        return start_cash, {}

    engine = Trading(
        ohlcv_df=global_ohlcv_df,
        date_range=window_date_range,
        start_cash=start_cash,
        top_n_movers=5,
        max_position=0.2,
        strategy_params=params
    )
    
    try:
        engine.run()
    except Exception as e:
        print(f"  Test run failed: {e}")
        return start_cash, {}

    return engine.portfolio.total_value, engine.portfolio.stats_dict()

def main():
    global global_ohlcv_df, global_date_range
    
    rm = ResultsManager("run_wfo")
    
    # 1. Load Data
    print("Loading data for WFO...")
    k_h = KrakenHistoricalData()
    start_dt = pd.to_datetime(START_DATE).tz_localize('UTC')
    end_dt = pd.to_datetime(END_DATE).tz_localize('UTC')
    
    global_ohlcv_df = k_h.get_ohlcv_df(SYMBOLS, interval=INTERVAL, start=start_dt, end=end_dt)
    
    if isinstance(global_ohlcv_df.columns, pd.MultiIndex):
        global_ohlcv_df.columns.names = [None] * global_ohlcv_df.columns.nlevels
    else:
        global_ohlcv_df.columns.name = None
    global_ohlcv_df.index.name = None
    global_date_range = global_ohlcv_df.index
    
    print(f"Data loaded: {len(global_ohlcv_df)} rows.")

    # 2. WFO Loop
    current_train_start = start_dt
    results = []
    current_capital = 10000 
    
    # Calculate total steps for progress bar
    temp_start = start_dt
    total_steps = 0
    while True:
        train_end = temp_start + timedelta(days=TRAIN_WINDOW_DAYS)
        test_start = train_end
        test_end = test_start + timedelta(days=TEST_WINDOW_DAYS)
        if test_end > end_dt:
            break
        total_steps += 1
        temp_start = temp_start + timedelta(days=TEST_WINDOW_DAYS)

    print(f"Starting WFO with {total_steps} windows...")
    
    with tqdm(total=total_steps, desc="WFO Progress") as pbar:
        while True:
            train_end = current_train_start + timedelta(days=TRAIN_WINDOW_DAYS)
            test_start = train_end
            test_end = test_start + timedelta(days=TEST_WINDOW_DAYS)
            
            if test_end > end_dt:
                break
                
            # print(f"\n=== WFO Step: Train [{current_train_start.date()} - {train_end.date()}] -> Test [{test_start.date()} - {test_end.date()}] ===")
            
            best_params = run_optimization(current_train_start, train_end)
            
            if best_params:
                final_value, stats = run_backtest(test_start, test_end, best_params, current_capital)
                
                profit = final_value - current_capital
                pct_return = (profit / current_capital) * 100
                
                # print(f"  Result: {current_capital:.2f} -> {final_value:.2f} ({pct_return:.2f}%)")
                
                check_point = {
                    'train_start': current_train_start.isoformat(),
                    'train_end': train_end.isoformat(),
                    'test_start': test_start.isoformat(),
                    'test_end': test_end.isoformat(),
                    'start_capital': current_capital,
                    'end_capital': final_value,
                    'profit': profit,
                    'params': best_params
                }
                results.append(check_point)
                current_capital = final_value
                pbar.set_postfix({"Current Capital": f"${current_capital:,.0f}"})
            else:
                print("  Optimization failed (no params found). Skipping window.")

            current_train_start = current_train_start + timedelta(days=TEST_WINDOW_DAYS) 
            pbar.update(1)

    # 3. Report
    print("\n\n=== Walk Forward Optimization Results ===")
    print(f"Overall Period: {START_DATE} to {END_DATE}")
    print(f"Total Return: {((current_capital - 10000)/10000)*100:.2f}%")
    print(f"Final Capital: ${current_capital:.2f}")
    
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        rm.save_metrics(df_res, "wfo_results.csv")
        
        # Also save the VERY BEST params from the last window as the recommendation
        best_overall_params = results[-1]['params']
        rm.save_params(best_overall_params)
        
        # Save Metadata
        metadata = {
            "script": "run_wfo",
            "timestamp": datetime.now().isoformat(),
            "symbols": SYMBOLS,
            "interval": INTERVAL,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "test_window_days": TEST_WINDOW_DAYS,
            "total_return_pct": ((current_capital - 10000)/10000)*100,
            "final_capital": current_capital
        }
        rm.save_metadata(metadata)
        
        print("\nWindow Details:")
        view_cols = ['test_start', 'test_end', 'start_capital', 'end_capital', 'profit']
        print(df_res[view_cols].to_string())
        
        # --- 4. Advanced Visualization (VectorBT) ---
        print("\nGenerating VectorBT aggregate plots...")
        try:
            # Create a simple vbt portfolio representing the walk-forward equity
            # We use the end_capital from each window to build a return series
            returns = df_res['end_capital'].pct_change().fillna(0)
            # Note: This is an approximation for visualization
            fig = vbt.Portfolio.from_returns(returns, init_cash=10000).plot()
            fig.update_layout(title="Walk-Forward Aggregate Equity (Approximate)")
            rm.save_plotly_figure(fig, "wfo_aggregate_equity")
            print("VectorBT plots saved.")
        except Exception as e:
            print(f"VectorBT plotting failed: {e}")

        # --- 5. Export to Excel ---
        print("\nExporting to Excel...")
        excel_data = {
            "Metadata": pd.DataFrame.from_dict(metadata, orient='index', columns=['Value']),
            "Window Results": df_res
        }
        # Add parameters as individual sheets or one merged sheet
        params_df = pd.DataFrame([r['params'] for r in results])
        params_df.index = [r['test_start'] for r in results]
        excel_data["Best Parameters"] = params_df
        
        excel_path = rm.save_excel(excel_data, "wfo_results.xlsx")
        print(f"Excel report saved to: {excel_path}")

        print(f"\nResults saved to: {rm.run_dir}")

if __name__ == "__main__":
    main()
