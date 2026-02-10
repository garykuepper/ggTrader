import optuna
import pandas as pd
import sys
import os
from datetime import datetime

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.trading import Trading
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.utils.results_manager import ResultsManager

try:
    from scripts.sensitivity_visualizer import plot_optimization_landscape
except ImportError:
    from sensitivity_visualizer import plot_optimization_landscape

# Global data for optimization
global_ohlcv_df = pd.DataFrame()
global_date_range = None


def objective(trial):
    params = {
        "adx_threshold": trial.suggest_int("adx_threshold", 15, 35),
        "adx_length": trial.suggest_int("adx_length", 10, 20),
        "sar_acceleration": trial.suggest_float(
            "sar_acceleration", 0.01, 0.05, step=0.005
        ),
        "sar_maximum": trial.suggest_float("sar_maximum", 0.1, 0.3, step=0.05),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 4.0, step=0.5),
        "atr_length": trial.suggest_int("atr_length", 10, 20),
        "use_dmp_cross": trial.suggest_categorical("use_dmp_cross", [True, False]),
    }

    # Configuration
    start_cash = 10000
    top_n_movers = 5
    max_position = 0.2

    global global_ohlcv_df, global_date_range

    if global_ohlcv_df.empty:
        return -float("inf")

    engine = Trading(
        ohlcv_df=global_ohlcv_df,
        date_range=global_date_range,
        start_cash=start_cash,
        top_n_movers=top_n_movers,
        max_position=max_position,
        strategy_params=params,
    )

    try:
        engine.run()
    except Exception as e:
        return -float("inf")

    return engine.portfolio.total_value - start_cash


if __name__ == "__main__":
    rm = ResultsManager("run_sensitivity")

    # --- Data Loading (Once) ---
    CONSTANTS = {
        "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
        "START_DATE": "2024-01-01",
        "END_DATE": "2024-06-01",
        "INTERVAL": "4h",
    }

    print("Loading data for optimization...")
    from ggTrader.utils.setup import load_data_and_setup

    # Load data using shared setup
    # Note: load_data_and_setup expects config dict
    try:
        global_ohlcv_df = load_data_and_setup(CONSTANTS)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Preprocess for Trading engine (remove MultiIndex levels if necessary)
    if isinstance(global_ohlcv_df.columns, pd.MultiIndex):
        global_ohlcv_df.columns.names = [None] * global_ohlcv_df.columns.nlevels
    else:
        global_ohlcv_df.columns.name = None
    global_ohlcv_df.index.name = None

    global_date_range = global_ohlcv_df.index
    print(f"Data loaded. Rows: {len(global_ohlcv_df)}")

    # --- Optimization ---
    study_name = "trading_sensitivity_analysis"
    storage_name = "sqlite:///" + str(rm.run_dir / "sensitivity_study.db")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
    )

    print("Starting optimization... (Press Ctrl+C to stop early)")
    try:
        study.optimize(objective, n_trials=20)
    except KeyboardInterrupt:
        print("Optimization stopped by user.")

    # --- Results ---
    print("\n--- Optimization Results ---")
    print(f"Best Value (Profit): ${study.best_value:.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save Best Params
    rm.save_params(study.best_params)

    # Save consolidated results (params + metrics) -> run_results.json
    metrics = {
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
    metadata = {
        "start_date": CONSTANTS["START_DATE"],
        "end_date": CONSTANTS["END_DATE"],
        "symbols_file": CONSTANTS["SYMBOLS_FILE"],
    }

    rm.save_run_results(params=study.best_params, metrics=metrics, metadata=metadata)

    # Save best value to performance_metrics table
    rm.db_manager.add_metrics(rm.run_id, metrics)

    # Save trials dataframe
    trials_df = study.trials_dataframe()
    rm.save_metrics(trials_df, "trials_dataframe.csv", save_csv=False)

    # --- Visualization ---
    try:
        print("\nGenerating plots...")
        plot_optimization_landscape(
            study,
            params_to_plot=["adx_threshold", "atr_multiplier"],
            metric_name="Net Profit",
            results_manager=rm,
        )
    except Exception as e:
        print(f"Plotting failed: {e}")
        # import traceback
        # traceback.print_exc()

    print(f"\nResults saved to: {rm.run_dir}")
