import argparse
import os
import sys

import pandas as pd

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.trading import Trading
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_and_setup

# --- USER CONFIGURATION ---
CONSTANTS = {
    "SYMBOLS": None,  # Set to a list like ["BTC/USD", "ETH/USD"] to override JSON
    "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    "START_CASH": 10000,
    "DEFAULT_PARAMS": {
        "adx_threshold": 25,
        "adx_length": 14,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": 3.0,
        "atr_length": 14,
        "use_dmp_cross": False,
    },
}


def main() -> None:
    """
    Main entry point for running a single backtest.
    """
    parser = argparse.ArgumentParser(description="Run a single ggTrader backtest")
    parser.add_argument("--params", type=str, help="Path to params.json")
    args = parser.parse_args()

    rm = ResultsManager("run_backtest")
    params = rm.load_params(args.params) if args.params else CONSTANTS["DEFAULT_PARAMS"]

    print("Loading data...")
    try:
        ohlcv = load_data_and_setup(CONSTANTS)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Running backtest...")
    try:
        # Log data summary to help debugging
        if hasattr(ohlcv, "columns") and isinstance(ohlcv.columns, pd.MultiIndex):
            symbols_found = ohlcv.columns.levels[0].tolist()
            print(f"Loaded {len(symbols_found)} symbols with {len(ohlcv)} data points.")

        engine = Trading(
            ohlcv_df=ohlcv,
            date_range=ohlcv.index,
            start_cash=CONSTANTS["START_CASH"],
            strategy_params=params,
        )
    except ValueError as e:
        print(f"Error initializing Trading engine: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during engine setup: {e}")
        return

    engine.run()

    print("Backtest complete. Processing results...")
    stats = engine.portfolio.stats_dict()

    # Save consolidated results (params + metrics) -> run_results.json
    rm.save_run_results(params=params, metrics=stats, metadata=CONSTANTS)

    # Print summary to console
    rm.print_summary(stats)
    print(f"Results saved to: {rm.run_dir}")


if __name__ == "__main__":
    main()
