"""Run a single ggTrader backtest using the vectorized FastBacktest engine."""

import argparse
import os
import sys

import pandas as pd

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import build_mover_mask, load_data_and_setup

# =====================================================================
# USER CONFIGURATION — edit these values to customize the backtest
# =====================================================================
CONSTANTS = {
    # Symbol pool (set SYMBOLS to None to use SYMBOLS_FILE instead)
    "SYMBOLS": None,
    "SYMBOLS_FILE": "data/top_20_USD_1095_movers.json",
    # Date range
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    # Portfolio
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.001,
    # Dynamic movers: set to 0 to disable, or e.g. 20 for top-20 daily
    "USE_MOVERS": 10,
    # Strategy parameters
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
# =====================================================================


def main() -> None:
    """Main entry point for running a single backtest."""
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

    if hasattr(ohlcv, "columns") and isinstance(ohlcv.columns, pd.MultiIndex):
        symbols_found = ohlcv.columns.levels[0].tolist()
        print(f"Loaded {len(symbols_found)} symbols with {len(ohlcv)} data points.")

    # Optional: build dynamic mover mask
    mover_mask = None
    top_n = CONSTANTS["USE_MOVERS"]
    if top_n > 0:
        print(f"Building dynamic top-{top_n} daily mover mask...")
        try:
            mover_mask = build_mover_mask(ohlcv, CONSTANTS, top_n=top_n)
            print(f"Mover mask built: {mover_mask.shape}")
        except Exception as e:
            print(f"Warning: could not build mover mask: {e}")

    print("Running backtest...")
    try:
        engine = FastBacktest(ohlcv, params, config=CONSTANTS, mover_mask=mover_mask)
        pf = engine.run()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error running backtest: {e}")
        return

    print("Backtest complete. Processing results...")
    stats = engine.get_stats()

    # Save consolidated results with new structure
    rm.save_run_results(params=params, metrics=stats, metadata=CONSTANTS)

    # Save VBT Plots (Dashboard)
    print("Saving VectorBT dashboard...")
    rm.save_vbt_dashboard(pf, "dashboard")

    # Print summary
    rm.print_summary(stats)
    print(f"Results saved to: {rm.run_dir}")


if __name__ == "__main__":
    main()
