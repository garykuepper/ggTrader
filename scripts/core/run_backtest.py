"""Run a single ggTrader backtest using the vectorized FastBacktest engine."""

import argparse
import os
import sys

import pandas as pd

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from ggTrader.core.orchestrator import run_backtest_orchestrator
from ggTrader.utils.results_manager import ResultsManager

# =====================================================================
# USER CONFIGURATION — edit these values to customize the backtest
# =====================================================================
CONSTANTS = {
    # Symbol pool (set SYMBOLS to None to use SYMBOLS_FILE instead)
    "SYMBOLS": None,
    "SYMBOLS_FILE": "data/top_30_USD_1095_movers.json",
    # Date range
    "START_DATE": "2023-01-01",
    "END_DATE": "2023-12-31",
    "INTERVAL": "4h",
    # Portfolio
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.004,
    "SLIPPAGE": 0.003,
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
    """Run a single backtest using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run a single ggTrader backtest")
    parser.add_argument("--params", type=str, help="Path to params.json")
    parser.add_argument("--progress", action="store_true", help="Show VectorBT progress bar")
    args = parser.parse_args()

    # Load parameters if provided, else use defaults
    params = CONSTANTS["DEFAULT_PARAMS"]
    if args.params:
        rm_temp = ResultsManager("temp")
        params = rm_temp.load_params(args.params)

    # Execute via orchestrator
    results = run_backtest_orchestrator(
        config=CONSTANTS, params=params, save_results=True, show_progress=args.progress
    )
    pf = results["portfolio"]
    stats = results["stats"]
    # --- Visualization ---
    print("Global Portfolio Stats:")
    # Convert stats to a DataFrame for a cleaner table view
    stats_df = pf.stats().to_frame(name="Portfolio Stats")
    print(stats_df)
    pf.plot(subplots=["drawdowns", "value", "cum_returns"]).show()


if __name__ == "__main__":
    main()
