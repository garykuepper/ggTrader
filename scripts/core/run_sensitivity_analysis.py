"""Run a vectorized sensitivity analysis (grid search) using FastBacktest."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is in path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.orchestrator import run_sensitivity_orchestrator

# =====================================================================
# USER CONFIGURATION — edit these values to customize the backtest
# =====================================================================
CONSTANTS = {
    # Symbol pool (set SYMBOLS to None to use SYMBOLS_FILE instead)
    "SYMBOLS": None,
    "SYMBOLS_FILE": "data/top_10_USD_1095_movers.json",
    # Date range
    "START_DATE": "2023-01-01",
    "END_DATE": "2023-12-31",
    "INTERVAL": "4h",
    # Portfolio
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.001,
    # Dynamic movers: set to 0 to disable, or e.g. 20 for top-20 daily
    "USE_MOVERS": 0,
    # # Strategy parameters
    # "DEFAULT_PARAMS": {
    #     "adx_threshold": 25,
    #     "adx_length": 14,
    #     "sar_acceleration": 0.02,
    #     "sar_maximum": 0.2,
    #     "atr_multiplier": 3.0,
    #     "atr_length": 14,
    #     "use_dmp_cross": False,
    # },
}
# =====================================================================


def main() -> None:
    """Run vectorized sensitivity analysis using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis")
    parser.add_argument(
        "--progress", action="store_true", default=True, help="Show progress bar"
    )
    args = parser.parse_args()

    # Parameter grid — VectorBT creates the Cartesian product
    params = {
        "adx_threshold": list(range(0, 50, 5)),
        "adx_length": list(range(8, 14, 2)),
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": list(np.arange(2.0, 5.0, 1)),
        "atr_length": 14,
        "use_dmp_cross": True,
    }

    run_sensitivity_orchestrator(
        config=CONSTANTS,
        param_grid=params,
        save_results=True,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
