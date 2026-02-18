"""Run Walk-Forward Optimization (WFO) using VectorBT time-series CV."""

import argparse
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from tabulate import tabulate

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from ggTrader.core.orchestrator import run_wfo_orchestrator

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
    "FEES": 0.004,
    "SLIPPAGE": 0.003,
    # Dynamic movers: set to 0 to disable, or e.g. 20 for top-20 daily
    "USE_MOVERS": 0,
    # WFO-specific configuration
    "N_SPLITS": 5,
    "TEST_RATIO": 0.2,
    # Minimum trades to accept a result in optimization
    "MIN_TRADES": 10,
    # Memory optimization: process in chunks of N parameter combinations
    "CHUNK_SIZE": 500,
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
    """Run Walk-Forward Optimization using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run WFO")
    parser.add_argument("--progress", action="store_true", default=True, help="Show progress bar")
    args = parser.parse_args()

    # Vectorized Parameter Grid
    params = {
        # entry
        "sar_acceleration": [0.02],
        "sar_maximum": [0.2],
        "use_dmp_cross": [False],
        "adx_threshold": list(range(5, 20, 5)),
        "adx_length": list(range(10, 20, 5)),
        # exit
        "atr_length": list(range(5, 15, 5)),
        "atr_multiplier": list(np.arange(0.01, 0.08, 0.015)),
    }

    run_wfo_orchestrator(
        config=CONSTANTS,
        param_grid=params,
        save_results=True,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
