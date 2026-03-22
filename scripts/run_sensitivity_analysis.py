"""Run a vectorized sensitivity analysis (grid search) using FastBacktest."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ggTrader.core.orchestrator import run_sensitivity_orchestrator

# =====================================================================
# USER CONFIGURATION — edit these values to customize the backtest
# =====================================================================
CONSTANTS = {
    # Symbol pool (set SYMBOLS to None to use SYMBOLS_FILE instead)
    "SYMBOLS": None,
    "SYMBOLS_FILE": "data/top_10_USD_2023-01-01_2025-12-31.json",
    # Date range
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    # Portfolio
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.004,
    "SLIPPAGE": 0.001,
    # Dynamic movers: set to 0 to disable, or e.g. 20 for top-20 daily
    "USE_MOVERS": 0,
    # Legacy — no longer used as primary filter; kept for backward compat
    "MIN_TRADES": 0,
    # Require at least 1 completed round-trip on the analysis window before ranking a combo
    "MIN_CLOSED_TRADES_TRAIN": 1,
    # Memory optimization: process in chunks of N parameter combinations
    "CHUNK_SIZE": 1000,
    # Strategy selection
    "ENTRY_STRATEGY": "psar_adx",  # "psar_adx", "ema_cross", "rsi_reversal"
    "EXIT_STRATEGY": "atr_trailing",  # "atr_trailing", "fixed_sl_tp"
    # Use vectorized signal generation (experimental)
    "USE_VECTORIZED": False,
}
# =====================================================================


def main() -> None:
    """Run vectorized sensitivity analysis using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis")
    parser.add_argument("--progress", action="store_true", default=True, help="Show progress bar")
    args = parser.parse_args()

    # Parameter grid — VectorBT creates the Cartesian product
    params = {
        "adx_threshold": list(range(15, 45, 5)),
        "adx_length": list(range(5, 40, 5)),
        "atr_length": list(range(5, 40, 5)),
        "atr_multiplier": list(np.arange(0.1, 1.1, 0.1)),
        "sar_acceleration": [0.02],
        "sar_maximum": [0.2],
        "use_dmp_cross": [True, False],
    }

    run_sensitivity_orchestrator(
        config=CONSTANTS,
        param_grid=params,
        save_results=True,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
