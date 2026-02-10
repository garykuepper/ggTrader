import argparse
import pandas as pd
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.core.optimization import WalkForwardOptimizer
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.config import load_symbols_from_json
from tabulate import tabulate

# --- USER CONFIGURATION ---
CONSTANTS = {
    "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    "TRAIN_DAYS": 365,
    "TEST_DAYS": 120,
    "N_TRIALS": 20,
    "N_JOBS": -1,
    "PARAM_RANGES": {
        "adx_threshold": {"type": "int", "min": 15, "max": 35},
        "adx_length": {"type": "int", "min": 10, "max": 20},
        "sar_acceleration": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.005},
        "sar_maximum": {"type": "float", "min": 0.1, "max": 0.3, "step": 0.05},
        "atr_multiplier": {"type": "float", "min": 1.5, "max": 4.0, "step": 0.5},
        "atr_length": {"type": "int", "min": 10, "max": 20},
        "use_dmp_cross": {"type": "categorical", "choices": [True, False]},
    },
}


def main():
    rm = ResultsManager("run_wfo")
    symbols = load_symbols_from_json(CONSTANTS["SYMBOLS_FILE"])
    k_h = KrakenHistoricalData()
    ohlcv = k_h.get_ohlcv_df(
        symbols,
        interval=CONSTANTS["INTERVAL"],
        start=pd.to_datetime(CONSTANTS["START_DATE"]).tz_localize("UTC"),
        end=pd.to_datetime(CONSTANTS["END_DATE"]).tz_localize("UTC"),
    )

    optimizer = WalkForwardOptimizer(
        ohlcv_df=ohlcv,
        train_days=CONSTANTS["TRAIN_DAYS"],
        test_days=CONSTANTS["TEST_DAYS"],
        n_trials=CONSTANTS["N_TRIALS"],
        n_jobs=CONSTANTS["N_JOBS"],
    )

    results_df = optimizer.run(CONSTANTS["PARAM_RANGES"])

    print("\n--- WFO Results ---")
    print(tabulate(results_df, headers="keys", tablefmt="github", floatfmt=".2f"))
    rm.save_metadata(CONSTANTS)
    rm.save_metrics(results_df, "wfo_results.csv")


if __name__ == "__main__":
    main()
