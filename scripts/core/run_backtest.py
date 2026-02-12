import argparse
from ggTrader.core.trading import Trading
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_and_setup

# --- USER CONFIGURATION ---
CONSTANTS = {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, help="Path to params.json")
    args = parser.parse_args()

    rm = ResultsManager("run_backtest")
    params = rm.load_params(args.params) if args.params else CONSTANTS["DEFAULT_PARAMS"]

    # Load data using shared setup
    ohlcv = load_data_and_setup(CONSTANTS)

    engine = Trading(
        ohlcv_df=ohlcv,
        date_range=ohlcv.index,
        start_cash=CONSTANTS["START_CASH"],
        strategy_params=params,
    )
    engine.run()

    stats = engine.portfolio.stats_dict()

    # Save consolidated results (params + metrics) -> run_results.json
    rm.save_run_results(params=params, metrics=stats, metadata=CONSTANTS)

    # Print summary to console
    rm.print_summary(stats)


if __name__ == "__main__":
    main()
