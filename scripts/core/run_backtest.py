import argparse
import pandas as pd
from ggTrader.core.trading import Trading
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.config import load_symbols_from_json

# --- USER CONFIGURATION ---
CONSTANTS = {
    "SYMBOLS_FILE": "data/top_50_consistent_movers.json",
    "START_DATE": "2025-01-01",
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

    symbols = load_symbols_from_json(CONSTANTS["SYMBOLS_FILE"])
    k_h = KrakenHistoricalData()
    ohlcv = k_h.get_ohlcv_df(
        symbols,
        interval=CONSTANTS["INTERVAL"],
        start=pd.to_datetime(CONSTANTS["START_DATE"]).tz_localize("UTC"),
        end=pd.to_datetime(CONSTANTS["END_DATE"]).tz_localize("UTC"),
    )

    engine = Trading(
        ohlcv_df=ohlcv,
        date_range=ohlcv.index,
        start_cash=CONSTANTS["START_CASH"],
        strategy_params=params,
    )
    engine.run()

    stats = engine.portfolio.stats_dict()
    rm.save_metadata(stats)
    print(f"Final Value: {stats['total_value']:.2f}")


if __name__ == "__main__":
    main()
