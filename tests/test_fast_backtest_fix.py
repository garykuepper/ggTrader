import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

import numpy as np
import pandas as pd

from ggTrader.core.fast_backtest import FastBacktest


def test_fast_backtest_run():
    # 1. Create dummy OHLCV data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="4h")
    symbols = ["BTC", "ETH"]

    # MultiIndex columns: (symbol, field)
    iterables = [symbols, ["open", "high", "low", "close", "volume"]]
    columns = pd.MultiIndex.from_product(iterables, names=["symbol", "field"])

    data = np.random.rand(100, 10) * 100
    ohlcv_df = pd.DataFrame(data, index=dates, columns=columns)

    # Ensure High is highest, Low is lowest
    for symbol in symbols:
        ohlcv_df[(symbol, "high")] = ohlcv_df[(symbol, "close")] + 5
        ohlcv_df[(symbol, "low")] = ohlcv_df[(symbol, "close")] - 5
        ohlcv_df[(symbol, "open")] = ohlcv_df[(symbol, "close")]  # simplify

    print("Created dummy OHLCV data.")

    # 2. Define params
    params = {
        "adx_length": 14,
        "adx_threshold": 25,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "use_dmp_cross": True,
        "atr_length": 14,
        "atr_multiplier": 3.0,
    }

    # 3. Instantiate FastBacktest
    config = {
        "START_CASH": 10000,
        "PORTFOLIO_SHARE": 0.5,
        "FEES": 0.001,
        "SLIPPAGE": 0.0005,
        "FREQ": "4h",
        "N_JOBS": 1,
        "USE_CASH_SHARING": True,
    }

    print("Instantiating FastBacktest...")
    engine = FastBacktest(ohlcv_df, params, config=config)

    # 4. Run backtest
    print("Running FastBacktest.run()...")
    try:
        pf = engine.run(show_progress=False)
        print("FastBacktest.run() completed successfully.")
        print(f"Portfolio Stats: {engine.get_stats()}")

        print("Testing profit_factor() to ensure no read-only errors...")
        # Note: profit_factor is usually on the trades accessor or via stats()
        try:
            # Try direct access if available or via trades
            pf_factor = pf.trades.profit_factor()
            print(f"Profit Factor (from trades): {pf_factor.mean()}")
        except Exception as e:
            print(f"pf.trades.profit_factor() failed: {e}")

        print("Testing pf.stats() which caused the original error...")
        stats = pf.stats()
        print("pf.stats() completed successfully.")

    except AttributeError as e:
        print(f"FAILED with AttributeError: {e}")
        raise
    except Exception as e:
        print(f"FAILED with Exception: {e}")
        raise


if __name__ == "__main__":
    test_fast_backtest_run()
