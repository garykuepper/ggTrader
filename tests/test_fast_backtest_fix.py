import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

import numpy as np
import pandas as pd

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.metrics import _profit_factor_raw, safe_portfolio_stats


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
    pf = engine.run(show_progress=False)
    stats = engine.get_stats()
    assert isinstance(stats["sharpe"], float)

    # profit factor via the writable-safe raw-PnL path (vbt's native accessor can
    # crash with "assignment destination is read-only" depending on numba state)
    pf_factor = _profit_factor_raw(pf)
    assert len(pf_factor) >= 1

    # full stats table via the safe wrapper that survives the same vbt crash
    s = safe_portfolio_stats(pf)
    assert "Profit Factor" in s.index
    assert "Expectancy" in s.index


if __name__ == "__main__":
    test_fast_backtest_run()
