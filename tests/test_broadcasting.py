"""Param-grid broadcasting across multiple symbols through the registry path."""

import numpy as np
import pandas as pd

from ggTrader.core.fast_backtest import FastBacktest


def test_broadcasting():
    # 1. Mock data: 2 symbols, uptrend with noise
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-01", periods=100, freq="1D")
    symbols = ["BTC", "ETH"]
    close_vals = np.linspace(100, 200, 100)[:, None] + rng.normal(0, 1, (100, 2))

    frames = {}
    for j, sym in enumerate(symbols):
        c = close_vals[:, j]
        frames[sym] = pd.DataFrame(
            {"open": c - 1, "high": c + 2, "low": c - 2, "close": c}, index=index
        )
    ohlcv = pd.concat(frames, axis=1)
    ohlcv.columns.names = ["symbol", "field"]

    # 2. Grid: 3 ADX thresholds x 1 ATR pair -> 3 combos x 2 symbols = 6 columns
    adx_thresholds = [15, 25, 35]
    engine = FastBacktest(
        ohlcv,
        {
            "adx_length": [14],
            "adx_threshold": adx_thresholds,
            "sar_acceleration": [0.02],
            "sar_maximum": [0.2],
            "use_dmp_cross": [False],
            "atr_length": [14],
            "atr_multiplier": [3.0],
        },
        config={"FREQ": "1d"},
    )
    engine.run(show_progress=False)

    expected_cols = len(adx_thresholds) * len(symbols)
    assert engine.entries.shape[1] == expected_cols
    assert isinstance(engine.entries.columns, pd.MultiIndex)
    assert engine.entries.columns.names == ["param_combo", "symbol"]
    assert engine.entries.dtypes.unique().tolist() == [np.dtype(bool)]
