import numpy as np
import pandas as pd

from ggTrader.indicators.signals import SignalFactory


def test_broadcasting():
    print("Setting up mock data...")
    # 1. Create Mock Data
    index = pd.date_range("2023-01-01", periods=100, freq="1D")
    # 2 symbols
    symbols = ["BTC", "ETH"]

    # Create price data that definitely triggers signals
    # Uptrend
    close_vals = np.linspace(100, 200, 100)
    # Add noise
    close_vals = close_vals[:, None] + np.random.randn(100, 2)

    close = pd.DataFrame(close_vals, index=index, columns=symbols)
    high = close + 2
    low = close - 2
    open_ = close - 1

    # 2. Run Factory with Broadcasting
    # We vary ADX threshold.
    # Param product: 3 thresholds x 2 signals = 6 columns.
    adx_thresholds = [15, 25, 35]

    print(f"Running SignalFactory with adx_thresholds={adx_thresholds}...")

    sf = SignalFactory.run(
        close,
        high,
        low,
        open_,
        adx_threshold=adx_thresholds,
        adx_length=14,
        atr_multiplier=[3.0],
        param_product=True,
    )

    # 3. Verify Output Shape
    # entries shape: (Time, Prod(Params) * Symbols)
    # vectorbt aligns output columns.
    # The columns should be a MultiIndex.
    print(f"Entries shape: {sf.entries.shape}")
    print(f"Columns: {sf.entries.columns}")

    expected_cols = len(adx_thresholds) * 1 * len(symbols)  # 3 * 1 * 2 = 6
    assert sf.entries.shape[1] == expected_cols, (
        f"Expected {expected_cols} columns, got {sf.entries.shape[1]}"
    )

    # 4. Check if we have MultiIndex columns with parameter levels
    if isinstance(sf.entries.columns, pd.MultiIndex):
        print("Columns are MultiIndex as expected.")
        # Check level names
        print(f"Level names: {sf.entries.columns.names}")
        # assert "adx_threshold" in sf.entries.columns.names
        # assert "atr_multiplier" in sf.entries.columns.names

    print("Broadcasting Test PASSED")


if __name__ == "__main__":
    test_broadcasting()
