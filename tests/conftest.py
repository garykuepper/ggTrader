import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=500, freq="4h")
    
    # Create two symbols
    symbols = ["BTC-USD", "ETH-USD"]
    data = {}
    
    for symbol in symbols:
        np.random.seed(42)
        close = 1000 + np.cumsum(np.random.randn(500) * 10)
        high = close + np.abs(np.random.randn(500) * 5)
        low = close - np.abs(np.random.randn(500) * 5)
        open_ = close.copy()
        volume = np.random.randint(100, 1000, 500)
        
        data[(symbol, "open")] = open_
        data[(symbol, "high")] = high
        data[(symbol, "low")] = low
        data[(symbol, "close")] = close
        data[(symbol, "volume")] = volume
    
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    
    return df
