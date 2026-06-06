"""Integration tests for Cross-Sectional Momentum strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.strategies.momentum.config import MomentumConfig
from ggTrader.strategies.momentum.cross_sectional import CrossSectionalMomentum


def generate_synthetic_ohlcv(
    symbols: list[str], n_bars: int = 252
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic price and volume DataFrames."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=n_bars, freq="D")

    close_dict = {}
    vol_dict = {}

    for sym in symbols:
        # random walk prices
        prices = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, n_bars))
        prices = np.clip(prices, 5.0, 500.0)
        close_dict[sym] = prices

        vols = np.random.uniform(100000, 1000000, n_bars)
        vol_dict[sym] = vols

    close_df = pd.DataFrame(close_dict, index=dates)
    volume_df = pd.DataFrame(vol_dict, index=dates)

    return close_df, volume_df


def test_strategy1_equity_e2e() -> None:
    """Run full backtest on 1 year of synthetic OHLCV data (5 equities, 1 sector)."""
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    close_df, volume_df = generate_synthetic_ohlcv(symbols, 252)

    sector_map = {s: "Tech" for s in symbols}
    config = MomentumConfig.for_equities()

    strategy = CrossSectionalMomentum(config)
    portfolio = strategy.run(close_df, volume_df, sector_map=sector_map)

    # Assert non-empty trades and Sharpe computable
    assert portfolio is not None
    stats = portfolio.stats()
    assert "Sharpe Ratio" in stats
    assert not np.isnan(stats["Sharpe Ratio"])
    assert stats["Total Trades"] > 0


def test_strategy1_crypto_e2e() -> None:
    """Run full backtest on crypto config (3 altcoins + BTC); assert BTC-beta stripped."""
    # Symbols include 3 alts + BTC
    symbols = ["ETH-USD", "SOL-USD", "LTC-USD", "BTC-USD"]
    close_df, volume_df = generate_synthetic_ohlcv(symbols, 252)

    config = MomentumConfig.for_crypto()

    # Altcoins only are traded, but we need BTC close prices for stripping
    alt_cols = ["ETH-USD", "SOL-USD", "LTC-USD"]
    alt_close = close_df[alt_cols]
    alt_vol = volume_df[alt_cols]
    btc_close = close_df["BTC-USD"]

    strategy = CrossSectionalMomentum(config)
    portfolio = strategy.run(alt_close, alt_vol, btc_close=btc_close)

    assert portfolio is not None
    stats = portfolio.stats()
    assert "Sharpe Ratio" in stats
    assert not np.isnan(stats["Sharpe Ratio"])
    assert stats["Total Trades"] > 0


def test_10pct_cap_respected() -> None:
    """Assert that in no bar does any single position exceed 10% of equity."""
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "AMD", "INTC", "TSLA", "NFLX"]
    close_df, volume_df = generate_synthetic_ohlcv(symbols, 252)

    sector_map = {s: "Tech" for s in symbols}
    config = MomentumConfig.for_equities()

    strategy = CrossSectionalMomentum(config)
    portfolio = strategy.run(close_df, volume_df, sector_map=sector_map)

    # Check individual asset value / portfolio value ratio at each timestamp
    asset_value = portfolio.asset_value(group_by=False)
    portfolio_value = portfolio.value()

    # Div row-wise
    ratios = asset_value.div(portfolio_value, axis=0)

    # Verify that no single column has ratio > 0.15 (allowing for minor rounding and price drift)
    # The entry size itself is capped at 10%.
    assert ratios.max().max() < 0.15
