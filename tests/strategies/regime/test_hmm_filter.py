"""Unit and integration tests for HMM regime filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from ggTrader.strategies.momentum.config import MomentumConfig
from ggTrader.strategies.momentum.cross_sectional import CrossSectionalMomentum
from ggTrader.strategies.regime.hmm_filter import load_regime_gate
from ggTrader.utils.result_db_manager import ResultDBManager


def generate_synthetic_ohlcv(
    symbols: list[str], n_bars: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic price and volume DataFrames."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=n_bars, freq="D", tz="UTC")

    close_dict = {}
    vol_dict = {}

    for sym in symbols:
        prices = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, n_bars))
        prices = np.clip(prices, 5.0, 500.0)
        close_dict[sym] = prices

        vols = np.random.uniform(100000, 1000000, n_bars)
        vol_dict[sym] = vols

    close_df = pd.DataFrame(close_dict, index=dates)
    volume_df = pd.DataFrame(vol_dict, index=dates)

    return close_df, volume_df


@pytest.fixture(scope="module")
def setup_db_regime_states():
    """Fixture to set up test records in regime_states table."""
    db = ResultDBManager()

    # Create table if it doesn't exist
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS regime_states (
            timestamp TIMESTAMPTZ NOT NULL,
            asset_class VARCHAR(16) NOT NULL,
            state_0_prob DOUBLE PRECISION,
            state_1_prob DOUBLE PRECISION,
            state_2_prob DOUBLE PRECISION,
            dominant_state INTEGER,
            PRIMARY KEY (timestamp, asset_class)
        );
    """
    with db.engine.connect() as conn:
        with conn.begin():
            conn.execute(text(create_table_sql))

    # Test dates
    dates = pd.date_range(start="2025-01-01", periods=10, freq="D", tz="UTC")

    # Insert test records
    # Row 0: Engaged Trend (state_0_prob = 0.8, dominant = 0) -> should be True
    # Row 1: Mean Reversion (state_0_prob = 0.2, dominant = 1) -> should be False
    # Row 2: Systemic Lapse (state_0_prob = 0.1, dominant = 2) -> should be False
    # Row 3: High state 0 but dominant 2 -> should be False
    # Row 4: state_0_prob = 0.64 (just below threshold 0.65) -> should be False
    # Row 5: state_0_prob = 0.66 (just above threshold 0.65) -> should be True
    # Rows 6-9: missing/not inserted
    test_records = [
        {"timestamp": dates[0], "state_0_prob": 0.8, "dominant_state": 0},
        {"timestamp": dates[1], "state_0_prob": 0.2, "dominant_state": 1},
        {"timestamp": dates[2], "state_0_prob": 0.1, "dominant_state": 2},
        {"timestamp": dates[3], "state_0_prob": 0.7, "dominant_state": 2},
        {"timestamp": dates[4], "state_0_prob": 0.64, "dominant_state": 0},
        {"timestamp": dates[5], "state_0_prob": 0.66, "dominant_state": 0},
    ]

    # Clean previous test entries if any
    with db.engine.connect() as conn:
        with conn.begin():
            conn.execute(text("DELETE FROM regime_states WHERE asset_class = 'test_crypto'"))

    # Insert new entries
    for rec in test_records:
        with db.engine.connect() as conn:
            with conn.begin():
                conn.execute(
                    text(
                        """
                        INSERT INTO regime_states (
                            timestamp, asset_class, state_0_prob,
                            state_1_prob, state_2_prob, dominant_state
                        )
                        VALUES (
                            :timestamp, 'test_crypto', :state_0_prob,
                            0.0, 0.0, :dominant_state
                        )
                        """
                    ),
                    rec,
                )

    yield dates

    # Clean up test entries
    with db.engine.connect() as conn:
        with conn.begin():
            conn.execute(text("DELETE FROM regime_states WHERE asset_class = 'test_crypto'"))


def test_gate_default_false(setup_db_regime_states) -> None:
    """Missing timestamps in database default to False when reindexed by the backtest."""
    dates = setup_db_regime_states
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[9])

    # Load gate (it only contains records for first 6 days)
    gate = load_regime_gate("test_crypto", start, end, engaged_threshold=0.65)

    # Reindex to full dates range
    aligned_gate = gate.reindex(dates, fill_value=False)

    # Days 6 to 9 should be False
    for i in range(6, 10):
        assert not aligned_gate.loc[dates[i]]


def test_gate_suppresses_lapse(setup_db_regime_states) -> None:
    """Systemic lapse dominant state always returns False gate, and threshold checks work."""
    dates = setup_db_regime_states
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[9])

    gate = load_regime_gate("test_crypto", start, end, engaged_threshold=0.65)
    aligned_gate = gate.reindex(dates, fill_value=False)

    # Row 0: Engaged Trend (state_0_prob = 0.8 >= 0.65) -> True
    assert aligned_gate.loc[dates[0]]

    # Row 1: Mean Reversion (state_0_prob = 0.2 < 0.65) -> False
    assert not aligned_gate.loc[dates[1]]

    # Row 2: Systemic Lapse (state_0_prob = 0.1 < 0.65, dominant = 2) -> False
    assert not aligned_gate.loc[dates[2]]

    # Row 3: High state 0 but dominant 2 -> False
    assert not aligned_gate.loc[dates[3]]

    # Row 4: state_0_prob = 0.64 < 0.65 -> False
    assert not aligned_gate.loc[dates[4]]

    # Row 5: state_0_prob = 0.66 >= 0.65 -> True
    assert aligned_gate.loc[dates[5]]


def test_filtered_entries_subset(setup_db_regime_states) -> None:
    """Filtered entries should strictly be a subset of raw entries."""
    dates = setup_db_regime_states
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[9])

    # Generate dummy entries DataFrame (10 rows, 2 assets)
    np.random.seed(42)
    raw_entries = pd.DataFrame(
        np.random.choice([True, False], size=(10, 2)),
        index=dates,
        columns=["A", "B"],
    )

    gate = load_regime_gate("test_crypto", start, end, engaged_threshold=0.65)
    aligned_gate = gate.reindex(raw_entries.index, fill_value=False)

    filtered_entries = raw_entries.multiply(aligned_gate, axis=0)

    # Check subset property: filtered_entries cannot be True where raw_entries is False
    # (filtered_entries & ~raw_entries) must be all False
    violation = filtered_entries & (~raw_entries)
    assert not violation.any().any()


def test_hmm_gate_integration(setup_db_regime_states) -> None:
    """Strategy run with filter has <= signal count compared to raw run."""
    dates = setup_db_regime_states
    symbols = ["ETH-USD", "SOL-USD", "LTC-USD", "BTC-USD"]
    # We need n_bars = len(dates) = 10.
    close_df, volume_df = generate_synthetic_ohlcv(symbols, 10)
    close_df.index = dates
    volume_df.index = dates

    config = MomentumConfig.for_crypto()
    # Temporarily override asset_class to query 'test_crypto'
    config.asset_class = "test_crypto"  # type: ignore

    alt_cols = ["ETH-USD", "SOL-USD", "LTC-USD"]
    alt_close = close_df[alt_cols]
    alt_vol = volume_df[alt_cols]
    btc_close = close_df["BTC-USD"]

    strategy = CrossSectionalMomentum(config)

    # Run without filter
    strategy.run(alt_close, alt_vol, btc_close=btc_close, hmm_filter_enabled=False)
    raw_signals = strategy.entries_df.sum().sum()

    # Run with filter
    strategy.run(alt_close, alt_vol, btc_close=btc_close, hmm_filter_enabled=True)
    filtered_signals = strategy.entries_df.sum().sum()

    assert filtered_signals <= raw_signals
