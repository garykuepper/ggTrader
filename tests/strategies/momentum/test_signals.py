"""Unit tests for signals.py (Strategy 1 Rank Synthesis & Signals)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.strategies.momentum.config import MomentumConfig
from ggTrader.strategies.momentum.signals import CrossSectionalSignalGenerator


def test_top_decile_count() -> None:
    """Verify that in a 100-asset universe, entry signals on any row <= 10."""
    config = MomentumConfig(entry_percentile=0.90, rank_mode="residual_btc")
    generator = CrossSectionalSignalGenerator(config)

    # Generate 100 assets over 20 timestamps with random data (no NaNs)
    np.random.seed(42)
    assets = [f"Asset_{i}" for i in range(100)]
    mom_df = pd.DataFrame(np.random.uniform(0, 1, (20, 100)), columns=assets)
    liq_df = pd.DataFrame(np.random.uniform(0, 1, (20, 100)), columns=assets)

    signals = generator.generate(mom_df, liq_df)

    # Verify that the number of True entries on any row is <= 10
    # Note: Because the first row after shift is filled with False, the count is 0.
    for i in range(len(signals)):
        true_count = signals.iloc[i].sum()
        assert true_count <= 10


def test_no_signal_on_warmup() -> None:
    """Verify no True signals in warmup rows."""
    config = MomentumConfig(entry_percentile=0.90)
    generator = CrossSectionalSignalGenerator(config)

    assets = ["A", "B", "C", "D", "E"]
    # Warmup of 5 bars
    mom_data = np.random.uniform(0, 1, (10, 5))
    mom_data[:5, :] = np.nan
    liq_data = np.random.uniform(0, 1, (10, 5))
    liq_data[:5, :] = np.nan

    mom_df = pd.DataFrame(mom_data, columns=assets)
    liq_df = pd.DataFrame(liq_data, columns=assets)

    signals = generator.generate(mom_df, liq_df)

    # First 5 rows + 1 shifted row = first 6 rows must be entirely False
    assert np.all(~signals.iloc[:6].values)


def test_shift_applied() -> None:
    """Verify that signal at bar i is based on data through bar i-1 only."""
    config = MomentumConfig(entry_percentile=0.60)
    generator = CrossSectionalSignalGenerator(config)

    assets = ["A", "B"]
    mom_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=assets)
    liq_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=assets)

    signals_1 = generator.generate(mom_df, liq_df)

    # Mutate row index 2 (bar i)
    mom_df_mutated = mom_df.copy()
    mom_df_mutated.iloc[2, :] += 10.0

    signals_2 = generator.generate(mom_df_mutated, liq_df)

    # The signal at index 2 must be identical in both runs (not affected by mutation at index 2)
    pd.testing.assert_series_equal(signals_1.iloc[2], signals_2.iloc[2])


def test_sector_rank_isolation() -> None:
    """Verify that equity rank signals do not cross sector boundaries."""
    config = MomentumConfig(entry_percentile=0.75, rank_mode="sector")
    generator = CrossSectionalSignalGenerator(config)

    # 4 assets: A1, A2 in SectorA; B1, B2 in SectorB
    assets = ["A1", "A2", "B1", "B2"]
    sector_map = {"A1": "SectorA", "A2": "SectorA", "B1": "SectorB", "B2": "SectorB"}

    # Set up factor values where Sector A has huge values and Sector B has tiny values.
    # In a global rank, B1/B2 would never trigger signals.
    # In a sector rank:
    # A1 (100) vs A2 (90) -> A1 is top
    # B1 (10) vs B2 (9) -> B1 is top
    # Both A1 and B1 should get rank 1.0 within their sectors, and thus should both trigger entries.
    mom_df = pd.DataFrame(
        [[100.0, 90.0, 10.0, 9.0], [100.0, 90.0, 10.0, 9.0], [100.0, 90.0, 10.0, 9.0]],
        columns=assets,
    )
    liq_df = pd.DataFrame(
        [[100.0, 90.0, 10.0, 9.0], [100.0, 90.0, 10.0, 9.0], [100.0, 90.0, 10.0, 9.0]],
        columns=assets,
    )

    signals = generator.generate(mom_df, liq_df, sector_map=sector_map)

    # At index 1 (corresponding to shifted index 0/1 calculations), both A1 and B1 should be True
    # (since their ranks are 1.0, which is >= 0.75 quantile)
    assert bool(signals.at[1, "A1"]) is True
    assert bool(signals.at[1, "B1"]) is True
    assert bool(signals.at[1, "A2"]) is False
    assert bool(signals.at[1, "B2"]) is False
