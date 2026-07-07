import json
from pathlib import Path

import pandas as pd
import pytest

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
from ggTrader.lab.strategies.registry import apply_sector_constraints
from ggTrader.lab.strategy import LabConfig


@pytest.fixture
def mock_sectors_file(tmp_path, monkeypatch):
    # Create a mock sp500_sectors.json inside tmp_path
    mock_data = {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "NVDA": "Information Technology",
        "JPM": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
    }
    sectors_file = tmp_path / "sp500_sectors.json"
    with open(sectors_file, "w") as f:
        json.dump(mock_data, f)

    # Monkeypatch Path.exists to return True and patch open() to read our temp file
    # for sp500_sectors.json
    original_exists = Path.exists

    def mock_exists(self):
        if "sp500_sectors.json" in str(self):
            return True
        return original_exists(self)

    original_open = open

    def mock_open(file, *args, **kwargs):
        if "sp500_sectors.json" in str(file):
            return original_open(sectors_file, *args, **kwargs)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr("builtins.open", mock_open)

    return mock_data


def test_apply_sector_constraints_pruning(mock_sectors_file):
    symbols = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "XOM"]
    # Tech: AAPL, MSFT, NVDA. Financials: JPM, BAC. Energy: XOM.

    # Cap at 1 per sector
    res = apply_sector_constraints(symbols, max_sec=1)
    assert res == ["AAPL", "JPM", "XOM"]

    # Cap at 2 per sector
    res2 = apply_sector_constraints(symbols, max_sec=2)
    assert res2 == ["AAPL", "MSFT", "JPM", "BAC", "XOM"]

    # Cap at 3 per sector
    res3 = apply_sector_constraints(symbols, max_sec=3)
    assert res3 == symbols


def test_cross_sectional_momentum_respects_sector_constraints(mock_sectors_file):
    cfg = LabConfig(top_n=5, lookback=2, skip=0, max_sector_count=1)
    strategy = CrossSectionalMomentum(cfg)

    # Mock data: daily prices for 6 symbols
    idx = pd.date_range("2021-01-01", periods=5, tz="UTC")
    prices = pd.DataFrame(
        {
            ("AAPL", "close"): [10, 11, 12, 13, 14],
            ("MSFT", "close"): [10, 10.8, 11.6, 12.4, 13.2],
            ("NVDA", "close"): [10, 10.6, 11.2, 11.8, 12.4],
            ("JPM", "close"): [10, 10.5, 11.0, 11.5, 12.0],
            ("BAC", "close"): [10, 10.4, 10.8, 11.2, 11.6],
            ("XOM", "close"): [10, 10.1, 10.2, 10.3, 10.4],
        },
        index=idx,
    )
    prices.columns.names = ["symbol", "field"]

    eligible = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "XOM"]

    plan = strategy.select(idx[4], prices, eligible)

    # Cap = 1, top_n = 5.
    selected_symbols = [p["symbol"] for p in plan]
    assert selected_symbols == ["AAPL", "JPM", "XOM"]


def test_ensemble_signal_respects_sector_constraints(mock_sectors_file):
    cfg = LabConfig(min_history_bars=2, max_sector_count=1)
    strategy = EnsembleSignal(cfg)

    # Mock price/volume data
    idx = pd.date_range("2021-01-01", periods=5, tz="UTC")
    data = pd.DataFrame(
        {
            ("AAPL", "close"): [10, 11, 12, 13, 14],
            ("MSFT", "close"): [10, 11, 12, 13, 14],
            ("JPM", "close"): [10, 11, 12, 13, 14],
        },
        index=idx,
    )
    data.columns.names = ["symbol", "field"]

    eligible = ["AAPL", "MSFT", "JPM"]

    plan = strategy.select(idx[4], data, eligible)

    # AAPL and MSFT are IT (cap 1). Only one should be selected.
    selected_symbols = [p["symbol"] for p in plan]
    assert "AAPL" in selected_symbols or "MSFT" in selected_symbols
    assert not ("AAPL" in selected_symbols and "MSFT" in selected_symbols)
    assert "JPM" in selected_symbols
