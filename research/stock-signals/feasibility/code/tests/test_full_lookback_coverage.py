"""Test complete lookback counting using synthetic membership and sessions."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "full_lookback_coverage.py"  # root-ok: sibling code
spec = importlib.util.spec_from_file_location("full_lookback_coverage", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_counts_same_day_and_complete_history_per_decision():
    days = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    history = pd.DataFrame(
        {"tickers": [["A", "B"]]}, index=pd.DatetimeIndex(["2021-01-04"], tz="UTC")
    )
    bars = pd.DataFrame(
        {"timestamp": [days[0], days[1], days[2], days[2]], "symbol": ["A", "A", "A", "B"]}
    )
    result = module.measure([{"date": "2021-01-06", "n_members": 2}], bars, history, days, window=3)
    assert result[0]["same_day"] == 2
    assert result[0]["complete_lookback"] == 1
    assert result[0]["insufficient_lookback"] == ["B"]


def test_rejects_non_session_decisions_and_duplicates():
    days = pd.to_datetime(["2021-01-04", "2021-01-05"])
    history = pd.DataFrame({"tickers": [["A"]]}, index=pd.DatetimeIndex(["2021-01-04"], tz="UTC"))
    bars = pd.DataFrame({"timestamp": [days[0]], "symbol": ["A"]})
    with pytest.raises(ValueError, match="calendar"):
        module.measure([{"date": "2021-01-09", "n_members": 1}], bars, history, days, window=1)
    with pytest.raises(ValueError, match="duplicate"):
        module.measure([], pd.concat([bars, bars]), history, days, window=1)
