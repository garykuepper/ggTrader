"""Check direct-ticker coverage and bar-quality diagnostics on invented data."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "finsaber_coverage.py"  # root-ok: sibling code
spec = importlib.util.spec_from_file_location("finsaber_coverage", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_distinguishes_missing_dates_from_invalid_bars():
    sessions = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    rows = [{"date": "2021-01-06", "missing": ["A", "B"]}]
    bars = pd.DataFrame(
        {
            "date": ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-06"],
            "symbol": ["A", "A", "A", "B"],
            "open": [10, 10, 10, 10],
            "high": [11, 11, 11, 11],
            "low": [9, 9, 9, 9],
            "close": [10, 10, 10, 10],
            "adjusted_close": [10, 10, 0, 10],
            "volume": [100, 100, 100, 100],
        }
    )
    result = module.measure(rows, bars, sessions, excluded=set(), window=3)
    assert result["A"] == {
        "missing_months": 1,
        "same_day": 1,
        "complete_dates": 1,
        "complete_valid_bars": 0,
    }
    assert result["B"]["same_day"] == 1
    assert result["B"]["complete_dates"] == 0


def test_duplicate_symbol_dates_are_rejected():
    bars = pd.DataFrame({"date": ["2021-01-04"] * 2, "symbol": ["A"] * 2})
    with pytest.raises(ValueError, match="duplicate"):
        module.measure([], bars, pd.to_datetime(["2021-01-04"]), excluded=set(), window=1)


def test_zero_volume_is_not_a_valid_lookback_bar():
    bars = pd.DataFrame(
        {
            "date": ["2021-01-04"],
            "symbol": ["A"],
            "open": [10],
            "high": [10],
            "low": [10],
            "close": [10],
            "adjusted_close": [10],
            "volume": [0],
        }
    )
    result = module.measure(
        [{"date": "2021-01-04", "missing": ["A"]}],
        bars,
        pd.to_datetime(["2021-01-04"]),
        excluded=set(),
        window=1,
    )
    assert result["A"]["same_day"] == result["A"]["complete_dates"] == 1
    assert result["A"]["complete_valid_bars"] == 0
