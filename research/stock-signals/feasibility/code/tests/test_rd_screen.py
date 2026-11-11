"""Check the R&D portfolio selection and fixed windows."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "rd_screen.py"
spec = importlib.util.spec_from_file_location("rd_screen", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_selects_high_rd_long_leg_and_rejects_duplicates():
    frame = pd.DataFrame(
        {
            "signalname": ["RD", "RD"],
            "port": ["05", "LS"],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "ret": [2.0, 90.0],
            "Nlong": [10, 10],
            "Nshort": [0, 10],
        }
    )
    selected = module.select_rd(frame)
    assert selected.iloc[0] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="duplicate"):
        module.select_rd(pd.concat([frame, frame.iloc[:1]]))


def test_fixed_windows_and_missing_month():
    dates = pd.date_range("1975-07-31", "2025-12-31", freq="ME")
    candidate = pd.Series(0.01, index=dates)
    market = pd.Series(0.005, index=dates)
    result = module.evaluate(candidate, market)
    assert result["full"]["comparison"]["months"] == 606
    assert result["since_1994"]["comparison"]["months"] == 384
    with pytest.raises(ValueError):
        module.evaluate(candidate.drop(dates[0]), market)
