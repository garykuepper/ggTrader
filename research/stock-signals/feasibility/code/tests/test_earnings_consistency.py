"""Check earnings-consistency portfolio selection."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "earnings_consistency_screen.py"  # root-ok: sibling
spec = importlib.util.spec_from_file_location("earnings_consistency_screen", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select_portfolio = module.select_portfolio

pytestmark = pytest.mark.study


def test_selects_long_high_group_and_rejects_duplicates():
    frame = pd.DataFrame(
        {
            "signalname": ["EarningsConsistency", "EarningsConsistency"],
            "port": ["05", "LS"],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "ret": [2.0, 90.0],
            "Nlong": [10, 10],
            "Nshort": [0, 10],
        }
    )
    assert select_portfolio(frame).iloc[0] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="duplicate"):
        select_portfolio(pd.concat([frame, frame.iloc[:1]]))


def test_rejects_missing_or_short_positions():
    frame = pd.DataFrame(
        {
            "signalname": ["EarningsConsistency"],
            "port": ["05"],
            "date": [pd.Timestamp("2020-01-31")],
            "ret": [2.0],
            "Nlong": [10],
            "Nshort": [1],
        }
    )
    with pytest.raises(ValueError, match="long-only"):
        select_portfolio(frame)
    with pytest.raises(ValueError, match="Missing"):
        select_portfolio(frame.iloc[:0])
