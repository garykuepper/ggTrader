"""Check adjustment diagnostics on invented dividend events."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "dividend_adjustment_audit.py"  # root-ok: sibling
spec = importlib.util.spec_from_file_location("dividend_adjustment_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_unchanged_adjustment_factor_is_flagged_and_changed_factor_is_not():
    sessions = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    source = pd.DataFrame(
        {
            "date": [sessions[0], sessions[1], sessions[0], sessions[1]],
            "symbol": ["A", "A", "B", "B"],
            "close": [100.0, 99.0, 100.0, 99.0],
            "adjusted_close": [100.0, 99.0, 99.0, 99.0],
        }
    )
    reference = pd.DataFrame(
        {
            "date": [sessions[1], sessions[1], sessions[2]],
            "symbol": ["A", "B", "A"],
            "dividend": [1.0, 1.0, 1.0],
        }
    )
    result = module.assess(source, reference, sessions)
    assert result["A"] == {
        "reference_events": 2,
        "checked": 1,
        "unchanged_factor_dates": ["2022-01-04"],
        "uncheckable_dates": ["2022-01-05"],
    }
    assert result["B"]["checked"] == 1
    assert not result["B"]["unchanged_factor_dates"]


def test_rejects_duplicate_symbol_dates():
    source = pd.DataFrame({"date": ["2022-01-03"] * 2, "symbol": ["A"] * 2})
    reference = pd.DataFrame({"date": [], "symbol": [], "dividend": []})
    with pytest.raises(ValueError, match="duplicate"):
        module.assess(source, reference, pd.to_datetime(["2022-01-03"]))
