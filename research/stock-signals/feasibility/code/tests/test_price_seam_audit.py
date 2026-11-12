"""Check price-vintage seam detection on invented prices."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "price_seam_audit.py"  # root-ok: sibling
spec = importlib.util.spec_from_file_location("price_seam_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_flags_ratio_shift_only_on_consecutive_matching_sessions():
    sessions = pd.date_range("2024-01-01", periods=4)
    tape = pd.DataFrame({"symbol": ["A"] * 3, "date": sessions[[0, 1, 3]], "price": [102, 100, 97]})
    ref = pd.DataFrame({"symbol": ["A"] * 4, "date": sessions, "price": [100] * 4})
    result = module.assess(tape, ref, sessions)
    assert result["matched_pairs"] == 1
    assert [(x["symbol"], x["date"]) for x in result["flags"]] == [("A", "2024-01-02")]


def test_duplicate_symbol_dates_are_rejected():
    sessions = pd.date_range("2024-01-01", periods=2)
    tape = pd.DataFrame({"symbol": ["A", "A"], "date": [sessions[0]] * 2, "price": [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        module.assess(tape, tape.iloc[:1], sessions)


def test_invalid_price_does_not_form_a_pair_or_a_flag():
    sessions = pd.date_range("2024-01-01", periods=3)
    tape = pd.DataFrame({"symbol": ["A"] * 3, "date": sessions, "price": [100, 0, 200]})
    ref = pd.DataFrame({"symbol": ["A"] * 3, "date": sessions, "price": [100] * 3})
    result = module.assess(tape, ref, sessions)
    assert result["invalid_matched_rows"] == 1
    assert result["matched_pairs"] == 0
    assert result["flags"] == []
