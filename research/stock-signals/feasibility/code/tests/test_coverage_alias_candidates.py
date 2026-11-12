"""Check that ticker links only count available pre-change rows."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "coverage_alias_candidates.py"  # root-ok: sibling code
spec = importlib.util.spec_from_file_location("coverage_alias_candidates", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_counts_only_missing_members_with_pre_change_successor_bars():
    rows = [
        {"date": "2021-01-29", "missing": ["OLD", "OTHER"]},
        {"date": "2021-02-26", "missing": ["OLD"]},
        {"date": "2021-03-31", "missing": ["OLD"]},
    ]
    bars = pd.DataFrame(
        {"symbol": ["NEW", "NEW"], "timestamp": pd.to_datetime(["2021-01-29", "2021-03-31"])}
    )
    result = module.count_candidates(rows, bars, {"OLD": ("NEW", "2021-03-01")})
    assert result == {"OLD": {"successor": "NEW", "eligible": 2, "present": 1}}


def test_rejects_duplicate_tape_rows():
    bars = pd.DataFrame(
        {"symbol": ["NEW", "NEW"], "timestamp": pd.to_datetime(["2021-01-29", "2021-01-29"])}
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.count_candidates([], bars, {})
