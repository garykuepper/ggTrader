"""Test rollup for the universe-wide dividend risk audit."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "dividend_universe_audit.py"  # root-ok: sibling
spec = importlib.util.spec_from_file_location("dividend_universe_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_summary_distinguishes_uncheckable_and_flagged_events():
    results = {
        "A": {
            "reference_events": 2,
            "checked": 1,
            "unchanged_factor_dates": ["2022-01-04"],
            "uncheckable_dates": ["2022-02-01"],
        },
        "B": {
            "reference_events": 1,
            "checked": 1,
            "unchanged_factor_dates": [],
            "uncheckable_dates": [],
        },
    }
    assert module.summarize(results) == {
        "symbols_with_events": 2,
        "events": 3,
        "checked_events": 2,
        "flagged_events": 1,
        "uncheckable_events": 1,
        "symbols_flagged": 1,
    }
