"""Test conservative name comparisons used for retrospective review leads."""

import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "ncen_name_review.py")
pytestmark = pytest.mark.study


def test_compares_names_without_guessing_aliases():
    rows = [
        {"symbol": "A", "name": "Example, Inc. Common Stock", "REGISTRANT_NAME": "EXAMPLE INC"},
        {"symbol": "B", "name": "Other Fund Common Stock", "REGISTRANT_NAME": "Different Fund"},
    ]
    result = module.summarize(rows)
    assert result["matching_labels"] == ["A"]
    assert result["review"][0]["symbol"] == "B"


def test_mixed_names_are_flagged_for_review():
    rows = [
        {"symbol": "A", "name": "Example Common Stock", "REGISTRANT_NAME": "Different"},
        {"symbol": "A", "name": "Example Common Stock", "REGISTRANT_NAME": "Example"},
    ]
    result = module.summarize(rows)
    assert result["matching_labels"] == []
    assert result["review"][0]["symbol"] == "A"
