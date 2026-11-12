"""Check the exact-period filing coverage diagnostic on invented data."""

import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "sec_income_coverage.py")
pytestmark = pytest.mark.study


def test_coverage_excludes_future_wrong_form_and_wrong_period():
    rows = [
        {
            "start": "2023-01-01",
            "end": "2023-12-31",
            "form": "10-K",
            "filed": "2024-02-01",
            "accn": "a",
        },
        {
            "start": "2023-01-01",
            "end": "2023-12-31",
            "form": "10-K",
            "filed": "2024-06-01",
            "accn": "b",
        },
        {
            "start": "2023-01-01",
            "end": "2023-09-30",
            "form": "10-K",
            "filed": "2023-11-01",
            "accn": "c",
        },
        {
            "start": "2023-04-01",
            "end": "2023-09-30",
            "form": "10-Q",
            "filed": "2023-11-01",
            "accn": "d",
        },
    ]
    document = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": rows}}}}}
    periods = {"annual": ("2023-01-01", "2023-12-31"), "nine_month": ("2023-01-01", "2023-09-30")}
    result = module.observations(document, "2024-05-01", periods)
    assert result == {"annual": [{"accession": "a", "filed": "2024-02-01"}], "nine_month": []}


def test_missing_concept_is_reported_as_no_observations():
    assert module.observations({}, "2024-05-01", {"annual": ("2023-01-01", "2023-12-31")}) == {
        "annual": []
    }
