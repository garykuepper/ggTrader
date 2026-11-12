"""Check quarter-specific SEC income normalization on invented filings."""

import pytest
import sec_annual_flows
import sec_facts
from filing_vintages import available_facts

pytestmark = pytest.mark.study


def test_quarter_is_not_confused_with_year_to_date_or_later_revision():
    doc = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Income": {
                    "units": {
                        "USD": [
                            {"accn": "a", "start": "2024-04-01", "end": "2024-06-30", "val": 10},
                            {"accn": "a", "start": "2024-01-01", "end": "2024-06-30", "val": 25},
                            {"accn": "b", "start": "2024-04-01", "end": "2024-06-30", "val": 12},
                        ]
                    }
                }
            }
        },
    }
    rows = sec_annual_flows.quarterly_flows(
        doc,
        "Income",
        "USD",
        {"a": "2024-08-01T12:00:00Z", "b": "2024-10-01T12:00:00Z"},
        {("2024-04-01", "2024-06-30")},
        expected_cik=123,
    )
    assert available_facts(rows, "2024-09-01T12:00:00Z").value.tolist() == [10]
    assert available_facts(rows, "2024-10-01T12:00:00Z").value.tolist() == [12]


def test_year_to_date_is_not_a_quarter():
    with pytest.raises(ValueError, match="quarter"):
        sec_annual_flows.quarterly_flows(
            {},
            "Income",
            "USD",
            {},
            {("2024-01-01", "2024-06-30")},
            expected_cik=123,
        )


def test_fourth_quarter_is_not_treated_as_an_annual_flow():
    doc = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Income": {
                    "units": {
                        "USD": [
                            {"accn": "a", "start": "2024-10-01", "end": "2024-12-31", "val": 10},
                        ]
                    }
                }
            }
        },
    }
    rows = sec_annual_flows.quarterly_flows(
        doc,
        "Income",
        "USD",
        {"a": "2025-02-01T12:00:00Z"},
        {("2024-10-01", "2024-12-31")},
        expected_cik=123,
    )
    with pytest.raises(ValueError, match="annual"):
        sec_facts.latest_annual_instant(
            rows, "2025-06-01T12:00:00Z", {("0000000123", "2024-12-31")}
        )
