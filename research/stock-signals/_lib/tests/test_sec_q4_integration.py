"""Check the fourth-quarter path with invented annual and nine-month filings."""

import pytest
import sec_annual_flows
import sec_q4
from filing_vintages import available_facts

pytestmark = pytest.mark.study


def test_q4_uses_only_nine_month_revision_available_at_decision():
    doc = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Income": {
                    "units": {
                        "USD": [
                            {"accn": "q3", "start": "2023-01-01", "end": "2023-09-30", "val": 90},
                            {
                                "accn": "annual",
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 120,
                            },
                            {
                                "accn": "q3_revision",
                                "start": "2023-01-01",
                                "end": "2023-09-30",
                                "val": 95,
                            },
                        ]
                    }
                }
            }
        },
    }
    times = {
        "q3": "2023-11-01T12:00:00Z",
        "annual": "2024-02-01T12:00:00Z",
        "q3_revision": "2024-04-01T12:00:00Z",
    }
    yearly = sec_annual_flows.annual_flows(
        doc, "Income", "USD", times, {("2023-01-01", "2023-12-31")}, expected_cik=123
    )
    ytd = sec_annual_flows.nine_month_flows(
        doc, "Income", "USD", times, {("2023-01-01", "2023-09-30")}, expected_cik=123
    )

    def q4(decision):
        a = available_facts(yearly, decision).iloc[0].to_dict()
        b = available_facts(ytd, decision).iloc[0].to_dict()
        return sec_q4.derive_q4(a, b, decision)

    earlier = q4("2024-03-01T12:00:00Z")
    later = q4("2024-04-01T12:00:00Z")
    assert earlier["value"] == 30
    assert later["value"] == 25
    assert [(row["accession"], row["value"]) for row in earlier["sources"]] == [
        ("annual", 120),
        ("q3", 90),
    ]
    assert [(row["accession"], row["value"]) for row in later["sources"]] == [
        ("annual", 120),
        ("q3_revision", 95),
    ]
    assert str(later["sources"][1]["available_at"]) == "2024-04-01 12:00:00+00:00"


def test_nine_month_loader_rejects_a_full_year():
    with pytest.raises(ValueError, match="nine-month"):
        sec_annual_flows.nine_month_flows(
            {}, "Income", "USD", {}, {("2023-01-01", "2023-12-31")}, expected_cik=123
        )
