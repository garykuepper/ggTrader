"""Check annual SEC flow normalization on invented filings."""

import pytest
import sec_annual_flows
from filing_vintages import available_facts

pytestmark = pytest.mark.study


def document():
    return {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"accn": "a", "start": "2023-01-01", "end": "2023-12-31", "val": 100},
                            {"accn": "a", "start": "2023-10-01", "end": "2023-12-31", "val": 30},
                            {"accn": "b", "start": "2023-01-01", "end": "2023-12-31", "val": 110},
                        ]
                    }
                }
            }
        },
    }


def test_only_explicit_annual_period_is_normalized_and_revisions_are_timed():
    times = {"a": "2024-02-01T12:00:00Z", "b": "2024-06-01T12:00:00Z"}
    data = sec_annual_flows.annual_flows(
        document(),
        "Revenues",
        "USD",
        times,
        {("2023-01-01", "2023-12-31")},
        expected_cik=123,
    )
    assert len(data) == 2
    assert available_facts(data, "2024-05-01T12:00:00Z").value.tolist() == [100]
    assert available_facts(data, "2024-06-01T12:00:00Z").value.tolist() == [110]


def test_annual_flow_accepts_zero_padded_document_cik():
    doc = document()
    doc["cik"] = "0000000123"
    times = {"a": "2024-02-01T12:00:00Z", "b": "2024-06-01T12:00:00Z"}
    data = sec_annual_flows.annual_flows(
        doc, "Revenues", "USD", times, {("2023-01-01", "2023-12-31")}, expected_cik=123
    )
    assert set(data.issuer_id) == {"0000000123"}


def test_rejects_quarter_or_unverified_period():
    with pytest.raises(ValueError):
        sec_annual_flows.annual_flows(
            document(),
            "Revenues",
            "USD",
            {},
            {("2023-10-01", "2023-12-31")},
            expected_cik=123,
        )
    with pytest.raises(ValueError):
        sec_annual_flows.annual_flows(
            document(),
            "Revenues",
            "USD",
            {},
            set(),
            expected_cik=123,
        )


def test_rejects_unverified_availability_and_wrong_issuer():
    periods = {("2023-01-01", "2023-12-31")}
    with pytest.raises(ValueError, match="availability"):
        sec_annual_flows.annual_flows(document(), "Revenues", "USD", {}, periods, expected_cik=123)
    with pytest.raises(ValueError, match="CIK"):
        sec_annual_flows.annual_flows(document(), "Revenues", "USD", {}, periods, expected_cik=456)


def test_rejects_different_annual_start_for_same_year_end():
    doc = document()
    doc["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
        {"accn": "c", "start": "2023-01-02", "end": "2023-12-31", "val": 999}
    )
    with pytest.raises(ValueError, match="start"):
        sec_annual_flows.annual_flows(
            doc, "Revenues", "USD", {}, {("2023-01-01", "2023-12-31")}, expected_cik=123
        )


def test_flow_key_retains_its_period_start_and_type():
    import json

    data = sec_annual_flows.annual_flows(
        document(),
        "Revenues",
        "USD",
        {"a": "2024-02-01T12:00:00Z", "b": "2024-06-01T12:00:00Z"},
        {("2023-01-01", "2023-12-31")},
        expected_cik=123,
    )
    assert json.loads(data.fact_key.iloc[0]) == [
        "us-gaap",
        "Revenues",
        "USD",
        "duration",
        "2023-01-01",
        "2023-12-31",
    ]


def test_different_annual_starts_cannot_be_stacked_silently():
    import pandas as pd
    import sec_facts

    one = document()
    two = document()
    two["facts"]["us-gaap"]["Revenues"]["units"]["USD"] = [
        {"accn": "c", "start": "2023-01-02", "end": "2023-12-31", "val": 999}
    ]
    a = sec_annual_flows.annual_flows(
        one,
        "Revenues",
        "USD",
        {"a": "2024-02-01T12:00:00Z", "b": "2024-06-01T12:00:00Z"},
        {("2023-01-01", "2023-12-31")},
        expected_cik=123,
    )
    b = sec_annual_flows.annual_flows(
        two,
        "Revenues",
        "USD",
        {"c": "2024-07-01T12:00:00Z"},
        {("2023-01-02", "2023-12-31")},
        expected_cik=123,
    )
    with pytest.raises(ValueError, match="Ambiguous annual period"):
        sec_facts.latest_annual_instant(
            pd.concat([a, b]), "2024-05-01T12:00:00Z", {("0000000123", "2023-12-31")}
        )
