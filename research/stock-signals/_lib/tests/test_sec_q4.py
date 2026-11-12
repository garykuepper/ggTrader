"""Check fourth-quarter derivation using invented financial facts."""

import json

import numpy as np
import pytest
import sec_q4

pytestmark = pytest.mark.study


def fact(start, end, value, available, accession):
    return {
        "issuer_id": "0000000123",
        "fact_key": json.dumps(["us-gaap", "Income", "USD", "duration", start, end]),
        "value": value,
        "available_at": available,
        "accession": accession,
    }


def test_q4_uses_available_annual_and_nine_month_amounts():
    annual = fact("2023-01-01", "2023-12-31", 120, "2024-02-01T12:00:00Z", "annual")
    nine = fact("2023-01-01", "2023-09-30", 90, "2023-11-01T12:00:00Z", "nine")
    result = sec_q4.derive_q4(annual, nine, "2024-02-01T12:00:00Z")
    assert result["value"] == 30
    assert (result["period_start"], result["period_end"]) == ("2023-10-01", "2023-12-31")
    assert result["source_accessions"] == ("annual", "nine")


def test_q4_rejects_future_or_mismatched_input():
    annual = fact("2023-01-01", "2023-12-31", 120, "2024-02-01T12:00:00Z", "annual")
    nine = fact("2023-01-01", "2023-09-30", 90, "2023-11-01T12:00:00Z", "nine")
    with pytest.raises(ValueError, match="available"):
        sec_q4.derive_q4(annual, nine, "2024-01-31T12:00:00Z")
    nine["issuer_id"] = "0000000456"
    with pytest.raises(ValueError, match="match"):
        sec_q4.derive_q4(annual, nine, "2024-03-01T12:00:00Z")


def test_q4_rejects_zero_issuer_even_when_both_facts_match():
    annual = fact("2023-01-01", "2023-12-31", 120, "2024-02-01T12:00:00Z", "annual")
    nine = fact("2023-01-01", "2023-09-30", 90, "2023-11-01T12:00:00Z", "nine")
    annual["issuer_id"] = nine["issuer_id"] = "0000000000"
    with pytest.raises(ValueError, match="issuer"):
        sec_q4.derive_q4(annual, nine, "2024-03-01T12:00:00Z")


def test_q4_rejects_nonmatching_fiscal_start_and_impossible_availability():
    annual = fact("2022-09-25", "2023-09-30", 120, "2023-11-01T12:00:00Z", "annual")
    nine = fact("2022-09-26", "2023-07-01", 90, "2023-08-01T12:00:00Z", "nine")
    with pytest.raises(ValueError, match="match"):
        sec_q4.derive_q4(annual, nine, "2023-12-01T12:00:00Z")
    nine = fact("2022-09-25", "2023-07-01", 90, "2023-06-01T12:00:00Z", "nine")
    with pytest.raises(ValueError, match="precedes"):
        sec_q4.derive_q4(annual, nine, "2023-12-01T12:00:00Z")


def test_q4_accepts_a_53_week_fiscal_year():
    annual = fact("2022-09-25", "2023-09-30", 120, "2023-11-01T12:00:00Z", "annual")
    nine = fact("2022-09-25", "2023-07-01", 90, "2023-08-01T12:00:00Z", "nine")
    result = sec_q4.derive_q4(annual, nine, "2023-12-01T12:00:00Z")
    assert result["period_start"] == "2023-07-02"
    assert result["value"] == 30


@pytest.mark.parametrize("unit", ["shares", "USD/shares"])
def test_q4_rejects_non_dollar_units(unit):
    annual = fact("2023-01-01", "2023-12-31", 120, "2024-02-01T12:00:00Z", "annual")
    nine = fact("2023-01-01", "2023-09-30", 90, "2023-11-01T12:00:00Z", "nine")
    for row in (annual, nine):
        key = json.loads(row["fact_key"])
        key[2] = unit
        row["fact_key"] = json.dumps(key)
    with pytest.raises(ValueError, match="USD"):
        sec_q4.derive_q4(annual, nine, "2024-03-01T12:00:00Z")


def test_q4_handles_numpy_integer_subtraction_without_wraparound():
    annual = fact(
        "2023-01-01",
        "2023-12-31",
        np.int64(9_000_000_000_000_000_000),
        "2024-02-01T12:00:00Z",
        "annual",
    )
    nine = fact(
        "2023-01-01",
        "2023-09-30",
        np.int64(-9_000_000_000_000_000_000),
        "2023-11-01T12:00:00Z",
        "nine",
    )
    assert (
        sec_q4.derive_q4(annual, nine, "2024-03-01T12:00:00Z")["value"]
        == 18_000_000_000_000_000_000
    )


def test_q4_later_nine_month_revision_requires_caller_to_verify_consistency():
    annual = fact("2023-01-01", "2023-12-31", 120, "2024-02-01T12:00:00Z", "annual")
    nine = fact("2023-01-01", "2023-09-30", 90, "2024-03-01T12:00:00Z", "revised")
    result = sec_q4.derive_q4(annual, nine, "2024-03-02T12:00:00Z")
    assert result["value"] == 30
    assert str(result["available_at"]) == "2024-03-01 12:00:00+00:00"
