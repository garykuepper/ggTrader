"""Check SEC instant facts with invented filing availability records."""

import pytest
import sec_facts
from filing_vintages import available_facts

pytestmark = pytest.mark.study


def _document():
    return {
        "cik": 320193,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"accn": "a1", "end": "2023-09-30", "val": 100},
                            {"accn": "a2", "end": "2023-09-30", "val": 110},
                        ]
                    }
                }
            }
        },
    }


def test_normalized_revisions_respect_independent_availability_times():
    times = {"a1": "2023-11-01T12:00:00Z", "a2": "2024-02-01T12:00:00Z"}
    rows = sec_facts.instant_facts(_document(), "Assets", "USD", times, expected_cik=320193)
    assert set(rows.issuer_id) == {"0000320193"}
    assert available_facts(rows, "2024-01-01T00:00:00Z").value.tolist() == [100]
    assert available_facts(rows, "2024-02-01T12:00:00Z").value.tolist() == [110]


def test_missing_availability_stops_instead_of_guessing():
    with pytest.raises(ValueError, match="availability"):
        sec_facts.instant_facts(
            _document(), "Assets", "USD", {"a1": "2023-11-01T12:00:00Z"}, expected_cik=320193
        )


def test_duration_fact_is_not_treated_as_instant():
    doc = _document()
    doc["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["start"] = "2023-01-01"
    with pytest.raises(ValueError, match="instant"):
        sec_facts.instant_facts(
            doc, "Assets", "USD", {"a1": "2023-11-01T12:00:00Z"}, expected_cik=320193
        )


def test_invalid_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone"):
        sec_facts.instant_facts(
            _document(),
            "Assets",
            "USD",
            {"a1": "2023-11-01", "a2": "2024-02-01"},
            expected_cik=320193,
        )


def test_rejects_availability_before_period_end():
    with pytest.raises(ValueError, match="period"):
        sec_facts.instant_facts(
            _document(),
            "Assets",
            "USD",
            {"a1": "2023-01-01T00:00:00Z", "a2": "2024-02-01T00:00:00Z"},
            expected_cik=320193,
        )


def test_rejects_availability_before_sec_filing_date():
    doc = _document()
    doc["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["filed"] = "2023-11-03"
    with pytest.raises(ValueError, match="filing date"):
        sec_facts.instant_facts(
            doc,
            "Assets",
            "USD",
            {"a1": "2023-11-02T23:00:00Z", "a2": "2024-02-01T00:00:00Z"},
            expected_cik=320193,
        )


def test_rejects_wrong_issuer_and_malformed_document():
    with pytest.raises(ValueError, match="CIK"):
        sec_facts.instant_facts(_document(), "Assets", "USD", {}, expected_cik=1)
    with pytest.raises(ValueError, match="document"):
        sec_facts.instant_facts(None, "Assets", "USD", {}, expected_cik=320193)


def test_zero_padded_sec_string_cik_preserves_issuer_identity():
    doc = _document()
    doc["cik"] = "0000320193"
    times = {"a1": "2023-11-01T12:00:00Z", "a2": "2024-02-01T12:00:00Z"}
    assert set(
        sec_facts.instant_facts(doc, "Assets", "USD", times, expected_cik=320193).issuer_id
    ) == {"0000320193"}


@pytest.mark.parametrize("cik", ["320193", "0000320194", " 0000320193", "0000000000", True])
def test_rejects_invalid_sec_cik_strings_and_booleans(cik):
    doc = _document()
    doc["cik"] = cik
    with pytest.raises(ValueError, match="CIK"):
        sec_facts.instant_facts(doc, "Assets", "USD", {}, expected_cik=320193)


def test_rejects_duplicate_accession_and_period_even_for_equal_values():
    doc = _document()
    facts = doc["facts"]["us-gaap"]["Assets"]["units"]["USD"]
    facts.append(dict(facts[0]))
    with pytest.raises(ValueError, match="Ambiguous"):
        sec_facts.instant_facts(
            doc,
            "Assets",
            "USD",
            {"a1": "2023-11-01T12:00:00Z", "a2": "2024-02-01T12:00:00Z"},
            expected_cik=320193,
        )


def test_latest_annual_instant_respects_fiscal_lag_and_filing_revision():
    doc = _document()
    doc["facts"]["us-gaap"]["Assets"]["units"]["USD"] = [
        {"accn": "old", "end": "2022-12-31", "val": 80},
        {"accn": "quarter", "end": "2023-09-30", "val": 999},
        {"accn": "annual", "end": "2023-12-31", "val": 100},
        {"accn": "revision", "end": "2023-12-31", "val": 110},
    ]
    times = {
        "old": "2023-02-01T12:00:00Z",
        "quarter": "2023-11-01T12:00:00Z",
        "annual": "2024-02-01T12:00:00Z",
        "revision": "2024-05-02T12:00:00Z",
    }
    rows = sec_facts.instant_facts(doc, "Assets", "USD", times, expected_cik=320193)
    year_ends = {("0000320193", "2022-12-31"), ("0000320193", "2023-12-31")}
    select = sec_facts.latest_annual_instant
    assert select(rows, "2024-04-29T21:00:00Z", year_ends).value.tolist() == [80]
    assert select(rows, "2024-04-30T21:00:00Z", year_ends).value.tolist() == [100]
    assert select(rows, "2024-05-02T12:00:00Z", year_ends).value.tolist() == [110]


def test_latest_annual_instant_does_not_guess_fiscal_year_ends():
    rows = sec_facts.instant_facts(
        _document(),
        "Assets",
        "USD",
        {"a1": "2023-11-01T12:00:00Z", "a2": "2024-02-01T12:00:00Z"},
        expected_cik=320193,
    )
    assert sec_facts.latest_annual_instant(rows, "2024-06-01T12:00:00Z", set()).empty


def test_latest_annual_does_not_fall_back_to_an_older_year():
    doc = _document()
    doc["facts"]["us-gaap"]["Assets"]["units"]["USD"] = [
        {"accn": "old", "end": "2022-12-31", "val": 80},
        {"accn": "late", "end": "2023-12-31", "val": 100},
    ]
    rows = sec_facts.instant_facts(
        doc,
        "Assets",
        "USD",
        {"old": "2023-02-01T12:00:00Z", "late": "2024-09-01T12:00:00Z"},
        expected_cik=320193,
    )
    ends = {("0000320193", "2022-12-31"), ("0000320193", "2023-12-31")}
    assert sec_facts.latest_annual_instant(rows, "2024-06-01T12:00:00Z", ends).empty


def test_latest_annual_rejects_future_mixed_concepts():
    rows = sec_facts.instant_facts(
        _document(),
        "Assets",
        "USD",
        {"a1": "2023-11-01T12:00:00Z", "a2": "2024-10-01T12:00:00Z"},
        expected_cik=320193,
    )
    rows.loc[1, "fact_key"] = '["us-gaap","Liabilities","USD","2023-09-30"]'
    with pytest.raises(ValueError, match="Mixed"):
        sec_facts.latest_annual_instant(rows, "2024-06-01T12:00:00Z", set())


def test_latest_annual_rejects_unpadded_issuer():
    rows = sec_facts.instant_facts(
        _document(),
        "Assets",
        "USD",
        {"a1": "2023-11-01T12:00:00Z", "a2": "2024-02-01T12:00:00Z"},
        expected_cik=320193,
    )
    with pytest.raises(ValueError, match="fiscal-year"):
        sec_facts.latest_annual_instant(rows, "2024-06-01T12:00:00Z", {("320193", "2023-09-30")})
