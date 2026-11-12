"""Check point-in-time selection of expected-growth training targets."""

import json

import expected_growth_training
import pandas as pd
import pytest
import sec_facts
from expected_growth_training import investment_target

pytestmark = pytest.mark.study


def test_filed_fiscal_year_ends_respects_filing_availability(facts):
    issuer = "0000000123"
    filings = pd.DataFrame(
        [
            {
                "issuer_id": issuer,
                "period_end": end,
                "accession": accession,
                "form": "10-K",
                "available_at": available,
            }
            for end, accession, available in [
                ("2021-12-31", "a", "2022-02-01T12:00:00Z"),
                ("2022-12-31", "b", "2023-02-01T12:00:00Z"),
                ("2023-12-31", "c", "2024-02-01T12:00:00Z"),
                ("2024-12-31", "d", "2025-05-01T12:00:00Z"),
            ]
        ]
    )
    get_ends = expected_growth_training.filed_fiscal_year_ends
    before = get_ends(filings, issuer=issuer, fit_at="2025-05-01T11:59:59Z")
    after = get_ends(filings, issuer=issuer, fit_at="2025-05-01T12:00:00Z")
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    assert before == set(ends)
    assert after == set(ends) | {"2024-12-31"}
    # A known calendar year-end can precede its late filing. The filed-only set
    # misses it and must not be mistaken for the complete known fiscal calendar.
    with pytest.raises(ValueError, match="known fiscal year ends"):
        target(
            facts, issuer=issuer, ends=ends, known_year_ends=before, fit_at="2025-05-01T11:59:59Z"
        )
    assert target(
        facts, issuer=issuer, ends=ends, known_year_ends=set(before), fit_at="2025-05-01T11:59:59Z"
    ) == pytest.approx(0)
    assert (
        target(
            facts,
            issuer=issuer,
            ends=ends,
            known_year_ends=set(before) | {"2024-12-31"},
            fit_at="2025-05-01T11:59:59Z",
        )
        is None
    )
    assert (
        target(
            facts,
            issuer=issuer,
            ends=ends,
            known_year_ends=set(after),
            fit_at="2025-05-01T12:00:00Z",
        )
        is None
    )


def test_known_fiscal_year_ends_rejects_bad_future_filing():
    rows = pd.DataFrame(
        [
            {
                "issuer_id": "0000000123",
                "period_end": "2025-12-31",
                "accession": "a",
                "form": "10-K",
                "available_at": "2025-01-01T00:00:00Z",
            }
        ]
    )
    with pytest.raises(ValueError, match="annual"):
        expected_growth_training.filed_fiscal_year_ends(
            rows, issuer="0000000123", fit_at="2024-01-01T00:00:00Z"
        )


@pytest.mark.parametrize("transition_form", ["10-KT", "10-KT/A"])
def test_known_fiscal_year_ends_keeps_transition_reports_for_the_correct_issuer(
    facts, transition_form
):
    rows = pd.DataFrame(
        [
            {
                "issuer_id": "0000000123",
                "period_end": "2022-06-30",
                "accession": "a",
                "form": transition_form,
                "available_at": "2022-08-01T12:00:00Z",
            },
            {
                "issuer_id": "0000000999",
                "period_end": "2023-12-31",
                "accession": "b",
                "form": "10-K",
                "available_at": "2024-02-01T12:00:00Z",
            },
        ]
    )
    transition = expected_growth_training.filed_fiscal_year_ends(
        rows, issuer="0000000123", fit_at="2024-05-01T12:00:00Z"
    )
    assert transition == {"2022-06-30"}
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    assert (
        target(
            facts,
            issuer="0000000123",
            ends=ends,
            known_year_ends=set(ends) | transition,
            fit_at="2024-05-01T12:00:00Z",
        )
        is None
    )


def test_known_fiscal_year_ends_rejects_duplicate_accessions_and_naive_times():
    row = {
        "issuer_id": "0000000123",
        "period_end": "2023-12-31",
        "accession": "a",
        "form": "10-K",
        "available_at": "2024-02-01T12:00:00Z",
    }
    with pytest.raises(ValueError, match="annual"):
        expected_growth_training.filed_fiscal_year_ends(
            pd.DataFrame([row, row]), issuer="0000000123", fit_at="2024-05-01T12:00:00Z"
        )
    row["available_at"] = "2024-02-01 12:00:00"
    with pytest.raises(ValueError, match="annual"):
        expected_growth_training.filed_fiscal_year_ends(
            pd.DataFrame([row]), issuer="0000000123", fit_at="2024-05-01T12:00:00Z"
        )


def target(facts, *, issuer, ends, fit_at, known_year_ends=None):
    if known_year_ends is None:
        known_year_ends = set(ends)
    return investment_target(
        facts,
        issuer=issuer,
        ends=ends,
        known_year_ends=known_year_ends,
        known_continuity_breaks={},
        fit_at=fit_at,
    )


@pytest.fixture
def facts():
    document = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"accn": "a", "end": "2021-12-31", "val": 100},
                            {"accn": "b", "end": "2022-12-31", "val": 120},
                            {"accn": "c", "end": "2023-12-31", "val": 132},
                            {"accn": "d", "end": "2023-12-31", "val": 144},
                        ]
                    }
                }
            }
        },
    }
    return sec_facts.instant_facts(
        document,
        "Assets",
        "USD",
        {
            "a": "2022-02-01T12:00:00Z",
            "b": "2023-02-01T12:00:00Z",
            "c": "2024-02-01T12:00:00Z",
            "d": "2025-02-01T12:00:00Z",
        },
        expected_cik=123,
    )


def test_target_requires_all_facts_at_cutoff_and_uses_available_version(facts):
    kwargs = {"issuer": "0000000123", "ends": ("2021-12-31", "2022-12-31", "2023-12-31")}
    late = facts.copy()
    late.loc[late.accession == "c", "available_at"] = "2024-05-01T12:00:00Z"
    assert target(late, fit_at="2024-05-01T11:59:59Z", **kwargs) is None
    assert target(late, fit_at="2024-05-01T12:00:00Z", **kwargs) == pytest.approx(-0.1)
    assert target(late, fit_at="2025-02-01T12:00:00Z", **kwargs) == pytest.approx(0)
    assert target(late, fit_at="2024-05-01T12:00:00Z", **kwargs) == pytest.approx(-0.1)


def test_target_evidence_keeps_selected_filing_versions(facts):
    kwargs = {
        "issuer": "0000000123",
        "ends": ("2021-12-31", "2022-12-31", "2023-12-31"),
        "known_year_ends": {"2021-12-31", "2022-12-31", "2023-12-31"},
        "known_continuity_breaks": {},
    }
    earlier = expected_growth_training.investment_target_evidence(
        facts, fit_at="2024-05-01T12:00:00Z", **kwargs
    )
    later = expected_growth_training.investment_target_evidence(
        facts, fit_at="2025-02-01T12:00:00Z", **kwargs
    )
    assert earlier["target"] == pytest.approx(-0.1)
    assert later["target"] == pytest.approx(0)
    assert earlier["issuer"] == kwargs["issuer"]
    assert earlier["fit_at"] == "2024-05-01T12:00:00+00:00"
    assert [row["accession"] for row in earlier["facts"]] == ["a", "b", "c"]
    assert [row["accession"] for row in later["facts"]] == ["a", "b", "d"]
    assert [row["period_end"] for row in earlier["facts"]] == list(kwargs["ends"])
    assert earlier["facts"][2]["available_at"] == "2024-02-01T12:00:00+00:00"
    assert earlier["facts"][2]["value"] == 132
    assert (
        expected_growth_training.investment_target_evidence(
            facts, fit_at="2024-04-29T12:00:00Z", **kwargs
        )
        is None
    )


def test_target_evidence_serializes_nullable_integer_values(facts):
    facts.loc[facts.accession == "a", "value"] = 2**53 + 1
    facts["value"] = facts["value"].astype("Int64")
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    result = expected_growth_training.investment_target_evidence(
        facts,
        issuer="0000000123",
        ends=ends,
        known_year_ends=set(ends),
        known_continuity_breaks={},
        fit_at="2024-05-01T12:00:00Z",
    )
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert [row["value"] for row in restored["facts"]] == [2**53 + 1, 120, 132]


def test_target_evidence_preserves_only_continuity_information_known_at_cutoff(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    result = expected_growth_training.investment_target_evidence(
        facts,
        issuer="0000000123",
        ends=ends,
        known_year_ends={*ends, "2020-12-31"},
        known_continuity_breaks={
            "2021-01-01": "2021-02-01T07:00:00-05:00",
            "2022-06-01": "2024-06-01T12:00:00Z",
        },
        fit_at="2024-05-01T12:00:00Z",
    )
    assert result["known_year_ends"] == ["2020-12-31", *ends]
    assert result["known_continuity_breaks"] == {"2021-01-01": "2021-02-01T12:00:00+00:00"}
    without_future = expected_growth_training.investment_target_evidence(
        facts,
        issuer="0000000123",
        ends=ends,
        known_year_ends={*ends, "2020-12-31"},
        known_continuity_breaks={"2021-01-01": "2021-02-01T12:00:00Z"},
        fit_at="2024-05-01T12:00:00Z",
    )
    assert result == without_future
    json.dumps(result, allow_nan=False)


def test_target_evidence_excludes_future_fiscal_ends(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    common = dict(
        issuer="0000000123",
        ends=ends,
        known_continuity_breaks={},
        fit_at="2024-05-01T12:00:00Z",
    )
    without_future = expected_growth_training.investment_target_evidence(
        facts, known_year_ends=set(ends), **common
    )
    with_future = expected_growth_training.investment_target_evidence(
        facts, known_year_ends={*ends, "2024-12-31"}, **common
    )
    assert with_future == without_future


def test_middle_year_revision_is_used_only_when_available(facts, monkeypatch):
    revised = facts.loc[facts.accession != "d"].copy()
    restatement = revised.loc[revised.accession == "b"].copy()
    restatement.loc[:, "accession"] = "restatement"
    restatement.loc[:, "available_at"] = "2024-06-01T12:00:00Z"
    restatement.loc[:, "value"] = 125
    revised = pd.concat([revised, restatement], ignore_index=True)
    received = []

    def capture(**values):
        received.append(values)
        return 0.0

    monkeypatch.setattr(expected_growth_training, "investment_change", capture)
    kwargs = {"issuer": "0000000123", "ends": ("2021-12-31", "2022-12-31", "2023-12-31")}
    for cutoff in ("2024-06-01T11:59:59Z", "2024-06-01T12:00:00Z", "2024-06-01T11:59:59Z"):
        target(revised, fit_at=cutoff, **kwargs)
    assert received == [
        {"prior_assets": 100, "assets": 120, "next_assets": 132},
        {"prior_assets": 100, "assets": 125, "next_assets": 132},
        {"prior_assets": 100, "assets": 120, "next_assets": 132},
    ]


def test_target_requires_four_calendar_months_on_new_york_date(facts):
    kwargs = {"issuer": "0000000123", "ends": ("2021-12-31", "2022-12-31", "2023-12-31")}
    assert target(facts, fit_at="2024-04-30T03:59:59Z", **kwargs) is None
    assert target(facts, fit_at="2024-04-30T04:00:00Z", **kwargs) == pytest.approx(-0.1)


def test_target_does_not_fall_back_when_newer_known_year_is_eligible(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    known = {*ends, "2024-12-31"}
    kwargs = {"issuer": "0000000123", "ends": ends, "known_year_ends": known}
    assert target(facts, fit_at="2025-04-30T03:59:59Z", **kwargs) == pytest.approx(0)
    assert target(facts, fit_at="2025-04-30T04:00:00Z", **kwargs) is None


def test_target_requires_verified_year_end_set(facts):
    kwargs = {
        "issuer": "0000000123",
        "ends": ("2021-12-31", "2022-12-31", "2023-12-31"),
        "fit_at": "2024-05-01T00:00:00Z",
    }
    with pytest.raises(TypeError):
        investment_target(facts, **kwargs)
    for invalid in ({"2021-12-31"}, {"2021-12-31", "2022-12-31", "2023-12-31", "20241231"}):
        with pytest.raises(ValueError, match="known fiscal"):
            target(facts, known_year_ends=invalid, **kwargs)


def test_target_rejects_a_known_intervening_fiscal_period(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    assert (
        target(
            facts,
            issuer="0000000123",
            ends=ends,
            known_year_ends={*ends, "2022-06-30"},
            fit_at="2024-05-01T00:00:00Z",
        )
        is None
    )


@pytest.mark.parametrize("event", ["2021-12-31", "2022-06-01", "2024-04-30"])
def test_known_entity_break_blocks_training_target(facts, event):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    assert (
        investment_target(
            facts,
            issuer="0000000123",
            ends=ends,
            known_year_ends=set(ends),
            known_continuity_breaks={event: "2024-04-30T12:00:00Z"},
            fit_at="2024-05-01T12:00:00Z",
        )
        is None
    )


def test_future_or_older_entity_break_does_not_block_target(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    assert investment_target(
        facts,
        issuer="0000000123",
        ends=ends,
        known_year_ends=set(ends),
        known_continuity_breaks={
            "2021-12-30": "2021-12-31T12:00:00Z",
            "2024-05-02": "2024-04-30T12:00:00Z",
        },
        fit_at="2024-05-01T12:00:00Z",
    ) == pytest.approx(-0.1)


def test_invalid_entity_break_set_is_rejected(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    with pytest.raises(ValueError, match="continuity"):
        investment_target(
            facts,
            issuer="0000000123",
            ends=ends,
            known_year_ends=set(ends),
            known_continuity_breaks={"20220430": "2024-04-30T12:00:00Z"},
            fit_at="2024-05-01T12:00:00Z",
        )


def test_entity_break_set_must_be_supplied(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    with pytest.raises(TypeError, match="known_continuity_breaks"):
        investment_target(
            facts,
            issuer="0000000123",
            ends=ends,
            known_year_ends=set(ends),
            fit_at="2024-05-01T12:00:00Z",
        )


def test_entity_break_only_blocks_after_its_verified_availability(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    kwargs = {
        "issuer": "0000000123",
        "ends": ends,
        "known_year_ends": set(ends),
        "known_continuity_breaks": {"2022-06-01": "2024-06-01T12:00:00Z"},
    }
    assert investment_target(facts, fit_at="2024-05-01T12:00:00Z", **kwargs) == pytest.approx(-0.1)
    assert investment_target(facts, fit_at="2024-06-01T12:00:00Z", **kwargs) is None


def test_entity_break_rejects_unzoned_availability(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    with pytest.raises(ValueError, match="continuity"):
        investment_target(
            facts,
            issuer="0000000123",
            ends=ends,
            known_year_ends=set(ends),
            known_continuity_breaks={"2022-06-01": "2024-06-01"},
            fit_at="2024-05-01T12:00:00Z",
        )


@pytest.mark.parametrize(
    "ends",
    [
        ("2021-12-31", "2023-12-31", "2022-12-31"),
        ("2020-12-31", "2022-12-31", "2023-12-31"),
        ("2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"),
    ],
)
def test_target_rejects_wrong_or_nonconsecutive_periods(facts, ends):
    with pytest.raises(ValueError):
        target(facts, issuer="0000000123", ends=ends, fit_at="2024-05-01T00:00:00Z")


def test_target_rejects_bad_identity_and_unzoned_time(facts):
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    with pytest.raises(ValueError):
        target(facts, issuer="123", ends=ends, fit_at="2024-05-01T00:00:00Z")
    with pytest.raises(ValueError):
        target(facts, issuer="0000000123", ends=ends, fit_at="2024-05-01")


def test_target_rejects_mixed_concept_and_impossible_publication_time(facts):
    kwargs = {
        "issuer": "0000000123",
        "ends": ("2021-12-31", "2022-12-31", "2023-12-31"),
        "fit_at": "2024-05-01T00:00:00Z",
    }
    mixed = facts.copy()
    mixed.loc[mixed.accession == "a", "fact_key"] = '["us-gaap","Liabilities","USD","2021-12-31"]'
    with pytest.raises(ValueError):
        target(mixed, **kwargs)
    impossible = facts.copy()
    impossible.loc[impossible.accession == "a", "available_at"] = "2021-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        target(impossible, **kwargs)


@pytest.mark.parametrize("problem", ["key", "timing", "ambiguous"])
def test_future_invalid_asset_versions_fail_before_cutoff(facts, problem):
    future = facts.copy()
    mask = future.accession == "d"
    if problem == "key":
        future.loc[mask, "fact_key"] = "invalid json"
    elif problem == "timing":
        future.loc[mask, "fact_key"] = '["us-gaap", "Assets", "USD", "2026-12-31"]'
    else:
        future.loc[mask, "fact_key"] = '["us-gaap", "Assets", "USD", "2023-12-31"]'
    with pytest.raises(ValueError, match="assets|Ambiguous"):
        target(
            future,
            issuer="0000000123",
            ends=("2021-12-31", "2022-12-31", "2023-12-31"),
            fit_at="2024-05-01T00:00:00Z",
        )
    with pytest.raises(ValueError, match="assets|Ambiguous"):
        target(
            future,
            issuer="0000000123",
            ends=("2021-12-31", "2022-12-31", "2023-12-31"),
            fit_at="2024-02-02T00:00:00Z",
        )


def test_target_accepts_52_and_53_week_year_ends(facts):
    import json

    changed = facts.copy()
    replacements = {
        "2021-12-31": "2021-09-25",
        "2022-12-31": "2022-09-24",
        "2023-12-31": "2023-09-30",
    }
    changed["fact_key"] = changed.fact_key.map(
        lambda raw: json.dumps([*json.loads(raw)[:-1], replacements[json.loads(raw)[-1]]])
    )
    assert target(
        changed,
        issuer="0000000123",
        ends=tuple(replacements.values()),
        fit_at="2024-05-01T00:00:00Z",
    ) == pytest.approx(-0.1)
