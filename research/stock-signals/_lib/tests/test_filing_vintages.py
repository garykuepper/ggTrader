"""Check point-in-time selection on invented company filings."""

import pandas as pd
import pytest
from filing_vintages import available_facts, available_versions

pytestmark = pytest.mark.study


def _facts():
    return pd.DataFrame(
        [
            ("100", "assets:USD:2025-12-31", "a1", "2026-02-01T12:00:00Z", 100.0),
            ("100", "assets:USD:2025-12-31", "a2", "2026-03-01T12:00:00Z", 110.0),
            ("200", "assets:USD:2025-12-31", "b1", "2026-02-01T12:00:00Z", 0.0),
            ("100", "assets:USD:2024-12-31", "a3", "2026-02-01T12:00:00Z", 90.0),
        ],
        columns=["issuer_id", "fact_key", "accession", "available_at", "value"],
    )


def test_filing_revision_is_available_only_at_its_timestamp():
    facts = _facts()
    early = available_facts(facts, "2026-02-20T00:00:00Z")
    assert set(early.accession) == {"a1", "a3", "b1"}
    assert early.loc[early.accession == "b1", "value"].iloc[0] == 0.0
    later = available_facts(facts, "2026-03-01T12:00:00Z")
    assert set(later.accession) == {"a2", "a3", "b1"}
    assert set(available_facts(facts, "2026-01-01T00:00:00Z").accession) == set()


def test_general_dynamics_reported_asset_revision_respects_supplied_dates():
    """Select independently transcribed values without claiming dissemination times."""
    facts = pd.DataFrame(
        [
            (
                "0000040533",
                "assets:USD:2019-12-31",
                "0000040533-20-000015",
                "2020-02-11T21:00:00Z",
                48_841_000_000,
            ),
            (
                "0000040533",
                "assets:USD:2019-12-31",
                "0000040533-21-000010",
                "2021-02-10T21:00:00Z",
                49_349_000_000,
            ),
        ],
        columns=["issuer_id", "fact_key", "accession", "available_at", "value"],
    )
    before = available_facts(facts, "2021-02-08T21:00:00Z")
    after = available_facts(facts, "2021-02-11T21:00:00Z")
    assert before.iloc[0]["value"] == 48_841_000_000
    assert after.iloc[0]["value"] == 49_349_000_000


def test_available_versions_preserves_older_filings_and_excludes_future():
    facts = _facts()
    original = facts.copy(deep=True)
    before = available_versions(facts, "2026-02-20T00:00:00Z")
    assert set(before.accession) == {"a1", "a3", "b1"}
    versions = available_versions(facts, "2026-03-01T12:00:00Z")
    assert set(versions.accession) == {"a1", "a2", "a3", "b1"}
    assert set(versions.loc[versions.issuer_id == "100", "accession"]) == {"a1", "a2", "a3"}
    assert list(versions.columns) == ["issuer_id", "fact_key", "accession", "available_at", "value"]
    assert versions.available_at.is_monotonic_increasing
    assert versions.loc[versions.accession.isin(["a1", "a2"]), "value"].tolist() == [100.0, 110.0]
    latest = versions.drop_duplicates(["issuer_id", "fact_key"], keep="last")
    pd.testing.assert_frame_equal(latest, available_facts(facts, "2026-03-01T12:00:00Z"))
    assert available_versions(facts, "2026-01-01T00:00:00Z").empty
    pd.testing.assert_frame_equal(facts, original)


def test_available_versions_rejects_ambiguous_future_data():
    facts = pd.concat([_facts(), _facts().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        available_versions(facts, "2026-02-20T00:00:00Z")


@pytest.mark.parametrize(
    "problem", ["naive_time", "missing_time", "duplicate", "tie", "missing_key", "nan"]
)
def test_rejects_ambiguous_or_unavailable_input(problem):
    facts = _facts()
    if problem == "naive_time":
        facts.loc[0, "available_at"] = "2026-02-01 12:00:00"
    elif problem == "missing_time":
        facts.loc[0, "available_at"] = None
    elif problem == "duplicate":
        facts = pd.concat([facts, facts.iloc[[0]]], ignore_index=True)
    elif problem == "tie":
        facts.loc[1, "available_at"] = facts.loc[0, "available_at"]
    elif problem == "missing_key":
        facts.loc[0, "fact_key"] = ""
    else:
        facts.loc[0, "value"] = float("nan")
    with pytest.raises(ValueError):
        available_facts(facts, "2026-04-01T00:00:00Z")


def test_rejects_naive_decision_time():
    with pytest.raises(ValueError, match="timezone"):
        available_facts(_facts(), "2026-02-20")


def test_timezone_offsets_and_input_order_do_not_change_the_selection():
    facts = _facts()
    expected = available_facts(facts, "2026-02-01T07:00:00-05:00")
    shuffled = available_facts(facts.iloc[::-1], "2026-02-01T12:00:00Z")
    assert set(expected.accession) == set(shuffled.accession) == {"a1", "a3", "b1"}
    assert set(available_facts(facts, "2026-02-01T11:59:59Z").accession) == set()


def test_rejects_context_collision_within_one_filing():
    facts = _facts()
    collision = facts.iloc[[0]].copy()
    collision["value"] = 999.0
    with pytest.raises(ValueError, match="Ambiguous"):
        available_facts(pd.concat([facts, collision]), "2026-04-01T00:00:00Z")


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_rejects_missing_nullable_value(dtype):
    facts = _facts()
    facts["value"] = pd.array([100, None, 0, 90], dtype=dtype)
    with pytest.raises(ValueError, match="Invalid fact value"):
        available_facts(facts, "2026-04-01T00:00:00Z")


@pytest.mark.parametrize("value", [True, "2e3", 1 + 0j, float("inf")])
def test_rejects_invalid_fact_values(value):
    facts = _facts()
    facts["value"] = facts["value"].astype(object)
    facts.at[1, "value"] = value
    with pytest.raises(ValueError, match="Invalid fact value"):
        available_facts(facts, "2026-04-01T00:00:00Z")


def test_rejects_different_times_for_same_filing():
    facts = _facts()
    facts.loc[3, "accession"] = "a1"
    facts.loc[3, "available_at"] = "2026-05-01T12:00:00Z"
    with pytest.raises(ValueError, match="Ambiguous"):
        available_facts(facts, "2026-04-01T00:00:00Z")
