"""Select previously normalized filing facts by documented availability time."""

import math
from numbers import Real

import pandas as pd


def _utc(value: object) -> pd.Timestamp:
    """Require an explicit timezone so a date is never mistaken for a publication time."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid availability timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Availability timestamp must include a timezone")
    return stamp.tz_convert("UTC")


def available_facts(facts: pd.DataFrame, decision_at: object) -> pd.DataFrame:
    """Select the latest available version of each issuer and caller-defined fact key.

    ``available_at`` must represent a defensible public-availability time, not merely
    an acceptance time. The caller must distinguish periods, units and contexts in
    ``fact_key`` and verify issuer identity and source provenance separately.
    Keys match exactly; the latest available version wins, including at the cutoff.
    Facts carry forward indefinitely; callers must enforce latency and freshness.
    """
    return available_versions(facts, decision_at).drop_duplicates(
        ["issuer_id", "fact_key"], keep="last"
    )


def available_versions(facts: pd.DataFrame, decision_at: object) -> pd.DataFrame:
    """Return all validated versions available by the cutoff, in time order.

    The caller must verify availability, identity, keys, freshness and comparability.
    Keeping two periods from one filing does not establish consistent definitions.
    Versions of one fact are not independent observations. All input rows are
    validated, including future rows. Equal timestamps retain input order.
    """
    required = ["issuer_id", "fact_key", "accession", "available_at", "value"]
    if facts.columns.has_duplicates or not set(required).issubset(facts.columns):
        raise ValueError("Missing or duplicate filing columns")
    cutoff = _utc(decision_at)
    data = facts.loc[:, required].copy()
    for column in ("issuer_id", "fact_key", "accession"):
        if not data[column].map(lambda v: isinstance(v, str) and bool(v) and v == v.strip()).all():
            raise ValueError(f"Invalid {column}")
    try:
        valid = all(
            isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            for value in data["value"]
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid fact value") from exc
    if not valid:
        raise ValueError("Invalid fact value")
    data["available_at"] = pd.to_datetime(data["available_at"].map(_utc), utc=True)
    keys = ["issuer_id", "fact_key"]
    if (
        data.duplicated(keys + ["accession"]).any()
        or data.duplicated(keys + ["available_at"]).any()
        or data.groupby(["issuer_id", "accession"])["available_at"].nunique().gt(1).any()
    ):
        raise ValueError("Ambiguous filing versions")
    data = data.loc[data["available_at"] <= cutoff]
    return data.sort_values("available_at", kind="stable")
