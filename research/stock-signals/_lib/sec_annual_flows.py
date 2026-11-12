"""Normalize explicitly identified annual, quarterly and nine-month SEC flows."""

import json
from datetime import date

import pandas as pd
from sec_facts import instant_facts


def annual_flows(
    document: dict,
    concept: str,
    unit: str,
    availability: dict[str, object],
    annual_periods: set[tuple[str, str]],
    *,
    expected_cik: int,
) -> pd.DataFrame:
    """Select verified full fiscal periods and normalize their filing vintages.

    The caller must independently establish issuer, fiscal periods and public
    availability. Periods must be 350–380 days inclusive; transition years are
    deliberately excluded. Filing types are not restricted, and exact duplicate
    records are rejected. A four-month eligibility rule must still be applied
    separately at the decision date.
    """
    return _flows(
        document, concept, unit, availability, annual_periods, expected_cik, 350, 380, "annual"
    )


def quarterly_flows(
    document: dict,
    concept: str,
    unit: str,
    availability: dict[str, object],
    quarterly_periods: set[tuple[str, str]],
    *,
    expected_cik: int,
) -> pd.DataFrame:
    """Normalize explicitly verified quarter periods; no announcement timing is inferred.

    Periods must be 70–110 days inclusive, excluding some retail 16/17-week quarters.
    This does not derive a missing fourth quarter from annual and year-to-date
    values or choose a quarter for a decision.
    """
    return _flows(
        document, concept, unit, availability, quarterly_periods, expected_cik, 70, 110, "quarter"
    )


def nine_month_flows(
    document: dict,
    concept: str,
    unit: str,
    availability: dict[str, object],
    periods: set[tuple[str, str]],
    *,
    expected_cik: int,
) -> pd.DataFrame:
    """Normalize verified nine-month fiscal periods for fourth-quarter derivation.

    The caller verifies fiscal-year start and publication times. The 240–300-day
    bound is a safety check, not proof that a period represents three quarters.
    """
    return _flows(
        document, concept, unit, availability, periods, expected_cik, 240, 300, "nine-month"
    )


def _flows(
    document: dict,
    concept: str,
    unit: str,
    availability: dict[str, object],
    periods: set[tuple[str, str]],
    expected_cik: int,
    min_days: int,
    max_days: int,
    label: str,
) -> pd.DataFrame:
    if not isinstance(periods, set) or not periods:
        raise ValueError(f"{label} periods must be explicitly supplied")
    ends = set()
    starts = {}
    for period in periods:
        if not isinstance(period, tuple) or len(period) != 2:
            raise ValueError(f"Invalid {label} period")
        start, end = period
        try:
            a, b = date.fromisoformat(start), date.fromisoformat(end)
            if (
                a.isoformat() != start
                or b.isoformat() != end
                or not min_days <= (b - a).days + 1 <= max_days
            ):
                raise ValueError(f"Invalid {label} period")
        except TypeError as exc:
            raise ValueError(f"Invalid {label} period") from exc
        if end in ends:
            raise ValueError(f"Multiple {label} starts for period end")
        ends.add(end)
        starts[end] = start
    try:
        entries = document["facts"]["us-gaap"][concept]["units"][unit]
        cik = document["cik"]
    except (TypeError, KeyError) as exc:
        raise ValueError("Invalid SEC document or concept") from exc
    if not isinstance(entries, list):
        raise ValueError("Invalid SEC observations")
    matched = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Invalid SEC observation")
        start, end = item.get("start"), item.get("end")
        if start is not None and not isinstance(start, str):
            raise ValueError("Invalid SEC observation period")
        if end is not None and not isinstance(end, str):
            raise ValueError("Invalid SEC observation period")
        if end in ends and start is not None and start != starts[end]:
            try:
                begin, finish = date.fromisoformat(start), date.fromisoformat(end)
                if begin.isoformat() != start:
                    raise ValueError("Invalid SEC observation period")
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid SEC observation period") from exc
            if min_days <= (finish - begin).days + 1 <= max_days:
                raise ValueError(f"Conflicting {label} period start")
        if (start, end) in periods:
            matched.append({key: value for key, value in item.items() if key != "start"})
    filtered = {"cik": cik, "facts": {"us-gaap": {concept: {"units": {unit: matched}}}}}
    result = instant_facts(filtered, concept, unit, availability, expected_cik=expected_cik)

    def flow_key(key: str) -> str:
        namespace, name, currency, end = json.loads(key)
        return json.dumps(
            [namespace, name, currency, "duration", starts[end], end], separators=(",", ":")
        )

    result["fact_key"] = result["fact_key"].map(flow_key)
    return result
