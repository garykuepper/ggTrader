"""Normalize SEC company-facts instant values with externally verified availability."""

import json
import math
import re
from datetime import date
from numbers import Real

import pandas as pd
from filing_vintages import _utc, available_facts


def instant_facts(
    document: dict, concept: str, unit: str, availability: dict[str, object], *, expected_cik: int
) -> pd.DataFrame:
    """Normalize instant facts; filing-date bounds are conservative, not proof of availability."""
    if not isinstance(document, dict):
        raise ValueError("Invalid SEC document")
    cik = document.get("cik")
    if isinstance(cik, str) and re.fullmatch(r"[0-9]{10}", cik):
        cik = int(cik)
    if (
        type(cik) is not int
        or not 0 < cik < 10**10
        or type(expected_cik) is not int
        or cik != expected_cik
    ):
        raise ValueError("Invalid or mismatched issuer CIK")
    try:
        observations = document["facts"]["us-gaap"][concept]["units"][unit]
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing concept or unit") from exc
    if not isinstance(observations, list) or not observations:
        raise ValueError("Missing concept observations")
    rows = []
    seen = set()
    for item in observations:
        try:
            accession, end, value = item["accn"], item["end"], item["val"]
            if "start" in item:
                raise ValueError("Expected instant fact")
            if not isinstance(accession, str) or not accession or accession != accession.strip():
                raise ValueError("Invalid accession")
            if date.fromisoformat(end).isoformat() != end:
                raise ValueError("Invalid period end")
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Invalid fact value")
            if accession not in availability:
                raise ValueError("Missing independent availability")
            timestamp = _utc(availability[accession])
            local_day = timestamp.tz_convert("America/New_York").date()
            if local_day < date.fromisoformat(end):
                raise ValueError("Availability precedes period end")
            if "filed" in item:
                filed = item["filed"]
                if date.fromisoformat(filed).isoformat() != filed:
                    raise ValueError("Invalid filing date")
                if local_day < date.fromisoformat(filed):
                    raise ValueError("Availability precedes filing date")
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Invalid instant fact") from exc
        if (accession, end) in seen:
            raise ValueError(f"Ambiguous {concept} fact for {cik}, {accession}, {end}")
        seen.add((accession, end))
        key = json.dumps(["us-gaap", concept, unit, end], separators=(",", ":"))
        rows.append((f"{cik:010d}", key, accession, timestamp, value))
    return pd.DataFrame(
        rows, columns=["issuer_id", "fact_key", "accession", "available_at", "value"]
    )


def latest_annual_instant(
    facts: pd.DataFrame, decision_at: object, fiscal_year_ends: set[tuple[str, str]]
) -> pd.DataFrame:
    """Select one eligible annual instant or flow per issuer for the prototype.

    Fiscal year ends must be independently verified issuer/date pairs; no quarterly
    fact is treated as annual merely because it exists. Availability times must be
    independently verified. A fiscal period becomes eligible four calendar months
    after its end, on the New York calendar date; this is not an execution-time rule.
    No earlier year substitutes for a missing latest eligible year. Input facts
    must all represent the same concept, unit and fact type. This exact-date rule
    can differ from a rule counting whole calendar months.
    """
    if not isinstance(fiscal_year_ends, set):
        raise ValueError("Invalid fiscal-year ends")
    for item in fiscal_year_ends:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("Invalid fiscal-year end")
        issuer, end = item
        try:
            if (
                not isinstance(issuer, str)
                or not re.fullmatch(r"[0-9]{10}", issuer)
                or date.fromisoformat(end).isoformat() != end
            ):
                raise ValueError("Invalid fiscal-year end")
        except TypeError as exc:
            raise ValueError("Invalid fiscal-year end") from exc
    selected = available_facts(facts, decision_at).copy()
    decision_day = _utc(decision_at).tz_convert("America/New_York").date()
    periods = {}
    kinds = set()
    for key in facts["fact_key"]:
        try:
            decoded = json.loads(key)
            if not isinstance(decoded, list) or decoded[0] != "us-gaap":
                raise ValueError("Invalid fact key")
            if len(decoded) == 4:
                namespace, concept, unit, end = decoded
                kind = "instant"
            elif len(decoded) == 6 and decoded[3] == "duration":
                namespace, concept, unit, _, start, end = decoded
                if not isinstance(start, str) or date.fromisoformat(start).isoformat() != start:
                    raise ValueError("Invalid fact key")
                if not 350 <= (date.fromisoformat(end) - date.fromisoformat(start)).days + 1 <= 380:
                    raise ValueError("Invalid annual flow duration")
                kind = "duration"
            else:
                raise ValueError("Invalid fact key")
            if not all(isinstance(v, str) and v for v in (concept, unit, end)):
                raise ValueError("Invalid fact key")
            if date.fromisoformat(end).isoformat() != end:
                raise ValueError("Invalid fact key")
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Invalid fact key or annual flow duration") from exc
        kinds.add((namespace, concept, unit, kind))
        periods[key] = end
    if len(kinds) > 1:
        raise ValueError("Mixed concepts or units")
    identities = facts[["issuer_id", "fact_key"]].copy()
    identities["period_end"] = identities["fact_key"].map(periods)
    if identities.groupby(["issuer_id", "period_end"])["fact_key"].nunique().gt(1).any():
        raise ValueError("Ambiguous annual period")
    selected["period_end"] = [periods[key] for key in selected["fact_key"]]
    latest = {}
    for issuer, end in fiscal_year_ends:
        if (pd.Timestamp(end) + pd.DateOffset(months=4)).date() <= decision_day:
            latest[issuer] = max(latest.get(issuer, end), end)
    eligible = [
        latest.get(issuer) == end
        for issuer, end in zip(selected["issuer_id"], selected["period_end"], strict=True)
    ]
    return (
        selected.loc[eligible]
        .sort_values("period_end", kind="stable")
        .drop_duplicates("issuer_id", keep="last")
    )
