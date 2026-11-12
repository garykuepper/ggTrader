"""Select point-in-time annual assets for an experimental training target."""

import json
import re
from datetime import date
from numbers import Integral

import pandas as pd
from expected_growth_features import investment_change
from filing_vintages import _utc, available_facts


def filed_fiscal_year_ends(filings: pd.DataFrame, *, issuer: str, fit_at: object) -> frozenset[str]:
    """Select period ends from annual reports available by the cutoff, inclusive.

    The caller must verify the issuer, annual-report forms, period identities,
    availability timestamps, source authenticity and completeness. A filing's
    acceptance time alone does not establish public availability. Even a complete
    set of reports omits a known fiscal year when its report is late or missing.
    The immutable result cannot be passed directly as the training guard's complete
    calendar; assemble that separately with independently verified year-ends known
    at the cutoff.
    Future records are validated but never selected. Only 10-K, 10-K/A, 10-KT and
    10-KT/A are accepted; other filing types need separate treatment. Transition
    reports are kept so the training guard can reject a fiscal-calendar change.
    """
    if not isinstance(issuer, str) or not re.fullmatch(r"[0-9]{10}", issuer) or int(issuer) == 0:
        raise ValueError("Invalid issuer")
    columns = ["issuer_id", "period_end", "accession", "form", "available_at"]
    if (
        not isinstance(filings, pd.DataFrame)
        or filings.columns.has_duplicates
        or not set(columns).issubset(filings.columns)
        or filings.duplicated(["issuer_id", "accession"]).any()
    ):
        raise ValueError("Invalid annual filings")
    cutoff = _utc(fit_at)
    selected: set[str] = set()
    for row in filings.loc[:, columns].itertuples(index=False):
        try:
            if (
                not isinstance(row.issuer_id, str)
                or not re.fullmatch(r"[0-9]{10}", row.issuer_id)
                or int(row.issuer_id) == 0
                or not isinstance(row.accession, str)
                or not row.accession
                or row.accession != row.accession.strip()
                or row.form not in ("10-K", "10-K/A", "10-KT", "10-KT/A")
            ):
                raise ValueError("Invalid annual filing")
            end = date.fromisoformat(row.period_end)
            available = _utc(row.available_at)
            if (
                end.isoformat() != row.period_end
                or available.tz_convert("America/New_York").date() < end
            ):
                raise ValueError("Invalid annual filing")
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid annual filing") from exc
        if row.issuer_id == issuer and available <= cutoff:
            selected.add(row.period_end)
    return frozenset(selected)


def investment_target(
    facts: pd.DataFrame,
    *,
    issuer: str,
    ends: tuple[str, str, str],
    known_year_ends: set[str],
    known_continuity_breaks: dict[str, object],
    fit_at: object,
) -> float | None:
    """Return the target while retaining compatibility with existing callers."""
    result = investment_target_evidence(
        facts,
        issuer=issuer,
        ends=ends,
        known_year_ends=known_year_ends,
        known_continuity_breaks=known_continuity_breaks,
        fit_at=fit_at,
    )
    return None if result is None else result["target"]


def investment_target_evidence(
    facts: pd.DataFrame,
    *,
    issuer: str,
    ends: tuple[str, str, str],
    known_year_ends: set[str],
    known_continuity_breaks: dict[str, object],
    fit_at: object,
) -> dict | None:
    """Calculate a target only when all three explicitly named annual facts exist.

    Availability times and the complete fiscal-year-end set for this issuer must be
    independently verified as of fit_at, including year-ends with late or missing
    filings; filed_fiscal_year_ends alone is not sufficient. The caller must also
    supply independently identified continuity breaks as effective-date to
    verified-availability-time mappings; an empty mapping is not evidence that none
    exist. Available breaks
    from the first period end through the fit date block the target, including
    those after the last period end. Availability at the cutoff is included.
    This checks spacing, the cutoff and the
    prototype's exact four-calendar-month rule
    on the New York date, not accounting comparability, issuer mapping,
    winsorization or predictor lag. Month-based conventions can differ.
    The issuer's input must contain only Assets facts. All its keys and times,
    including future rows, are validated. Latest available revisions are used and
    missing years return None. The result records each selected filing and its
    availability time, supplied year ends through the cutoff date and breaks
    available at the cutoff;
    these records do not verify completeness or accounting comparability.
    Integer values remain exact; other real values are recorded as Python floats.
    """
    if not isinstance(issuer, str) or not re.fullmatch(r"[0-9]{10}", issuer) or int(issuer) == 0:
        raise ValueError("Invalid issuer")
    if not isinstance(ends, tuple) or len(ends) != 3:
        raise ValueError("Invalid fiscal year ends")
    try:
        dates = [date.fromisoformat(value) for value in ends]
        if any(value.isoformat() != raw for value, raw in zip(dates, ends, strict=True)) or not all(
            350 <= (later - earlier).days <= 380
            for earlier, later in zip(dates, dates[1:], strict=False)
        ):
            raise ValueError("Invalid fiscal year ends")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid fiscal year ends") from exc
    if not isinstance(known_year_ends, set) or not set(ends).issubset(known_year_ends):
        raise ValueError("Invalid known fiscal year ends")
    try:
        known_dates = []
        for raw in known_year_ends:
            parsed = date.fromisoformat(raw)
            if parsed.isoformat() != raw:
                raise ValueError("Invalid known fiscal year ends")
            known_dates.append(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid known fiscal year ends") from exc
    if not isinstance(known_continuity_breaks, dict):
        raise ValueError("Invalid continuity breaks")
    try:
        breaks = []
        for raw, available_at in known_continuity_breaks.items():
            day = date.fromisoformat(raw)
            if day.isoformat() != raw:
                raise ValueError("Invalid continuity breaks")
            breaks.append((day, _utc(available_at)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid continuity breaks") from exc
    if not isinstance(facts, pd.DataFrame):
        raise ValueError("Invalid filing facts")
    selected = available_facts(facts, fit_at)
    period_keys = {}
    for row in facts.loc[facts.issuer_id == issuer].itertuples():
        try:
            key = json.loads(row.fact_key)
            if (
                not isinstance(key, list)
                or len(key) != 4
                or key[:3] != ["us-gaap", "Assets", "USD"]
            ):
                raise ValueError("Invalid assets key")
            end = date.fromisoformat(key[3])
            if (
                end.isoformat() != key[3]
                or _utc(row.available_at).tz_convert("America/New_York").date() < end
            ):
                raise ValueError("Invalid assets timing")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("Invalid assets key or timing") from exc
        if key[3] in period_keys and period_keys[key[3]] != row.fact_key:
            raise ValueError("Ambiguous assets period")
        period_keys[key[3]] = row.fact_key
    versions = {
        json.loads(row.fact_key)[3]: row
        for row in selected.loc[selected.issuer_id == issuer].itertuples()
    }
    fit_time = _utc(fit_at)
    fit_day = fit_time.tz_convert("America/New_York").date()
    eligible = [
        day
        for day in known_dates
        if (pd.Timestamp(day) + pd.DateOffset(months=4)).date() <= fit_day
    ]
    if (
        not eligible
        or max(eligible) != dates[-1]
        or any(dates[0] < day < dates[-1] and day not in dates for day in known_dates)
        or any(dates[0] <= day <= fit_day and available <= fit_time for day, available in breaks)
    ):
        return None
    if any(end not in versions for end in ends):
        return None
    rows = [versions[end] for end in ends]
    return {
        "issuer": issuer,
        "fit_at": fit_time.isoformat(),
        "known_year_ends": sorted(day.isoformat() for day in known_dates if day <= fit_day),
        "known_continuity_breaks": {
            day.isoformat(): available.isoformat()
            for day, available in sorted(breaks)
            if available <= fit_time
        },
        "target": investment_change(
            prior_assets=rows[0].value, assets=rows[1].value, next_assets=rows[2].value
        ),
        "facts": [
            {
                "period_end": end,
                "accession": row.accession,
                "available_at": row.available_at.isoformat(),
                "value": int(row.value) if isinstance(row.value, Integral) else float(row.value),
            }
            for end, row in zip(ends, rows, strict=True)
        ],
    }
