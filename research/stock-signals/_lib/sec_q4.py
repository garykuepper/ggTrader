"""Derive a fourth-quarter flow from matched, available SEC facts."""

import json
import math
import re
from datetime import date, timedelta
from numbers import Integral, Real

from filing_vintages import _utc


def derive_q4(annual: dict, nine_month: dict, decision_at: object) -> dict:
    """Subtract a verified nine-month flow from the matching annual flow.

    Inputs must be the caller's point-in-time versions of the same issuer,
    concept, unit and fiscal year. Times must be independently verified public
    availability, not inferred from filing or acceptance dates. USD amounts
    only; callers must verify the two filing versions are accounting-consistent.
    This does not establish accounting-tag equivalence. Quarters over 110 days
    are rejected. Returned source details preserve the supplied versions, but do
    not prove their authenticity or timing.
    """
    cutoff = _utc(decision_at)
    parsed = []
    for row in (annual, nine_month):
        try:
            issuer, accession = row["issuer_id"], row["accession"]
            key = json.loads(row["fact_key"])
            value, available = row["value"], _utc(row["available_at"])
            if (
                not isinstance(issuer, str)
                or not re.fullmatch(r"[0-9]{10}", issuer)
                or int(issuer) == 0
            ):
                raise ValueError("Invalid issuer")
            if not isinstance(accession, str) or not accession or accession != accession.strip():
                raise ValueError("Invalid accession")
            if (
                not isinstance(key, list)
                or len(key) != 6
                or key[0] != "us-gaap"
                or key[3] != "duration"
            ):
                raise ValueError("Invalid flow key")
            if not all(isinstance(s, str) and s for s in (key[1], key[2], key[4], key[5])):
                raise ValueError("Invalid flow key")
            if key[2] != "USD":
                raise ValueError("Only USD flow amounts are supported")
            start, end = date.fromisoformat(key[4]), date.fromisoformat(key[5])
            if start.isoformat() != key[4] or end.isoformat() != key[5] or start > end:
                raise ValueError("Invalid period")
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Invalid flow value")
            if available.tz_convert("America/New_York").date() < end:
                raise ValueError("Availability precedes period end")
            if available > cutoff:
                raise ValueError("Fact not yet available")
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Invalid flow input") from exc
        parsed.append((issuer, accession, key, value, available, start, end))
    a, b = parsed
    if a[0] != b[0] or a[2][:3] != b[2][:3] or a[5] != b[5]:
        raise ValueError("Flow inputs do not match")
    annual_days, nine_days, quarter_days = (
        (a[6] - a[5]).days + 1,
        (b[6] - b[5]).days + 1,
        (a[6] - b[6]).days,
    )
    if not (350 <= annual_days <= 380 and 240 <= nine_days <= 300 and 70 <= quarter_days <= 110):
        raise ValueError("Invalid fiscal periods")
    value = (int(a[3]) if isinstance(a[3], Integral) else a[3]) - (
        int(b[3]) if isinstance(b[3], Integral) else b[3]
    )
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError("Invalid derived value") from exc
    if not finite:
        raise ValueError("Invalid derived value")
    return {
        "issuer_id": a[0],
        "concept": a[2][1],
        "unit": a[2][2],
        "period_start": (b[6] + timedelta(days=1)).isoformat(),
        "period_end": a[6].isoformat(),
        "value": value,
        "available_at": max(a[4], b[4]),
        "source_accessions": (a[1], b[1]),
        "sources": [
            {
                "accession": row[1],
                "period_start": row[5].isoformat(),
                "period_end": row[6].isoformat(),
                "value": int(row[3]) if isinstance(row[3], Integral) else float(row[3]),
                "available_at": row[4],
            }
            for row in (a, b)
        ],
    }
