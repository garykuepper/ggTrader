"""Payout table: why each held dead stock stopped trading, from filings, with suggested losses."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

import pandas as pd
import sec_filings

EVIDENCE_CLASSES = (
    "bankruptcy_filed",
    "acquired_or_merged",
    "delisted_other",
    "no_8k_found",
    "no_filing_match",
    "name_unknown",
    "ambiguous_match",
)
# The classes that rest on 8-K items actually read from a filing near the exit.
DOCUMENTED_CLASSES = ("bankruptcy_filed", "acquired_or_merged", "delisted_other")

CLASS_NAME_UNKNOWN = "name_unknown"
CLASS_NO_FILING_MATCH = "no_filing_match"
CLASS_AMBIGUOUS = "ambiguous_match"

CONFIDENCE_CONFIRMED = "confirmed"
CONFIDENCE_NAME_ONLY = "name_only"
CONFIDENCE_NONE = ""
CONFIDENCES = (CONFIDENCE_CONFIRMED, CONFIDENCE_NAME_ONLY)
RULE_VERSION_NONE = ""
RULE_VERSIONS = sec_filings.RULE_VERSIONS
V11_LEAD_NOTE = (
    "v1.1-only matches are leads: the company name matched only after the extra cleaning "
    "rules and was not otherwise verified"
)

BASIS_FILING = "filing_evidence"
BASIS_LITERATURE = "literature_fill"
BASIS_NONE = "none"
# Plain reasons a class gets no number; none of them says what holders were or were not paid.
ACQUIRED_NOTE = (
    "filings report an acquisition or merger event; payment amount and form not verified"
)
NO_LOSS_DEFAULT = "no loss can be set from the filings"
NO_LOSS_REASONS = {
    "delisted_other": "filings report a delisting notice only; no loss can be set from them",
    "no_8k_found": "no acquisition, bankruptcy or delisting item found near the exit; no loss set",
    "no_filing_match": "no Form 25 notice matched the company name; no loss can be set",
    "name_unknown": "no company name in our data to match; no loss can be set",
    "ambiguous_match": "the name matched more than one company; no loss can be set",
}
BASES = (BASIS_FILING, BASIS_LITERATURE, BASIS_NONE)

# Events statuses that mean the filing list was read in full; any other value is a gap.
EVENTS_STATUS_COMPLETE = (
    sec_filings.EVENTS_STATUS_OK,
    sec_filings.EVENTS_STATUS_NO_OLDER_PAGES_NEEDED,
)
EVENTS_UNAVAILABLE_NOTE = (
    "A stock with one of these statuses has no evidence because its filing list could not be "
    "read in full; that does not mean no 8-K was filed."
)

CROSS_CONFLICT = "conflict"
CROSS_AGREES = "agrees"
CROSS_NOT_COMPARABLE = "not_comparable"
LABEL_CALM = "calm_like_deal"
LABEL_COLLAPSE = "collapse_like"

# CRSP exchange codes: 1 NYSE, 2 AMEX, 3 Nasdaq.
# Fills for the return a CRSP file leaves missing when a stock is dropped for bad performance.
# The papers sit in gg-library as ``ec9f9e4b98`` (Shumway 1997) and ``c5d15fa3c6`` (Shumway and
# Warther 1999); both are marked "unverified" there (status of the copy, not of the numbers).
LITERATURE_SOURCE_STATUS = "unverified library copy"
LITERATURE_MEANING = (
    "missing-delisting-return fill used in the literature, not a measurement of these stocks"
)
LITERATURE_FILLS: dict[tuple[int, ...], dict[str, Any]] = {
    (1, 2): {
        "loss": 0.30,
        "source_status": LITERATURE_SOURCE_STATUS,
        "meaning": LITERATURE_MEANING,
        "library_id": "ec9f9e4b98",
        "library_sha256": "ec9f9e4b9815b57b0c1900c2c908a0e466b0d0e03b5b0eb29fddb130d1b00895",
        "doi": "10.1111/j.1540-6261.1997.tb03818.x",
        "source": (
            "Shumway (1997), The Delisting Bias in CRSP Data, Journal of Finance 52(1), 327-340"
        ),
        "locators": [
            "p. 340 (library PDF page 15): omitted delisting returns average about -30 percent; "
            "researchers can test with -0.3 in place of the missing performance delisting return",
            "p. 329 (library PDF page 4), Table I: NYSE/AMEX performance delisting average "
            "-41.56 percent where a return exists (11.7 percent of cases)",
        ],
    },
    (3,): {
        "loss": 0.55,
        "source_status": LITERATURE_SOURCE_STATUS,
        "meaning": LITERATURE_MEANING,
        "library_id": "c5d15fa3c6",
        "library_sha256": "c5d15fa3c61b701786f80850b9a8dc825aba78b0ef9594c2ec4781a237358c55",
        "doi": "10.1111/0022-1082.00192",
        "source": (
            "Shumway and Warther (1999), The Delisting Bias in CRSP's Nasdaq Data and Its "
            "Implications for the Size Effect, Journal of Finance 54(6), 2361-2379"
        ),
        "locators": [
            "p. 2361 (library PDF page 2), abstract: a corrected return of -55 percent for "
            "missing performance-related delisting returns",
            "p. 2370 (library PDF page 11), equation 3: -55 percent after the wider bid-ask gap",
            "p. 2378 (library PDF page 19), conclusion: use -55 percent wherever CRSP shows a "
            "missing return for a Nasdaq performance delisting",
        ],
    },
}

TABLE_COLUMNS = (
    "permno",
    "symbol",
    "window",
    "universe",
    "exit_date",
    "last_close",
    "stock_name",
    "exchcd",
    "price_label",
    "evidence_class",
    "match_confidence",
    "match_rule_version",
    "form25_accession",
    "form25_provision",
    "form25_security",
    "event_accessions",
    "events_status",
    "suggested_loss",
    "basis",
    "loss_note",
    "cross_check",
)


def clean_exchange(exchcd: Any) -> int | None:
    """The exchange code as an int, or ``None`` when it is not a finite whole number.

    Text (``"3"``, ``"NYSE"``), ``None``, NaN, infinity, booleans and fractions are all unknown:
    a code is only trusted when it arrives as a real number, so a stray string cannot pick a fill.
    """
    if isinstance(exchcd, bool) or not isinstance(exchcd, numbers.Real):
        return None
    value = float(exchcd)
    if not math.isfinite(value) or value != int(value):
        return None
    return int(value)


def loss_decision(evidence_class: str, exchcd: Any) -> tuple[float | None, str, str]:
    """The suggested loss, its basis and a plain reason for the choice.

    Only two cases get a number: a stock bought out or merged (nothing lost on the filing's
    word) and a bankruptcy on an exchange the literature has a fill for. Everything else keeps
    no number, so a later rerun can try the whole range (1.0, 0.5, 0.0) for it.

    Args:
        evidence_class: one of ``EVIDENCE_CLASSES``.
        exchcd: the CRSP exchange code of the stock (``None`` or NaN when unknown).

    The basis says where the number came from (``filing_evidence``, ``literature_fill`` or
    ``none``); how sure the filing match is lives in ``match_confidence``, not here.

    Returns:
        ``(loss, basis, note)`` with ``loss`` ``None`` and basis ``"none"`` when there is no number.
    """
    code = clean_exchange(exchcd)
    loss: float | None
    if evidence_class == sec_filings.EVIDENCE_ACQUIRED:
        loss, basis, note = (
            0.0,
            BASIS_FILING,
            "filings report an acquisition or merger event; payment amount and form not verified",
        )
    elif evidence_class == sec_filings.EVIDENCE_BANKRUPTCY:
        fill = next((f for codes, f in LITERATURE_FILLS.items() if code in codes), None)
        if fill is None:
            return (
                None,
                BASIS_NONE,
                f"filings report a bankruptcy item but exchange code {code} has no literature fill",
            )
        loss, basis = fill["loss"], BASIS_LITERATURE
        note = (
            f"filings report a bankruptcy item; loss is a literature fill for exchange code "
            f"{code}, not a measurement of this stock"
        )
    else:
        return (
            None,
            BASIS_NONE,
            f"{evidence_class}: {NO_LOSS_REASONS.get(evidence_class, NO_LOSS_DEFAULT)}",
        )
    return loss, basis, note


def suggested_loss(evidence_class: str, exchcd: Any) -> tuple[float | None, str]:
    """The suggested loss and its basis (see ``loss_decision`` for the rules and the reason)."""
    loss, basis, _ = loss_decision(evidence_class, exchcd)
    return loss, basis


def _by_permno(mapping: Mapping[Any, Any] | None) -> dict[int, Any]:
    """Re-key a mapping by integer permno (the files carry them as text)."""
    return {int(k): v for k, v in (mapping or {}).items()}


def _is_named(row: Mapping[str, Any]) -> bool:
    name = row.get("stock_name")
    return row.get("name_source") != "none" and isinstance(name, str) and bool(name.strip())


def classify(row: Mapping[str, Any], match: Mapping[str, Any] | None, events: list[dict]) -> str:
    """The evidence class of one roster stock (see the mapping in the study brief)."""
    if not _is_named(row) or (match or {}).get("status") == CLASS_NAME_UNKNOWN:
        return CLASS_NAME_UNKNOWN
    if match is None:
        return CLASS_NO_FILING_MATCH  # named, but the Form 25 phase has no record of a match
    status = match.get("status")
    if status == sec_filings.STATUS_AMBIGUOUS:
        return CLASS_AMBIGUOUS
    if status == sec_filings.STATUS_NONE:
        return CLASS_NO_FILING_MATCH
    if status == sec_filings.STATUS_MATCHED and len(match.get("ciks") or []) == 1:
        return sec_filings.evidence_class(events)
    return CLASS_AMBIGUOUS  # a "match" without exactly one issuer is not usable


def match_confidence(match: Mapping[str, Any] | None, evidence_class: str) -> str:
    """``confirmed`` (v1 match with a readable common-stock notice), ``name_only`` or blank.

    A stock matched only under rule v1.1 is an identity lead and never ``confirmed``, even when
    its Form 25 notice was readable for that CIK (the notice check verifies the chosen CIK, not
    that it is the roster stock).
    """
    if match is None or evidence_class not in (*DOCUMENTED_CLASSES, sec_filings.EVIDENCE_NONE):
        return CONFIDENCE_NONE
    confirmation = match.get("confirmation")
    version = match.get("rule_version")
    if confirmation == "confirmed_common":
        if version == sec_filings.RULE_VERSION_V1:
            return CONFIDENCE_CONFIRMED
        # Rule v1.1 (or any match lacking an explicit v1 tag) stays a name-only lead.
        return CONFIDENCE_NAME_ONLY
    if confirmation == "unconfirmed_unreadable":
        return CONFIDENCE_NAME_ONLY
    return CONFIDENCE_NONE


def match_rule_version(match: Mapping[str, Any] | None, evidence_class: str) -> str:
    """``v1``, ``v1.1``, or blank when the stock has no usable Form 25 match."""
    if match is None or evidence_class not in (*DOCUMENTED_CLASSES, sec_filings.EVIDENCE_NONE):
        return RULE_VERSION_NONE
    version = match.get("rule_version")
    if isinstance(version, str) and version in RULE_VERSIONS:
        return version
    return RULE_VERSION_NONE


def cross_check(evidence_class: str, price_label: Any) -> str:
    """Compare the filing's story with the price path's label.

    A bankruptcy whose price stayed calm, or a buyout whose price collapsed, is a ``conflict``;
    the matching pairs ``agree``; everything else cannot be compared.
    """
    if evidence_class == sec_filings.EVIDENCE_BANKRUPTCY:
        if price_label == LABEL_CALM:
            return CROSS_CONFLICT
        return CROSS_AGREES if price_label == LABEL_COLLAPSE else CROSS_NOT_COMPARABLE
    if evidence_class == sec_filings.EVIDENCE_ACQUIRED:
        if price_label == LABEL_COLLAPSE:
            return CROSS_CONFLICT
        return CROSS_AGREES if price_label == LABEL_CALM else CROSS_NOT_COMPARABLE
    return CROSS_NOT_COMPARABLE


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def build_payout_table(
    roster: pd.DataFrame,
    matches: Mapping[Any, Mapping[str, Any]],
    notices: Mapping[Any, Mapping[str, Any]],
    events: Mapping[Any, list[dict]],
    events_status: Mapping[Any, str] | None = None,
) -> pd.DataFrame:
    """One row per roster row: why it stopped trading, on what evidence, and a suggested loss.

    Args:
        roster: the roster (one row per held dead stock).
        matches: per-stock Form 25 results from ``_form25_matches.json``, keyed by permno.
        notices: optional per-stock deciding notices, keyed by permno; when absent the notice
            inside the stock's match record is used.
        events: per-stock 8-K events around the exit, keyed by permno.
        events_status: optional per-stock status of the events download, keyed by permno.

    Returns:
        A frame with ``TABLE_COLUMNS``; no roster row is dropped.
    """
    match_by = _by_permno(matches)
    notice_by = _by_permno(notices)
    events_by = _by_permno(events)
    status_by = _by_permno(events_status)
    records: list[dict[str, Any]] = []
    for row in roster.to_dict("records"):
        permno = int(row["permno"])
        match = match_by.get(permno)
        stock_events = events_by.get(permno, [])
        evidence = classify(row, match, stock_events)
        confidence = match_confidence(match, evidence)
        loss, basis, note = loss_decision(evidence, row.get("exchcd"))
        notice = notice_by.get(permno) or (match or {}).get("notice") or {}
        details = notice.get("notice") or {}
        accession = notice.get("accession") or (
            ((match or {}).get("accessions") or [""])[0] if confidence else ""
        )
        has_events = evidence in (*DOCUMENTED_CLASSES, sec_filings.EVIDENCE_NONE)
        records.append(
            {
                "permno": permno,
                "symbol": row["symbol"],
                "window": row.get("window"),
                "universe": row.get("universe"),
                "exit_date": row.get("exit_date"),
                "last_close": row.get("last_close"),
                "stock_name": _text(row.get("stock_name")),
                "exchcd": clean_exchange(row.get("exchcd")),
                "price_label": row.get("price_label"),
                "evidence_class": evidence,
                "match_confidence": confidence,
                "match_rule_version": match_rule_version(match, evidence),
                "form25_accession": _text(accession),
                "form25_provision": _text(details.get("rule_provision")),
                "form25_security": _text(details.get("security_description")),
                "event_accessions": ";".join(e["accession"] for e in stock_events),
                "events_status": status_by.get(permno, "") if has_events else "",
                "suggested_loss": loss,
                "basis": basis,
                "loss_note": note,
                "cross_check": cross_check(evidence, row.get("price_label")),
            }
        )
    table = pd.DataFrame.from_records(records, columns=list(TABLE_COLUMNS))
    table["exchcd"] = table["exchcd"].astype("Int64")
    table["suggested_loss"] = table["suggested_loss"].astype(float)
    return table


def coverage_summary(
    table: pd.DataFrame, *, items_missing: int = 0, quarters_skipped: int = 0
) -> dict[str, Any]:
    """Counts of the table: per class, per confidence, how many have a loss, and the conflicts.

    Confirmed and name-only matches are always counted apart, and ``name_only`` is further split
    into v1.1-only identity leads (``name_only_v1_1``), v1 matches whose notice was unreadable
    (``name_only_v1_unreadable``) and the rest (``name_only_other``: a missing or unknown rule
    version); the three always add up to ``name_only``.

    Args:
        table: the output of ``build_payout_table``.
        items_missing: number of 8-K events whose item list was empty (from the events fetch).
        quarters_skipped: quarterly SEC indexes skipped by the Form 25 phase.
    """
    classes = table["evidence_class"]
    confidence = table["match_confidence"]
    rule_version = table["match_rule_version"].fillna("")
    has_loss = table["suggested_loss"].notna()
    by_class = {name: int((classes == name).sum()) for name in EVIDENCE_CLASSES}
    statuses = table["events_status"].fillna("")
    bad = statuses[(statuses != "") & ~statuses.isin(EVENTS_STATUS_COMPLETE)]
    unavailable = {str(k): int(v) for k, v in bad.value_counts().items()}
    name_only_mask = confidence == CONFIDENCE_NAME_ONLY
    name_only_v1_1 = int((name_only_mask & (rule_version == sec_filings.RULE_VERSION_V11)).sum())
    name_only_v1_unreadable = int(
        (name_only_mask & (rule_version == sec_filings.RULE_VERSION_V1)).sum()
    )
    # Everything else that is name-only (no or unknown rule version, any other combination), so
    # the three counts always add up to ``name_only``.
    name_only_other = int(
        (
            name_only_mask
            & (rule_version != sec_filings.RULE_VERSION_V11)
            & (rule_version != sec_filings.RULE_VERSION_V1)
        ).sum()
    )
    return {
        "roster_rows": len(table),
        "by_class": by_class,
        "documented_evidence": sum(by_class[name] for name in DOCUMENTED_CLASSES),
        "by_match_confidence": {
            CONFIDENCE_CONFIRMED: int((confidence == CONFIDENCE_CONFIRMED).sum()),
            CONFIDENCE_NAME_ONLY: int((confidence == CONFIDENCE_NAME_ONLY).sum()),
            "none": int((confidence == CONFIDENCE_NONE).sum()),
        },
        "name_only_v1_1": name_only_v1_1,
        "name_only_v1_unreadable": name_only_v1_unreadable,
        "name_only_other": name_only_other,
        "match_confidence_note": V11_LEAD_NOTE,
        "by_match_rule_version": {
            **{v: int((rule_version == v).sum()) for v in RULE_VERSIONS},
            "none": int((rule_version == RULE_VERSION_NONE).sum()),
        },
        "by_class_and_confidence": {
            name: {c: int(((classes == name) & (confidence == c)).sum()) for c in CONFIDENCES}
            for name in EVIDENCE_CLASSES
        },
        "by_class_and_rule_version": {
            name: {v: int(((classes == name) & (rule_version == v)).sum()) for v in RULE_VERSIONS}
            for name in EVIDENCE_CLASSES
        },
        "with_suggested_loss": int(has_loss.sum()),
        "with_suggested_loss_confirmed": int((has_loss & (confidence == "confirmed")).sum()),
        "with_suggested_loss_name_only": int((has_loss & (confidence == "name_only")).sum()),
        "with_suggested_loss_by_rule_version": {
            v: int((has_loss & (rule_version == v)).sum()) for v in RULE_VERSIONS
        },
        "by_basis": {basis: int((table["basis"] == basis).sum()) for basis in BASES},
        "by_basis_and_confidence": {
            basis: {
                **{
                    c: int(((table["basis"] == basis) & (confidence == c)).sum())
                    for c in CONFIDENCES
                },
                "none": int(((table["basis"] == basis) & (confidence == CONFIDENCE_NONE)).sum()),
            }
            for basis in BASES
        },
        "conflicts": int((table["cross_check"] == CROSS_CONFLICT).sum()),
        "agrees": int((table["cross_check"] == CROSS_AGREES).sum()),
        "events_fetch_errors": int((table["events_status"] == "fetch_error").sum()),
        "events_unavailable_or_incomplete": {
            "total": sum(unavailable.values()),
            "by_status": dict(sorted(unavailable.items())),
            "note": EVENTS_UNAVAILABLE_NOTE,
        },
        "items_missing": int(items_missing),
        "quarters_skipped": int(quarters_skipped),
    }
