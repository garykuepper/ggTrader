"""Select periodic filing metadata from a complete SEC submissions history."""

import re
from datetime import date

_LISTING_FORMS = frozenset({"8-A12B", "8-A12B/A", "25", "25/A", "25-NSE", "25-NSE/A"})


def _check_cik(submissions: dict, expected_cik: int) -> None:
    if not isinstance(submissions, dict) or type(expected_cik) is not int:
        raise ValueError("Invalid CIK")
    cik = submissions.get("cik")
    if not isinstance(cik, str) or not re.fullmatch(r"[0-9]{1,10}", cik):
        raise ValueError("Invalid CIK")
    if not 0 < expected_cik < 10**10 or int(cik) != expected_cik:
        raise ValueError("Mismatched CIK")


def _date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid filing date") from exc
    if parsed.isoformat() != value:
        raise ValueError("Invalid filing date")
    return parsed


def _submission_pages(submissions: dict, pages: dict) -> list:
    try:
        filings = submissions["filings"]
        recent, references = filings["recent"], filings["files"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid submissions") from exc
    if not isinstance(references, list) or not isinstance(pages, dict):
        raise ValueError("Invalid submissions pages")
    selected_pages = [(recent, None, None)]
    names = set()
    for ref in references:
        try:
            name = ref["name"]
            start, end = _date(ref["filingFrom"]), _date(ref["filingTo"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid page reference") from exc
        if (
            not isinstance(name, str)
            or name in (".", "..")
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            or name in names
            or start > end
        ):
            raise ValueError("Invalid page reference")
        names.add(name)
        if name not in pages:
            raise ValueError(f"Missing historical page: {name}")
        selected_pages.append((pages[name], (start, end), name))
    return selected_pages


def latest_periodic_filing(
    submissions: dict,
    pages: dict[str, dict],
    filing_date_cutoff: str,
    *,
    expected_cik: int,
    form: str | None = None,
    allow_range_discrepancies: bool = False,
) -> dict:
    """Select 10-K/10-Q by inclusive filing date, not public availability time.

    Caller must verify that the SEC root and archive bytes are authentic and complete.
    All referenced pages are required. Earlier range discrepancies stop selection
    unless explicitly allowed; if allowed, every discrepancy is returned for review
    and the caller must resolve it before relying on a selected filing.
    Optional form selects 10-K or 10-Q, by filing date rather than report date.
    Amendments and transition forms are excluded.
    A same-day filing may have become public after the decision; independently
    verify its availability before using any of its facts. The report date is
    issuer-reported metadata, not independently verified fiscal-period identity.
    """
    _check_cik(submissions, expected_cik)
    if form not in (None, "10-K", "10-Q") or type(allow_range_discrepancies) is not bool:
        raise ValueError("Invalid filing selection options")
    day = _date(filing_date_cutoff)
    selected_pages = _submission_pages(submissions, pages)
    fields = ("filingDate", "form", "accessionNumber", "primaryDocument", "reportDate")
    candidates = []
    seen = set()
    observed_dates = []
    discrepancies = []
    prior_discrepancies = []
    for page, bounds, page_name in selected_pages:
        try:
            columns = [page[field] for field in fields]
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid filing columns") from exc
        if not all(isinstance(col, list) for col in columns) or len({len(c) for c in columns}) != 1:
            raise ValueError("Invalid filing columns")
        for filed, row_form, accession, document, reported in zip(*columns, strict=True):
            filed_date = _date(filed)
            observed_dates.append(filed_date)
            if bounds is not None and not bounds[0] <= filed_date <= bounds[1]:
                discrepancy = {"page": page_name, "filing_date": filed, "accession": accession}
                if filed_date <= day and not allow_range_discrepancies:
                    raise ValueError("Filing outside referenced page range before cutoff")
                (prior_discrepancies if filed_date <= day else discrepancies).append(discrepancy)
            if not isinstance(row_form, str) or not row_form or row_form != row_form.strip():
                raise ValueError("Invalid filing form")
            if not isinstance(accession, str) or not re.fullmatch(
                r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession
            ):
                raise ValueError("Invalid accession")
            if accession in seen:
                raise ValueError("Duplicate accession")
            seen.add(accession)
            if (
                row_form in ("10-K", "10-Q")
                and (form is None or form == row_form)
                and filed_date <= day
            ):
                candidates.append((filed_date, row_form, accession, document, reported))
    if not observed_dates or min(observed_dates) > day:
        raise ValueError("Cutoff precedes observed filing history")
    if not candidates:
        return {
            "filing": None,
            "prior_range_discrepancies": prior_discrepancies,
            "future_range_discrepancies": discrepancies,
        }
    latest = max(row[0] for row in candidates)
    rows = [row for row in candidates if row[0] == latest]
    if len(rows) != 1:
        raise ValueError("Ambiguous same-day periodic filings")
    filed, form, accession, document, reported = rows[0]
    try:
        report_date = _date(reported)
    except ValueError as exc:
        raise ValueError("Invalid report date") from exc
    if report_date > filed:
        raise ValueError("Invalid report date: after filing date")
    if (
        not isinstance(document, str)
        or document in (".", "..")
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", document)
    ):
        raise ValueError("Invalid filing metadata")
    filing = {
        "filing_date": filed.isoformat(),
        "report_date": report_date.isoformat(),
        "form": form,
        "accession": accession,
        "document": document,
    }
    return {
        "filing": filing,
        "prior_range_discrepancies": prior_discrepancies,
        "future_range_discrepancies": discrepancies,
    }


def recent_listing_filings(
    submissions: dict, filing_date_cutoff: str, *, expected_cik: int
) -> dict:
    """Inventory candidate forms in the recent block through an inclusive filing date.

    This does not read older pages, establish complete history, identify a security
    class or determine an event's effective or public-availability time. Filings on
    the cutoff date are included even if published after a decision. The caller must
    verify source provenance, documents, event timing, amendments and older pages.
    ``has_older_pages`` describes the supplied snapshot, not what was known then;
    ``earliest_recent_filing_date`` includes only dates at or before the cutoff.
    """
    _check_cik(submissions, expected_cik)
    cutoff = _date(filing_date_cutoff)
    try:
        filings = submissions["filings"]
        recent, older = filings["recent"], filings["files"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid submissions") from exc
    if not isinstance(older, list):
        raise ValueError("Invalid submissions pages")
    dates, candidates, _ = _listing_page(recent, cutoff, set())
    return {
        "filings": candidates,
        "has_older_pages": bool(older),
        "earliest_recent_filing_date": min(dates).isoformat() if dates else None,
    }


def _listing_page(page: dict, cutoff: date, seen: set) -> tuple:
    try:
        columns = [
            page[key] for key in ("filingDate", "form", "accessionNumber", "primaryDocument")
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid filing columns") from exc
    if (
        not all(isinstance(column, list) for column in columns)
        or len({len(c) for c in columns}) != 1
    ):
        raise ValueError("Invalid filing columns")
    dates, candidates, observed = [], [], []
    for filed, form, accession, document in zip(*columns, strict=True):
        day = _date(filed)
        if day <= cutoff:
            dates.append(day)
        if not isinstance(form, str) or not form or form != form.strip():
            raise ValueError("Invalid filing form")
        if not isinstance(accession, str) or not re.fullmatch(
            r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession
        ):
            raise ValueError("Invalid accession")
        if accession in seen:
            raise ValueError("Duplicate accession")
        seen.add(accession)
        observed.append((day, accession))
        if day > cutoff or form not in _LISTING_FORMS:
            continue
        if (
            not isinstance(document, str)
            or not document
            or any(
                part in ("", ".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                for part in document.split("/")
            )
        ):
            raise ValueError("Invalid document path")
        candidates.append(
            {"filing_date": filed, "form": form, "accession": accession, "document": document}
        )
    candidates.sort(key=lambda row: (row["filing_date"], row["accession"]))
    return dates, candidates, observed


def listing_filings(
    submissions: dict,
    pages: dict[str, dict],
    filing_date_cutoff: str,
    *,
    expected_cik: int,
    allow_range_discrepancies: bool = False,
) -> dict:
    """Inventory candidate listing filings across all supplied referenced pages.

    The cutoff is an inclusive filing date, not a public availability or event
    time. Even a complete set of referenced pages does not prove SEC history or
    event coverage is complete. Callers must resolve all range discrepancies and
    verify source provenance, issuer, security class and the actual event.
    Future rows are validated for malformed metadata and duplicates, but are not
    returned. Page names and ranges are from the supplied snapshot, not a claim
    about what was known at the cutoff. An empty result is not proof that no
    events existed, including when the cutoff precedes all observed filings.
    """
    _check_cik(submissions, expected_cik)
    if type(allow_range_discrepancies) is not bool:
        raise ValueError("Invalid filing selection options")
    cutoff = _date(filing_date_cutoff)
    seen, dates, candidates, discrepancies = set(), [], [], []
    for page, bounds, page_name in _submission_pages(submissions, pages):
        page_dates, page_candidates, observed = _listing_page(page, cutoff, seen)
        dates.extend(page_dates)
        candidates.extend(page_candidates)
        for day, accession in observed:
            if bounds is not None and day <= cutoff and not bounds[0] <= day <= bounds[1]:
                if not allow_range_discrepancies:
                    raise ValueError("Filing outside referenced page range before cutoff")
                discrepancies.append(
                    {"page": page_name, "filing_date": day.isoformat(), "accession": accession}
                )
    candidates.sort(key=lambda row: (row["filing_date"], row["accession"]))
    return {
        "filings": candidates,
        "range_discrepancies": discrepancies,
        "earliest_observed_filing_date": min(dates).isoformat() if dates else None,
    }
