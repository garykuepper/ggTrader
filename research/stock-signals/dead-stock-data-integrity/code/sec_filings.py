"""Read dead stocks' SEC Form 25 notices and Form 8-K exit events."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any
from xml.etree import ElementTree

from sec_client import SecClient, SecFetchError, SecNotFound
from sec_form25 import parse_bound_exchange_notice
from sec_master_index import parse_listing_archive

MASTER_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.zip"
FILING_DIR_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
NOTICE_DOCUMENT = "primary_doc.xml"
# Forms that announce a stock leaving an exchange; 8-A12B (a listing) is dropped on purpose.
FORM25_FORMS: frozenset[str] = frozenset({"25", "25/A", "25-NSE", "25-NSE/A"})
# Current-report forms inspected for exit evidence around the delisting date.
FORM8K_FORMS: frozenset[str] = frozenset({"8-K", "8-K/A"})
# Default symmetric window (in days) around the exit date for Form 8-K evidence.
EVENT_WINDOW_DAYS = 90
_SUBMISSIONS_FIELDS = ("form", "filingDate", "items", "accessionNumber", "primaryDocument")
_SAFE_PAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# Closed set of outcome codes returned by fetch_events_with_status.
EVENTS_STATUS_OK = "ok"
EVENTS_STATUS_NO_OLDER_PAGES_NEEDED = "no_older_pages_needed"
EVENTS_STATUS_NOT_FOUND = "not_found"
EVENTS_STATUS_OLDER_PAGE_NOT_FOUND = "older_page_not_found"
EVENTS_STATUSES: frozenset[str] = frozenset(
    {
        EVENTS_STATUS_OK,
        EVENTS_STATUS_NO_OLDER_PAGES_NEEDED,
        EVENTS_STATUS_NOT_FOUND,
        EVENTS_STATUS_OLDER_PAGE_NOT_FOUND,
    }
)
# Exit-evidence classes for Form 8-K items, applied in fixed priority order below.
EVIDENCE_BANKRUPTCY = "bankruptcy_filed"
EVIDENCE_ACQUIRED = "acquired_or_merged"
EVIDENCE_DELISTED_OTHER = "delisted_other"
EVIDENCE_NONE = "no_8k_found"
# Fixed priority table from the Task 7 brief; first matching rule wins across all window events.
# Item meanings (as understood; verify against the SEC Form 8-K instructions before use):
#   1.03: bankruptcy or receivership
#   2.01: completion of an acquisition or disposition of assets
#   3.01: notice of delisting
#   3.03: material change to the rights of shareholders
#   5.01: change in control of the company
EVIDENCE_ITEM_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    (EVIDENCE_BANKRUPTCY, frozenset({"1.03"})),
    (EVIDENCE_ACQUIRED, frozenset({"2.01", "3.03", "5.01"})),
    (EVIDENCE_DELISTED_OTHER, frozenset({"3.01"})),
)
# The window around the exit date in which a Form 25 can belong to the exit (brief section 5).
BEFORE_DAYS = 30
AFTER_DAYS = 180
# The McGill company name is cut off at this many characters.
TRUNCATED_NAME_LENGTH = 16
STATUS_MATCHED = "matched"
# A notice file larger than this is not parsed here (the shared parser has the same ceiling).
MAX_NOTICE_BYTES = 1024 * 1024
STATUS_READABLE = "readable"
STATUS_UNREADABLE = "unreadable"
STATUS_ISSUER_MISMATCH = "issuer_mismatch"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NONE = "none"
# A security description counts as common stock when it has one of these words and none of
# the words that mark a different kind of security.
COMMON_WORDS = ("COMMON", "ORDINARY SHARES")
NOT_COMMON_WORDS = ("PREFERRED", "WARRANT", "NOTE", "DEBENTURE", "RIGHTS")
# Trailing words dropped from a cleaned-up name (after the short forms below are spelled out).
SUFFIX_WORDS: frozenset[str] = frozenset(
    {"INC", "CORP", "CO", "COMPANY", "LTD", "LLC", "LP", "PLC"}
)
# Whole-word short forms spelled out the same way on the roster side and the filer side. The
# first block is the owner-approved base; the second was chosen by reading only the 133
# roster names, never an SEC result (see the brief's dated clarification of 2026-11-24).
ABBREVIATIONS: dict[str, str] = {
    "CP": "CORP",
    "CORPORATION": "CORP",
    "INCORPORATED": "INC",
    "INDS": "INDUSTRIES",
    "INTL": "INTERNATIONAL",
    "HLDGS": "HOLDINGS",
    "GRP": "GROUP",
    "TECH": "TECHNOLOGY",
    "TECHS": "TECHNOLOGIES",
    "SYS": "SYSTEMS",
    "PHARMACEUTICAL": "PHARMACEUTICALS",
    "DRG": "DRUG",
    "STRS": "STORES",
    "AMERN": "AMERICAN",
    "NATL": "NATIONAL",
    # Added after reading the roster names only.
    "PHARMA": "PHARMACEUTICALS",
    "PHARMS": "PHARMACEUTICALS",
    "NAT": "NATURAL",
    "ELEC": "ELECTRIC",
    "MED": "MEDICAL",
    "CONTL": "CONTINENTAL",
    "EXPLOR": "EXPLORATION",
    "RES": "RESOURCES",
    "PETE": "PETROLEUM",
    "COMM": "COMMUNICATIONS",
    "SOLU": "SOLUTIONS",
    "SOL": "SOLUTIONS",
    "PAYMNT": "PAYMENT",
    "SOUTHN": "SOUTHERN",
    "FINL": "FINANCIAL",
    "INS": "INSURANCE",
}


def master_index_url(year: int, quarter: int) -> str:
    """The SEC quarterly master index archive for one year and quarter."""
    return MASTER_INDEX_URL.format(year=year, quarter=quarter)


def quarters_covering(start: date, end: date) -> list[tuple[int, int]]:
    """Every (year, quarter) that holds at least one day from ``start`` to ``end``, in order."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    quarters: list[tuple[int, int]] = []
    year, quarter = start.year, (start.month - 1) // 3 + 1
    last = (end.year, (end.month - 1) // 3 + 1)
    while (year, quarter) <= last:
        quarters.append((year, quarter))
        year, quarter = (year + 1, 1) if quarter == 4 else (year, quarter + 1)
    return quarters


def load_quarter(client: SecClient, year: int, quarter: int) -> list[dict]:
    """Download one quarter's index and return its Form 25 filings.

    Raises:
        SecFetchError: if the download fails (including a missing quarter).
        ValueError: if the archive does not parse.
    """
    content = client.get(master_index_url(year, quarter))
    return [f for f in parse_listing_archive(content) if f["form"] in FORM25_FORMS]


def form25_candidates(client: SecClient, start: date, end: date) -> tuple[list[dict], list[dict]]:
    """Collect the Form 25 filings of every quarter from ``start`` to ``end``.

    A quarter that fails to download or parse is logged and the run goes on.

    Returns:
        ``(candidates, skipped)``; each skipped entry is ``{"quarter": (year, quarter),
        "reason": text}``.
    """
    candidates: list[dict] = []
    skipped: list[dict] = []
    for year, quarter in quarters_covering(start, end):
        try:
            candidates.extend(load_quarter(client, year, quarter))
        except (SecFetchError, ValueError) as exc:
            skipped.append({"quarter": (year, quarter), "reason": f"{type(exc).__name__}: {exc}"})
    return candidates, skipped


def normalize_name(name: str) -> str:
    """Clean a company name the same way for the roster and for SEC filers.

    Capitals, punctuation to spaces, spaces collapsed, whole-word short forms spelled out
    (``ABBREVIATIONS``), then trailing suffix words (``INC``, ``CORP`` and so on) dropped. The
    last word is never dropped, so a name that is only ``Inc.`` stays ``INC``.
    """
    words = re.sub(r"[^A-Z0-9]+", " ", name.upper()).split()
    words = [ABBREVIATIONS.get(word, word) for word in words]
    while len(words) > 1 and words[-1] in SUFFIX_WORDS:
        words.pop()
    return " ".join(words)


def _as_date(value: str | date) -> date:
    """Read an ISO date string or pass a date through."""
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _name_matches(roster_name: str, raw_length: int, filer_name: str) -> bool:
    """Exact match, or prefix match when the roster name is a known 16-character cut-off."""
    cleaned = normalize_name(filer_name)
    wanted = normalize_name(roster_name)
    if raw_length == TRUNCATED_NAME_LENGTH:
        return cleaned.startswith(wanted)
    return cleaned == wanted


def match_roster_row(
    row: Mapping[str, Any],
    candidates: list[dict],
    before_days: int = BEFORE_DAYS,
    after_days: int = AFTER_DAYS,
) -> dict:
    """Match one roster stock to the Form 25 filings that name it near its exit date.

    A candidate counts when one of its indexed parties has the stock's cleaned-up name (exact,
    or by prefix when the roster name is a 16-character cut-off) and it was filed from
    ``before_days`` before to ``after_days`` after the exit date, both ends included.

    Returns:
        ``{"status", "ciks", "accessions"}``: one distinct CIK is ``matched``, two or more is
        ``ambiguous``, none (or an unnamed row) is ``none``. Accessions are in filing order;
        ``by_cik`` lists the accessions of each matched CIK. A name match is only a lead: the
        caller must still check that the CIK is the notice's issuer (an exchange or another
        party named in the index can share a name).

    A complete legal name of exactly 16 characters is indistinguishable from a cut-off one and
    matches by prefix too; two different filers that share the prefix make it ``ambiguous``.
    """
    name = row.get("stock_name")
    if row.get("name_source") == "none" or not isinstance(name, str) or not name.strip():
        return {"status": STATUS_NONE, "ciks": [], "accessions": [], "by_cik": {}}
    exit_date = _as_date(row["exit_date"])
    first = exit_date - timedelta(days=before_days)
    last = exit_date + timedelta(days=after_days)
    ciks: set[str] = set()
    hits: list[tuple[str, str]] = []
    by_cik: dict[str, list[tuple[str, str]]] = {}
    for filing in candidates:
        if not first <= _as_date(filing["filing_date"]) <= last:
            continue
        matching = [
            p["cik"] for p in filing["parties"] if _name_matches(name, len(name), p["name"])
        ]
        if matching:
            ciks.update(matching)
            hits.append((filing["filing_date"], filing["accession"]))
            for cik in matching:
                by_cik.setdefault(cik, []).append((filing["filing_date"], filing["accession"]))
    status = STATUS_NONE if not ciks else STATUS_MATCHED if len(ciks) == 1 else STATUS_AMBIGUOUS
    return {
        "status": status,
        "ciks": sorted(ciks),
        "accessions": [accession for _, accession in sorted(hits)],
        "by_cik": {cik: [a for _, a in sorted(found)] for cik, found in sorted(by_cik.items())},
    }


def notice_is_common_stock(security_description: str) -> bool:
    """True when the described security looks like common (or ordinary) shares."""
    text = security_description.upper()
    return any(w in text for w in COMMON_WORDS) and not any(w in text for w in NOT_COMMON_WORDS)


def _issuer_cik(document: bytes) -> int | None:
    """The issuer CIK printed in an exchange notice, or None when it cannot be read simply.

    This only lets the caller tell a notice about another issuer from a notice that is broken;
    the shared parser still does the real validation.
    """
    if len(document) > MAX_NOTICE_BYTES:
        return None
    try:
        text = document.decode("utf-8-sig")
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            return None
        nodes = ElementTree.fromstring(text).findall("issuer/cik")
        return int(nodes[0].text.strip()) if len(nodes) == 1 and nodes[0].text else None
    except (ElementTree.ParseError, UnicodeError, ValueError):
        return None


def read_notice(client: SecClient, accession: str, cik: str) -> dict:
    """Fetch and read one exchange-filed Form 25 notice, recording both file hashes.

    Returns:
        ``{"status": "readable", "accession", "index_sha256", "document_sha256",
        "document_path", "notice": {...}}``; ``{"status": "issuer_mismatch", "accession",
        "found_issuer_cik", "reason"}`` when a well-formed notice names a different issuer than
        ``cik``; or ``{"status": "unreadable", "accession", "reason"}`` when a file is missing
        or the notice is not an exchange-filed XML one (an issuer-filed HTML Form 25 is not read).
    """
    nodash = accession.replace("-", "")
    index_url = FILING_DIR_URL.format(cik=int(cik), accession_nodash=nodash)
    try:
        index_bytes = client.get(index_url)
        document_bytes = client.get(index_url + NOTICE_DOCUMENT)
        found = _issuer_cik(document_bytes)
        if found is not None and found != int(cik):
            return {
                "status": STATUS_ISSUER_MISMATCH,
                "accession": accession,
                "found_issuer_cik": f"{found:010d}",
                "reason": f"notice issuer CIK {found:010d} is not {int(cik):010d}",
            }
        bound = parse_bound_exchange_notice(
            index_bytes,
            document_bytes,
            expected_accession=accession,
            expected_cik=int(cik),
            expected_index_sha256=hashlib.sha256(index_bytes).hexdigest(),
            expected_document_sha256=hashlib.sha256(document_bytes).hexdigest(),
        )
    except (SecFetchError, ValueError) as exc:
        return {
            "status": STATUS_UNREADABLE,
            "accession": accession,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {"status": STATUS_READABLE, **bound}


def submissions_url(cik: str) -> str:
    """Return the SEC EDGAR submissions JSON URL for ``cik``, zero-padded to 10 digits.

    Args:
        cik: Numeric CIK string (padded or unpadded).

    Returns:
        HTTPS URL on ``data.sec.gov`` for the company's submissions JSON.

    Raises:
        ValueError: If ``cik`` is not a positive integer under 10 digits.
    """
    value = int(cik)
    if not 0 < value < 10**10:
        raise ValueError(f"CIK out of range: {cik!r}")
    return SUBMISSIONS_URL.format(cik=value)


def _validate_page_name(name: Any) -> str:
    """Ensure an older submissions page filename cannot escape ``/submissions/``."""
    if not isinstance(name, str) or name in (".", "..") or not _SAFE_PAGE_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid submissions page name: {name!r}")
    return name


def older_page_urls(submissions: dict[str, Any]) -> list[str]:
    """Return the SEC URLs for older submissions pages listed in ``filings.files``.

    Args:
        submissions: Parsed top-level SEC submissions JSON object.

    Returns:
        List of ``https://data.sec.gov/submissions/<name>`` URLs in listed order.

    Raises:
        ValueError: If an entry in ``filings.files`` lacks a safe filename.
    """
    filings = submissions.get("filings")
    if not isinstance(filings, Mapping):
        return []
    files = filings.get("files")
    if not isinstance(files, list):
        return []
    urls: list[str] = []
    for entry in files:
        if not isinstance(entry, Mapping) or "name" not in entry:
            raise ValueError(f"Invalid submissions file entry: {entry!r}")
        name = _validate_page_name(entry["name"])
        urls.append(SUBMISSIONS_PAGE_URL.format(name=name))
    return urls


def _parse_items(raw: Any) -> list[str]:
    """Normalize an 8-K ``items`` cell into a list of stripped non-empty item codes.

    Handles both comma-separated strings (brief fixtures) and lists of strings (live EDGAR JSON).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, list):
        codes: list[str] = []
        for elem in raw:
            if isinstance(elem, str):
                codes.extend(part.strip() for part in elem.split(",") if part.strip())
        return codes
    raise ValueError(f"Unexpected 8-K items value: {raw!r}")


def _try_date(value: Any) -> date | None:
    """Parse an ISO date string or pass a ``date`` through; return None when blank or invalid."""
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _earliest_filing_date(block: Mapping[str, Any] | None) -> date | None:
    """Return the earliest valid ``filingDate`` in a submissions block, or None if empty."""
    if not isinstance(block, Mapping):
        return None
    raw_dates = block.get("filingDate")
    if not isinstance(raw_dates, list):
        return None
    parsed = [d for raw in raw_dates if (d := _try_date(raw)) is not None]
    return min(parsed) if parsed else None


def _page_overlaps_window(entry: Mapping[str, Any], start: date, end: date) -> bool:
    """True unless an older page's valid ``filingFrom``/``filingTo`` lies outside the window."""
    page_from = _try_date(entry.get("filingFrom"))
    page_to = _try_date(entry.get("filingTo"))
    # Fetch conservatively whenever range metadata is missing or inverted.
    if page_from is None or page_to is None or page_from > page_to:
        return True
    return page_from <= end and page_to >= start


def _iter_block_rows(block: Mapping[str, Any] | None) -> list[tuple[Any, Any, Any, Any, Any]]:
    """Validate and zip the five parallel filing arrays from a ``recent`` or archive block."""
    if not isinstance(block, Mapping) or not block:
        return []
    try:
        columns = [block[field] for field in _SUBMISSIONS_FIELDS]
    except KeyError as exc:
        raise ValueError(f"Submissions block missing required array: {exc}") from exc
    if not all(isinstance(col, list) for col in columns) or len({len(c) for c in columns}) != 1:
        raise ValueError("Submissions block parallel arrays must be lists of equal length")
    return list(zip(*columns, strict=True))


def extract_8k_events(
    submissions: dict[str, Any],
    pages: list[dict[str, Any]],
    around: date,
    days: int = EVENT_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """Extract Form 8-K and 8-K/A filings filed within ``days`` of ``around``.

    Reads the parallel arrays ``form``, ``filingDate``, ``items``, ``accessionNumber`` and
    ``primaryDocument`` from ``submissions["filings"]["recent"]`` and each older page in
    ``pages``. Events with empty ``items`` are preserved so callers can count ``items_missing``.

    Args:
        submissions: Parsed top-level SEC submissions JSON object.
        pages: Parsed older submissions archive JSON objects.
        around: Center date for the event window (typically the stock's exit date).
        days: Symmetric half-window in calendar days (both ends included).

    Returns:
        List of ``{"accession", "filing_date", "items", "primary_document"}`` dicts sorted by
        ``filing_date`` then ``accession``, deduplicated by accession number.

    Raises:
        ValueError: If ``days`` is negative or a submissions block has malformed parallel arrays.
    """
    if days < 0:
        raise ValueError(f"days must be non-negative, got {days}")
    center = _as_date(around)
    window_start = center - timedelta(days=days)
    window_end = center + timedelta(days=days)

    filings = submissions.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    blocks: list[Mapping[str, Any] | None] = [recent, *pages]

    events: list[dict[str, Any]] = []
    seen_accessions: set[str] = set()
    for block in blocks:
        for form, filed, raw_items, accession, document in _iter_block_rows(block):
            if not isinstance(form, str) or form.strip() not in FORM8K_FORMS:
                continue
            filed_date = _try_date(filed)
            if filed_date is None or not window_start <= filed_date <= window_end:
                continue
            acc_str = "" if accession is None else str(accession).strip()
            # Avoid double-counting a boundary filing that appears in both recent and page 1.
            if acc_str and acc_str in seen_accessions:
                continue
            if acc_str:
                seen_accessions.add(acc_str)
            events.append(
                {
                    "accession": acc_str,
                    "filing_date": filed_date.isoformat(),
                    "items": _parse_items(raw_items),
                    "primary_document": "" if document is None else str(document),
                }
            )

    events.sort(key=lambda event: (event["filing_date"], event["accession"]))
    return events


def evidence_class(events: list[dict[str, Any]]) -> str:
    """Classify a stock's exit from its window 8-K items using ``EVIDENCE_ITEM_RULES``.

    Applies the fixed priority table across all item codes present in ``events``:
    ``1.03`` -> ``bankruptcy_filed``; else any of ``2.01``, ``3.03``, ``5.01`` ->
    ``acquired_or_merged``; else ``3.01`` -> ``delisted_other``; else ``no_8k_found``.
    """
    all_items: set[str] = set()
    for event in events:
        items = event.get("items") or ()
        all_items.update(code.strip() for code in items if isinstance(code, str) and code.strip())
    for label, codes in EVIDENCE_ITEM_RULES:
        if all_items & codes:
            return label
    return EVIDENCE_NONE


def count_items_missing(events: list[dict[str, Any]]) -> int:
    """Count extracted 8-K events whose ``items`` list is empty."""
    return sum(1 for event in events if not event.get("items"))


def fetch_events_with_status(
    client: SecClient,
    cik: str,
    around: date,
    days: int = EVENT_WINDOW_DAYS,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch 8-K/8-K/A events near ``around`` and report the retrieval status.

    Older archive pages listed in ``filings.files`` are fetched only when the ``recent`` block
    does not already reach strictly before ``around - days`` (and when a page's optional
    ``filingFrom``/``filingTo`` range overlaps the window). An HTTP 404 (``SecNotFound``) on the
    main submissions URL returns ``([], "not_found")`` instead of raising; a 404 on a needed
    older page skips that page and returns ``(events, "older_page_not_found")``.

    Args:
        client: ``SecClient`` (or fake with ``get(url) -> bytes``).
        cik: Company CIK string.
        around: Center date of the search window.
        days: Symmetric half-window in calendar days.

    Returns:
        ``(events, status)`` where ``events`` is sorted by ``filing_date`` then ``accession`` and
        ``status`` is in ``EVENTS_STATUSES``: ``"ok"``, ``"no_older_pages_needed"``,
        ``"not_found"``, or ``"older_page_not_found"``.

    Raises:
        SecFetchError: On non-404 HTTP or transport errors.
        ValueError: On invalid CIK, negative ``days``, or malformed JSON/arrays.
    """
    if days < 0:
        raise ValueError(f"days must be non-negative, got {days}")
    center = _as_date(around)
    window_start = center - timedelta(days=days)
    window_end = center + timedelta(days=days)

    try:
        raw = client.get(submissions_url(cik))
    except SecNotFound:
        return [], EVENTS_STATUS_NOT_FOUND

    submissions = json.loads(raw.decode("utf-8"))
    if not isinstance(submissions, dict):
        raise ValueError(f"Submissions payload is not a JSON object for CIK {cik!r}")

    older_urls = older_page_urls(submissions)
    if not older_urls:
        return extract_8k_events(submissions, [], center, days=days), EVENTS_STATUS_OK

    filings = submissions.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    earliest_recent = _earliest_filing_date(recent)
    # Require strictly earlier than window_start so boundary-day filings on page 1 are not missed.
    if earliest_recent is not None and earliest_recent < window_start:
        return (
            extract_8k_events(submissions, [], center, days=days),
            EVENTS_STATUS_NO_OLDER_PAGES_NEEDED,
        )

    file_entries = filings["files"] if isinstance(filings, Mapping) else []
    needed_urls = [
        url
        for entry, url in zip(file_entries, older_urls, strict=True)
        if _page_overlaps_window(entry, window_start, window_end)
    ]
    if not needed_urls:
        return (
            extract_8k_events(submissions, [], center, days=days),
            EVENTS_STATUS_NO_OLDER_PAGES_NEEDED,
        )

    pages: list[dict[str, Any]] = []
    missing_older_page = False
    for page_url in needed_urls:
        try:
            page_raw = client.get(page_url)
        except SecNotFound:
            missing_older_page = True
            continue
        page_doc = json.loads(page_raw.decode("utf-8"))
        if not isinstance(page_doc, dict):
            raise ValueError(f"Older submissions page is not a JSON object: {page_url}")
        pages.append(page_doc)

    events = extract_8k_events(submissions, pages, center, days=days)
    status = EVENTS_STATUS_OLDER_PAGE_NOT_FOUND if missing_older_page else EVENTS_STATUS_OK
    return events, status


def fetch_events(
    client: SecClient,
    cik: str,
    around: date,
    days: int = EVENT_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """Fetch 8-K and 8-K/A events for ``cik`` within ``days`` of ``around``, sorted by date."""
    events, _ = fetch_events_with_status(client, cik, around, days=days)
    return events
