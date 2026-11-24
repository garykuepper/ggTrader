"""Find each dead stock's SEC Form 25 notice: quarterly index, name matching, notice reading."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

from sec_client import SecClient, SecFetchError
from sec_form25 import parse_bound_exchange_notice
from sec_master_index import parse_listing_archive

MASTER_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.zip"
FILING_DIR_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"
NOTICE_DOCUMENT = "primary_doc.xml"
# Forms that announce a stock leaving an exchange; 8-A12B (a listing) is dropped on purpose.
FORM25_FORMS: frozenset[str] = frozenset({"25", "25/A", "25-NSE", "25-NSE/A"})
# The window around the exit date in which a Form 25 can belong to the exit (brief section 5).
BEFORE_DAYS = 30
AFTER_DAYS = 180
# The McGill company name is cut off at this many characters.
TRUNCATED_NAME_LENGTH = 16
STATUS_MATCHED = "matched"
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
        ``ambiguous``, none (or an unnamed row) is ``none``. Accessions are in filing order.
    """
    name = row.get("stock_name")
    if row.get("name_source") == "none" or not isinstance(name, str) or not name.strip():
        return {"status": STATUS_NONE, "ciks": [], "accessions": []}
    exit_date = _as_date(row["exit_date"])
    first = exit_date - timedelta(days=before_days)
    last = exit_date + timedelta(days=after_days)
    ciks: set[str] = set()
    hits: list[tuple[str, str]] = []
    for filing in candidates:
        if not first <= _as_date(filing["filing_date"]) <= last:
            continue
        matching = [
            p["cik"] for p in filing["parties"] if _name_matches(name, len(name), p["name"])
        ]
        if matching:
            ciks.update(matching)
            hits.append((filing["filing_date"], filing["accession"]))
    status = STATUS_NONE if not ciks else STATUS_MATCHED if len(ciks) == 1 else STATUS_AMBIGUOUS
    return {
        "status": status,
        "ciks": sorted(ciks),
        "accessions": [accession for _, accession in sorted(hits)],
    }


def notice_is_common_stock(security_description: str) -> bool:
    """True when the described security looks like common (or ordinary) shares."""
    text = security_description.upper()
    return any(w in text for w in COMMON_WORDS) and not any(w in text for w in NOT_COMMON_WORDS)


def read_notice(client: SecClient, accession: str, cik: str) -> dict:
    """Fetch and read one exchange-filed Form 25 notice, recording both file hashes.

    Returns:
        ``{"status": "readable", "accession", "index_sha256", "document_sha256",
        "document_path", "notice": {...}}``, or ``{"status": "unreadable", "accession",
        "reason"}`` when a file is missing or the notice is not an exchange-filed XML one (an
        issuer-filed HTML Form 25 is not read).
    """
    nodash = accession.replace("-", "")
    index_url = FILING_DIR_URL.format(cik=int(cik), accession_nodash=nodash)
    try:
        index_bytes = client.get(index_url)
        document_bytes = client.get(index_url + NOTICE_DOCUMENT)
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
            "status": "unreadable",
            "accession": accession,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "readable", **bound}
