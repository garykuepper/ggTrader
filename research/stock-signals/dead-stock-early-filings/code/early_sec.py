"""SEC index search, filing-header parsing, and 8-K item classification for early-window deaths.

No network calls or file reads occur at import time. Callers supply an opener or cache
directory; live SEC requests are made only by the controller driver.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
import zlib
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

import early_match
import pandas as pd
from ef_sec_client import (
    DEFAULT_MAX_BYTES,
    SecClient,
    SecNotFound,
    SecUserAgentMissing,
    _atomic_write_bytes,
    _url_key,
    _validate_url,
)

DEFAULT_CACHE_DIR = Path("/media/thesix/ggTrader-data/sec-early-window")
DEFAULT_MIN_INTERVAL = 0.2
DEFAULT_CONTACT_PATH = Path("~/.config/ggtrader/sec_contact")
ENV_CONTACT_PATH = "SEC_CONTACT_PATH"

BEFORE_DAYS = 30
AFTER_DAYS = 180

MASTER_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.zip"
FILING_ARCHIVE_URL = "https://www.sec.gov/Archives/{file_path}"
MASTER_INDEX_HEADER = "CIK|Company Name|Form Type|Date Filed|Filename"
MASTER_INDEX_FILENAME = "master.idx"
FILENAME_SUFFIX = ".txt"
MAX_ARCHIVE_BYTES = 30_000_000
MAX_INDEX_BYTES = 64_000_000
MAX_HEADER_LINES = 50
VALID_QUARTERS: frozenset[int] = frozenset({1, 2, 3, 4})

# Forms read for exit events; 8-A12B / 8-A12G are exchange listings and ignored as exit events.
FORM8K_FORMS: frozenset[str] = frozenset({"8-K", "8-K/A"})
FORM25_FORMS: frozenset[str] = frozenset({"25", "25/A", "25-NSE", "25-NSE/A"})
IGNORED_LISTING_FORMS: frozenset[str] = frozenset({"8-A12B", "8-A12B/A", "8-A12G", "8-A12G/A"})
RELEVANT_EVENT_FORMS: frozenset[str] = FORM8K_FORMS | FORM25_FORMS
# Whitelisted periodic report and NT 10-K/10-Q late-notice form codes (brief amendment 5).
_PERIODIC_REPORT_BASE_FORMS: tuple[str, ...] = (
    "10-K",
    "10-K405",
    "10-KSB",
    "10-KSB40",
    "10-KT",
    "10-K405T",
    "10-Q",
    "10-QSB",
    "10-QT",
    "NT 10-K",
    "NT 10-Q",
    "NT 10-KT",
    "NT 10-QT",
)
PERIODIC_REPORT_FORMS: frozenset[str] = frozenset(
    {form for base in _PERIODIC_REPORT_BASE_FORMS for form in (base, f"{base}/A")}
)

# Registered stock labels and 8-K item label constants from section 1 + amendment 2 of the brief.
LABEL_AMBIGUOUS_MATCH = "ambiguous_match"
LABEL_BANKRUPTCY = "bankruptcy"
LABEL_EVIDENCE_UNRESOLVED = "evidence_unresolved"
LABEL_ACQUISITION_TYPE = "acquisition_type"
LABEL_DELISTING_NOTICE_ONLY = "delisting_notice_only"
LABEL_KEPT_FILING = "kept_filing"
LABEL_NO_EVENT_FOUND = "no_event_found"
LABEL_NO_COMPANY_MATCH = "no_company_match"
LABEL_NAME_UNKNOWN = "name_unknown"
LABEL_OTHER = "other"

# Registered confidence tiers (higher rank = higher confidence).
CONFIDENCE_CHECKED = "checked"
CONFIDENCE_NAME_ONLY = "name_only"
CONFIDENCE_FORMER_NAME = "former_name"
CONFIDENCE_RANK: dict[str, int] = {
    CONFIDENCE_FORMER_NAME: 1,
    CONFIDENCE_NAME_ONLY: 2,
    CONFIDENCE_CHECKED: 3,
}

# Effective date of the SEC's renumbered Form 8-K item list (2026-11-25 brief amendment).
NUMBERED_ITEMS_START = date(2004, 8, 23)

# Registered 8-K item text labels (count in every era, matched by whole normalised text).
BANKRUPTCY_TEXT_ITEMS: frozenset[str] = frozenset(
    {
        "bankruptcy or receivership",
    }
)
ACQUISITION_TEXT_ITEMS: frozenset[str] = frozenset(
    {
        "changes in control of registrant",
        "acquisition or disposition of assets",
        "completion of acquisition or disposition of assets",
    }
)
DELISTING_TEXT_ITEMS: frozenset[str] = frozenset(
    {
        "notice of delisting or failure to satisfy a continued listing rule or standard; "
        "transfer of listing",
    }
)

# Registered 8-K numeric item codes (count only for filings dated on or after 2004-08-23).
BANKRUPTCY_NUMBER_ITEMS: frozenset[str] = frozenset({"1.03"})
ACQUISITION_NUMBER_ITEMS: frozenset[str] = frozenset({"2.01", "5.01"})
DELISTING_NUMBER_ITEMS: frozenset[str] = frozenset({"3.01"})

# Combined registered sets kept for reference/inspection.
BANKRUPTCY_ITEMS: frozenset[str] = BANKRUPTCY_TEXT_ITEMS | BANKRUPTCY_NUMBER_ITEMS
ACQUISITION_ITEMS: frozenset[str] = ACQUISITION_TEXT_ITEMS | ACQUISITION_NUMBER_ITEMS
DELISTING_ITEMS: frozenset[str] = DELISTING_TEXT_ITEMS | DELISTING_NUMBER_ITEMS

_UNSET: Any = object()
_NUMBERED_ITEM_RE = re.compile(r"\d+\.\d{2}")
# A numeric declaration is strictly one code or a list separated by commas / 'and' / whitespace.
_ITEM_CODE_LIST_RE = re.compile(
    r"^\d+\.\d{2}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)\d+\.\d{2})*$",
    re.IGNORECASE,
)
_ITEM_INFO_LINE_RE = re.compile(r"^\s*ITEM\s+INFORMATION\s*:\s*(.*)$", re.IGNORECASE)
_ITEMS_TAG_LINE_RE = re.compile(r"^\s*<ITEMS>\s*(.*?)(?:</ITEMS>)?\s*$", re.IGNORECASE)
_HEADER_FIELD_RE = re.compile(r"^\s*([A-Z][A-Z0-9 /_-]*?)\s*:\s*(.*?)\s*$")
_SEC_HEADER_OPEN_RE = re.compile(r"<SEC-HEADER>", re.IGNORECASE)
_SEC_HEADER_CLOSE_RE = re.compile(r"^[ \t]*</SEC-HEADER>[ \t]*\r?$", re.IGNORECASE | re.MULTILINE)
_DOCUMENT_OPEN_RE = re.compile(r"^[ \t]*<DOCUMENT>[ \t]*\r?$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True, slots=True)
class IndexRow:
    """One filing row parsed from an SEC quarterly ``master.zip`` index archive."""

    cik: int
    company_name: str
    form: str
    filed: date
    file_path: str

    @property
    def accession(self) -> str:
        """Accession identifier extracted from the trailing ``<accession>.txt`` path."""
        return self.file_path.rsplit("/", 1)[-1].removesuffix(FILENAME_SUFFIX)


@dataclass(frozen=True, slots=True)
class Filing:
    """One matched SEC filing from the quarterly master index."""

    form: str
    company_name: str
    cik: int
    filed: date
    file_path: str
    confidence: str = CONFIDENCE_NAME_ONLY
    allow_checked: bool = True

    @property
    def accession(self) -> str:
        """Accession identifier extracted from the trailing ``<accession>.txt`` path."""
        return self.file_path.rsplit("/", 1)[-1].removesuffix(FILENAME_SUFFIX)


@dataclass(frozen=True, slots=True)
class FilingHeader:
    """Structured metadata extracted from an EDGAR SGML filing header."""

    form: str | None
    filed: date | None
    company_name: str | None
    cik: int | None
    former_names: tuple[str, ...]
    items: frozenset[str]
    complete: bool = False


def load_contact_line(
    environ: Mapping[str, str] | None = None,
    *,
    contact_path: Path | None = None,
) -> str:
    """Read the SEC fair-access contact line from a path without logging or echoing it.

    Path resolution order: explicit ``contact_path``, then ``SEC_CONTACT_PATH`` from
    ``environ`` (defaults to ``os.environ``), then ``~/.config/ggtrader/sec_contact``.

    Raises:
        SecUserAgentMissing: If the file cannot be read or contains no non-empty line.
    """
    env = os.environ if environ is None else environ
    if contact_path is not None:
        resolved = Path(contact_path).expanduser()
    else:
        raw_env_path = env.get(ENV_CONTACT_PATH, "")
        env_path = raw_env_path.strip() if isinstance(raw_env_path, str) else ""
        resolved = Path(env_path).expanduser() if env_path else DEFAULT_CONTACT_PATH.expanduser()

    try:
        content = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise SecUserAgentMissing(
            f"SEC contact file could not be read at configured path ({resolved.name})."
        ) from exc

    for line in content.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    raise SecUserAgentMissing(f"SEC contact file is empty or whitespace-only ({resolved.name}).")


class EarlySecClient:
    """Study wrapper around ``ef_sec_client.SecClient`` with persistent 404 caching.

    Uses the frozen study defaults (``/media/thesix/ggTrader-data/sec-early-window`` and
    ``min_interval=0.2`` for 5 requests/second) and writes ``<sha256(url)>.notfound.json``
    markers so HTTP 404 responses are never requested a second time. Never logs or writes
    the User-Agent contact line.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._client = SecClient(
            user_agent=user_agent,
            cache_dir=self._cache_dir,
            min_interval=min_interval,
            opener=opener,
            sleep=sleep,
            clock=clock,
            max_bytes=max_bytes,
        )
        self.not_found_hits = 0
        self.not_found_recorded = 0

    @property
    def client(self) -> SecClient:
        """Underlying byte-identical ``ef_sec_client.SecClient`` instance."""
        return self._client

    @property
    def cache_hits(self) -> int:
        """Count of cached 200 responses served without hitting the opener."""
        return self._client.cache_hits

    @property
    def network_fetches(self) -> int:
        """Count of 200 responses fulfilled by the opener and written to cache."""
        return self._client.network_fetches

    @property
    def corrupt_cache_refetches(self) -> int:
        """Count of corrupted cached entries refetched by the underlying client."""
        return self._client.corrupt_cache_refetches

    def _not_found_path(self, url: str) -> Path:
        return self._cache_dir / f"{_url_key(url)}.notfound.json"

    def _has_valid_not_found_marker(self, url: str) -> bool:
        marker_path = self._not_found_path(url)
        if not marker_path.exists():
            return False
        try:
            meta = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker_path.unlink(missing_ok=True)
            return False
        if isinstance(meta, Mapping) and meta.get("url") == url and meta.get("status") == 404:
            return True
        marker_path.unlink(missing_ok=True)
        return False

    def _record_not_found(self, url: str) -> None:
        marker_path = self._not_found_path(url)
        # Never persist User-Agent or headers in not-found markers.
        meta = {
            "url": url,
            "status": 404,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write_bytes(marker_path, payload)
        self.not_found_recorded += 1

    def is_not_found(self, url: str) -> bool:
        """Return True if ``url`` has a valid recorded 404 marker on disk."""
        _validate_url(url)
        return self._has_valid_not_found_marker(url)

    def get(self, url: str) -> bytes:
        """Fetch ``url``, honouring cached 200 bodies and recorded 404 markers."""
        _validate_url(url)
        if self._has_valid_not_found_marker(url):
            self.not_found_hits += 1
            raise SecNotFound(f"SEC URL previously recorded as not found (HTTP 404): {url}")
        try:
            return self._client.get(url)
        except SecNotFound:
            self._record_not_found(url)
            raise

    def manifest(self) -> dict[str, str]:
        """Return ``{url: sha256}`` for all verified cached 200 bodies."""
        return self._client.manifest()


def _as_date(value: date | datetime | pd.Timestamp | str) -> date:
    """Normalise a ``pd.Timestamp``, ``datetime``, ``date``, or ISO string to ``date``."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def master_index_url(year: int, quarter: int) -> str:
    """Return the SEC quarterly ``master.zip`` URL for one ``(year, quarter)``."""
    if year <= 0 or quarter not in VALID_QUARTERS:
        raise ValueError(
            f"Invalid quarter ({year}, {quarter}); expected positive year and quarter in 1..4"
        )
    return MASTER_INDEX_URL.format(year=year, quarter=quarter)


def filing_url(file_path: str) -> str:
    """Return the SEC EDGAR HTTPS URL for a ``master.idx`` relative ``file_path``."""
    cleaned = file_path.strip().lstrip("/")
    if not cleaned or "://" in cleaned or ".." in cleaned.split("/"):
        raise ValueError(f"Invalid SEC archive file_path: {file_path!r}")
    return FILING_ARCHIVE_URL.format(file_path=cleaned)


def quarters_covering(start: date, end: date) -> list[tuple[int, int]]:
    """Every ``(year, quarter)`` containing at least one day in ``[start, end]``, in order."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    quarters: list[tuple[int, int]] = []
    year, quarter = start.year, (start.month - 1) // 3 + 1
    last = (end.year, (end.month - 1) // 3 + 1)
    while (year, quarter) <= last:
        quarters.append((year, quarter))
        year, quarter = (year + 1, 1) if quarter == 4 else (year, quarter + 1)
    return quarters


def quarters_for(last_bar: pd.Timestamp) -> list[tuple[int, int]]:
    """Return ``(year, quarter)`` pairs covering 30 days before to 180 days after ``last_bar``."""
    last_day = _as_date(last_bar)
    return quarters_covering(
        last_day - timedelta(days=BEFORE_DAYS),
        last_day + timedelta(days=AFTER_DAYS),
    )


class _BoundedStream(io.RawIOBase):
    """Wrap a zip member stream so uncompressed bytes cannot silently exceed the ceiling."""

    def __init__(self, raw: BinaryIO, max_bytes: int) -> None:
        self._raw = raw
        self._max_bytes = max_bytes
        self._remaining = max_bytes

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        limit = min(len(buffer), self._remaining + 1)
        chunk = self._raw.read(limit)
        if not chunk:
            return 0
        if len(chunk) > self._remaining:
            raise ValueError(f"Uncompressed master-index archive exceeds {self._max_bytes} bytes")
        self._remaining -= len(chunk)
        buffer[: len(chunk)] = chunk
        return len(chunk)


def _parse_data_line(raw_line: str, line_no: int, date_cache: dict[str, date]) -> IndexRow:
    """Parse and validate one pipe-delimited filing row from ``master.idx``."""
    parts = raw_line.rstrip("\r\n").split("|")
    if len(parts) != 5:
        raise ValueError(
            f"Malformed master-index line {line_no}: expected 5 fields, got {len(parts)}"
        )
    cik_raw, name_raw, form_raw, filed_raw, filename_raw = parts
    if not (cik_raw.isascii() and cik_raw.isdigit() and 1 <= len(cik_raw) <= 10):
        raise ValueError(f"Malformed master-index line {line_no}: invalid CIK {cik_raw!r}")
    cik = int(cik_raw)
    if cik == 0:
        raise ValueError(f"Malformed master-index line {line_no}: CIK must be positive")

    company_name = name_raw.strip()
    form = form_raw.strip()
    if not form:
        raise ValueError(f"Malformed master-index line {line_no}: empty form type")

    file_path = filename_raw.strip()
    if not file_path:
        raise ValueError(f"Malformed master-index line {line_no}: empty filename")
    basename = file_path.rsplit("/", 1)[-1]
    if not basename.endswith(FILENAME_SUFFIX):
        raise ValueError(
            f"Malformed master-index line {line_no}: filename must end with "
            f"{FILENAME_SUFFIX!r}: {filename_raw!r}"
        )
    accession = basename.removesuffix(FILENAME_SUFFIX)
    if not accession.strip():
        raise ValueError(
            f"Malformed master-index line {line_no}: empty accession in filename {filename_raw!r}"
        )

    filed = date_cache.get(filed_raw)
    if filed is None:
        try:
            parsed = date.fromisoformat(filed_raw)
        except ValueError as exc:
            raise ValueError(
                f"Malformed master-index line {line_no}: invalid date {filed_raw!r}"
            ) from exc
        if parsed.isoformat() != filed_raw:
            raise ValueError(
                f"Malformed master-index line {line_no}: non-canonical date {filed_raw!r}"
            )
        date_cache[filed_raw] = parsed
        filed = parsed

    return IndexRow(
        cik=cik,
        company_name=company_name,
        form=form,
        filed=filed,
        file_path=file_path,
    )


@contextmanager
def _open_verified_index(content: bytes) -> Iterator[tuple[io.TextIOWrapper, int]]:
    """Validate archive size, zip structure, header, and separator, then yield the row stream."""
    if not isinstance(content, bytes):
        raise ValueError("Invalid master-index archive")
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"Master-index archive exceeds {MAX_ARCHIVE_BYTES} bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            info = archive.infolist()
            if (
                len(info) != 1
                or info[0].filename != MASTER_INDEX_FILENAME
                or info[0].file_size > MAX_INDEX_BYTES
            ):
                raise ValueError("Invalid master-index archive contents")
            with (
                archive.open(info[0]) as raw,
                io.TextIOWrapper(
                    io.BufferedReader(_BoundedStream(raw, MAX_INDEX_BYTES)),
                    encoding="utf-8-sig",
                    newline="",
                ) as stream,
            ):
                line_no = 0
                for _ in range(MAX_HEADER_LINES):
                    header_line = stream.readline()
                    if not header_line:
                        raise ValueError("Missing master-index header")
                    line_no += 1
                    if header_line.rstrip("\r\n") == MASTER_INDEX_HEADER:
                        break
                else:
                    raise ValueError("Missing master-index header")

                sep_line = stream.readline()
                line_no += 1
                sep = sep_line.strip()
                if not sep or any(ch != "-" for ch in sep):
                    raise ValueError("Invalid master-index separator")

                yield stream, line_no
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError, UnicodeError) as exc:
        raise ValueError("Invalid master-index archive") from exc


def iter_master_index(content: bytes) -> Iterator[IndexRow]:
    """Stream every validated ``IndexRow`` from one quarterly ``master.zip`` archive."""
    with _open_verified_index(content) as (stream, line_no):
        date_cache: dict[str, date] = {}
        for raw_line in stream:
            line_no += 1
            yield _parse_data_line(raw_line, line_no, date_cache)


def parse_master_index(content: bytes) -> list[IndexRow]:
    """Parse every filing row from a quarterly SEC ``master.zip`` archive."""
    return list(iter_master_index(content))


def clean_company_name(name: str) -> str:
    """Clean a company name using the registered v1.1 rule (brief amendment 3)."""
    return early_match.normalize_name(name)


def _unpack_index_row(
    row: IndexRow | Filing | Mapping[str, Any],
) -> tuple[str, str, int, date, str]:
    """Read ``(form, company_name, cik, filed, file_path)`` from an index row or mapping."""
    if isinstance(row, (IndexRow, Filing)):
        return row.form, row.company_name, row.cik, row.filed, row.file_path
    if isinstance(row, Mapping):
        form = str(row["form"]).strip()
        company_name = str(row.get("company_name", row.get("name", ""))).strip()
        cik = int(row["cik"])
        filed_raw = row.get("filed", row.get("date"))
        if filed_raw is None:
            raise ValueError("Index row mapping missing 'filed' or 'date'")
        filed = _as_date(filed_raw)
        file_path = str(row.get("file_path", row.get("filename", ""))).strip()
        return form, company_name, cik, filed, file_path
    raise TypeError(f"Unsupported index row type: {type(row).__name__}")


def find_filings(
    index_rows: Iterable[IndexRow | Filing | Mapping[str, Any]],
    company_name: str,
    former_names: Sequence[str] = (),
    *,
    forms: Collection[str] | None = None,
) -> list[Filing]:
    """Search quarterly master-index rows for ``company_name`` and ``former_names``.

    Matches via ``early_match`` (amendment 3 v1.1 rule, 16-character cut-off prefix, added
    short forms, and roster-side trailing state-code drop). Primary-name matches receive
    ``confidence="name_only"`` (with ``allow_checked=False`` when matched only through
    rules 2 to 4); former-name matches receive ``confidence="former_name"`` (lower
    confidence and never promoted to ``"checked"``). Results are sorted by
    ``(filed, form, cik, file_path)``.
    """
    primary_raw = company_name.strip() if isinstance(company_name, str) else ""
    primary_has_key = bool(early_match.roster_name_key(primary_raw))
    cleaned_formers = [
        raw_f.strip()
        for raw_f in former_names
        if isinstance(raw_f, str) and early_match.roster_name_key(raw_f.strip())
    ]
    if not primary_has_key and not cleaned_formers:
        return []

    allowed_forms = frozenset(forms) if forms is not None else None
    primary_len = len(primary_raw)
    former_entries = [(f_name, len(f_name)) for f_name in cleaned_formers]
    match_cache: dict[str, tuple[str, bool] | None] = {}
    matched: list[Filing] = []

    for row in index_rows:
        form, raw_name, cik, filed, file_path = _unpack_index_row(row)
        if allowed_forms is not None and form not in allowed_forms:
            continue
        if not raw_name:
            continue
        cached = match_cache.get(raw_name, _UNSET)
        if cached is _UNSET:
            primary_match = (
                early_match.classify_match(primary_raw, primary_len, raw_name)
                if primary_has_key
                else None
            )
            if primary_match is not None:
                cached = (CONFIDENCE_NAME_ONLY, primary_match == early_match.MATCH_RULE1)
            elif any(
                early_match.name_matches(f_name, f_len, raw_name)
                for f_name, f_len in former_entries
            ):
                cached = (CONFIDENCE_FORMER_NAME, False)
            else:
                cached = None
            match_cache[raw_name] = cached
        if cached is None:
            continue

        confidence, allow_checked = cached
        matched.append(
            Filing(
                form=form,
                company_name=raw_name,
                cik=cik,
                filed=filed,
                file_path=file_path,
                confidence=confidence,
                allow_checked=allow_checked,
            )
        )

    matched.sort(key=lambda filing: (filing.filed, filing.form, filing.cik, filing.file_path))
    return matched


def _extract_header_section_with_status(text: str) -> tuple[str, bool]:
    """Isolate the SGML header before ``</SEC-HEADER>`` or ``<DOCUMENT>`` and report completion.

    A header is complete only when a standalone closing ``</SEC-HEADER>`` delimiter line
    (case-insensitive, optional leading/trailing horizontal whitespace) appears before any
    ``<DOCUMENT>`` line. Quoted mentions of ``</SEC-HEADER>`` inside other text do not count.
    """
    doc_first = _DOCUMENT_OPEN_RE.search(text)
    open_match = (
        _SEC_HEADER_OPEN_RE.search(text, 0, doc_first.start())
        if doc_first is not None
        else _SEC_HEADER_OPEN_RE.search(text)
    )
    start = open_match.start() if open_match is not None else 0
    doc_match = _DOCUMENT_OPEN_RE.search(text, start)
    close_match = (
        _SEC_HEADER_CLOSE_RE.search(text, start, doc_match.start())
        if doc_match is not None
        else _SEC_HEADER_CLOSE_RE.search(text, start)
    )
    if close_match is not None:
        return text[start : close_match.end()], True
    if doc_match is not None:
        return text[start : doc_match.start()], False
    return text[start:], False


def _extract_header_section(text: str) -> str:
    """Isolate the SGML header portion before ``</SEC-HEADER>`` or ``<DOCUMENT>``."""
    return _extract_header_section_with_status(text)[0]


def _parse_header_date(raw: str) -> date | None:
    """Parse an EDGAR header date in ``YYYYMMDD`` or ``YYYY-MM-DD`` format."""
    cleaned = raw.strip()
    if len(cleaned) == 8 and cleaned.isascii() and cleaned.isdigit():
        try:
            return date(int(cleaned[:4]), int(cleaned[4:6]), int(cleaned[6:8]))
        except ValueError:
            return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _extract_filed_date(header: str) -> date | None:
    """Read the first valid ``FILED AS OF DATE:`` value from an SGML header block."""
    for line in header.splitlines():
        match = _HEADER_FIELD_RE.match(line)
        if match is None:
            continue
        key = " ".join(match.group(1).upper().split())
        if key == "FILED AS OF DATE":
            value = " ".join(match.group(2).split())
            return _parse_header_date(value) if value else None
    return None


def _numbers_allowed(filed: date | datetime | pd.Timestamp | str | None) -> bool:
    """Return True only when ``filed`` is a readable date on or after 2004-08-23."""
    if filed is None or filed is pd.NaT:
        return False
    if isinstance(filed, str):
        parsed = _parse_header_date(filed)
        return parsed is not None and parsed >= NUMBERED_ITEMS_START
    try:
        return _as_date(filed) >= NUMBERED_ITEMS_START
    except (TypeError, ValueError):
        return False


def _parse_item_declaration(raw_value: str) -> tuple[bool, list[str]]:
    """Classify one item value as a strict numeric code list or a whole normalised text label.

    Numbers count only when the whole normalised value is a single item code or a list of
    item codes separated by commas, ``and``, or whitespace. Any other value (including a
    description containing a number such as ``"Other events (agreement section 2.01)"``) is
    returned as whole normalised text only.
    """
    collapsed = " ".join(raw_value.split())
    if not collapsed:
        return False, []
    if _ITEM_CODE_LIST_RE.fullmatch(collapsed):
        return True, _NUMBERED_ITEM_RE.findall(collapsed)
    normalised = collapsed.rstrip(".").lower()
    return False, [normalised] if normalised else []


def parse_8k_items(
    header_text: str,
    *,
    filed: Any = _UNSET,
) -> frozenset[str]:
    """Extract era-eligible 8-K item labels or numbered codes from an EDGAR filing header.

    Text labels are extracted in every era and matched by whole normalised text. Numeric
    item declarations (a single code or list of codes separated by commas, ``and``, or
    whitespace) are extracted only when the filing date is on or after 2004-08-23; when
    ``filed`` is omitted, ``FILED AS OF DATE:`` is read from ``header_text``. Missing or
    pre-2004-08-23 dates yield text labels only (numbers are excluded). Body text after
    ``</SEC-HEADER>`` or ``<DOCUMENT>`` is ignored.
    """
    if not isinstance(header_text, str) or not header_text.strip():
        return frozenset()
    header = _extract_header_section(header_text)
    if filed is _UNSET:
        effective_filed: date | datetime | pd.Timestamp | str | None = _extract_filed_date(header)
    else:
        effective_filed = filed
    allow_numbers = _numbers_allowed(effective_filed)

    items: set[str] = set()
    for line in header.splitlines():
        match = _ITEM_INFO_LINE_RE.match(line)
        if match is None:
            match = _ITEMS_TAG_LINE_RE.match(line)
        if match is None:
            continue
        is_numeric, tokens = _parse_item_declaration(match.group(1))
        if is_numeric and not allow_numbers:
            continue
        items.update(tokens)
    return frozenset(items)


def parse_filing_header(text: str) -> FilingHeader:
    """Parse form, filed date, company name, CIK, former names, items, and completion."""
    header, complete = _extract_header_section_with_status(text)
    form: str | None = None
    fallback_form: str | None = None
    filed: date | None = None
    company_name: str | None = None
    cik: int | None = None
    former_names: list[str] = []
    seen_formers: set[str] = set()

    for line in header.splitlines():
        match = _HEADER_FIELD_RE.match(line)
        if match is None:
            continue
        key = " ".join(match.group(1).upper().split())
        value = " ".join(match.group(2).split())
        if not value:
            continue
        if key == "CONFORMED SUBMISSION TYPE" and form is None:
            form = value
        elif key == "FORM TYPE" and fallback_form is None:
            fallback_form = value
        elif key == "FILED AS OF DATE" and filed is None:
            filed = _parse_header_date(value)
        elif key == "COMPANY CONFORMED NAME" and company_name is None:
            company_name = value
        elif key == "CENTRAL INDEX KEY" and cik is None:
            if value.isascii() and value.isdigit():
                parsed_cik = int(value)
                if parsed_cik > 0:
                    cik = parsed_cik
        elif key == "FORMER CONFORMED NAME":
            if value not in seen_formers:
                seen_formers.add(value)
                former_names.append(value)

    return FilingHeader(
        form=form if form is not None else fallback_form,
        filed=filed,
        company_name=company_name,
        cik=cik,
        former_names=tuple(former_names),
        items=parse_8k_items(header, filed=filed),
        complete=complete,
    )


def label_items(
    items: frozenset[str],
    *,
    filed: date | datetime | pd.Timestamp | str | None = None,
) -> str:
    """Map 8-K items to bankruptcy, acquisition_type, delisting_notice_only, or other.

    Applies the registered mapping and 2026-11-25 amendment in priority order
    (``bankruptcy`` > ``acquisition_type`` > ``delisting_notice_only`` > ``other``).
    Text labels count in every era by whole normalised text. Numeric item declarations
    count only when ``filed`` is on or after 2004-08-23; undated calls and pre-cutoff
    dates treat numeric codes as ``other``. Numbers embedded inside descriptions are
    never extracted.
    """
    allow_numbers = _numbers_allowed(filed)
    text_tokens: set[str] = set()
    numeric_tokens: set[str] = set()
    for raw in items:
        if not isinstance(raw, str):
            continue
        is_numeric, tokens = _parse_item_declaration(raw)
        if is_numeric:
            if allow_numbers:
                numeric_tokens.update(tokens)
        else:
            text_tokens.update(tokens)

    if (text_tokens & BANKRUPTCY_TEXT_ITEMS) or (numeric_tokens & BANKRUPTCY_NUMBER_ITEMS):
        return LABEL_BANKRUPTCY
    if (text_tokens & ACQUISITION_TEXT_ITEMS) or (numeric_tokens & ACQUISITION_NUMBER_ITEMS):
        return LABEL_ACQUISITION_TYPE
    if (text_tokens & DELISTING_TEXT_ITEMS) or (numeric_tokens & DELISTING_NUMBER_ITEMS):
        return LABEL_DELISTING_NOTICE_ONLY
    return LABEL_OTHER


def label_filing_header(header: FilingHeader) -> str:
    """Classify one parsed filing header using its own ``filed`` date for the era cutoff."""
    return label_items(header.items, filed=header.filed)


def combine_item_labels(labels: Iterable[str]) -> str:
    """Combine per-filing 8-K labels in registered priority order."""
    seen = set(labels)
    if LABEL_BANKRUPTCY in seen:
        return LABEL_BANKRUPTCY
    if LABEL_ACQUISITION_TYPE in seen:
        return LABEL_ACQUISITION_TYPE
    if LABEL_DELISTING_NOTICE_ONLY in seen:
        return LABEL_DELISTING_NOTICE_ONLY
    return LABEL_OTHER


def resolve_confidence(
    match_confidence: str,
    filing: Filing,
    header: FilingHeader | None,
) -> str:
    """Resolve the confidence tier after inspecting a filing header.

    A ``former_name`` match or a match via amendment 3 rules 2 to 4 (``filing.allow_checked``
    is False) is never promoted to ``checked``. A rule-1 ``name_only`` match is promoted to
    ``checked`` only when ``header`` confirms both the same CIK and the same form type as the
    master-index row.
    """
    if match_confidence == CONFIDENCE_FORMER_NAME:
        return CONFIDENCE_FORMER_NAME
    if (
        match_confidence == CONFIDENCE_NAME_ONLY
        and filing.allow_checked
        and header is not None
        and header.cik is not None
        and header.cik == filing.cik
        and header.form is not None
        and header.form.upper() == filing.form.upper()
    ):
        return CONFIDENCE_CHECKED
    return match_confidence


def is_periodic_report_form(form: str) -> bool:
    """Return True if ``form`` is a whitelisted periodic report or NT 10-K/10-Q notice.

    Case-insensitive, whole-string match (after collapsing whitespace) against
    ``PERIODIC_REPORT_FORMS``: ``10-K``, ``10-K405``, ``10-KSB``, ``10-KSB40``, ``10-KT``,
    ``10-K405T``, ``10-Q``, ``10-QSB``, ``10-QT``, ``NT 10-K``, ``NT 10-Q``, ``NT 10-KT``,
    ``NT 10-QT``, each optionally followed by ``/A``. Everything else—including ``NT 10-D``,
    ``NT 11-K``, ``10-12B``/``10-12G``, ``10-C``, ``10-D``, ``11-K``, ownership forms,
    SEC letters, Form 15 variants, 8-K, Form 25, S-x, 424B, and lookalikes—returns False.
    """
    if not isinstance(form, str):
        return False
    normalized = " ".join(form.strip().upper().split())
    return normalized in PERIODIC_REPORT_FORMS


def has_kept_filing(
    filing_dates: Sequence[
        date | datetime | pd.Timestamp | str | tuple[date | datetime | pd.Timestamp | str, str]
    ],
    last_bar: pd.Timestamp | date,
    *,
    after_days: int = AFTER_DAYS,
) -> bool:
    """Return True if any qualifying filing is dated more than ``after_days`` after ``last_bar``.

    Entries may be ``(filed, form)`` pairs—in which case only periodic reports matching
    ``is_periodic_report_form(form)`` count (brief amendment 5)—or bare filing dates already
    filtered by the caller.
    """
    if after_days < 0:
        raise ValueError(f"after_days must be non-negative, got {after_days}")
    cutoff = _as_date(last_bar) + timedelta(days=after_days)
    for entry in filing_dates:
        if isinstance(entry, tuple):
            filed, form = entry
            if not is_periodic_report_form(form):
                continue
            if _as_date(filed) > cutoff:
                return True
        elif _as_date(entry) > cutoff:
            return True
    return False


def classify_company_events(
    filings: Sequence[
        FilingHeader | tuple[date | datetime | pd.Timestamp | str | None, Iterable[str]]
    ],
    *,
    has_form25: bool,
    filing_dates: Sequence[
        date | datetime | pd.Timestamp | str | tuple[date | datetime | pd.Timestamp | str, str]
    ],
    last_bar: pd.Timestamp | date,
    after_days: int = AFTER_DAYS,
    ambiguous_match: bool = False,
    evidence_unresolved: bool = False,
) -> str:
    """Apply the registered per-stock label precedence (amendment 2) using each filing's own date.

    Each entry in ``filings`` is either a ``FilingHeader`` or a ``(filed, items)`` pair;
    its items are classified against its own filing date before combining across filings.
    Order: ``ambiguous_match`` > ``bankruptcy`` > ``evidence_unresolved`` > ``kept_filing``
    (any qualifying periodic report or NT 10-K/10-Q notice strictly > 180 days after
    ``last_bar``; bare dates are accepted as a compatibility input assumed already filtered) >
    ``acquisition_type`` > ``delisting_notice_only`` (Form 25 or delisting 8-K item) >
    ``no_event_found``.
    """
    if ambiguous_match:
        return LABEL_AMBIGUOUS_MATCH

    per_filing_labels: list[str] = []
    for entry in filings:
        if isinstance(entry, FilingHeader):
            per_filing_labels.append(label_filing_header(entry))
        else:
            entry_filed, entry_items = entry
            per_filing_labels.append(label_items(frozenset(entry_items), filed=entry_filed))

    item_label = combine_item_labels(per_filing_labels)
    if item_label == LABEL_BANKRUPTCY:
        return LABEL_BANKRUPTCY
    if evidence_unresolved:
        return LABEL_EVIDENCE_UNRESOLVED
    if has_kept_filing(filing_dates, last_bar, after_days=after_days):
        return LABEL_KEPT_FILING
    if item_label == LABEL_ACQUISITION_TYPE:
        return LABEL_ACQUISITION_TYPE
    if item_label == LABEL_DELISTING_NOTICE_ONLY or has_form25:
        return LABEL_DELISTING_NOTICE_ONLY
    return LABEL_NO_EVENT_FOUND


__all__ = [
    "ACQUISITION_ITEMS",
    "ACQUISITION_NUMBER_ITEMS",
    "ACQUISITION_TEXT_ITEMS",
    "AFTER_DAYS",
    "BANKRUPTCY_ITEMS",
    "BANKRUPTCY_NUMBER_ITEMS",
    "BANKRUPTCY_TEXT_ITEMS",
    "BEFORE_DAYS",
    "CONFIDENCE_CHECKED",
    "CONFIDENCE_FORMER_NAME",
    "CONFIDENCE_NAME_ONLY",
    "CONFIDENCE_RANK",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CONTACT_PATH",
    "DEFAULT_MIN_INTERVAL",
    "DELISTING_ITEMS",
    "DELISTING_NUMBER_ITEMS",
    "DELISTING_TEXT_ITEMS",
    "ENV_CONTACT_PATH",
    "FILING_ARCHIVE_URL",
    "FORM8K_FORMS",
    "FORM25_FORMS",
    "IGNORED_LISTING_FORMS",
    "LABEL_ACQUISITION_TYPE",
    "LABEL_AMBIGUOUS_MATCH",
    "LABEL_BANKRUPTCY",
    "LABEL_DELISTING_NOTICE_ONLY",
    "LABEL_EVIDENCE_UNRESOLVED",
    "LABEL_KEPT_FILING",
    "LABEL_NAME_UNKNOWN",
    "LABEL_NO_COMPANY_MATCH",
    "LABEL_NO_EVENT_FOUND",
    "LABEL_OTHER",
    "MASTER_INDEX_URL",
    "NUMBERED_ITEMS_START",
    "PERIODIC_REPORT_FORMS",
    "RELEVANT_EVENT_FORMS",
    "EarlySecClient",
    "Filing",
    "FilingHeader",
    "IndexRow",
    "classify_company_events",
    "clean_company_name",
    "combine_item_labels",
    "filing_url",
    "find_filings",
    "has_kept_filing",
    "is_periodic_report_form",
    "iter_master_index",
    "label_filing_header",
    "label_items",
    "load_contact_line",
    "master_index_url",
    "parse_8k_items",
    "parse_filing_header",
    "parse_master_index",
    "quarters_covering",
    "quarters_for",
    "resolve_confidence",
]
