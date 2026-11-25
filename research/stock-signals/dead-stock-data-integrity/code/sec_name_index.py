"""Stream all-form company name lookups over cached SEC quarterly master-index archives."""

from __future__ import annotations

import io
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any, BinaryIO

from sec_client import SecClient
from sec_filings import master_index_url

MASTER_INDEX_HEADER = "CIK|Company Name|Form Type|Date Filed|Filename"
MASTER_INDEX_FILENAME = "master.idx"
FILENAME_SUFFIX = ".txt"
MAX_ARCHIVE_BYTES = 30_000_000
MAX_INDEX_BYTES = 64_000_000
MAX_HEADER_LINES = 50
VALID_QUARTERS: frozenset[int] = frozenset({1, 2, 3, 4})


@dataclass(frozen=True, slots=True)
class IndexedFiling:
    """One filing row parsed from an SEC quarterly master-index archive."""

    cik: int
    name: str
    form: str
    filed: date
    accession: str


class MissingArchive(RuntimeError):
    """Raised when a requested quarterly SEC index archive is absent, invalid, or unreadable."""


class _BoundedStream(io.RawIOBase):
    """Wrap a zip member stream so uncompressed bytes cannot silently exceed the ceiling."""

    def __init__(self, raw: BinaryIO, max_bytes: int) -> None:
        self._raw = raw
        self._max_bytes = max_bytes
        self._remaining = max_bytes

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        # Read at most remaining + 1 bytes so crossing the cap fails immediately.
        limit = min(len(buffer), self._remaining + 1)
        chunk = self._raw.read(limit)
        if not chunk:
            return 0
        if len(chunk) > self._remaining:
            raise ValueError(f"Uncompressed master-index archive exceeds {self._max_bytes} bytes")
        self._remaining -= len(chunk)
        buffer[: len(chunk)] = chunk
        return len(chunk)


def _parse_data_line(raw_line: str, line_no: int, date_cache: dict[str, date]) -> IndexedFiling:
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
    name = name_raw.strip()
    if not name:
        raise ValueError(f"Malformed master-index line {line_no}: empty company name")
    form = form_raw.strip()
    if not form:
        raise ValueError(f"Malformed master-index line {line_no}: empty form type")

    filename = filename_raw.strip()
    if not filename:
        raise ValueError(f"Malformed master-index line {line_no}: empty filename")
    basename = filename.rsplit("/", 1)[-1]
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

    # Quarters hold ~65 distinct filing dates across hundreds of thousands of rows; caching valid
    # ISO strings keeps full validation while avoiding redundant parsing on every row.
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

    return IndexedFiling(cik=cik, name=name, form=form, filed=filed, accession=accession)


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


def _iter_all_forms(content: bytes) -> Iterator[IndexedFiling]:
    """Stream every validated filing row from one ``master.zip`` archive."""
    with _open_verified_index(content) as (stream, line_no):
        date_cache: dict[str, date] = {}
        for raw_line in stream:
            line_no += 1
            yield _parse_data_line(raw_line, line_no, date_cache)


def _iter_archive_forms(label: str, content: bytes) -> Iterator[IndexedFiling]:
    """Stream rows from one quarter-labelled archive, wrapping archive errors with the quarter."""
    try:
        yield from _iter_all_forms(content)
    except (ValueError, zipfile.BadZipFile, zlib.error, EOFError, OSError, UnicodeError) as exc:
        raise MissingArchive(f"Invalid SEC archive for {label}: {exc}") from exc


def parse_all_forms(content: bytes) -> list[IndexedFiling]:
    """Parse every filing row from a quarterly SEC ``master.zip`` archive.

    Applies the same archive size ceiling, uncompressed size ceiling, single-member check,
    header check, and dashed separator check as the shared listing parser, while retaining
    every SEC form type.

    Args:
        content: Raw bytes of a quarterly ``master.zip`` archive.

    Returns:
        List of ``IndexedFiling`` records in archive order.

    Raises:
        ValueError: If the archive is oversized, corrupt, truncated, missing the expected
            header/separator, or contains a malformed data line (with the 1-based line number).
    """
    return list(_iter_all_forms(content))


def find_filers(
    archives: Iterable[tuple[str, bytes]],
    matcher: Callable[[str], Mapping[str, frozenset[str]]],
) -> dict[int, dict[str, frozenset[str]]]:
    """Pass 1: stream archives and map each matching CIK to matched roster keys and rule labels.

    Streams one archive at a time and never retains parsed archive rows across iterations.
    ``matcher`` owns all name cleaning, prefix matching, and rule labelling; results are memoized
    per distinct raw company name within one archive only so memory stays bounded.

    Args:
        archives: Iterable of ``(quarter_label, archive_bytes)`` pairs.
        matcher: Callable taking a raw SEC filer name and returning a mapping from roster key
            to the set of rule labels under which that name matched (empty mapping if none).

    Returns:
        Mapping from integer CIK to a mapping from roster key to the union of rule labels across
        all matched index lines for that CIK.

    Raises:
        MissingArchive: If any archive fails size, zip, header, or row validation (with the
            quarter label in the message and the original error chained).
    """
    matches: dict[int, dict[str, frozenset[str]]] = {}
    for label, content in archives:
        # Memoize matcher output within one archive only so memory stays bounded across quarters.
        seen_names: dict[str, tuple[tuple[str, frozenset[str]], ...]] = {}
        for filing in _iter_archive_forms(label, content):
            matched_entries = seen_names.get(filing.name)
            if matched_entries is None:
                raw_match = matcher(filing.name)
                matched_entries = tuple(
                    (roster_key, frozenset(labels))
                    for roster_key, labels in raw_match.items()
                    if labels
                )
                seen_names[filing.name] = matched_entries
            if matched_entries:
                by_key = matches.setdefault(filing.cik, {})
                for roster_key, labels in matched_entries:
                    existing = by_key.get(roster_key)
                    by_key[roster_key] = labels if existing is None else existing | labels
    return matches


def filings_of(
    archives: Iterable[tuple[str, bytes]],
    ciks: set[int],
) -> dict[int, list[IndexedFiling]]:
    """Pass 2: stream archives and collect every filing for ``ciks``, sorted by filing date.

    Callers pass a fresh iterator for ``archives`` so the two-pass workflow never holds more
    than one archive in memory at a time.

    Args:
        archives: Iterable of ``(quarter_label, archive_bytes)`` pairs.
        ciks: Set of integer CIKs whose filings should be retained.

    Returns:
        Mapping from each requested CIK to all of its ``IndexedFiling`` records sorted by
        ``(filed, form, name, accession)``.

    Raises:
        MissingArchive: If any archive fails size, zip, header, or row validation (with the
            quarter label in the message and the original error chained).
    """
    if not ciks:
        return {}

    by_cik: dict[int, list[IndexedFiling]] = {cik: [] for cik in ciks}
    for label, content in archives:
        for filing in _iter_archive_forms(label, content):
            if filing.cik in ciks:
                by_cik[filing.cik].append(filing)

    for filings in by_cik.values():
        filings.sort(
            key=lambda filing: (filing.filed, filing.form, filing.name, filing.accession),
        )
    return by_cik


def cached_archives(
    client: SecClient,
    quarters: Sequence[tuple[int, int]],
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(f"{year}Q{quarter}", bytes)`` from ``client``'s verified cache, never downloading.

    Reads exclusively through ``SecClient._read_verified_cache`` so SHA-256 and URL metadata
    are checked on every entry and no network opener can ever be invoked, then validates the
    archive size ceiling, zip structure, single ``master.idx`` member, header, and separator
    before yielding.

    Args:
        client: ``SecClient`` pointing at the cache directory holding quarterly ``master.zip``
            archives.
        quarters: Sequence of ``(year, quarter)`` tuples as returned by
            ``sec_filings.quarters_covering``.

    Yields:
        ``(quarter_label, archive_bytes)`` for each requested quarter in order.

    Raises:
        ValueError: If any ``(year, quarter)`` pair is invalid.
        MissingArchive: If any quarter is absent from cache, empty, fails checksum/metadata
            verification, exceeds the archive size ceiling, or fails zip/header validation.
    """
    for year, quarter in quarters:
        if year <= 0 or quarter not in VALID_QUARTERS:
            raise ValueError(
                f"Invalid quarter ({year}, {quarter}); expected positive year and quarter in 1..4"
            )
        label = f"{year}Q{quarter}"
        url = master_index_url(year, quarter)
        # Verified cache-only read: checks .meta.json URL and SHA-256 without touching any opener.
        body = client._read_verified_cache(url)
        if body is None or not body:
            raise MissingArchive(f"Cached SEC archive missing or unreadable for {label} ({url})")
        # Refuse oversized or structurally invalid cached bodies before yielding to callers.
        try:
            with _open_verified_index(body):
                pass
        except ValueError as exc:
            raise MissingArchive(f"Cached SEC archive invalid for {label} ({url}): {exc}") from exc
        yield label, body
