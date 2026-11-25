"""Stream all-form company name lookups over cached SEC quarterly master-index archives."""

from __future__ import annotations

import io
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, BinaryIO

from sec_client import SecClient
from sec_filings import master_index_url, normalize_name

MASTER_INDEX_HEADER = "CIK|Company Name|Form Type|Date Filed|Filename"
MASTER_INDEX_FILENAME = "master.idx"
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


class MissingArchive(RuntimeError):
    """Raised when a requested quarterly SEC index archive is absent or unreadable in cache."""


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
    if not filename_raw.strip():
        raise ValueError(f"Malformed master-index line {line_no}: empty filename")

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

    return IndexedFiling(cik=cik, name=name, form=form, filed=filed)


def _iter_all_forms(content: bytes) -> Iterator[IndexedFiling]:
    """Stream every validated filing row from one ``master.zip`` archive."""
    if not isinstance(content, bytes) or len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Invalid master-index archive")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            info = archive.infolist()
            if (
                len(info) != 1
                or info[0].filename != MASTER_INDEX_FILENAME
                or info[0].file_size > MAX_INDEX_BYTES
            ):
                raise ValueError("Invalid master-index archive contents")
            with archive.open(info[0]) as raw:
                bounded = io.BufferedReader(_BoundedStream(raw, MAX_INDEX_BYTES))
                stream = io.TextIOWrapper(bounded, encoding="utf-8-sig", newline="")
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

                date_cache: dict[str, date] = {}
                for raw_line in stream:
                    line_no += 1
                    yield _parse_data_line(raw_line, line_no, date_cache)
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError, UnicodeError) as exc:
        raise ValueError("Invalid master-index archive") from exc


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
    wanted: Mapping[str, str],
    normalizers: Sequence[Callable[[str], str]] = (normalize_name,),
) -> dict[int, set[str]]:
    """Pass 1: stream archives and map each matching filer CIK to the roster keys it matched.

    Streams one archive at a time and never retains parsed archive rows across iterations. A
    filer row matches when any function in ``normalizers`` maps its company name to a key in
    ``wanted``.

    Args:
        archives: Iterable of ``(quarter_label, archive_bytes)`` pairs.
        wanted: Mapping from normalized company name to the roster key that requested it.
        normalizers: Name-cleaning callables applied to each filer's company name.

    Returns:
        Mapping from integer CIK to the set of matched roster keys.

    Raises:
        ValueError: If ``normalizers`` is empty or any archive fails validation.
    """
    if not normalizers:
        raise ValueError("normalizers must contain at least one callable")
    if not wanted:
        return {}

    matches: dict[int, set[str]] = {}
    for _label, content in archives:
        # Memoize name normalization within one archive only so memory stays bounded.
        seen_names: dict[str, tuple[str, ...]] = {}
        for filing in _iter_all_forms(content):
            matched_keys = seen_names.get(filing.name)
            if matched_keys is None:
                found: list[str] = []
                for normalize in normalizers:
                    norm = normalize(filing.name)
                    if norm and norm in wanted:
                        found.append(wanted[norm])
                matched_keys = tuple(found)
                seen_names[filing.name] = matched_keys
            if matched_keys:
                matches.setdefault(filing.cik, set()).update(matched_keys)
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
        ``(filed, form, name)``.

    Raises:
        ValueError: If any archive fails validation.
    """
    if not ciks:
        return {}

    by_cik: dict[int, list[IndexedFiling]] = {cik: [] for cik in ciks}
    for _label, content in archives:
        for filing in _iter_all_forms(content):
            if filing.cik in ciks:
                by_cik[filing.cik].append(filing)

    for filings in by_cik.values():
        filings.sort(key=lambda filing: (filing.filed, filing.form, filing.name))
    return by_cik


def cached_archives(
    client: SecClient,
    quarters: Sequence[tuple[int, int]],
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(f"{year}Q{quarter}", bytes)`` from ``client``'s verified cache, never downloading.

    Reads exclusively through ``SecClient._read_verified_cache`` so SHA-256 and URL metadata
    are checked on every entry and no network opener can ever be invoked.

    Args:
        client: ``SecClient`` pointing at the cache directory holding quarterly ``master.zip``
            archives.
        quarters: Sequence of ``(year, quarter)`` tuples as returned by
            ``sec_filings.quarters_covering``.

    Yields:
        ``(quarter_label, archive_bytes)`` for each requested quarter in order.

    Raises:
        ValueError: If any ``(year, quarter)`` pair is invalid.
        MissingArchive: If any quarter is absent from cache, empty, or fails checksum/metadata
            verification (for example a cut-short body).
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
        yield label, body
