"""Parse and load headers from verified SEC filing bodies."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from psycopg2.extensions import connection, cursor

from ggTrader.data.kaggle_import.guards import backup_fresh, free_space_ok
from ggTrader.data.kaggle_import.loadlog import (
    LoadFailed,
    begin_load,
    commit_chunk,
    complete_load,
    record_files,
)
from ggTrader.data.sec_import.cache import CacheEntry, manifest_sha256

CHUNK_SIZE = 200


class HeaderIncomplete(ValueError):
    """A filing header is missing required or valid fields."""


@dataclass(frozen=True)
class HeaderRow:
    """One company in a filing header."""

    accession: str
    cik: int
    form_type: str
    filed_date: date
    items: tuple[str, ...]


def read_header_text(path: Path) -> str:
    """Read only the header, decoding the cached bytes as Latin-1."""
    lines = []
    with path.open("rb") as source:
        for line in source:
            lines.append(line.decode("latin-1"))
            if line.strip() == b"</SEC-HEADER>":
                return "".join(lines)
    raise HeaderIncomplete("Missing closing SEC header line")


def parse_header(text: str) -> list[HeaderRow]:
    """Extract required filing fields and unique company keys."""
    fields: dict[str, list[str]] = {}
    closed = False
    # Newline-only split: latin-1 text can hold characters that str.splitlines() treats as breaks.
    if "\x00" in text:
        raise HeaderIncomplete("NUL character in header")
    for line in text.split("\n"):
        if line.strip() == "</SEC-HEADER>":
            closed = True
            break
        key, sep, value = line.strip().partition(":")
        if sep:
            fields.setdefault(key, []).append(value.strip())
    try:
        if not closed:
            raise ValueError("Missing closing SEC header line")
        accession = fields["ACCESSION NUMBER"][0]
        form = fields["CONFORMED SUBMISSION TYPE"][0]
        filed = fields["FILED AS OF DATE"][0]
        if not accession or not form or not re.fullmatch(r"\d{8}", filed):
            raise ValueError("Missing or invalid accession, form or date")
        filed_date = date(int(filed[:4]), int(filed[4:6]), int(filed[6:]))
        keys = fields["CENTRAL INDEX KEY"]
        if not keys:
            raise ValueError("Missing CIK")
        ciks = []
        for key in keys:
            if not key.isascii() or not key.isdigit() or int(key) > 9223372036854775807:
                raise ValueError("Invalid CIK")
            if int(key) not in ciks:
                ciks.append(int(key))
        items = tuple(item for item in fields.get("ITEM INFORMATION", []) if item)
        return [HeaderRow(accession, cik, form, filed_date, items) for cik in ciks]
    except (KeyError, IndexError, ValueError) as exc:
        raise HeaderIncomplete(f"Incomplete header: {exc}") from exc


def load_headers(
    conn: connection,
    entries: Sequence[CacheEntry],
    *,
    backup_dir: Path,
    db_free_path: Path,
    margin_bytes: int | None = None,
) -> int:
    """Load verified filing headers in resumable chunks."""
    backup_fresh(backup_dir)
    selected = sorted((e for e in entries if e.kind == "filing_text"), key=lambda e: e.url_hash)
    if not selected:
        raise LoadFailed("No filing text entries found; check the --folder paths")
    if len({e.url_hash for e in selected}) != len(selected):
        raise LoadFailed("Duplicate cache entries; merge the folders first")
    for entry in selected:
        with entry.body_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != entry.content_sha256:
            raise LoadFailed(f"Hash mismatch: {entry.body_path}")
    options = {} if margin_bytes is None else {"margin_bytes": margin_bytes}
    if not free_space_ok(db_free_path, sum(e.bytes for e in selected), **options):
        raise LoadFailed("Insufficient free space on database drive")
    digest = manifest_sha256(selected)
    chunks = [selected[i : i + CHUNK_SIZE] for i in range(0, len(selected), CHUNK_SIZE)]
    keys = [f"headers-{i:04d}" for i in range(len(chunks))]
    load_id = begin_load(
        conn,
        dataset="sec-headers",
        kaggle_slug="sec-headers",
        file_path=Path("sec-headers"),
        file_sha256=digest,
        parser_version="sec-headers-1",
        content_sha256=digest,
        expected_chunks=keys,
        derivation_identity={"chunk_size": CHUNK_SIZE},
    )
    record_files(conn, load_id, [(str(e.body_path), e.content_sha256, e.bytes) for e in selected])
    # The same filing can be cached under two companies' folders. Identical copies are kept once;
    # copies that disagree stop the load, because one of them would be silently wrong.
    seen: dict[tuple[str, int], HeaderRow] = {}
    skipped = quarantined = 0
    for key, chunk in zip(keys, chunks, strict=True):
        rows: list[HeaderRow] = []
        bad: list[tuple[str, dict, str]] = []
        for entry in chunk:
            try:
                parsed = parse_header(read_header_text(entry.body_path))
            except (HeaderIncomplete, UnicodeError) as exc:
                path = str(entry.body_path)
                bad.append((path, {"path": path, "reason": str(exc)}, str(exc)))
                continue
            for row in parsed:
                earlier = seen.setdefault((row.accession, row.cik), row)
                if earlier != row:
                    raise LoadFailed(
                        f"Filing {row.accession} for CIK {row.cik} differs between cache files"
                    )
                if earlier is row:
                    rows.append(row)
                else:
                    skipped += 1
        quarantined += len(bad)

        def write(cur: cursor, batch: list[HeaderRow] = rows) -> int:
            cur.executemany(
                "INSERT INTO sec.filing_headers "
                "(load_id, accession, cik, form_type, filed_date, items) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                [
                    (load_id, r.accession, r.cik, r.form_type, r.filed_date, list(r.items))
                    for r in batch
                ],
            )
            return len(batch)

        commit_chunk(conn, load_id, key, len(rows), write, quarantine=bad)
    complete_load(
        conn,
        load_id,
        {
            "bodies_total": len(selected),
            "bodies_quarantined": quarantined,
            "duplicate_rows_skipped": skipped,
        },
        expected_chunks=keys,
        coverage=("not_a_stock_source", "SEC filing headers"),
    )
    return load_id
