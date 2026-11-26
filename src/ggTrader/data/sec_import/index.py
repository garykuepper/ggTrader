"""Import cached SEC quarterly filing indexes."""

import hashlib
import io
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

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


def quarter_of(url: str) -> str:
    """Extract a quarter from a quarterly master index URL."""
    match = re.search(r"/full-index/(\d{4})/QTR([1-4])/master\.zip$", urlsplit(url).path)
    if not match:
        raise ValueError(f"Invalid quarterly index URL: {url}")
    return f"{match[1]}Q{match[2]}"


def read_master_zip(path: Path) -> str:
    """Read the Latin-1 master index inside a cache body."""
    with ZipFile(path) as archive:
        return archive.read("master.idx").decode("latin-1")


def parse_master_idx(text: str, quarter: str) -> tuple[list[tuple], list[tuple[str, dict, str]]]:
    """Parse rows after the separator and retain invalid lines for quarantine."""
    lines = text.splitlines()
    separator = next(
        (i for i, line in enumerate(lines) if re.fullmatch(r"-{3,}", line.strip())), None
    )
    if separator is None:
        raise ValueError(f"Missing index separator for {quarter}")
    rows: list[tuple] = []
    bad: list[tuple[str, dict, str]] = []
    for number, line in enumerate(lines[separator + 1 :], start=separator + 2):
        fields = line.split("|")
        try:
            if len(fields) != 5 or any(not field.strip() for field in fields):
                raise ValueError("expected five nonempty fields")
            cik, company, form, filed, filename = fields
            if not cik.isascii() or not cik.isdigit():
                raise ValueError("invalid CIK")
            cik_value = int(cik)
            if cik_value > 9223372036854775807:
                raise ValueError("CIK exceeds bigint")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", filed):
                raise ValueError("invalid date")
            rows.append((quarter, cik_value, company, form, date.fromisoformat(filed), filename))
        except ValueError as exc:
            bad.append((str(number), {"line": line}, str(exc)))
    return rows, bad


def _copy_escape(value: str) -> str:
    """Escape a value for PostgreSQL COPY text format."""
    return (
        value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    )


def load_index(
    conn: connection,
    entries: Sequence[CacheEntry],
    *,
    backup_dir: Path,
    db_free_path: Path,
    only_quarters: Sequence[str] | None = None,
    margin_bytes: int | None = None,
) -> int:
    """Load selected verified quarterly indexes in resumable transactions."""
    backup_fresh(backup_dir)
    selected: dict[str, CacheEntry] = {}
    for entry in entries:
        if entry.kind != "quarterly_index":
            continue
        quarter = quarter_of(entry.url)
        if only_quarters is not None and quarter not in only_quarters:
            continue
        if quarter in selected:
            raise LoadFailed(f"Duplicate quarter: {quarter}")
        selected[quarter] = entry
    selected = dict(sorted(selected.items()))
    for entry in selected.values():
        if hashlib.sha256(entry.body_path.read_bytes()).hexdigest() != entry.content_sha256:
            raise LoadFailed(f"Hash mismatch: {entry.body_path}")
    options = {} if margin_bytes is None else {"margin_bytes": margin_bytes}
    if not free_space_ok(db_free_path, sum(e.bytes for e in selected.values()), **options):
        raise LoadFailed("Insufficient free space on database drive")
    digest = manifest_sha256(list(selected.values()))
    keys = list(selected)
    load_id = begin_load(
        conn,
        dataset="sec-index",
        kaggle_slug="sec-index",
        file_path=Path("sec-index"),
        file_sha256=digest,
        parser_version="sec-index-1",
        content_sha256=digest,
        expected_chunks=keys,
    )
    record_files(
        conn, load_id, [(str(e.body_path), e.content_sha256, e.bytes) for e in selected.values()]
    )
    for quarter, entry in selected.items():
        rows, bad = parse_master_idx(read_master_zip(entry.body_path), quarter)

        def write(cur: cursor, batch: list[tuple] = rows) -> int:
            buffer = io.StringIO()
            for row in batch:
                buffer.write(
                    "\t".join(_copy_escape(str(value)) for value in (load_id, *row)) + "\n"
                )
            buffer.seek(0)
            cur.copy_expert(
                "COPY sec.index_entries (load_id, quarter, cik, company_name, "
                "form_type, date_filed, filename) FROM STDIN",
                buffer,
            )
            return len(batch)

        commit_chunk(conn, load_id, quarter, len(rows), write, quarantine=bad)
    complete_load(
        conn,
        load_id,
        {},
        expected_chunks=keys,
        coverage=("not_a_stock_source", "SEC quarterly filing index"),
    )
    return load_id
