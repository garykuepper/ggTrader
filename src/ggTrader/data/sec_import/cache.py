"""Read and register existing SEC download caches without changing them."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from psycopg2.extensions import connection, cursor

from ggTrader.data.kaggle_import.guards import backup_fresh, free_space_ok
from ggTrader.data.kaggle_import.loadlog import (
    LoadFailed,
    begin_load,
    commit_chunk,
    complete_load,
    record_files,
)

CACHE_NAME = re.compile(r"^([0-9a-f]{64})\.(body|meta\.json)$")
CHUNK_SIZE = 200


class CacheCorrupt(Exception):
    """A cache pair is incomplete or does not match its metadata."""


class CacheConflict(Exception):
    """Two caches contain different content for the same URL."""


@dataclass(frozen=True)
class CacheEntry:
    """A verified cached response."""

    url_hash: str
    url: str
    content_sha256: str
    fetched_at: datetime
    status: int
    bytes: int
    body_path: Path
    kind: str


def classify_url(url: str) -> str:
    """Classify the source URL by its SEC path."""
    parts = urlsplit(url)
    path = parts.path.lower()
    if "/full-index/" in path and path.endswith("/master.zip"):
        return "quarterly_index"
    if (
        parts.hostname == "data.sec.gov"
        and path.startswith("/submissions/cik")
        and path.endswith(".json")
    ):
        return "submission_list"
    if "/archives/edgar/data/" in path and path.endswith("/"):
        return "directory_listing"
    if path.endswith(".xml"):
        return "form25_xml"
    if "/archives/edgar/data/" in path and path.endswith(".txt"):
        return "filing_text"
    return "other"


def stray_files(folder: Path) -> list[str]:
    """List names outside the cache naming convention."""
    return sorted(path.name for path in folder.iterdir() if not CACHE_NAME.fullmatch(path.name))


def scan_cache(folder: Path) -> list[CacheEntry]:
    """Verify every cache pair and return its metadata."""
    pairs: dict[str, set[str]] = {}
    for path in folder.iterdir():
        match = CACHE_NAME.fullmatch(path.name)
        if match:
            pairs.setdefault(match[1], set()).add(match[2])
    entries = []
    for stem, extensions in sorted(pairs.items()):
        body = folder / f"{stem}.body"
        meta = folder / f"{stem}.meta.json"
        if extensions != {"body", "meta.json"}:
            raise CacheCorrupt(str(body if "body" in extensions else meta))
        try:
            with meta.open("r", encoding="utf-8") as source:
                note = json.load(source)
            with body.open("rb") as source:
                content = source.read()
            url = note["url"]
            status = note["status"]
            if not isinstance(url, str) or isinstance(status, bool) or not isinstance(status, int):
                raise ValueError("url must be text and status an integer")
            digest = hashlib.sha256(content).hexdigest()
            fetched = datetime.fromisoformat(note["fetched_at"])
            if (
                hashlib.sha256(url.encode("utf-8")).hexdigest() != stem
                or digest != note["sha256"]
                or fetched.utcoffset() is None
            ):
                raise ValueError("hash or timestamp mismatch")
            entries.append(
                CacheEntry(
                    stem,
                    url,
                    digest,
                    fetched,
                    status,
                    len(content),
                    body,
                    classify_url(url),
                )
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise CacheCorrupt(str(body)) from exc
    return entries


def merge_caches(*groups: list[CacheEntry]) -> list[CacheEntry]:
    """Deduplicate URLs, rejecting different bodies for the same URL."""
    merged: dict[str, CacheEntry] = {}
    for group in groups:
        for entry in group:
            previous = merged.setdefault(entry.url_hash, entry)
            if previous.content_sha256 != entry.content_sha256:
                raise CacheConflict(entry.url_hash)
    return sorted(merged.values(), key=lambda entry: entry.url_hash)


def manifest_sha256(entries: Sequence[CacheEntry]) -> str:
    """Hash the sorted URL and content hash pairs."""
    manifest = "\n".join(sorted(f"{entry.url_hash}:{entry.content_sha256}" for entry in entries))
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def load_cache(
    conn: connection,
    folders: Sequence[Path],
    *,
    backup_dir: Path,
    db_free_path: Path,
    margin_bytes: int | None = None,
) -> int:
    """Register verified responses in resumable chunks, then publish coverage."""
    backup_fresh(backup_dir)
    folders = [Path(folder).resolve() for folder in folders]
    groups = [scan_cache(folder) for folder in folders]
    strays = {str(folder): stray_files(folder) for folder in folders}
    entries = merge_caches(*groups)
    if not entries:
        raise LoadFailed("No cache entries found in the folders; check the --folder paths")
    needed = sum(entry.bytes for entry in entries)
    options = {} if margin_bytes is None else {"margin_bytes": margin_bytes}
    if not free_space_ok(db_free_path, needed, **options):
        raise LoadFailed("Insufficient free space on database drive")
    digest = manifest_sha256(entries)
    # Resuming must see the same folders in the same order, the same chunk layout and the same
    # fetch notes, none of which the body hashes alone can show.
    notes = "\n".join(
        f"{index}:{e.url_hash}:{e.fetched_at.isoformat()}:{e.status}"
        for index, group in enumerate(groups)
        for e in group
    )
    identity = {
        "folders": [str(folder) for folder in folders],
        "chunk_size": CHUNK_SIZE,
        "notes_sha256": hashlib.sha256(notes.encode("utf-8")).hexdigest(),
    }
    chunks = [entries[i : i + CHUNK_SIZE] for i in range(0, len(entries), CHUNK_SIZE)]
    keys = [f"fetched-{index:04d}" for index in range(len(chunks))]
    load_id = begin_load(
        conn,
        dataset="sec-cache",
        kaggle_slug="sec-cache",
        file_path=Path("sec-cache"),
        file_sha256=digest,
        parser_version="sec-cache-1",
        content_sha256=digest,
        expected_chunks=keys,
        derivation_identity=identity,
    )
    record_files(
        conn,
        load_id,
        [(str(e.body_path), e.content_sha256, e.bytes) for group in groups for e in group],
    )
    for key, chunk in zip(keys, chunks, strict=True):

        def write(cur: cursor, batch: list[CacheEntry] = chunk) -> int:
            cur.executemany(
                "INSERT INTO sec.fetched_files "
                "(load_id,url_hash,url,content_sha256,fetched_at,status,bytes,path,kind) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (
                        load_id,
                        e.url_hash,
                        e.url,
                        e.content_sha256,
                        e.fetched_at,
                        e.status,
                        e.bytes,
                        str(e.body_path),
                        e.kind,
                    )
                    for e in batch
                ],
            )
            return len(batch)

        commit_chunk(conn, load_id, key, len(chunk), write)
    complete_load(
        conn,
        load_id,
        {"stray_files": strays},
        expected_chunks=keys,
        coverage=("not_a_stock_source", "SEC download cache, not a stock source"),
    )
    return load_id
