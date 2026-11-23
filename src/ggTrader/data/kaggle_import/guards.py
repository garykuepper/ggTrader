"""Preflight checks for imports."""

import hashlib
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


class HashMismatch(Exception):
    """The input does not match its recorded hash."""


class HashMissing(HashMismatch):
    """The input has no recorded hash."""


class BackupStale(Exception):
    """No sufficiently recent backup exists."""


def verify_sha256(path: Path, sums_file: Path) -> str:
    """Verify a file against its relative path or bare name in SHA256SUMS."""
    entries = {}
    for line in sums_file.read_text().splitlines():
        parts = line.split("  ", 1)
        if len(parts) == 2:
            entries[parts[1]] = parts[0]
    resolved = path.resolve()
    expected = next(
        (
            digest
            for name, digest in entries.items()
            if (sums_file.parent / name).resolve() == resolved
        ),
        None,
    )
    if expected is None:
        expected = entries.get(path.name)
    if expected is None:
        raise HashMissing(f"No SHA-256 recorded for {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    actual = hasher.hexdigest()
    if actual.lower() != expected.lower():
        raise HashMismatch(f"{path}: expected {expected}, got {actual}")
    return actual


def free_space_ok(path: Path, needed_bytes: int, margin_bytes: int = 40 * 1024**3) -> bool:
    """Check projected free space."""
    return shutil.disk_usage(path).free - needed_bytes >= margin_bytes


def backup_fresh(backup_dir: Path, max_age_days: int = 7, now: datetime | None = None) -> Path:
    """Return the newest recent dump."""
    dumps = list(backup_dir.glob("*.dump"))
    if not dumps:
        raise BackupStale(f"No backup dump in {backup_dir}")
    newest = max(dumps, key=lambda item: item.stat().st_mtime)
    current = now if now is not None else datetime.now(timezone.utc)
    modified = datetime.fromtimestamp(newest.stat().st_mtime, tz=current.tzinfo)
    if current - modified > timedelta(days=max_age_days):
        raise BackupStale(f"Backup {newest} is stale")
    return newest
