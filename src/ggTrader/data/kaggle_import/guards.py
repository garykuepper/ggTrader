"""Preflight checks for imports."""

import hashlib
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

_REDIRECT_KEYS = frozenset(
    {
        "dbname",
        "database",
        "host",
        "hostaddr",
        "port",
        "user",
        "service",
        "options",
        "target_session_attrs",
    }
)


def safe_error(exc: Exception) -> str:
    """Describe an exception while redacting connection strings and fields."""
    message = re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://\S+", "[connection redacted]", str(exc))
    message = re.sub(
        r"(?i)\b(?:password|passwd|pwd|user|host|hostaddr|port|dbname|database)\s*=\s*(?:'[^']*'|\"[^\"]*\"|\S+)",
        "[connection field redacted]",
        message,
    )
    return f"{type(exc).__name__}: {message}"


def checked_pg_url(url: str, *, expect_db: str | None = None) -> str:
    """Reject connection URLs whose options could redirect a named database target."""
    if not isinstance(url, str) or any(ch.isspace() or ord(ch) < 32 for ch in url) or "#" in url:
        raise ValueError("Invalid PostgreSQL connection target")
    try:
        parts = urlsplit(url)
        database = unquote(parts.path.removeprefix("/"))
        options = parse_qsl(parts.query, keep_blank_values=True)
        _ = parts.port
    except (TypeError, ValueError):
        raise ValueError("Invalid PostgreSQL connection target") from None
    if (
        parts.scheme not in ("postgresql", "postgres")
        or not parts.path.startswith("/")
        or not database
        or "/" in database
        or "\x00" in database
        or any(key.lower() in _REDIRECT_KEYS for key, _ in options)
    ):
        raise ValueError("Invalid or redirecting PostgreSQL connection target")
    if expect_db is not None and database != expect_db:
        raise ValueError("PostgreSQL database target does not match the expected database")
    return url


def assert_connected_database(conn: Any, expected: str) -> str:
    """Refuse to proceed unless the server confirms the expected database."""
    with conn.cursor() as cur:
        cur.execute("select current_database()")
        row = cur.fetchone()
    if row is None or row[0] != expected:
        raise ValueError("Connected database does not match the expected database")
    return str(row[0])


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
    size = newest.stat().st_size
    if size < 1024**2:
        raise BackupStale(f"Backup {newest} is too small: {size} bytes")
    return newest
