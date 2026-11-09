"""``ggt db membership export``: a restorable dump of schema ``universe`` (schema and data).

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, section 8 and Appendix B11.
``pg_dump --schema=universe --format=custom`` captures the tables, the append-only triggers, the
functions and the grants that a per-table dump would miss. Restore is: create the extension
(``CREATE EXTENSION btree_gist`` on the target, which the dump does not carry), then
``pg_restore``. The file goes to ``data/exports/universe/`` (git-ignored through ``data/exports/``;
dump files are never committed).

The database password is handed to ``pg_dump`` through ``PGPASSWORD`` in the child's environment,
never on the command line, and nothing here prints or logs a connection string: error text from
``pg_dump`` is passed through :func:`ggTrader.data.membership.redact` first. The dump is a read
(``pg_dump`` takes ``ACCESS SHARE`` locks only), so it may run against the live database.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ggTrader.data.membership import redact
from ggTrader.data.universe_db import loader_url
from ggTrader.utils.paths import REPO_ROOT

#: Default folder for dumps; ``data/exports/`` is git-ignored.
EXPORT_DIR = REPO_ROOT / "data" / "exports" / "universe"
#: The schema that is dumped (nothing else is).
EXPORT_SCHEMA = "universe"
#: Seconds to wait for ``pg_dump`` (the schema is a few MB at most).
DUMP_TIMEOUT_SECONDS = 300
#: How long ``pg_dump`` may wait for a table lock before giving up.
LOCK_WAIT_TIMEOUT = "30s"
#: Hash read size in bytes.
_CHUNK = 1 << 20

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class ExportError(RuntimeError):
    """The dump could not be made; the message never contains the connection string."""


@dataclass(frozen=True)
class ExportResult:
    """The finished dump: where it is, how big, and its sha256."""

    path: Path
    size_bytes: int
    sha256: str


def dump_filename(now: datetime) -> str:
    """``universe_YYYYMMDDTHHMMSSZ.dump`` for an aware instant (the name sorts by time)."""
    if now.tzinfo is None:
        raise ValueError("the export timestamp must be timezone-aware")
    return f"universe_{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.dump"


def dump_command(
    url: str, out_file: Path, *, pg_dump: str = "pg_dump"
) -> tuple[list[str], dict[str, str]]:
    """``(argv, extra_environment)`` for ``pg_dump``; the password only ever goes in the latter."""
    parts = urlparse(url)
    dbname = unquote(parts.path.lstrip("/"))
    if not dbname:
        raise ExportError("the database URL names no database")
    argv = [
        pg_dump,
        "--format=custom",
        f"--schema={EXPORT_SCHEMA}",
        f"--lock-wait-timeout={LOCK_WAIT_TIMEOUT}",
        f"--file={out_file}",
    ]
    if parts.hostname:
        argv.append(f"--host={parts.hostname}")
    if parts.port:
        argv.append(f"--port={parts.port}")
    if parts.username:
        argv.append(f"--username={unquote(parts.username)}")
    argv.append(f"--dbname={dbname}")
    env: dict[str, str] = {}
    if parts.password:
        env["PGPASSWORD"] = unquote(parts.password)
    sslmode = parse_qs(parts.query).get("sslmode")
    if sslmode:
        env["PGSSLMODE"] = sslmode[-1]
    return argv, env


def file_sha256(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def export_universe(
    url: str | None = None,
    *,
    out_dir: Path = EXPORT_DIR,
    now: datetime | None = None,
    runner: Runner = subprocess.run,
    pg_dump: str = "pg_dump",
) -> ExportResult:
    """Dump schema ``universe`` to a new timestamped file and return its path, size and sha256.

    Refuses to overwrite an existing file. The dump is written to a ``.partial`` name and renamed
    only when ``pg_dump`` succeeded and produced a non-empty file, so a failed run leaves nothing
    that looks like a backup. ``runner`` is ``subprocess.run``-compatible (tests inject one).
    """
    target_url = url or loader_url()
    instant = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / dump_filename(instant)
    if final.exists():
        raise ExportError(f"{final} already exists; wait a second or choose another --out-dir")
    partial = final.with_name(final.name + ".partial")
    argv, extra_env = dump_command(target_url, partial, pg_dump=pg_dump)
    try:
        proc = runner(
            argv,
            env={**os.environ, **extra_env},
            capture_output=True,
            text=True,
            check=False,
            timeout=DUMP_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        raise ExportError(f"{pg_dump!r} is not installed or not on PATH") from None
    except subprocess.TimeoutExpired:
        partial.unlink(missing_ok=True)
        raise ExportError(f"pg_dump did not finish in {DUMP_TIMEOUT_SECONDS} s") from None
    if proc.returncode != 0:
        partial.unlink(missing_ok=True)
        lines = [ln for ln in (proc.stderr or "").strip().splitlines() if ln.strip()]
        message = redact(" | ".join(lines[:3]) or "no error text", target_url)
        raise ExportError(f"pg_dump failed with exit {proc.returncode}: {message}")
    if not partial.exists() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise ExportError("pg_dump reported success but wrote no data")
    partial.rename(final)
    return ExportResult(final, final.stat().st_size, file_sha256(final))
