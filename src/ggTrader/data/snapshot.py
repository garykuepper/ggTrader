"""Read and verify the frozen Parquet tape snapshot against ``data/snapshots/MANIFEST.json``.

The snapshot is the default research source for the vectorbt engine rebuild (spec
docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md, sections 5.7 and 7.3): a
long-format dump of the ``ohlcv`` table (naive-UTC midnight stamps, as stored) written by
``scripts/make_tape_snapshot.py``. This module only reads. It refuses to hand out a file
whose sha256, row count or date range disagrees with the manifest, so a corrupted or
silently replaced tape cannot feed a cited result.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
MANIFEST_NAME = "MANIFEST.json"
#: Environment variable that points research at another snapshot file.
SNAPSHOT_ENV_VAR = "GGT_SNAPSHOT"
FIELDS = ("open", "high", "low", "close", "volume")
_HASH_BLOCK = 1 << 20


class SnapshotIntegrityError(RuntimeError):
    """The snapshot file does not match its manifest entry (or has none)."""


@dataclass(frozen=True)
class SnapshotInfo:
    """What was verified about a snapshot file; written into run JSON as provenance."""

    path: Path
    sha256: str
    content_sha256: Optional[str]
    rows: int
    symbols: int
    date_min: str
    date_max: str
    db_session_timezone: Optional[str]
    deep_checked: bool

    def as_dict(self) -> dict[str, Any]:
        """JSON-able form for run provenance."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "content_sha256": self.content_sha256,
            "rows": self.rows,
            "symbols": self.symbols,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "deep_checked": self.deep_checked,
        }


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(flat: pd.DataFrame) -> str:
    """Writer-independent hash of the data (sorted rows, little-endian float64 bytes).

    Must stay byte-compatible with ``content_sha256`` in ``scripts/make_tape_snapshot.py``,
    which produced the manifest value (a test checks the real tape against it).
    """
    flat = flat.sort_values(["symbol", "timestamp"], kind="mergesort")
    digest = hashlib.sha256()
    digest.update("\n".join(flat["symbol"].astype(str)).encode())
    digest.update(flat["timestamp"].to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes())
    for col in FIELDS:
        digest.update(flat[col].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


def read_manifest(directory: Path = DEFAULT_SNAPSHOT_DIR) -> dict[str, Any]:
    """Parse ``MANIFEST.json`` in ``directory``; raises ``FileNotFoundError`` if absent."""
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"snapshot manifest not found: {path}")
    return json.loads(path.read_text())


def manifest_entry(path: Path) -> dict[str, Any]:
    """The manifest entry for the parquet ``path`` (matched on file name, manifest beside it)."""
    path = Path(path)
    manifest = read_manifest(path.parent)
    for entry in manifest.get("snapshots", []):
        if Path(entry["path"]).name == path.name:
            return entry
    raise SnapshotIntegrityError(
        f"no manifest entry for {path.name} in {path.parent}/{MANIFEST_NAME}"
    )


def default_snapshot_path(directory: Path = DEFAULT_SNAPSHOT_DIR) -> Path:
    """The default research snapshot: ``$GGT_SNAPSHOT`` if set, else the manifest's last entry."""
    override = os.environ.get(SNAPSHOT_ENV_VAR)
    if override:
        return Path(override)
    manifest = read_manifest(directory)
    entries = manifest.get("snapshots", [])
    if not entries:
        raise SnapshotIntegrityError(f"{MANIFEST_NAME} lists no snapshots")
    return Path(directory) / Path(entries[-1]["path"]).name


def _check(label: str, actual: Any, expected: Any, path: Path) -> None:
    if actual != expected:
        raise SnapshotIntegrityError(
            f"{path.name}: {label} mismatch (file {actual!r}, manifest {expected!r}); "
            "refusing to use this snapshot"
        )


def load_snapshot_long(
    path: Optional[Path] = None,
    symbols: Optional[Iterable[str]] = None,
    *,
    deep: bool = False,
) -> tuple[pd.DataFrame, SnapshotInfo]:
    """Verify then read a snapshot as long rows, optionally filtered to ``symbols``.

    Checks, in order: file sha256 against the manifest, then (after reading) row count and
    date range. ``deep=True`` also recomputes ``content_sha256`` (slower: ~1 s). The symbol
    filter is applied only after verification, so the manifest always describes the whole file.

    Returns the long frame (``timestamp`` naive UTC midnight, ``symbol``, open/high/low/close/
    volume as float64) and a :class:`SnapshotInfo`. Raises :class:`SnapshotIntegrityError` on any
    disagreement; never repairs or falls back to another file.
    """
    path = default_snapshot_path() if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}")
    entry = manifest_entry(path)

    actual_sha = sha256_file(path)
    _check("sha256", actual_sha, entry["sha256"], path)

    flat = pd.read_parquet(path)
    flat["timestamp"] = pd.DatetimeIndex(flat["timestamp"]).astype("datetime64[ns]")
    _check("rows", len(flat), entry["rows"], path)
    _check("date_min", str(flat["timestamp"].min().date()), entry["date_min"], path)
    _check("date_max", str(flat["timestamp"].max().date()), entry["date_max"], path)
    if "symbols" in entry:
        _check("symbols", int(flat["symbol"].nunique()), entry["symbols"], path)

    content_hash = None
    if deep:
        content_hash = content_sha256(flat)
        _check("content_sha256", content_hash, entry["content_sha256"], path)

    info = SnapshotInfo(
        path=path,
        sha256=actual_sha,
        content_sha256=content_hash,
        rows=len(flat),
        symbols=int(flat["symbol"].nunique()),
        date_min=entry["date_min"],
        date_max=entry["date_max"],
        db_session_timezone=entry.get("db_session_timezone"),
        deep_checked=deep,
    )
    if symbols is not None:
        flat = flat[flat["symbol"].isin(set(symbols))].reset_index(drop=True)
    return flat, info
