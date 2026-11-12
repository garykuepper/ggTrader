"""Hash records for frozen inputs: describe a file, verify a set of files, hash a value."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ggTrader.utils.json_safe import canonical_hash as canonical_hash


class FrozenInputError(ValueError):
    """A frozen input is missing or no longer matches its recorded hash."""


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's bytes (streamed, so large files are fine)."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    """Describe one file as ``{"path", "sha256", "bytes"}`` for a manifest."""
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def verify_frozen(expected: Mapping[str, str], root: str | Path) -> None:
    """Check that every file under ``root`` still has its recorded SHA-256.

    Args:
        expected: relative file name to the hex SHA-256 it was frozen with.
        root: the folder the names are relative to.

    Raises:
        FrozenInputError: naming every file that is missing or differs.
    """
    base = Path(root)
    problems: list[str] = []
    for name, want in sorted(expected.items()):
        path = base / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif sha256_file(path) != want:
            problems.append(f"{name}: hash differs from the frozen record")
    if problems:
        raise FrozenInputError("frozen inputs changed: " + "; ".join(problems))
