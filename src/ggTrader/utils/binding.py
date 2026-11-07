"""Bind a result to the code that made it: git commit, dirty flag, diff hash, lock hash."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

from ggTrader.utils.paths import REPO_ROOT

LOCK_FILE = "uv.lock"


def _git(root: Path, *args: str) -> str | None:
    """Run git in ``root``; ``None`` when git is missing, fails or times out."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_sha(root: Path = REPO_ROOT) -> str | None:
    """The HEAD commit hash, or ``None`` outside a git checkout."""
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else None


def git_dirty(root: Path = REPO_ROOT) -> bool:
    """True when tracked files differ from HEAD or untracked files exist."""
    return bool(_git(root, "status", "--porcelain"))


def diff_sha256(root: Path = REPO_ROOT) -> str | None:
    """SHA-256 of ``git diff HEAD`` (tracked changes), or ``None`` when there are none."""
    out = _git(root, "diff", "HEAD")
    return hashlib.sha256(out.encode("utf-8")).hexdigest() if out else None


def lock_sha256(root: Path = REPO_ROOT) -> str | None:
    """SHA-256 of ``uv.lock``, or ``None`` when the file is missing."""
    try:
        return hashlib.sha256((root / LOCK_FILE).read_bytes()).hexdigest()
    except OSError:
        return None


def binding(root: Path = REPO_ROOT) -> dict[str, Any]:
    """The code binding of a result: where the numbers came from, without importing the engine."""
    return {
        "git_sha": git_sha(root),
        "git_dirty": git_dirty(root),
        "diff_sha256": diff_sha256(root),
        "lock_sha256": lock_sha256(root),
        "python": platform.python_version(),
    }
