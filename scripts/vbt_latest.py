"""Pick the vectorbt release the upgrade-watch workflow should try, or print nothing.

The newest non-pre-release, non-yanked ``vectorbt`` that (a) is newer than the version in
``uv.lock``, (b) was uploaded at least ``--cooldown`` days ago, (c) is newer than the
highest version in the rejected file (rejecting the newest release of a burst also rules out its
older members; ``--force`` still names any version) and (d) has a ``vectorbt-rust`` release with
the same version (the pair is atomic).
Prints ``version=<x>`` (empty when there is nothing to try) so the workflow can append it
to ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.request
from collections.abc import Collection, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REJECTED_FILE = REPO_ROOT / "docs" / "vbt_upgrade_rejected.txt"
PYPI_URL = "https://pypi.org/pypi/{name}/json"
_PLAIN_VERSION = re.compile(r"^\d+(\.\d+)*$")


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def locked_version(lock_path: Path, package: str = "vectorbt") -> str:
    """The version of ``package`` recorded in a uv.lock file."""
    try:
        data = tomllib.loads(Path(lock_path).read_text())
    except OSError as exc:
        raise ValueError(f"cannot read {lock_path}: {exc}") from exc
    for entry in data.get("package", []):
        if entry["name"] == package:
            return str(entry["version"])
    raise ValueError(f"{package} is not in {lock_path}")


def read_rejected(path: Path) -> set[str]:
    """Versions the owner or an agent REJECTED (one per line, ``#`` starts a comment)."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return set()
    return {line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()}


def _uploaded(files: list[dict[str, Any]]) -> datetime | None:
    """Earliest upload time of a release's files, ``None`` if every file is yanked."""
    times = [
        datetime.fromisoformat(f["upload_time_iso_8601"].replace("Z", "+00:00"))
        for f in files
        if not f.get("yanked")
    ]
    return min(times) if times else None


def pick_version(
    vbt_releases: Mapping[str, list[dict[str, Any]]],
    rust_versions: Collection[str],
    locked: str,
    *,
    now: datetime,
    cooldown_days: int,
    force: str = "",
    rejected: Collection[str] = (),
) -> str:
    """The vectorbt version to try, or ``""``. See the module docstring for the rules.

    Raises:
        ValueError: ``force`` names a version that is not on PyPI or has no vectorbt-rust.
    """
    if force:
        if force not in vbt_releases:
            raise ValueError(f"{force} is not on PyPI")
        if force not in rust_versions:
            raise ValueError(f"vectorbt-rust {force} does not exist; the pair is atomic")
        return force
    cutoff = now - timedelta(days=cooldown_days)
    # P8 R12: a rejected newest release must not make the next older one the proposal.
    floor = max((_key(v) for v in rejected if _PLAIN_VERSION.match(v)), default=())
    candidates = []
    for version, files in vbt_releases.items():
        if not _PLAIN_VERSION.match(version) or _key(version) <= _key(locked):
            continue
        if _key(version) <= floor:
            continue
        uploaded = _uploaded(files)
        if uploaded is None or uploaded > cutoff or version not in rust_versions:
            continue
        candidates.append(version)
    return max(candidates, key=_key, default="")


def _fetch(name: str) -> dict[str, Any]:
    with urllib.request.urlopen(PYPI_URL.format(name=name), timeout=30) as response:  # noqa: S310
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=REPO_ROOT / "uv.lock", help="uv.lock path")
    parser.add_argument("--cooldown", type=int, default=3, help="minimum release age in days")
    parser.add_argument(
        "--rejected-file",
        type=Path,
        default=REJECTED_FILE,
        help="versions already REJECTED, one per line (skipped unless --force)",
    )
    parser.add_argument("--force", default="", help="try this version, skipping the cooldown")
    args = parser.parse_args(argv)
    try:
        version = pick_version(
            _fetch("vectorbt")["releases"],
            set(_fetch("vectorbt-rust")["releases"]),
            locked_version(args.lock),
            now=datetime.now(UTC),
            cooldown_days=args.cooldown,
            force=args.force.strip(),
            rejected=read_rejected(args.rejected_file),
        )
    except (ValueError, OSError) as exc:
        print(f"vbt_latest failed: {exc}", file=sys.stderr)
        return 1
    print(f"version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
