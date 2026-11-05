#!/usr/bin/env python3
"""Tooling for the docs/research -> research/ reorganization (manifest loading and checking)."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = "docs/research"
NEW_ROOT = "research"
# research/<category>/<study>/<file> is the deepest permitted placement.
MAX_COMPONENTS = 4
FROZEN_EXTENSIONS = frozenset({".json", ".csv", ".txt", ".py", ".tgz"})
ARTIFACT_PREFIX = "artifacts-"
# The three briefs that collide with a same-named report once they share a study folder.
BRIEF_RENAMES = frozenset(
    {
        "2026-10-23-century-leverage.md",
        "2026-09-28-cross-asset-sleeve.md",
        "2026-09-30-static-fractional-leverage.md",
    }
)
BRIEF_SUFFIX = "-brief"
BRIEFS_DIR = f"{OLD_ROOT}/briefs/"
# Hash-pinned by other result files or checked at run time: must move byte-for-byte.
PINNED_PATHS: tuple[str, ...] = tuple(
    f"{OLD_ROOT}/{name}"
    for name in (
        "briefs/2026-10-21-event-timing-audit.md",
        "briefs/2026-10-21-event-timing-audit-sources.md",
        "briefs/2026-10-22-event-timing-source-conflict-proposal.md",
        "event-timing-audit-implementation.md",
        "2026-10-22-event-evidence-resolution-workbook.md",
        "_event_evidence_resolution_workbook.json",
        "_event_audit_metadata_inputs.json",
        "_cost_charter_comparison_results.json",
        "_event_evidence_export_file_search.json",
        "_insider_identity_inventory_resolution_results.json",
        "_insider_identity_inventory_acquisition.json",
        "_production_replica_results.json",
    )
)
MANIFEST_COLUMNS = ("current_path", "proposed_path", "folder", "kind", "notes")


@dataclass(frozen=True)
class ManifestRow:
    """One file move: where it is, where it goes, and whether its bytes must not change."""

    current_path: str
    proposed_path: str
    kind: str
    notes: str
    frozen: bool
    pinned: bool

    @classmethod
    def build(cls, current_path: str, proposed_path: str, kind: str, notes: str) -> ManifestRow:
        """Create a row, computing `pinned` and `frozen` from the paths.

        Args:
            current_path: Repo-relative path under docs/research.
            proposed_path: Repo-relative destination under research/.
            kind: Draft classification (report, result, brief, ...).
            notes: Free-text notes from the manifest.

        Returns:
            The row with `pinned` and `frozen` derived from `current_path`.
        """
        return cls(
            current_path=current_path,
            proposed_path=proposed_path,
            kind=kind,
            notes=notes,
            frozen=is_frozen(current_path),
            pinned=is_pinned(current_path),
        )


def is_pinned(current_path: str) -> bool:
    """Return True when the path is one of the twelve hash-pinned files."""
    return current_path in PINNED_PATHS


def is_frozen(current_path: str) -> bool:
    """Return True when a file must be moved byte-for-byte unchanged.

    Frozen files are the pinned twelve, every machine artifact by extension, and anything
    inside an `artifacts-*` package.
    """
    path = PurePosixPath(current_path)
    return (
        is_pinned(current_path)
        or path.suffix in FROZEN_EXTENSIONS
        or any(part.startswith(ARTIFACT_PREFIX) for part in path.parts[:-1])
    )


def load_manifest(path: Path) -> list[ManifestRow]:
    """Load a manifest CSV (columns: current_path, proposed_path, folder, kind, notes).

    Args:
        path: CSV file to read.

    Returns:
        One row per CSV record, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return [
            ManifestRow.build(r["current_path"], r["proposed_path"], r["kind"], r["notes"])
            for r in reader
        ]


def _basename_problem(row: ManifestRow) -> str | None:
    """Return a problem message when the basename changes in a way the layout does not allow."""
    old = PurePosixPath(row.current_path).name
    new = PurePosixPath(row.proposed_path).name
    if old == new:
        return None
    if (
        old in BRIEF_RENAMES
        and row.current_path.startswith(BRIEFS_DIR)
        and new == f"{PurePosixPath(old).stem}{BRIEF_SUFFIX}.md"
    ):
        return None
    return f"basename changed without permission: {row.current_path} -> {row.proposed_path}"


def check_manifest(rows: list[ManifestRow], tracked: set[str]) -> list[str]:
    """Check a manifest against the tracked file list and the layout rules.

    Args:
        rows: Parsed manifest rows.
        tracked: Every tracked file under docs/research (repo-relative, POSIX separators).

    Returns:
        Problem descriptions; an empty list means the manifest passes.
    """
    problems: list[str] = []
    seen_current: set[str] = set()
    seen_proposed: dict[str, str] = {}
    for row in rows:
        if not row.current_path.startswith(f"{OLD_ROOT}/"):
            problems.append(f"current_path outside docs/research: {row.current_path}")
        if row.current_path in seen_current:
            problems.append(f"duplicate current_path: {row.current_path}")
        seen_current.add(row.current_path)
        if row.current_path not in tracked and row.current_path.startswith(f"{OLD_ROOT}/"):
            problems.append(f"current_path is not tracked: {row.current_path}")
        if row.proposed_path in seen_proposed:
            problems.append(
                f"duplicate proposed_path {row.proposed_path} "
                f"({seen_proposed[row.proposed_path]} and {row.current_path})"
            )
        seen_proposed[row.proposed_path] = row.current_path
        if not row.proposed_path.startswith(f"{NEW_ROOT}/"):
            problems.append(f"proposed_path does not start with research/: {row.proposed_path}")
        if len(PurePosixPath(row.proposed_path).parts) > MAX_COMPONENTS:
            problems.append(f"depth exceeds research/<category>/<study>/: {row.proposed_path}")
        basename = _basename_problem(row)
        if basename:
            problems.append(basename)
    problems.extend(
        f"tracked file not in manifest: {path}" for path in sorted(tracked - seen_current)
    )
    return problems


def tracked_research_files(root: Path = REPO_ROOT) -> set[str]:
    """List tracked files under docs/research using read-only git.

    Args:
        root: Repository root to run git in.

    Returns:
        Repo-relative POSIX paths.

    Raises:
        subprocess.CalledProcessError: If git fails.
    """
    out = subprocess.run(
        ["git", "ls-files", OLD_ROOT],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line for line in out.splitlines() if line}


def _cmd_manifest_check(args: argparse.Namespace) -> int:
    """Run `manifest-check`: print problems and return 1 if any."""
    rows = load_manifest(Path(args.manifest))
    problems = check_manifest(rows, tracked_research_files())
    for problem in problems:
        print(f"PROBLEM {problem}")
    if problems:
        print(f"FAIL {len(problems)} problems in {len(rows)} rows")
        return 1
    print(f"OK {len(rows)} rows")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (later tasks add subcommands here)."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "manifest-check", help="Check a manifest CSV against git ls-files docs/research."
    )
    check.add_argument("--manifest", required=True, help="Path to the manifest CSV.")
    check.set_defaults(func=_cmd_manifest_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
