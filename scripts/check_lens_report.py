#!/usr/bin/env python3
"""Check an investigation lens report for required sections, real paths and banned phrases."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_HEADINGS = (
    "Falsifiers",
    "Steelman",
    "Questions and answers",
    "Verification",
    "Done checklist",
)
BANNED_PHRASES = (
    "retire the sleeve",
    "shut down the project",
    "end paper trading",
    "should deploy",
    "we recommend deploying",
    "open an IRA",
)
REPO_PATH_RE = re.compile(r"`((?:docs|src|scripts|tests|results)/[^`\s]*)`")
LINE_SUFFIX_RE = re.compile(r":\d+(?:-\d+)?$")
GLOB_CHARS = set("*?[]{}")
H2_RE = re.compile(r"^##\s+(.*?)\s*$")


def _section_bodies(lines: list[str]) -> dict[str, str]:
    """Map each level-2 heading text to the text beneath it."""
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        match = H2_RE.match(line)
        if match:
            current = match.group(1)
            bodies.setdefault(current, [])
        elif current is not None:
            bodies[current].append(line)
    return {name: "\n".join(body) for name, body in bodies.items()}


def _check_headings(lines: list[str]) -> list[str]:
    """Report missing or empty required sections."""
    problems: list[str] = []
    bodies = _section_bodies(lines)
    for name in REQUIRED_HEADINGS:
        if name not in bodies:
            problems.append(f"missing heading: ## {name}")
        elif not bodies[name].strip():
            problems.append(f"empty section: ## {name}")
    return problems


def _check_paths(text: str, repo_root: Path) -> list[str]:
    """Report backticked repo-relative paths that do not exist under repo_root."""
    problems: list[str] = []
    seen: set[str] = set()
    for raw in REPO_PATH_RE.findall(text):
        if "://" in raw or GLOB_CHARS & set(raw):
            continue
        rel = LINE_SUFFIX_RE.sub("", raw)
        if rel in seen:
            continue
        seen.add(rel)
        if not (repo_root / rel).exists():
            problems.append(f"cited path does not exist: {rel}")
    return problems


def _check_banned(lines: list[str]) -> list[str]:
    """Report banned phrases, skipping blockquote lines that quote owner instructions."""
    problems: list[str] = []
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith(">"):
            continue
        lowered = line.lower()
        for phrase in BANNED_PHRASES:
            if phrase.lower() in lowered:
                problems.append(f"line {number}: banned phrase: {phrase}")
    return problems


def check_report(path: Path, repo_root: Path, paths_only: bool = False) -> list[str]:
    """Validate a lens report and return its problems (empty means pass).

    Args:
        path: Markdown report to check.
        repo_root: Repository root that cited paths are resolved against.
        paths_only: Skip heading rules; check only cited paths and banned phrases.

    Returns:
        One human-readable string per problem found.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    problems: list[str] = []
    if not paths_only:
        problems += _check_headings(lines)
    problems += _check_paths(text, Path(repo_root))
    problems += _check_banned(lines)
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the checker from the command line; exit 0 on pass, 1 on any problem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="lens report markdown file")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root for resolving cited paths (default: this repo)",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="skip heading rules; check only cited paths and banned phrases",
    )
    args = parser.parse_args(argv)
    problems = check_report(args.report, args.repo_root, paths_only=args.paths_only)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
