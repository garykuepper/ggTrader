#!/usr/bin/env python3
"""Tooling for the docs/research -> research/ reorganization (manifest, ledger, rewrite, move)."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = "docs/research"
NEW_ROOT = "research"
# research/<category>/<study>/<file> is the deepest permitted placement.
MAX_COMPONENTS = 4
# A large study folder may group its files one level deeper (spec addendum 2026-11-05b).
EXTRA_COMPONENTS = 1
# Study folders (relative to research/) that use that extra level.
DEFAULT_ALLOW_DEEPER: tuple[str, ...] = ("audits/root-cause-2026-11-04",)
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
HASH_CHUNK_BYTES = 1 << 20
# Where consumers of docs/research live; docs/ is scanned minus the research tree itself.
INBOUND_SCAN_ROOTS: tuple[str, ...] = ("scripts", "src", "tests", ".claude", "docs")
INBOUND_SCAN_FILES: tuple[str, ...] = ("AGENTS.md", "README.md")
# The reorganization's own spec/plan/manifest/draft/ledger files all carry this name prefix.
REORG_DOC_PREFIX = "2026-11-05-research-reorg"
REORG_SELF_PATHS = frozenset({"scripts/research_reorg.py", "tests/scripts/test_research_reorg.py"})
SKIP_DIR_NAMES = frozenset({"__pycache__", ".git", ".venv", "node_modules"})
# Git-ignored skill eval scratch: never in a rewrite scope, and absent from a clean checkout.
SKIP_DIR_SUFFIX = "-workspace"
# A Markdown inline link target, with an optional "title" part.
LINK_PATTERN = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")

# --- Reference rewriter constants ---
SCOPES: tuple[str, ...] = ("moved-markdown", "fixed-markdown", "code")
CODE_SCOPE_DIRS: tuple[str, ...] = ("scripts/", "src/", "tests/", ".claude/")
CODE_SCOPE_FILES = frozenset({"AGENTS.md"})
# Never rewritten in any scope: these describe the old paths on purpose.
REWRITE_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "docs/superpowers/plans/2026-11-05-research-reorg",
    "docs/superpowers/plans/2026-11-05-research-reorganization",
    "docs/superpowers/specs/2026-11-05-research-reorganization",
)
# Never rewritten either: byte-restored so its sha256 equals the digest a frozen workbook recorded.
REWRITE_EXCLUDED_PATHS = frozenset(
    {"docs/superpowers/plans/2026-10-22-event-evidence-resolution.md"}
)
# Characters a repo path is made of; a token is the longest run of these.
PATH_CHAR_CLASS = r"A-Za-z0-9_\-./"
PATH_RUN = re.compile(rf"[{PATH_CHAR_CLASS}]*")
# An old-tree path token in code; the lookbehind stops a match inside a longer word.
CODE_TOKEN = re.compile(rf"(?<![A-Za-z0-9_\-])docs/research[{PATH_CHAR_CLASS}]*")
# A whole code span that is a citation: docs/research/<path> with an optional line suffix,
# which may list several ranges (`:12`, `:12-20`, `:27-42, 360-403`).
CITATION = re.compile(
    rf"^(docs/research/[{PATH_CHAR_CLASS}]+?)(:\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)?$"
)
# Inline link or image target: `](target "title")`; only the target group is rewritten.
# The target may hold one level of balanced parentheses; the title may be "", '' or ().
INLINE_LINK = re.compile(
    r"\]\((?P<lead>[ \t]*)(?P<target><[^<>\n]*>|(?:[^\s()<>]|\([^\s()<>]*\))+)"
    r"(?P<title>[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?(?P<trail>[ \t]*)\)"
)
# Reference-style link definition `[label]: target "title"` (footnotes `[^n]:` excluded).
REF_DEFINITION = re.compile(
    r"^(?P<indent> {0,3})\[(?!\^)(?P<label>[^\]\n]+)\]:[ \t]*(?P<target><[^<>\n]*>|\S+)"
    r"(?P<rest>(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?[ \t]*\r?)$"
)
# Any link opener `](` with a rough target, to report forms INLINE_LINK cannot parse.
LINK_OPENER = re.compile(r"\]\(\s*<?([^\s)>]*)")
# A fence opener: three or more backticks or tildes, optionally indented (list items).
FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
GLOB_CHARS = frozenset("*?[")
# `<slug>` and `{name}` placeholders: a pattern, not a path.
TEMPLATE_CHARS = frozenset("<{")
FORMAT_NEXT_CHARS = frozenset("{%")
FSTRING_START = re.compile(r"(?<![A-Za-z0-9_])(?:[rRbB]?[fF]|[fF][rRbB])[\"']")
# `"docs" / "research"`, `Path("docs") / "research"`, `os.path.join("docs", "research")`.
SPLIT_CONSTRUCTION = re.compile(r"""["']docs["']\s*\)?\s*[,/]\s*["']research["']""")
PARENT_ARITHMETIC = re.compile(r"\.parents?\b")
SNIPPET_MAX = 200
SNIPPET_BEFORE = 80
REASON_DIRECTORY = "directory citation"
REASON_GLOB = "glob citation"
REASON_NOT_IN_MANIFEST = "not in manifest"
REASON_PROSE = "prose mention"
REASON_FORMAT = "format string"
REASON_BASE_PATH = "base path"
REASON_SPLIT = "split path construction"
REASON_PARENT = "parent arithmetic"
REASON_FENCED = "fenced code"
REASON_UNSUPPORTED = "unsupported link form"
STATUS_OK = "ok"
STATUS_FAILED = "failed"

# --- Move and verify constants ---
PYCACHE_DIR = "__pycache__"
VERIFY_STAGES: tuple[str, ...] = ("moved", "rewritten", "final")
# The rewrite manifest Task 9 wrote; its presence means the tree is past the move-only step.
DEFAULT_REWRITE_MANIFEST = "docs/superpowers/plans/2026-11-05-research-reorg-rewrite-manifest.json"
# New (non-manifest) files `verify --stage final` accepts under research/, relative to research/
# and matched with fnmatch (so `*` also crosses `/`). Later tasks add their own via --allow-new.
DEFAULT_ALLOW_NEW: tuple[str, ...] = (
    "README.md",
    "MOVED.csv",
    "audits/root-cause-2026-11-04/README.md",
    "audits/root-cause-2026-11-04/reference-sheet.md",
    "audits/root-cause-2026-11-04/findings/2026-11-05-findings-plain-english.md",
    "audits/root-cause-2026-11-04/reviews/2026-11-05-direction-reviews-consolidated.md",
)
REWRITABLE_SUFFIX = ".md"


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


def _depth_limit(path: str, allow_deeper: tuple[str, ...]) -> int:
    """Return the most path components a repo-relative path under research/ may have."""
    inside = path.removeprefix(f"{NEW_ROOT}/")
    if path.startswith(f"{NEW_ROOT}/") and any(
        inside.startswith(f"{prefix.strip('/')}/") for prefix in allow_deeper
    ):
        return MAX_COMPONENTS + EXTRA_COMPONENTS
    return MAX_COMPONENTS


def check_manifest(
    rows: list[ManifestRow],
    tracked: set[str],
    allow_deeper: tuple[str, ...] = DEFAULT_ALLOW_DEEPER,
) -> list[str]:
    """Check a manifest against the tracked file list and the layout rules.

    Args:
        rows: Parsed manifest rows.
        tracked: Every tracked file under docs/research (repo-relative, POSIX separators).
        allow_deeper: Study folders (relative to research/) whose files may sit one level
            deeper, in a sub-folder.

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
        if len(PurePosixPath(row.proposed_path).parts) > _depth_limit(
            row.proposed_path, allow_deeper
        ):
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
    return tracked_files_under(OLD_ROOT, root)


def tracked_files_under(prefix: str, root: Path = REPO_ROOT) -> set[str]:
    """List files git tracks (index) under `prefix` using read-only git.

    Args:
        prefix: Repo-relative folder, e.g. `research`.
        root: Repository root to run git in.

    Returns:
        Repo-relative POSIX paths.

    Raises:
        subprocess.CalledProcessError: If git fails.
    """
    out = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line for line in out.splitlines() if line}


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reorg_own_file(rel: str) -> bool:
    """Return True for the reorganization's own files, which mention docs/research by design."""
    return rel in REORG_SELF_PATHS or PurePosixPath(rel).name.startswith(REORG_DOC_PREFIX)


def _iter_scan_files(root: Path) -> list[str]:
    """List repo-relative paths of candidate consumer files, sorted, outside the research tree."""
    found: set[str] = {name for name in INBOUND_SCAN_FILES if (root / name).is_file()}
    for top in INBOUND_SCAN_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            dirs = path.relative_to(root).parts[:-1]
            if SKIP_DIR_NAMES.intersection(dirs) or any(d.endswith(SKIP_DIR_SUFFIX) for d in dirs):
                continue
            rel = path.relative_to(root).as_posix()
            if rel.startswith(f"{OLD_ROOT}/") or _is_reorg_own_file(rel):
                continue
            found.add(rel)
    return sorted(found)


def scan_inbound_references(root: Path) -> dict[str, int]:
    """Count literal `docs/research` occurrences in files outside the research tree.

    Read-only text scan; binary or unreadable files are skipped.

    Args:
        root: Repository root.

    Returns:
        Mapping of repo-relative file to occurrence count (files with none are omitted).
    """
    counts: dict[str, int] = {}
    for rel in _iter_scan_files(root):
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits = text.count(OLD_ROOT)
        if hits:
            counts[rel] = hits
    return counts


def find_dead_links(rows: list[ManifestRow], root: Path) -> list[dict[str, Any]]:
    """List relative Markdown links in manifest files whose target does not exist.

    URLs, mailto/other schemes and pure `#anchors` are ignored, `#fragment` is stripped, and
    fenced code blocks are skipped.

    Args:
        rows: Manifest rows; only `.md` files are inspected.
        root: Repository root.

    Returns:
        `{file, line, link}` records in manifest then line order.
    """
    dead: list[dict[str, Any]] = []
    for row in rows:
        if not row.current_path.endswith(".md"):
            continue
        try:
            lines = (root / row.current_path).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        parent = PurePosixPath(row.current_path).parent.as_posix()
        in_fence = False
        for number, line in enumerate(lines, start=1):
            if FENCE_PATTERN.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for match in LINK_PATTERN.finditer(line):
                link = match.group(1)
                target = unquote(link.split("#", 1)[0])
                if not target or URL_SCHEME.match(link):
                    continue
                resolved = (
                    target.lstrip("/")
                    if target.startswith("/")
                    else posixpath.normpath(posixpath.join(parent, target))
                )
                if not (root / resolved).exists():
                    dead.append({"file": row.current_path, "line": number, "link": link})
    return dead


def build_ledger(rows: list[ManifestRow], root: Path, git_head: str) -> dict[str, Any]:
    """Record the pre-move state so a later step can prove nothing changed.

    Args:
        rows: Manifest rows, in manifest order.
        root: Repository root the rows' current paths resolve against.
        git_head: Revision the ledger was taken at.

    Returns:
        A JSON-serializable dict: `source_revision`, `files`, `inbound_references`,
        `existing_dead_links`.

    Raises:
        FileNotFoundError: If a manifest file is missing on disk.
    """
    files = []
    for row in rows:
        path = root / row.current_path
        files.append(
            {
                "current_path": row.current_path,
                "proposed_path": row.proposed_path,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "frozen": row.frozen,
                "pinned": row.pinned,
            }
        )
    return {
        "source_revision": git_head,
        "files": files,
        "inbound_references": scan_inbound_references(root),
        "existing_dead_links": find_dead_links(rows, root),
    }


@dataclass(frozen=True)
class Unresolved:
    """An old-tree mention the rewriter could not map safely; left unchanged for a person."""

    file: str
    line: int
    text: str
    reason: str


def _snippet(line: str, start: int) -> str:
    """Return the line (or a window around `start` in a long line) for a report entry."""
    clean = line.rstrip("\r")
    if len(clean.strip()) <= SNIPPET_MAX:
        return clean.strip()
    lo = max(0, start - SNIPPET_BEFORE)
    return f"...{clean[lo : lo + SNIPPET_MAX]}..."


def _key_directories(mapping: dict[str, str]) -> frozenset[str]:
    """Return every ancestor directory of a mapping key (e.g. docs/research/briefs)."""
    dirs: set[str] = set()
    for key in mapping:
        parent = posixpath.dirname(key)
        while parent:
            dirs.add(parent)
            parent = posixpath.dirname(parent)
    return frozenset(dirs)


def _path_reason(core: str, key_dirs: frozenset[str]) -> str:
    """Classify an unmapped old-tree path as a directory citation or a missing file."""
    if core.endswith("/") or core.rstrip("/") in key_dirs:
        return REASON_DIRECTORY
    return REASON_NOT_IN_MANIFEST


def _classify(
    line: str, start: int, mapping: dict[str, str], key_dirs: frozenset[str], code: bool
) -> str:
    """Give the reason an old-tree mention at `line[start:]` was left unresolved.

    Args:
        line: The line holding the mention.
        start: Index of the literal `docs/research` in `line`.
        mapping: Old path to new path.
        key_dirs: Ancestor directories of the mapping keys.
        code: True for code files, which add format-string and base-path reasons.

    Returns:
        One of the REASON_* strings.
    """
    end = PATH_RUN.match(line, start + len(OLD_ROOT)).end()  # type: ignore[union-attr]
    core = line[start:end].rstrip(".")
    nxt = line[end : end + 1]
    if code and nxt in FORMAT_NEXT_CHARS and nxt:
        if FSTRING_START.search(line) or ".format(" in line or nxt == "%" or core in mapping:
            return REASON_FORMAT
    if nxt and (nxt in GLOB_CHARS or nxt in TEMPLATE_CHARS):
        return REASON_GLOB
    if core in mapping:
        return REASON_PROSE
    if code and core == OLD_ROOT and line[start - 1 : start] in {'"', "'"} and nxt in {'"', "'"}:
        return REASON_BASE_PATH
    return _path_reason(core, key_dirs)


def _old_mentions(line: str) -> list[int]:
    """Return the start index of every literal `docs/research` in a line."""
    return [m.start() for m in re.finditer(re.escape(OLD_ROOT), line)]


def _fence_opener(line: str) -> tuple[str, int] | None:
    """Return (fence char, length) when the line opens a fenced code block."""
    match = FENCE_OPEN.match(line)
    if not match:
        return None
    run = match.group(1)
    # A backtick fence's info string cannot contain a backtick (that is inline code).
    if run[0] == "`" and "`" in match.group(2):
        return None
    return run[0], len(run)


def _fence_closes(line: str, fence: tuple[str, int]) -> bool:
    """Return True when the line closes `fence`: same char, at least as long, nothing after."""
    char, length = fence
    return re.match(rf"^[ \t]*{re.escape(char)}{{{length},}}[ \t]*\r?$", line) is not None


def _code_span_segments(line: str) -> list[tuple[bool, str]]:
    """Split a line into (is_code_span, text) pieces; an unmatched backtick run is plain text."""
    pieces: list[tuple[bool, str]] = []
    plain_start = 0
    i = 0
    while i < len(line):
        if line[i] != "`":
            i += 1
            continue
        j = i
        while j < len(line) and line[j] == "`":
            j += 1
        width = j - i
        k = j
        close = -1
        while k < len(line):
            if line[k] != "`":
                k += 1
                continue
            m = k
            while m < len(line) and line[m] == "`":
                m += 1
            if m - k == width:
                close = k
                break
            k = m
        if close < 0:
            i = j
            continue
        if plain_start < i:
            pieces.append((False, line[plain_start:i]))
        pieces.append((True, line[i : close + width]))
        i = close + width
        plain_start = i
    if plain_start < len(line):
        pieces.append((False, line[plain_start:]))
    return pieces


def _rewrite_citation(span: str, mapping: dict[str, str]) -> str:
    """Rewrite a code span that is exactly a mapped `docs/research/...` citation."""
    width = len(span) - len(span.lstrip("`"))
    inner = span[width:-width]
    match = CITATION.match(inner.strip())
    if not match or match.group(1) not in mapping:
        return span
    new_inner = inner.replace(match.group(0), mapping[match.group(1)] + (match.group(2) or ""), 1)
    return span[:width] + new_inner + span[-width:]


def _escapes_root(path: str) -> bool:
    """Return True for a normalized path that leaves the repository root."""
    return path == ".." or path.startswith("../")


def _rewrite_link_target(
    raw: str,
    old_dir: str,
    new_dir: str,
    mapping: dict[str, str],
    key_dirs: frozenset[str],
    exists: Callable[[str], bool] | None,
) -> tuple[str, str | None]:
    """Recompute one link target for the file's new directory.

    Args:
        raw: The target as written (may be `<...>`, may carry `#fragment`).
        old_dir: Directory of the file before the move (repo-relative, "" for the root).
        new_dir: Directory of the file after the move.
        mapping: Old path to new path.
        key_dirs: Ancestor directories of the mapping keys.
        exists: Optional repo-relative existence check, used to recognise a link that
            was already rewritten for the new directory (makes a rerun a no-op).

    Returns:
        (new target, unresolved reason or None). The target is `raw` when nothing changes.
    """
    angle = raw.startswith("<") and raw.endswith(">")
    target = raw[1:-1] if angle else raw
    if not target or target.startswith("#") or URL_SCHEME.match(target):
        return raw, None
    path, sep, fragment = target.partition("#")
    path, qsep, query = path.partition("?")
    if not path:
        return raw, None
    lookup = unquote(path)
    absolute = lookup.startswith("/")
    resolved = posixpath.normpath(
        lookup.lstrip("/") if absolute else posixpath.join(old_dir, lookup)
    )
    mapped = resolved in mapping
    under_old = resolved == OLD_ROOT or resolved.startswith(f"{OLD_ROOT}/")
    if not mapped and under_old:
        # Unmapped old-tree target: report it and keep the raw text, so no new
        # `docs/research` literal is ever introduced by a recomputed link.
        return raw, _path_reason(resolved, key_dirs)
    if not mapped and old_dir == new_dir:
        return raw, None
    if not mapped and exists is not None and not absolute:
        here = posixpath.normpath(posixpath.join(new_dir, lookup))
        old_ok = not _escapes_root(resolved) and exists(resolved)
        if not old_ok and not _escapes_root(here) and exists(here):
            return raw, None  # already written relative to the new directory
    dest = mapping[resolved] if mapped else resolved
    new_path = "/" + dest if absolute else posixpath.relpath(dest, new_dir or ".")
    if path.endswith("/") and not new_path.endswith("/"):
        new_path += "/"
    if lookup != path:
        new_path = quote(new_path, safe="/-._~")
    new_target = new_path + (qsep + query) + (sep + fragment if sep else "")
    return (f"<{new_target}>" if angle else new_target), None


def _unsupported_links(
    piece: str,
    parsed_starts: set[int],
    old_dir: str,
    new_dir: str,
) -> list[int]:
    """Return positions of `](` link openers INLINE_LINK could not parse but that matter.

    A form matters when its rough target resolves under the old tree, or when the file moves
    and the target is a relative path (it would need recomputing).
    """
    found: list[int] = []
    for match in LINK_OPENER.finditer(piece):
        if match.start() in parsed_starts:
            continue
        target = match.group(1)
        if not target or target.startswith("#") or URL_SCHEME.match(target):
            continue
        path = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not path:
            continue
        absolute = path.startswith("/")
        resolved = posixpath.normpath(
            path.lstrip("/") if absolute else posixpath.join(old_dir, path)
        )
        under_old = resolved == OLD_ROOT or resolved.startswith(f"{OLD_ROOT}/")
        if under_old or OLD_ROOT in target or (old_dir != new_dir and not absolute):
            found.append(match.start())
    return found


def _rewrite_markdown_counted(
    text: str,
    old_file: str,
    new_file: str,
    mapping: dict[str, str],
    exists: Callable[[str], bool] | None = None,
) -> tuple[str, list[Unresolved], int]:
    """Implement `rewrite_markdown`, also returning the number of replacements made."""
    key_dirs = _key_directories(mapping)
    old_dir = posixpath.dirname(old_file)
    new_dir = posixpath.dirname(new_file)
    out: list[str] = []
    unresolved: list[Unresolved] = []
    count = 0
    fence: tuple[str, int] | None = None

    def _note(number: int, line: str, start: int, reason: str) -> None:
        unresolved.append(Unresolved("", number, _snippet(line, start), reason))

    def _swap_target(match: re.Match[str], number: int, line: str) -> str:
        """Return the match text with its `target` group recomputed (or unchanged)."""
        nonlocal count
        raw = match.group("target")
        new_raw, reason = _rewrite_link_target(raw, old_dir, new_dir, mapping, key_dirs, exists)
        # A literal old path in `raw` is reported by the final scan; avoid a duplicate.
        if reason is not None and OLD_ROOT not in raw:
            _note(number, line, match.start(), reason)
        if new_raw == raw:
            return match.group(0)
        count += 1
        start, end = match.span("target")
        base = match.start()
        whole = match.group(0)
        return whole[: start - base] + new_raw + whole[end - base :]

    for number, line in enumerate(text.split("\n"), start=1):
        if fence is None:
            fence = _fence_opener(line)
            opened = fence is not None
        else:
            opened = False
        if fence is not None:
            # Fenced code is never modified, but old-tree mentions are listed for review.
            for start in _old_mentions(line):
                _note(number, line, start, REASON_FENCED)
            if not opened and _fence_closes(line, fence):
                fence = None
            out.append(line)
            continue
        definition = REF_DEFINITION.match(line)
        if definition:
            new_line = _swap_target(definition, number, line)
        else:
            pieces: list[str] = []
            for is_code, piece in _code_span_segments(line):
                if is_code:
                    new_piece = _rewrite_citation(piece, mapping)
                    count += new_piece != piece
                    pieces.append(new_piece)
                    continue
                parsed = {m.start() for m in INLINE_LINK.finditer(piece)}
                for start in _unsupported_links(piece, parsed, old_dir, new_dir):
                    _note(number, line, start, REASON_UNSUPPORTED)
                pieces.append(
                    INLINE_LINK.sub(lambda m, n=number, ln=line: _swap_target(m, n, ln), piece)
                )
            new_line = "".join(pieces)
        for start in _old_mentions(new_line):
            reason = _classify(new_line, start, mapping, key_dirs, code=False)
            _note(number, new_line, start, reason)
        out.append(new_line)
    return "\n".join(out), unresolved, count


def rewrite_markdown(
    text: str,
    old_file: str,
    new_file: str,
    mapping: dict[str, str],
    *,
    exists: Callable[[str], bool] | None = None,
) -> tuple[str, list[Unresolved]]:
    """Rewrite old-tree references in one Markdown file for its (possibly new) location.

    Backticked `docs/research/X[:N[-M]]` citations whose X is mapped become the new path
    (suffix kept). Inline link targets are resolved against `old_file`'s directory, mapped
    when the target moved, and recomputed relative to `new_file`'s directory, keeping
    `#fragment` and title. URLs, `mailto:`, pure anchors, bare filenames and fenced code are
    untouched. Every `docs/research` still present outside fences is reported.

    Args:
        text: File contents (line endings are preserved).
        old_file: Repo-relative path the file had when its links were written.
        new_file: Repo-relative path the file has after the move (same as old if not moved).
        mapping: Old repo-relative path to new repo-relative path.
        exists: Optional repo-relative existence check (see `_rewrite_link_target`).

    Returns:
        (new text, unresolved mentions with an empty `file`).
    """
    new_text, unresolved, _ = _rewrite_markdown_counted(text, old_file, new_file, mapping, exists)
    return new_text, unresolved


def _rewrite_code_counted(text: str, mapping: dict[str, str]) -> tuple[str, list[Unresolved], int]:
    """Implement `rewrite_code`, also returning the number of replacements made."""
    key_dirs = _key_directories(mapping)
    out: list[str] = []
    unresolved: list[Unresolved] = []
    count = 0
    for number, line in enumerate(text.split("\n"), start=1):

        def _token(match: re.Match[str]) -> str:
            nonlocal count
            token = match.group(0)
            core = token.rstrip(".")
            # A mapped path followed by `{` or `%` continues at run time: flag, never edit.
            if core not in mapping or line[match.end() : match.end() + 1] in FORMAT_NEXT_CHARS:
                return token
            count += 1
            return mapping[core] + token[len(core) :]

        new_line = CODE_TOKEN.sub(_token, line)
        if SPLIT_CONSTRUCTION.search(new_line):
            unresolved.append(Unresolved("", number, _snippet(new_line, 0), REASON_SPLIT))
        elif PARENT_ARITHMETIC.search(new_line) and "research" in new_line:
            unresolved.append(Unresolved("", number, _snippet(new_line, 0), REASON_PARENT))
        for start in _old_mentions(new_line):
            reason = _classify(new_line, start, mapping, key_dirs, code=True)
            unresolved.append(Unresolved("", number, _snippet(new_line, start), reason))
        out.append(new_line)
    return "\n".join(out), unresolved, count


def rewrite_code(text: str, mapping: dict[str, str]) -> tuple[str, list[Unresolved]]:
    """Replace exact old-tree paths in code and flag constructions that need a person.

    A token is the longest run of path characters starting at `docs/research`; it is
    replaced only when it (minus trailing sentence dots) is a mapping key, so a longer
    unmapped token is never partly rewritten. Directory prefixes, globs, f-strings and
    format strings, a bare `"docs/research"` base path, `"docs" / "research"` style
    split constructions and `.parent`/`parents` arithmetic near `research` are reported
    and left unchanged.

    Args:
        text: File contents (line endings are preserved).
        mapping: Old repo-relative path to new repo-relative path.

    Returns:
        (new text, unresolved mentions with an empty `file`).
    """
    new_text, unresolved, _ = _rewrite_code_counted(text, mapping)
    return new_text, unresolved


@dataclass(frozen=True)
class RewriteTarget:
    """One file a rewrite scope reads: where it is now, and its old and new locations."""

    path: str
    old_file: str
    new_file: str


def git_tracked_files(root: Path) -> list[str]:
    """List every tracked file (repo-relative POSIX paths) with read-only `git ls-files`.

    Args:
        root: Repository root.

    Returns:
        Sorted paths; untracked and git-ignored files (e.g. skill workspaces) are absent.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return sorted(p for p in out.split("\0") if p)


def _is_rewrite_excluded(rel: str, excludes: list[str]) -> bool:
    """Return True for always-excluded reorganization files and `--exclude` matches."""
    if (
        rel in REORG_SELF_PATHS
        or rel in REWRITE_EXCLUDED_PATHS
        or rel.startswith(REWRITE_EXCLUDED_PREFIXES)
    ):
        return True
    return any(rel == e.rstrip("/") or rel.startswith(e.rstrip("/") + "/") for e in excludes)


def _in_code_scope(rel: str) -> bool:
    """Return True for a path the `code` scope owns."""
    return rel in CODE_SCOPE_FILES or rel.startswith(CODE_SCOPE_DIRS)


def _in_research_tree(rel: str) -> bool:
    """Return True for a path under the old or the new research tree."""
    return rel.startswith((f"{OLD_ROOT}/", f"{NEW_ROOT}/"))


def scope_targets(
    scope: str, rows: list[ManifestRow], root: Path, excludes: list[str]
) -> tuple[list[RewriteTarget], list[str], list[str]]:
    """Resolve a rewrite scope to the files it reads.

    Args:
        scope: One of SCOPES.
        rows: Manifest rows.
        root: Repository root.
        excludes: Extra repo-relative paths (files or directories) to leave alone.

    Returns:
        (targets, excluded paths, missing paths). Symlinks are excluded, never followed.

    Raises:
        ValueError: If `scope` is unknown.
    """
    excluded: list[str] = []
    missing: list[str] = []
    targets: list[RewriteTarget] = []
    if scope == "moved-markdown":
        for row in rows:
            if row.frozen or not row.current_path.endswith(".md"):
                continue
            if _is_rewrite_excluded(row.current_path, excludes) or _is_rewrite_excluded(
                row.proposed_path, excludes
            ):
                excluded.append(row.current_path)
                continue
            proposed = root / row.proposed_path
            path = row.proposed_path if proposed.is_file() else row.current_path
            if (root / path).is_symlink():
                excluded.append(path)
            elif not (root / path).is_file():
                missing.append(row.current_path)
            else:
                targets.append(RewriteTarget(path, row.current_path, row.proposed_path))
        return targets, excluded, missing
    if scope == "code":
        candidates = [p for p in git_tracked_files(root) if _in_code_scope(p)]
    elif scope == "fixed-markdown":
        candidates = [
            p
            for p in git_tracked_files(root)
            if p.endswith(".md") and not _in_code_scope(p) and not _in_research_tree(p)
        ]
    else:
        raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")
    for rel in candidates:
        if _in_research_tree(rel):
            continue
        if _is_rewrite_excluded(rel, excludes) or (root / rel).is_symlink():
            excluded.append(rel)
        elif not (root / rel).is_file():
            missing.append(rel)
        else:
            targets.append(RewriteTarget(rel, rel, rel))
    return targets, excluded, missing


def _frozen_paths(rows: list[ManifestRow]) -> frozenset[str]:
    """Return both locations of every frozen manifest row."""
    return frozenset(p for r in rows if r.frozen for p in (r.current_path, r.proposed_path))


def _repo_exists(root: Path) -> Callable[[str], bool]:
    """Return an existence check for repo-relative paths that never looks outside `root`."""

    def _exists(rel: str) -> bool:
        return not _escapes_root(rel) and (root / rel).exists()

    return _exists


def write_atomic(path: Path, data: bytes) -> None:
    """Replace a file's bytes atomically, keeping its permission bits.

    The data goes to a temporary file in the same directory, is flushed to disk, and is
    then moved over the original with `os.replace`, so a crash leaves either the old or the
    new bytes, never a partial file.

    Args:
        path: Existing file to replace.
        data: New contents.

    Raises:
        OSError: If writing or replacing fails (the temporary file is removed).
    """
    mode = stat.S_IMODE(path.stat().st_mode)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def run_rewrite(
    scope: str,
    rows: list[ManifestRow],
    root: Path,
    dry_run: bool,
    excludes: list[str],
) -> dict[str, Any]:
    """Rewrite one scope (or report what would change) and return the report dict.

    Every target is read and rewritten in memory first. A pre-flight check then fails the
    run, before anything is written, when a file is missing, unreadable, not UTF-8, frozen,
    or (outside moved-markdown) inside the research tree. Only then are changed files
    written, each atomically. A dry run performs the same checks and writes nothing.

    Args:
        scope: One of SCOPES.
        rows: Manifest rows; the mapping is current_path -> proposed_path.
        root: Repository root.
        dry_run: When True nothing is written.
        excludes: Extra paths to leave alone.

    Returns:
        The JSON-serializable report; `status` is "ok" or "failed" with `error` text.
    """
    mapping = {r.current_path: r.proposed_path for r in rows}
    frozen = _frozen_paths(rows)
    exists = _repo_exists(root)
    targets, excluded, missing = scope_targets(scope, rows, root, excludes)
    changed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    undecodable: list[str] = []
    unreadable: list[str] = []
    protected: list[str] = []
    pending: list[tuple[str, str]] = []
    for target in targets:
        if target.path in frozen or (scope != "moved-markdown" and _in_research_tree(target.path)):
            protected.append(target.path)
            continue
        try:
            data = (root / target.path).read_bytes()
        except OSError as exc:
            unreadable.append(f"{target.path}: {exc.strerror or exc}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            undecodable.append(target.path)
            continue
        if scope == "code":
            new_text, found, count = _rewrite_code_counted(text, mapping)
        else:
            new_text, found, count = _rewrite_markdown_counted(
                text, target.old_file, target.new_file, mapping, exists
            )
        unresolved.extend(asdict(replace(u, file=target.path)) for u in found)
        if new_text != text:
            changed.append({"file": target.path, "replacements": count})
            pending.append((target.path, new_text))
    problems = (
        [f"missing: {p}" for p in missing]
        + [f"unreadable: {p}" for p in unreadable]
        + [f"not UTF-8: {p}" for p in undecodable]
        + [f"protected (frozen or research tree): {p}" for p in protected]
    )
    written: list[str] = []
    error = ""
    if problems:
        error = "pre-flight failed, nothing written: " + "; ".join(sorted(problems))
    elif not dry_run:
        for rel, new_text in sorted(pending):
            try:
                write_atomic(root / rel, new_text.encode("utf-8"))
            except OSError as exc:
                error = f"write failed for {rel}: {exc}; already written: {len(written)} files"
                break
            written.append(rel)
    return {
        "scope": scope,
        "dry_run": dry_run,
        "status": STATUS_FAILED if error else STATUS_OK,
        "error": error,
        "changed": sorted(changed, key=lambda c: c["file"]),
        "written": written,
        "unresolved": sorted(
            unresolved, key=lambda u: (u["file"], u["line"], u["text"], u["reason"])
        ),
        "skipped_frozen": sum(r.frozen for r in rows) if scope == "moved-markdown" else 0,
        "excluded": sorted(excluded),
        "skipped_undecodable": sorted(undecodable),
        "skipped_unreadable": sorted(unreadable),
        "skipped_missing": sorted(missing),
    }


def _cmd_rewrite(args: argparse.Namespace) -> int:
    """Run `rewrite`: rewrite one scope (or dry-run it) and always write the JSON report."""
    try:
        rows = load_manifest(Path(args.manifest))
        report = run_rewrite(args.scope, rows, Path(args.root), args.dry_run, args.exclude or [])
    except Exception as exc:  # noqa: BLE001 - any failure must still produce a report
        report = {
            "scope": args.scope,
            "dry_run": args.dry_run,
            "status": STATUS_FAILED,
            "error": f"{type(exc).__name__}: {exc}",
            "changed": [],
            "written": [],
            "unresolved": [],
        }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.report).write_text(text, encoding="utf-8")
    reasons = Counter(u["reason"] for u in report["unresolved"])
    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(
        f"{mode} scope={args.scope} status={report['status']} "
        f"changed={len(report['changed'])} files written={len(report['written'])} "
        f"replacements={sum(c['replacements'] for c in report['changed'])} "
        f"unresolved={len(report['unresolved'])} {dict(sorted(reasons.items()))}"
    )
    if report["status"] != STATUS_OK:
        print(f"FAIL {report['error']}")
        return 1
    return 0


class MoveRefusedError(RuntimeError):
    """Raised when the move pre-flight finds a problem; nothing has been moved."""


def _move_pairs(rows: list[ManifestRow]) -> list[tuple[str, str]]:
    """Return (source, destination) for every row that actually moves, in manifest order."""
    return [(r.current_path, r.proposed_path) for r in rows if r.current_path != r.proposed_path]


def _move_preflight(pairs: list[tuple[str, str]], root: Path) -> list[str]:
    """Check every move before any is made: sources present, destinations free and unique."""
    problems: list[str] = []
    seen: set[str] = set()
    sources = {src for src, _ in pairs}
    for src, dst in pairs:
        source = root / src
        if not source.is_file() or source.is_symlink():
            problems.append(f"source missing: {src}")
        dest = root / dst
        if dest.exists() or dest.is_symlink():
            problems.append(f"destination exists: {dst}")
        if dst in seen:
            problems.append(f"destination used twice: {dst}")
        if dst in sources:
            problems.append(f"destination is another row's source: {dst}")
        seen.add(dst)
    return problems


def apply_moves(rows: list[ManifestRow], root: Path, dry_run: bool) -> list[tuple[str, str]]:
    """Move every manifest file to its destination with `git mv`, in manifest order.

    The pre-flight checks all rows before the first move: a missing source, an existing
    destination or a repeated destination refuses the whole run, so nothing is moved.
    Rows whose current and proposed paths are equal are skipped. Destination folders are
    created first; nothing is ever overwritten.

    Args:
        rows: Manifest rows.
        root: Repository root the paths resolve against.
        dry_run: When True, print the plan and change nothing.

    Returns:
        The (source, destination) pairs moved, or that would be moved in a dry run.

    Raises:
        MoveRefusedError: If the pre-flight finds any problem.
        subprocess.CalledProcessError: If a `git mv` fails part-way (the message on stderr
            says how far the run got).
    """
    pairs = _move_pairs(rows)
    problems = _move_preflight(pairs, root)
    if problems:
        raise MoveRefusedError(
            f"move refused, nothing moved ({len(problems)} problems): " + "; ".join(problems)
        )
    if dry_run:
        for src, dst in pairs:
            print(f"PLAN {src} -> {dst}")
        return pairs
    for done, (src, dst) in enumerate(pairs):
        (root / dst).parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "mv", "--", src, dst], cwd=root, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"FAIL git mv {src} -> {dst} after {done} of {len(pairs)} moves: "
                f"{exc.stderr.strip()}",
                file=sys.stderr,
            )
            raise
    return pairs


def remove_leftover_dirs(root: Path) -> list[str]:
    """Delete `__pycache__` folders and empty folders left under docs/research.

    A `__pycache__` folder is removed only when it holds nothing but `.pyc` files. No other
    file is ever deleted.

    Args:
        root: Repository root.

    Returns:
        Repo-relative paths of the removed folders.
    """
    base = root / OLD_ROOT
    if not base.is_dir():
        return []
    removed: list[str] = []
    for cache in sorted(base.rglob(PYCACHE_DIR), reverse=True):
        if not cache.is_dir() or cache.is_symlink():
            continue
        contents = list(cache.rglob("*"))
        if all(p.is_file() and p.suffix == ".pyc" for p in contents):
            shutil.rmtree(cache)
            removed.append(cache.relative_to(root).as_posix())
    dirs = [p for p in base.rglob("*") if p.is_dir() and not p.is_symlink()]
    for folder in sorted(dirs, key=lambda p: len(p.parts), reverse=True) + [base]:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            removed.append(folder.relative_to(root).as_posix())
    return removed


def _files_under(base: Path, root: Path) -> list[str]:
    """List repo-relative files under `base`, ignoring `__pycache__` folders."""
    if not base.is_dir():
        return []
    return sorted(
        p.relative_to(root).as_posix()
        for p in base.rglob("*")
        if (p.is_file() or p.is_symlink()) and PYCACHE_DIR not in p.relative_to(root).parts
    )


def _matches_allow_new(path: str, allow_new: tuple[str, ...]) -> bool:
    """Return True when a repo-relative path under research/ matches an allow-new pattern."""
    inside = path.removeprefix(f"{NEW_ROOT}/")
    return any(fnmatch.fnmatchcase(inside, p.removeprefix(f"{NEW_ROOT}/")) for p in allow_new)


def _verify_tree(
    rows: list[ManifestRow],
    ledger: dict[str, Any],
    root: Path,
    allowed_changes: frozenset[str] | None,
    allow_new: tuple[str, ...] = (),
    allow_deeper: tuple[str, ...] = DEFAULT_ALLOW_DEEPER,
) -> list[str]:
    """Shared tree checks for the `moved`, `rewritten` and `final` gates.

    A destination whose sha256 differs from the ledger is a problem unless its proposed
    path is in `allowed_changes` and its row is a non-frozen file. `moved` passes None, so
    every byte must match. Files under research/ that are not manifest destinations are a
    problem unless they match an `allow_new` pattern (only the `final` gate passes any).
    """
    problems: list[str] = []
    allowed = allowed_changes or frozenset()
    by_proposed = {r.proposed_path: r for r in rows}
    for listed in sorted(allowed):
        row = by_proposed.get(listed)
        if row is None:
            problems.append(f"listed in the rewrite manifest but not in manifest: {listed}")
        elif row.frozen or PurePosixPath(listed).suffix != REWRITABLE_SUFFIX:
            problems.append(f"cannot be rewritten (frozen or not Markdown): {listed}")
    entries = {e["current_path"]: e for e in ledger.get("files", [])}
    if len(entries) != len(rows):
        problems.append(f"ledger has {len(entries)} files but the manifest has {len(rows)} rows")
    for row in rows:
        entry = entries.get(row.current_path)
        dest = root / row.proposed_path
        if entry is None:
            problems.append(f"not in ledger: {row.current_path}")
        elif entry.get("proposed_path") != row.proposed_path:
            problems.append(
                f"ledger destination differs: {row.current_path} -> "
                f"{entry.get('proposed_path')} (manifest {row.proposed_path})"
            )
        if not dest.is_file():
            problems.append(f"missing destination: {row.proposed_path}")
        elif entry is not None and sha256_file(dest) != entry["sha256"]:
            if row.frozen:
                problems.append(f"sha256 differs from ledger (frozen file): {row.proposed_path}")
            elif allowed_changes is None:
                problems.append(f"sha256 differs from ledger: {row.proposed_path}")
            elif row.proposed_path not in allowed:
                problems.append(f"unlisted change: {row.proposed_path}")
        if not row.proposed_path.startswith(f"{NEW_ROOT}/") or len(
            PurePosixPath(row.proposed_path).parts
        ) > _depth_limit(row.proposed_path, allow_deeper):
            problems.append(
                f"depth exceeds research/<category>/<study>/<file>: {row.proposed_path}"
            )
    on_disk = _files_under(root / NEW_ROOT, root)
    expected = {r.proposed_path for r in rows}
    allowed_extra = {p for p in on_disk if p not in expected and _matches_allow_new(p, allow_new)}
    if len(on_disk) - len(allowed_extra) != len(rows):
        problems.append(
            f"{len(on_disk) - len(allowed_extra)} files under {NEW_ROOT}/ (excluding "
            f"{len(allowed_extra)} allowed new) but {len(rows)} manifest rows"
        )
    problems.extend(
        f"unexpected file under {NEW_ROOT}/: {p}"
        for p in on_disk
        if p not in expected and p not in allowed_extra
    )
    tracked = sorted(tracked_research_files(root))
    problems.extend(f"still tracked under {OLD_ROOT}: {p}" for p in tracked)
    tracked_set = set(tracked)
    problems.extend(
        f"untracked leftover under {OLD_ROOT}: {p}"
        for p in _files_under(root / OLD_ROOT, root)
        if p not in tracked_set
    )
    return problems


def verify_moved(rows: list[ManifestRow], ledger: dict[str, Any], root: Path) -> list[str]:
    """Check the tree right after the move-only step.

    Checks: every destination exists and matches its ledger sha256 (all files, frozen or
    not); the number of files under research/ equals the number of manifest rows; no
    destination is deeper than research/<category>/<study>/<file>; nothing is tracked
    under docs/research; and no untracked file is left there (`__pycache__` is ignored).

    Args:
        rows: Manifest rows.
        ledger: The pre-move ledger (`files` entries keyed by `current_path`).
        root: Repository root.

    Returns:
        Problem descriptions; an empty list means the gate passes.
    """
    return _verify_tree(rows, ledger, root, None)


def verify_rewritten(
    rows: list[ManifestRow],
    ledger: dict[str, Any],
    root: Path,
    rewrite_listed: set[str] | frozenset[str],
) -> list[str]:
    """Check the tree after the reference rewrite of the mutable Markdown.

    Same checks as `verify_moved`, except a destination may differ from its ledger sha256
    when its proposed path is listed in the rewrite manifest. A frozen or pinned file must
    still match exactly even if listed, and a listed path that is unknown, frozen or not
    Markdown is itself a problem.

    Args:
        rows: Manifest rows.
        ledger: The pre-move ledger.
        root: Repository root.
        rewrite_listed: Proposed paths the rewrite manifest says were edited.

    Returns:
        Problem descriptions; an empty list means the gate passes.
    """
    return _verify_tree(rows, ledger, root, frozenset(rewrite_listed))


def verify_final(
    rows: list[ManifestRow],
    ledger: dict[str, Any],
    root: Path,
    rewrite_listed: set[str] | frozenset[str],
    allow_new: tuple[str, ...] = DEFAULT_ALLOW_NEW,
    expected_pinned: int = len(PINNED_PATHS),
    require_new: tuple[str, ...] = (),
    allow_deeper: tuple[str, ...] = DEFAULT_ALLOW_DEEPER,
) -> list[str]:
    """Check the finished tree: everything `verify_rewritten` checks, plus tracked-file rules.

    Extra checks: no duplicate proposed paths; every destination is tracked; the tracked
    files under research/ are exactly the manifest destinations plus files matching
    `allow_new`; no tracked file is deeper than research/<category>/<study>/<file> (the
    `artifacts-*` packages are study folders, so they share the limit; a study folder named
    in `allow_deeper` may add one sub-folder level); the pinned files
    still match their ledger digests; nothing is tracked under docs/research and the folder
    itself is gone.

    Args:
        rows: Manifest rows.
        ledger: The pre-move ledger.
        root: Repository root.
        rewrite_listed: Proposed paths the rewrite manifest says were edited.
        allow_new: Patterns (relative to research/, fnmatch) for files added after the move.
        expected_pinned: How many pinned rows the manifest must carry (twelve in production).
        require_new: Paths (relative to research/) that must exist and be tracked.
        allow_deeper: Study folders (relative to research/) whose files may sit one level
            deeper, in a sub-folder.

    Returns:
        Problem descriptions; an empty list means the gate passes.
    """
    problems = _verify_tree(rows, ledger, root, frozenset(rewrite_listed), allow_new, allow_deeper)
    counts = Counter(r.proposed_path for r in rows)
    problems.extend(f"duplicate proposed path: {p}" for p, n in sorted(counts.items()) if n > 1)
    tracked = tracked_files_under(NEW_ROOT, root)
    for row in rows:
        if row.proposed_path not in tracked:
            problems.append(f"destination not tracked: {row.proposed_path}")
    for path in sorted(tracked):
        if path not in counts and not _matches_allow_new(path, allow_new):
            problems.append(f"unexpected tracked file under {NEW_ROOT}/: {path}")
        if len(PurePosixPath(path).parts) > _depth_limit(path, allow_deeper):
            problems.append(f"tracked file deeper than research/<category>/<study>/<file>: {path}")
    pinned_rows = [r for r in rows if r.pinned]
    if len(pinned_rows) != expected_pinned:
        problems.append(
            f"{len(pinned_rows)} pinned rows in the manifest, expected {expected_pinned}"
        )
    entries = {e["current_path"]: e for e in ledger.get("files", [])}
    for row in pinned_rows:
        entry = entries.get(row.current_path)
        dest = root / row.proposed_path
        if entry is None or not entry.get("pinned"):
            problems.append(f"ledger does not mark this file pinned: {row.current_path}")
        elif dest.is_file() and sha256_file(dest) != entry["sha256"]:
            problems.append(f"pinned digest differs from ledger: {row.proposed_path}")
    for required in require_new:
        rel = f"{NEW_ROOT}/{required.removeprefix(f'{NEW_ROOT}/')}"
        if not (root / rel).is_file():
            problems.append(f"required new file missing: {rel}")
        elif rel not in tracked:
            problems.append(f"required new file not tracked: {rel}")
    if (root / OLD_ROOT).exists():
        problems.append(f"{OLD_ROOT} still exists")
    return problems


def load_rewrite_manifest(path: Path) -> set[str]:
    """Read the rewrite manifest: proposed paths of the edited files.

    Accepts `{"files": [path, ...]}`, a bare list of paths, or the `rewrite` report shape
    (`{"changed": [{"file": path, ...}, ...]}`).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(p) for p in data}
    if "files" in data:
        return {str(p) for p in data["files"]}
    return {str(c["file"]) for c in data.get("changed", [])}


def _cmd_move(args: argparse.Namespace) -> int:
    """Run `move`: preflight, then move (or print the plan) and clear leftover folders."""
    rows = load_manifest(Path(args.manifest))
    root = Path(args.root)
    try:
        pairs = apply_moves(rows, root, args.dry_run)
    except MoveRefusedError as exc:
        print(f"FAIL {exc}")
        return 1
    renamed = sum(PurePosixPath(s).name != PurePosixPath(d).name for s, d in pairs)
    if args.dry_run:
        print(f"DRY-RUN {len(pairs)} moves planned, {renamed} basename changes, 0 collisions")
        return 0
    for folder in remove_leftover_dirs(root):
        print(f"REMOVED {folder}")
    print(f"MOVED {len(pairs)} files, {renamed} basename changes")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Run `verify`: check one stage's gate and print problems one per line."""
    rows = load_manifest(Path(args.manifest))
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    root = Path(args.root)
    if args.stage in ("rewritten", "final"):
        if not args.rewrite_manifest:
            print(f"FAIL --rewrite-manifest is required for stage {args.stage}")
            return 1
        listed = load_rewrite_manifest(Path(args.rewrite_manifest))
        if args.stage == "final":
            allow_new = (*DEFAULT_ALLOW_NEW, *(args.allow_new or ()))
            require_new = tuple(args.require_new or ())
            allow_deeper = (*DEFAULT_ALLOW_DEEPER, *(args.allow_deeper or ()))
            problems = verify_final(
                rows,
                ledger,
                root,
                listed,
                allow_new,
                require_new=require_new,
                allow_deeper=allow_deeper,
            )
            summary = (
                f"{len(rows)} manifest files, {len(listed)} listed edits, "
                f"{len(tracked_files_under(NEW_ROOT, root))} tracked files under {NEW_ROOT}/"
            )
        else:
            problems = verify_rewritten(rows, ledger, root, listed)
            summary = f"{len(rows)} files checked, {len(listed)} listed edits"
    else:
        problems = verify_moved(rows, ledger, root)
        summary = f"{len(rows)} files byte-identical to the ledger"
    for problem in problems:
        print(f"PROBLEM {problem}")
    if problems:
        print(f"FAIL stage={args.stage} {len(problems)} problems")
        if args.stage == "moved" and (root / DEFAULT_REWRITE_MANIFEST).is_file():
            print(
                "NOTE stage moved is only valid before the rewrite step; a rewrite manifest "
                f"exists at {DEFAULT_REWRITE_MANIFEST}, so use --stage rewritten or final."
            )
        return 1
    print(f"OK stage={args.stage} {summary}")
    return 0


def current_git_head(root: Path = REPO_ROOT) -> str:
    """Return `git rev-parse HEAD` for the repository at `root`."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _cmd_ledger(args: argparse.Namespace) -> int:
    """Run `ledger`: write the deterministic provenance ledger JSON."""
    rows = load_manifest(Path(args.manifest))
    ledger = build_ledger(rows, Path(args.root), current_git_head())
    text = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"OK ledger {len(rows)} files, {len(ledger['inbound_references'])} referencing files")
    return 0


MOVED_COLUMNS = ("old_path", "new_path", "renamed")


def moved_csv_text(rows: list[ManifestRow]) -> str:
    """Render the MOVED.csv navigation table: one row per manifest row, sorted by old path.

    `renamed` is `yes` only when the basename changed (the three `-brief` renames).
    The table is for navigation only; it never rescues a path that is missing on disk.

    Args:
        rows: Manifest rows.

    Returns:
        CSV text with LF line endings, identical for any input order.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MOVED_COLUMNS)
    for row in sorted(rows, key=lambda r: (r.current_path, r.proposed_path)):
        changed = PurePosixPath(row.current_path).name != PurePosixPath(row.proposed_path).name
        writer.writerow((row.current_path, row.proposed_path, "yes" if changed else "no"))
    return buffer.getvalue()


def _cmd_moved_csv(args: argparse.Namespace) -> int:
    """Run `moved-csv`: write the old-path to new-path table."""
    rows = load_manifest(Path(args.manifest))
    Path(args.out).write_text(moved_csv_text(rows), encoding="utf-8", newline="")
    print(f"OK moved-csv {len(rows)} rows")
    return 0


def _cmd_manifest_check(args: argparse.Namespace) -> int:
    """Run `manifest-check`: print problems and return 1 if any."""
    rows = load_manifest(Path(args.manifest))
    allow_deeper = (*DEFAULT_ALLOW_DEEPER, *(args.allow_deeper or ()))
    problems = check_manifest(rows, tracked_research_files(), allow_deeper)
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
    check.add_argument(
        "--allow-deeper",
        action="append",
        metavar="PREFIX",
        help=(
            "Study folder (relative to research/; repeatable) whose files may sit one level "
            "deeper, in a sub-folder. Adds to the default: " + ", ".join(DEFAULT_ALLOW_DEEPER)
        ),
    )
    check.set_defaults(func=_cmd_manifest_check)
    ledger = sub.add_parser("ledger", help="Write the pre-move provenance ledger (JSON).")
    ledger.add_argument("--manifest", required=True, help="Path to the manifest CSV.")
    ledger.add_argument("--out", required=True, help="Path of the ledger JSON to write.")
    ledger.add_argument(
        "--root", default=str(REPO_ROOT), help="Repository root (default: this repository)."
    )
    ledger.set_defaults(func=_cmd_ledger)
    rewrite = sub.add_parser(
        "rewrite", help="Rewrite old docs/research references in one scope (or dry-run it)."
    )
    rewrite.add_argument("--manifest", required=True, help="Path to the manifest CSV.")
    rewrite.add_argument("--scope", required=True, choices=SCOPES, help="Which files to rewrite.")
    rewrite.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    rewrite.add_argument("--report", required=True, help="Path of the JSON report to write.")
    rewrite.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Repo-relative file or directory to leave alone (repeatable).",
    )
    rewrite.add_argument(
        "--root", default=str(REPO_ROOT), help="Repository root (default: this repository)."
    )
    rewrite.set_defaults(func=_cmd_rewrite)
    move = sub.add_parser("move", help="git mv every manifest file to its destination.")
    move.add_argument("--manifest", required=True, help="Path to the manifest CSV.")
    move.add_argument("--dry-run", action="store_true", help="Print the plan; change nothing.")
    move.add_argument(
        "--root", default=str(REPO_ROOT), help="Repository root (default: this repository)."
    )
    move.set_defaults(func=_cmd_move)
    moved = sub.add_parser("moved-csv", help="Write research/MOVED.csv (old path, new path).")
    moved.add_argument("--manifest", required=True, help="Path to the manifest CSV.")
    moved.add_argument("--out", required=True, help="Path of the CSV to write.")
    moved.set_defaults(func=_cmd_moved_csv)
    verify = sub.add_parser("verify", help="Check one stage's gate (exit 1 on any problem).")
    verify.add_argument(
        "--stage",
        required=True,
        choices=VERIFY_STAGES,
        help=(
            "Gate to check: moved (valid only before the rewrite step), rewritten or "
            "final (valid after it)."
        ),
    )
    verify.add_argument("--manifest", required=True, help="Path to the manifest CSV.")
    verify.add_argument("--ledger", required=True, help="Path to the pre-move ledger JSON.")
    verify.add_argument(
        "--rewrite-manifest",
        help="JSON listing the edited files (required for --stage rewritten and final).",
    )
    verify.add_argument(
        "--require-new",
        action="append",
        metavar="PATH",
        help=(
            "File under research/ (relative path) that must exist and be tracked; "
            "for --stage final, repeatable."
        ),
    )
    verify.add_argument(
        "--allow-new",
        action="append",
        metavar="PATTERN",
        help=(
            "Extra file pattern allowed under research/ for --stage final (relative to "
            "research/, fnmatch; repeatable). Adds to the defaults: " + ", ".join(DEFAULT_ALLOW_NEW)
        ),
    )
    verify.add_argument(
        "--allow-deeper",
        action="append",
        metavar="PREFIX",
        help=(
            "Study folder (relative to research/; repeatable) whose files may sit one level "
            "deeper, in a sub-folder. Adds to the default: " + ", ".join(DEFAULT_ALLOW_DEEPER)
        ),
    )
    verify.add_argument(
        "--root", default=str(REPO_ROOT), help="Repository root (default: this repository)."
    )
    verify.set_defaults(func=_cmd_verify)
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
