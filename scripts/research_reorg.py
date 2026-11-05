#!/usr/bin/env python3
"""Tooling for the docs/research -> research/ reorganization (manifest, ledger, rewriter)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import posixpath
import re
import subprocess
import sys
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
# Characters a repo path is made of; a token is the longest run of these.
PATH_CHAR_CLASS = r"A-Za-z0-9_\-./"
PATH_RUN = re.compile(rf"[{PATH_CHAR_CLASS}]*")
# An old-tree path token in code; the lookbehind stops a match inside a longer word.
CODE_TOKEN = re.compile(rf"(?<![A-Za-z0-9_\-])docs/research[{PATH_CHAR_CLASS}]*")
# A whole code span that is a citation: docs/research/<path> with an optional :N or :N-M.
CITATION = re.compile(rf"^(docs/research/[{PATH_CHAR_CLASS}]+?)(:\d+(?:-\d+)?)?$")
# Inline link or image target: `](target "title")`; only the target group is rewritten.
INLINE_LINK = re.compile(
    r"\]\((?P<lead>[ \t]*)(?P<target><[^<>\n]*>|[^\s()<>]+)"
    r"(?P<title>[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'))?(?P<trail>[ \t]*)\)"
)
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
        if FSTRING_START.search(line) or ".format(" in line or nxt == "%":
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
    if not path:
        return raw, None
    lookup = unquote(path)
    absolute = lookup.startswith("/")
    resolved = posixpath.normpath(
        lookup.lstrip("/") if absolute else posixpath.join(old_dir, lookup)
    )
    mapped = resolved in mapping
    under_old = resolved == OLD_ROOT or resolved.startswith(f"{OLD_ROOT}/")
    reason = None if mapped or not under_old else _path_reason(resolved, key_dirs)
    if not mapped and old_dir == new_dir:
        return raw, reason
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
    new_target = new_path + (sep + fragment if sep else "")
    return (f"<{new_target}>" if angle else new_target), reason


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
    for number, line in enumerate(text.split("\n"), start=1):
        if fence is not None:
            if _fence_closes(line, fence):
                fence = None
            out.append(line)
            continue
        fence = _fence_opener(line)
        if fence is not None:
            out.append(line)
            continue
        pieces: list[str] = []
        for is_code, piece in _code_span_segments(line):
            if is_code:
                new_piece = _rewrite_citation(piece, mapping)
                count += new_piece != piece
                pieces.append(new_piece)
                continue

            def _link(match: re.Match[str]) -> str:
                nonlocal count
                raw = match.group("target")
                new_raw, reason = _rewrite_link_target(
                    raw, old_dir, new_dir, mapping, key_dirs, exists
                )
                if reason is not None and OLD_ROOT not in new_raw:
                    unresolved.append(Unresolved("", number, _snippet(line, match.start()), reason))
                if new_raw == raw:
                    return match.group(0)
                count += 1
                start, end = match.span("target")
                base = match.start()
                whole = match.group(0)
                return whole[: start - base] + new_raw + whole[end - base :]

            pieces.append(INLINE_LINK.sub(_link, piece))
        new_line = "".join(pieces)
        for start in _old_mentions(new_line):
            reason = _classify(new_line, start, mapping, key_dirs, code=False)
            unresolved.append(Unresolved("", number, _snippet(new_line, start), reason))
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
            if core not in mapping:
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
    if rel in REORG_SELF_PATHS or rel.startswith(REWRITE_EXCLUDED_PREFIXES):
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


def run_rewrite(
    scope: str,
    rows: list[ManifestRow],
    root: Path,
    dry_run: bool,
    excludes: list[str],
) -> dict[str, Any]:
    """Rewrite one scope (or report what would change) and return the report dict.

    Args:
        scope: One of SCOPES.
        rows: Manifest rows; the mapping is current_path -> proposed_path.
        root: Repository root.
        dry_run: When True nothing is written.
        excludes: Extra paths to leave alone.

    Returns:
        The JSON-serializable report.

    Raises:
        RuntimeError: If a write would touch a frozen file or the research tree outside
            the moved-markdown scope (a bug guard; never expected).
    """
    mapping = {r.current_path: r.proposed_path for r in rows}
    frozen = _frozen_paths(rows)
    exists = _repo_exists(root)
    targets, excluded, missing = scope_targets(scope, rows, root, excludes)
    changed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    undecodable: list[str] = []
    for target in targets:
        file_path = root / target.path
        try:
            text = file_path.read_bytes().decode("utf-8")
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
        if new_text == text:
            continue
        changed.append({"file": target.path, "replacements": count})
        if dry_run:
            continue
        if target.path in frozen or (scope != "moved-markdown" and _in_research_tree(target.path)):
            raise RuntimeError(f"refusing to write protected file {target.path}")
        file_path.write_bytes(new_text.encode("utf-8"))
    return {
        "scope": scope,
        "dry_run": dry_run,
        "changed": sorted(changed, key=lambda c: c["file"]),
        "unresolved": sorted(
            unresolved, key=lambda u: (u["file"], u["line"], u["text"], u["reason"])
        ),
        "skipped_frozen": sum(r.frozen for r in rows) if scope == "moved-markdown" else 0,
        "excluded": sorted(excluded),
        "skipped_undecodable": sorted(undecodable),
        "skipped_missing": sorted(missing),
    }


def _cmd_rewrite(args: argparse.Namespace) -> int:
    """Run `rewrite`: rewrite one scope (or dry-run it) and write the JSON report."""
    rows = load_manifest(Path(args.manifest))
    report = run_rewrite(args.scope, rows, Path(args.root), args.dry_run, args.exclude or [])
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.report).write_text(text, encoding="utf-8")
    reasons = Counter(u["reason"] for u in report["unresolved"])
    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(
        f"{mode} scope={args.scope} changed={len(report['changed'])} files "
        f"replacements={sum(c['replacements'] for c in report['changed'])} "
        f"unresolved={len(report['unresolved'])} {dict(sorted(reasons.items()))}"
    )
    for rel in report["skipped_undecodable"]:
        print(f"WARN not UTF-8, skipped: {rel}")
    for rel in report["skipped_missing"]:
        print(f"FAIL missing file: {rel}")
    return 1 if report["skipped_missing"] else 0


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
