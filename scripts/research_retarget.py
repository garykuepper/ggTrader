#!/usr/bin/env python3
"""Second move of research/ files into sub-folders, with exact reference retargeting."""

from __future__ import annotations

import argparse
import csv
import io
import json
import posixpath
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

# Tests import this module as `scripts.research_retarget`; the CLI runs it from scripts/.
try:
    from scripts import research_reorg as reorg
except ModuleNotFoundError:  # pragma: no cover - exercised by `python scripts/research_retarget.py`
    import research_reorg as reorg

# Re-used from the first-generation tool so both tools resolve links identically.
Unresolved = reorg.Unresolved
MoveRefusedError = reorg.MoveRefusedError
_rewrite_link_target = reorg._rewrite_link_target
_unsupported_links = reorg._unsupported_links
_code_span_segments = reorg._code_span_segments
_fence_opener = reorg._fence_opener
_fence_closes = reorg._fence_closes
_snippet = reorg._snippet

REPO_ROOT = reorg.REPO_ROOT
NEW_ROOT = reorg.NEW_ROOT
DEFAULT_ALLOW_DEEPER = reorg.DEFAULT_ALLOW_DEEPER
PATH_CHAR_CLASS = reorg.PATH_CHAR_CLASS
INLINE_LINK = reorg.INLINE_LINK
REF_DEFINITION = reorg.REF_DEFINITION
URL_SCHEME = reorg.URL_SCHEME

MOVE_COLUMNS = ("old_path", "new_path")
DEFAULT_MANIFEST = "docs/superpowers/plans/2026-11-05-research-reorg-manifest.csv"
PLAN_PATH = "docs/superpowers/plans/2026-11-05-research-readability-and-subfolders.md"
MOVES_CSV = "docs/superpowers/plans/2026-11-05-research-subfolders-moves.csv"
MOVES_HEADER = ",".join(MOVE_COLUMNS)
SELF_PATHS = frozenset({"scripts/research_retarget.py", "tests/scripts/test_research_retarget.py"})
# Never rewritten: working notes, the price data and generated results.
EXCLUDED_PREFIXES: tuple[str, ...] = (".superpowers/", "data/", "results/")
MARKDOWN_SUFFIX = ".md"
REASON_FENCED = reorg.REASON_FENCED
REASON_UNSUPPORTED = reorg.REASON_UNSUPPORTED
REASON_INDIRECT = "reference-style or html link"
DEFAULT_BASE = "HEAD"
# `:12`, `:12-20`, `:12, 15`, `:27-42, 360-403` after a path.
LINE_SUFFIX = r":\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*"
# An HTML attribute that carries a link target.
HTML_LINK = re.compile(
    r"""\b(?:href|src)\s*=\s*(?P<quote>["'])(?P<target>[^"'\n]*)(?P=quote)""", re.I
)
JSON_INDENT = 2


# --- Move list ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveRow:
    """One file move: its current repo-relative path and the path it moves to."""

    old: str
    new: str


def load_moves(path: Path) -> list[MoveRow]:
    """Read a moves CSV (header `old_path,new_path`, repo-relative POSIX paths).

    Args:
        path: CSV file to read.

    Returns:
        One row per record, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing or a path is blank.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MOVE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        for number, record in enumerate(reader, start=1):
            # A short row leaves None fields; extra columns land under the None key.
            if None in record or any(record[c] is None for c in MOVE_COLUMNS):
                raise ValueError(f"{path}: row {number} needs old_path and new_path")
            rows.append(MoveRow(record["old_path"].strip(), record["new_path"].strip()))
    for row in rows:
        if not row.old or not row.new:
            raise ValueError(f"{path}: blank path in row {row}")
    return rows


def _path_problem(path: str) -> str | None:
    """Return a message when a path is not a normalized repo-relative POSIX path."""
    if path.startswith("/") or "\\" in path or posixpath.normpath(path) != path:
        return f"path is not a normalized repo-relative path: {path}"
    return None


def check_moves(
    rows: list[MoveRow],
    tracked: set[str],
    allow_deeper: tuple[str, ...] = DEFAULT_ALLOW_DEEPER,
) -> list[str]:
    """Check a move list against the tracked files and the layout rules.

    Args:
        rows: Parsed move rows.
        tracked: Every tracked file in the repository (repo-relative POSIX paths).
        allow_deeper: Study folders (relative to research/) whose files may sit one level
            deeper, in a sub-folder.

    Returns:
        Problem descriptions; an empty list means the moves pass.
    """
    problems: list[str] = []
    sources = {r.old for r in rows}
    seen_old: set[str] = set()
    seen_new: set[str] = set()
    for row in rows:
        for path in (row.old, row.new):
            bad = _path_problem(path)
            if bad:
                problems.append(bad)
        if row.old in seen_old:
            problems.append(f"source listed twice: {row.old}")
        seen_old.add(row.old)
        if row.old not in tracked:
            problems.append(f"source is not tracked: {row.old}")
        if row.old == row.new:
            problems.append(f"source equals destination: {row.old}")
            continue
        if row.new in tracked:
            problems.append(f"destination already exists: {row.new}")
        if row.new in seen_new:
            problems.append(f"destination used twice: {row.new}")
        seen_new.add(row.new)
        if row.new in sources:
            problems.append(f"destination is another row's source: {row.new}")
        if not row.new.startswith(f"{NEW_ROOT}/"):
            problems.append(f"destination outside {NEW_ROOT}/: {row.new}")
        if PurePosixPath(row.old).name != PurePosixPath(row.new).name:
            problems.append(f"basename changed: {row.old} -> {row.new}")
        limit = reorg._depth_limit(row.new, allow_deeper)
        if len(PurePosixPath(row.new).parts) > limit:
            problems.append(f"depth exceeds {limit} components: {row.new}")
    return problems


def apply_moves(
    rows: list[MoveRow],
    root: Path,
    dry_run: bool,
    allow_deeper: tuple[str, ...] = DEFAULT_ALLOW_DEEPER,
) -> list[tuple[str, str]]:
    """Move every file with `git mv`, after a preflight over all rows.

    Args:
        rows: Move rows.
        root: Repository root the paths resolve against.
        dry_run: When True, print the plan and touch nothing.
        allow_deeper: See `check_moves`.

    Returns:
        The (old, new) pairs moved, or that would be moved in a dry run.

    Raises:
        MoveRefusedError: If the preflight finds any problem (nothing is moved).
        subprocess.CalledProcessError: If a `git mv` fails part-way.
    """
    problems = check_moves(rows, set(reorg.git_tracked_files(root)), allow_deeper)
    for row in rows:
        source, dest = root / row.old, root / row.new
        if source.is_symlink() or not source.is_file():
            problems.append(f"source is not a regular file on disk: {row.old}")
        if row.old != row.new and (dest.exists() or dest.is_symlink()):
            problems.append(f"destination exists on disk: {row.new}")
    problems = list(dict.fromkeys(problems))
    if problems:
        raise MoveRefusedError(
            f"move refused, nothing moved ({len(problems)} problems): " + "; ".join(problems)
        )
    pairs = [(r.old, r.new) for r in rows]
    if dry_run:
        for old, new in pairs:
            print(f"PLAN {old} -> {new}")
        return pairs
    created: list[Path] = []
    for done, (old, new) in enumerate(pairs):
        parent = (root / new).parent
        missing = [p for p in (parent, *parent.parents) if not p.exists() and root in p.parents]
        try:
            created.extend(missing)  # recorded first so a partial mkdir is still cleaned up
            parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "mv", "--", old, new], cwd=root, check=True, capture_output=True, text=True
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            undone, stuck = _undo_moves(pairs[:done], root, created)
            hint = (
                f"undid {undone} completed moves"
                if not stuck
                else f"could not undo {len(stuck)} moves ({'; '.join(stuck)}); "
                "move them back by hand"
            )
            message = (
                f"git mv {old} -> {new} failed after {done} of {len(pairs)} moves "
                f"({str(getattr(exc, 'stderr', None) or exc).strip()}); {hint}; "
                "check `git status` before retrying"
            )
            raise MoveRefusedError(message) from exc
    return pairs


def _undo_moves(
    done: list[tuple[str, str]], root: Path, created: list[Path]
) -> tuple[int, list[str]]:
    """Reverse completed moves (last first) and drop the folders this run created.

    Returns:
        (moves undone, descriptions of moves that could not be undone).
    """
    undone = 0
    stuck: list[str] = []
    for old, new in reversed(done):
        try:
            subprocess.run(
                ["git", "mv", "--", new, old], cwd=root, check=True, capture_output=True, text=True
            )
            undone += 1
        except subprocess.CalledProcessError:
            stuck.append(f"{new} -> {old}")
    for folder in sorted(set(created), key=lambda p: len(p.parts), reverse=True):
        try:
            folder.rmdir()
        except OSError:
            pass  # not empty or already gone: leave it, nothing was lost
    return undone, stuck


# --- Text rewriting -----------------------------------------------------------------------


@lru_cache(maxsize=16)
def _token_regex(keys: tuple[str, ...]) -> re.Pattern[str]:
    """Build the matcher for exact repo-path tokens (longest key first).

    A token must not start inside a longer path (no path character before it) and must not
    run on into a longer path (no path character after it, except a closing sentence dot).
    """
    alternation = "|".join(re.escape(k) for k in keys)
    return re.compile(
        rf"(?<![{PATH_CHAR_CLASS}])(?P<path>{alternation})(?P<suffix>{LINE_SUFFIX})?"
        r"(?![A-Za-z0-9_\-/]|\.[A-Za-z0-9_\-/])"
    )


def _token_matcher(mapping: dict[str, str]) -> re.Pattern[str] | None:
    """Return the token regex for a mapping, or None when it has no keys."""
    if not mapping:
        return None
    return _token_regex(tuple(sorted(mapping, key=lambda k: (-len(k), k))))


def _rewrite_tokens(
    piece: str, matcher: re.Pattern[str] | None, mapping: dict[str, str]
) -> tuple[str, int]:
    """Replace every exact mapped path token in `piece`; return (text, replacements)."""
    if matcher is None:
        return piece, 0
    count = 0

    def _swap(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return mapping[match.group("path")] + (match.group("suffix") or "")

    return matcher.sub(_swap, piece), count


def _needs_attention(target: str, old_dir: str, new_dir: str, mapping: dict[str, str]) -> bool:
    """Return True when a link we will not rewrite points at a moved file or would break."""
    if not target or target.startswith("#") or URL_SCHEME.match(target):
        return False
    path = target.split("#", 1)[0].split("?", 1)[0]
    if not path:
        return False
    absolute = path.startswith("/")
    resolved = posixpath.normpath(path.lstrip("/") if absolute else posixpath.join(old_dir, path))
    return resolved in mapping or (old_dir != new_dir and not absolute)


def _rewrite_markdown_text(
    text: str,
    file_old: str,
    file_new: str,
    mapping: dict[str, str],
    exists: Callable[[str], bool] | None,
) -> tuple[str, list[Unresolved], int]:
    """Rewrite one Markdown file: exact path tokens plus recomputed inline links."""
    matcher = _token_matcher(mapping)
    old_dir, new_dir = posixpath.dirname(file_old), posixpath.dirname(file_new)
    unresolved: list[Unresolved] = []
    out: list[str] = []
    count = 0
    fence: tuple[str, int] | None = None

    def _note(number: int, line: str, start: int, reason: str) -> None:
        unresolved.append(Unresolved("", number, _snippet(line, start), reason))

    def _retarget(match: re.Match[str], number: int, line: str) -> str:
        nonlocal count
        raw = match.group("target")
        new_raw, reason = _rewrite_link_target(raw, old_dir, new_dir, mapping, frozenset(), exists)
        # Only a moved file's dead old-tree link matters here: it cannot be recomputed.
        if reason is not None and old_dir != new_dir:
            _note(number, line, match.start(), reason)
        if new_raw == raw:
            return match.group(0)
        count += 1
        start, end = match.span("target")
        whole, base = match.group(0), match.start()
        return whole[: start - base] + new_raw + whole[end - base :]

    def _plain(piece: str, number: int, line: str) -> str:
        """Rewrite a stretch of ordinary text: links by position, tokens everywhere else."""
        nonlocal count
        parsed = {m.start() for m in INLINE_LINK.finditer(piece)}
        for start in _unsupported_links(piece, parsed, old_dir, new_dir):
            _note(number, line, start, REASON_UNSUPPORTED)
        spans: list[tuple[int, int, str, re.Match[str]]] = [
            (m.start(), m.end(), "link", m) for m in INLINE_LINK.finditer(piece)
        ]
        for html in HTML_LINK.finditer(piece):
            if _needs_attention(html.group("target"), old_dir, new_dir, mapping):
                _note(number, line, html.start(), REASON_INDIRECT)
            spans.append((html.start("target"), html.end("target"), "html", html))
        spans.sort(key=lambda s: s[0])
        pieces: list[str] = []
        cursor = 0
        for start, end, kind, match in spans:
            if start < cursor:
                continue
            gap, found = _rewrite_tokens(piece[cursor:start], matcher, mapping)
            count += found
            pieces.append(gap)
            pieces.append(_retarget(match, number, line) if kind == "link" else piece[start:end])
            cursor = end
        gap, found = _rewrite_tokens(piece[cursor:], matcher, mapping)
        count += found
        pieces.append(gap)
        return "".join(pieces)

    for number, line in enumerate(text.split("\n"), start=1):
        if fence is None:
            fence = _fence_opener(line)
            opened = fence is not None
        else:
            opened = False
        if fence is not None:
            # Fenced code is never modified; exact mentions are listed for a person.
            if matcher is not None:
                for match in matcher.finditer(line):
                    _note(number, line, match.start(), REASON_FENCED)
            if not opened and _fence_closes(line, fence):
                fence = None
            out.append(line)
            continue
        definition = REF_DEFINITION.match(line)
        if definition:
            if _needs_attention(definition.group("target"), old_dir, new_dir, mapping):
                _note(number, line, definition.start("target"), REASON_INDIRECT)
            out.append(line)
            continue
        pieces = []
        for is_code, piece in _code_span_segments(line):
            if is_code:
                new_piece, found = _rewrite_tokens(piece, matcher, mapping)
                count += found
                pieces.append(new_piece)
            else:
                pieces.append(_plain(piece, number, line))
        out.append("".join(pieces))
    return "\n".join(out), unresolved, count


def _rewrite_text_counted(
    text: str,
    file_old: str,
    file_new: str,
    mapping: dict[str, str],
    exists: Callable[[str], bool] | None = None,
) -> tuple[str, list[Unresolved], int]:
    """Implement `rewrite_text`, also returning the number of replacements made."""
    if PurePosixPath(file_new).suffix.lower() == MARKDOWN_SUFFIX:
        return _rewrite_markdown_text(text, file_old, file_new, mapping, exists)
    new_text, count = _rewrite_tokens(text, _token_matcher(mapping), mapping)
    return new_text, [], count


def rewrite_text(
    text: str,
    file_old: str,
    file_new: str,
    mapping: dict[str, str],
    *,
    exists: Callable[[str], bool] | None = None,
) -> tuple[str, list[Unresolved]]:
    """Rewrite references to moved files in one text file.

    Markdown files get exact repo-path tokens (backticked or plain, with an optional
    `:N`, `:N-M` or `:N,M` suffix) rewritten, and every inline link target recomputed:
    resolved against the file's old directory, mapped, and written relative to its new
    directory (`#fragment` and title kept). Links from a moved file to an unmoved target
    are recomputed too. Other text files get exact tokens only. A token counts only when it
    equals a mapping key, so directory mentions are untouched. URLs, `mailto:`, pure anchors
    and fenced code are left alone; reference-style and HTML links that point at a moved file
    (or would break because the file moved) are reported, never guessed.

    Args:
        text: File contents (line endings are preserved).
        file_old: Repo-relative path the file had when its links were written.
        file_new: Repo-relative path the file has after the move (same as old if unmoved).
        mapping: Old repo-relative path to new repo-relative path.
        exists: Optional repo-relative existence check; lets a rerun recognise a link that
            was already written for the new directory.

    Returns:
        (new text, unresolved mentions with an empty `file`).
    """
    new_text, unresolved, _ = _rewrite_text_counted(text, file_old, file_new, mapping, exists)
    return new_text, unresolved


# --- Rewrite run --------------------------------------------------------------------------


def _frozen_set(manifest: Path | None, mapping: dict[str, str]) -> frozenset[str]:
    """Return every path (before and after the second move) the first manifest freezes."""
    if manifest is None:
        return frozenset()
    paths = reorg._frozen_paths(reorg.load_manifest(manifest))
    return frozenset(paths | {mapping[p] for p in paths if p in mapping})


def _is_frozen(path: str, frozen: Iterable[str]) -> bool:
    """Return True for a manifest-frozen path or any machine artifact inside research/."""
    return path in frozen or (path.startswith(f"{NEW_ROOT}/") and reorg.is_frozen(path))


def _virtual_exists(
    root: Path, tracked: list[str], mapping: dict[str, str]
) -> Callable[[str], bool]:
    """Return an existence check that sees the tree as it is once every move is done."""
    files = {mapping.get(p, p) for p in tracked}
    folders = {parent for p in files for parent in map(str, PurePosixPath(p).parents)}

    def _exists(rel: str) -> bool:
        if reorg._escapes_root(rel):
            return False
        if rel in mapping:
            return False
        return rel in files or rel in folders or (root / rel).exists()

    return _exists


def _skip_reason(final: str, current: str, extra: list[str]) -> str | None:
    """Return why a file is out of the rewrite scope, or None when it is in scope."""
    for path in (final, current):
        if (
            reorg._is_reorg_own_file(path)
            or path in SELF_PATHS
            or path in (PLAN_PATH, MOVES_CSV)
            or path.startswith(EXCLUDED_PREFIXES)
            or reorg._is_rewrite_excluded(path, extra)
        ):
            return "excluded"
    return None


def run_rewrite(
    root: Path,
    rows: list[MoveRow],
    dry_run: bool,
    report: Path | None = None,
    *,
    manifest: Path | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Rewrite references to moved files across the tracked text files.

    Scope: every tracked file that decodes as UTF-8, except frozen paths (from the
    first-generation manifest, plus machine artifacts inside research/), the reorganization's
    own files, this plan, `exclude` paths and anything under `.superpowers/`, `data/`,
    `results/`. A moved file is read at its new location (or its old one before the move) and
    its links are recomputed from its old directory to its new one.

    Args:
        root: Repository root.
        rows: Move rows (the mapping is old -> new).
        dry_run: When True nothing is written, except the report if one is given.
        report: Optional path of the JSON report to write.
        manifest: First-generation manifest CSV; its frozen rows are never rewritten.
        exclude: Extra repo-relative files or folders to leave alone (e.g. the moves CSV).

    Returns:
        The report: `status`, `error`, `changed`, `written`, `unresolved`, `excluded` and
        `rewrite_manifest` (`{"files": [...], "hand_edited": []}`, final paths).
    """
    mapping = {r.old: r.new for r in rows}
    reverse = {r.new: r.old for r in rows}
    frozen = _frozen_set(manifest, mapping)
    extra = list(exclude)
    tracked = reorg.git_tracked_files(root)
    exists = _virtual_exists(root, tracked, mapping)
    excluded: list[str] = []
    skipped_frozen = 0
    undecodable: list[str] = []
    unreadable: list[str] = []
    changed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []
    for current in tracked:
        final = mapping.get(current, current)
        file_old = reverse.get(current, current)
        if _skip_reason(final, current, extra) is not None or (root / current).is_symlink():
            excluded.append(final)
            continue
        if _is_frozen(final, frozen) or _is_frozen(current, frozen):
            excluded.append(final)
            skipped_frozen += 1
            continue
        try:
            data = (root / current).read_bytes()
        except OSError as exc:
            unreadable.append(f"{current}: {exc.strerror or exc}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            undecodable.append(final)
            continue
        if text.split("\n", 1)[0].rstrip("\r") == MOVES_HEADER:
            excluded.append(final)  # any moves CSV describes old paths on purpose
            continue
        new_text, found, count = _rewrite_text_counted(text, file_old, final, mapping, exists)
        unresolved.extend(asdict(replace(u, file=final)) for u in found)
        if new_text != text:
            changed.append({"file": final, "replacements": count})
            pending.append((current, new_text))
    written: list[str] = []
    error = ""
    if unreadable:
        error = "pre-flight failed, nothing written: unreadable " + "; ".join(sorted(unreadable))
    elif not dry_run:
        for current, new_text in sorted(pending):
            try:
                reorg.write_atomic(root / current, new_text.encode("utf-8"))
            except OSError as exc:
                error = f"write failed for {current}: {exc}; already written: {len(written)} files"
                break
            written.append(mapping.get(current, current))
    result: dict[str, Any] = {
        "dry_run": dry_run,
        "status": reorg.STATUS_FAILED if error else reorg.STATUS_OK,
        "error": error,
        "changed": sorted(changed, key=lambda c: c["file"]),
        "written": sorted(written),
        "unresolved": sorted(
            unresolved, key=lambda u: (u["file"], u["line"], u["text"], u["reason"])
        ),
        "skipped_frozen": skipped_frozen,
        "excluded": sorted(excluded),
        "skipped_undecodable": sorted(undecodable),
        "skipped_unreadable": sorted(unreadable),
        "rewrite_manifest": {"files": sorted(c["file"] for c in changed), "hand_edited": []},
    }
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(_dump_json(result), encoding="utf-8")
    return result


# --- Snapshot and verification ------------------------------------------------------------


def snapshot_moves(rows: list[MoveRow], root: Path) -> dict[str, str]:
    """Return {old path: sha256} for every moved file, read from the working tree.

    Raises:
        FileNotFoundError: If an old path is not a file on disk.
    """
    return {r.old: reorg.sha256_file(root / r.old) for r in rows}


def _git_lines(root: Path, *args: str) -> list[str]:
    """Run a read-only git command and return its NUL-separated output as a list."""
    out = subprocess.run(
        ["git", *args, "-z"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return [p for p in out.split("\0") if p]


def changed_since(root: Path, base: str) -> set[str]:
    """Return paths changed in the working tree or index vs `base`, plus untracked files."""
    changed = set(_git_lines(root, "diff", "--name-only", "--no-renames", base))
    changed.update(_git_lines(root, "ls-files", "--others", "--exclude-standard"))
    return changed


def verify_moved_tree(
    rows: list[MoveRow],
    before: dict[str, str],
    rewrite_files: set[str],
    root: Path,
    *,
    base: str = DEFAULT_BASE,
    frozen: Iterable[str] = (),
    allow: Iterable[str] = (),
) -> list[str]:
    """Check the tree after the second move and rewrite against the pre-move snapshot.

    Every moved file's sha256 must equal its snapshot unless its new path is a non-frozen
    Markdown file in `rewrite_files`; a frozen moved file must always be equal. Every file
    changed against `base` must be a moved file (old or new path), listed in `rewrite_files`
    or in `allow`. Every listed rewrite file must be a tracked, non-frozen UTF-8 text file
    that really differs from `base`.

    Args:
        rows: Move rows.
        before: `{old path: sha256}` from `snapshot_moves`.
        rewrite_files: Final paths of the files the rewrite changed.
        root: Repository root.
        base: Revision to compare the working tree with.
        frozen: Paths (old or new) the first manifest freezes.
        allow: Extra paths allowed to differ (e.g. the moves CSV).

    Returns:
        Problem descriptions, in a stable order; empty when the tree is clean.
    """
    frozen_set = frozenset(frozen)
    problems: list[str] = []
    moved_paths = {p for r in rows for p in (r.old, r.new)}
    new_to_old = {r.new: r.old for r in rows}
    for row in rows:
        dest = root / row.new
        if not dest.is_file():
            problems.append(f"moved file missing: {row.new}")
            continue
        expected = before.get(row.old)
        if expected is None:
            problems.append(f"no snapshot hash for: {row.old}")
        elif reorg.sha256_file(dest) != expected:
            is_frozen = _is_frozen(row.new, frozen_set) or _is_frozen(row.old, frozen_set)
            if is_frozen:
                problems.append(f"frozen moved file changed: {row.new}")
            elif not row.new.endswith(MARKDOWN_SUFFIX):
                problems.append(f"moved non-Markdown file changed: {row.new}")
            elif row.new not in rewrite_files:
                problems.append(f"moved file changed and not listed: {row.new}")
    changed = changed_since(root, base)
    allowed = moved_paths | rewrite_files | set(allow)
    problems.extend(
        f"changed file is neither moved nor listed: {path}" for path in sorted(changed - allowed)
    )
    tracked = set(reorg.git_tracked_files(root))
    for path in sorted(rewrite_files):
        if _is_frozen(path, frozen_set) or _is_frozen(new_to_old.get(path, path), frozen_set):
            problems.append(f"listed rewrite file is frozen: {path}")
            continue
        file = root / path
        if path not in tracked or not file.is_file():
            problems.append(f"listed rewrite file is not a tracked file: {path}")
            continue
        try:
            file.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"listed rewrite file is not UTF-8 text: {path}")
            continue
        if path in new_to_old:
            differs = reorg.sha256_file(file) != before.get(new_to_old[path])
        else:
            differs = path in changed
        if not differs:
            problems.append(f"listed rewrite file does not differ from the base: {path}")
    return problems


# --- Manifest update ----------------------------------------------------------------------


def _dump_json(data: Any) -> str:
    """Render JSON the way the reorganization files are written (2-space indent, final LF)."""
    return json.dumps(data, indent=JSON_INDENT, ensure_ascii=False) + "\n"


def _write_if_changed(path: Path, text: str) -> bool:
    """Atomically replace `path` with `text` unless it already holds exactly that."""
    data = text.encode("utf-8")
    if path.read_bytes() == data:
        return False
    reorg.write_atomic(path, data)
    return True


def _map_paths(paths: list[str], mapping: dict[str, str]) -> list[str]:
    """Map moved entries of a path list; keep the list sorted if it was sorted."""
    mapped = [mapping.get(p, p) for p in paths]
    return sorted(mapped) if paths == sorted(paths) else mapped


def _is_allow_new(path: str) -> bool:
    """Return True when a path is one the first-generation verify allows without a row."""
    return reorg._matches_allow_new(path, reorg.DEFAULT_ALLOW_NEW)


def update_manifest(
    rows: list[MoveRow],
    manifest: Path,
    ledger: Path | None = None,
    rewrite_manifest: Path | None = None,
) -> dict[str, int]:
    """Point the first-generation records at the second-move paths (atomic, idempotent).

    Rewrites `proposed_path` and `folder` of every moved row in the manifest CSV, the
    per-file `proposed_path` in the ledger JSON, and the `files` and `hand_edited` lists of
    the rewrite manifest JSON. Nothing is written when any move has no matching row.

    Args:
        rows: Move rows (old -> new).
        manifest: First-generation manifest CSV.
        ledger: Optional ledger JSON.
        rewrite_manifest: Optional first-generation rewrite manifest JSON.

    Returns:
        Counts: updated rows (`manifest`, `ledger`, `rewrite_manifest`) and `unlisted`
        (allow-new moves that never had a manifest row and were skipped).

    Raises:
        ValueError: If a move's old path is neither the `proposed_path` of exactly one row
            nor already the new path (a stale or conflicting row), unless the destination
            matches the first-generation allow-new list; or the header differs.
    """
    mapping = {r.old: r.new for r in rows}
    with manifest.open(newline="", encoding="utf-8") as handle:
        table = list(csv.reader(handle))
    header, body = table[0], table[1:]
    if tuple(header) != reorg.MANIFEST_COLUMNS:
        raise ValueError(f"{manifest}: unexpected header {header}")
    proposed_col, folder_col = header.index("proposed_path"), header.index("folder")
    errors: list[str] = []
    updates = 0
    unlisted = 0
    for move in rows:
        hits = [r for r in body if r[proposed_col] == move.old]
        done = [r for r in body if r[proposed_col] == move.new]
        if len(hits) > 1:
            errors.append(f"{len(hits)} manifest rows have proposed_path {move.old}")
        elif not hits and not done:
            if _is_allow_new(move.new):
                # Added after the first-generation manifest, so it never had a row.
                print(f"NOTE unlisted allow-new move: {move.old} -> {move.new}")
                unlisted += 1
            else:
                errors.append(f"no manifest row has proposed_path {move.old} (move to {move.new})")
        for row in hits:
            row[proposed_col] = move.new
            row[folder_col] = posixpath.dirname(move.new)
            updates += 1
    if errors:
        raise ValueError("; ".join(errors))
    buffer = io.StringIO()
    # The first-generation file leaves its header bare and quotes every data field.
    csv.writer(buffer, lineterminator="\n").writerow(header)
    csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(body)
    counts = {"manifest": updates, "ledger": 0, "rewrite_manifest": 0, "unlisted": unlisted}
    _write_if_changed(manifest, buffer.getvalue())
    if ledger is not None:
        data = json.loads(ledger.read_text(encoding="utf-8"))
        for entry in data.get("files", []):
            if entry.get("proposed_path") in mapping:
                entry["proposed_path"] = mapping[entry["proposed_path"]]
                counts["ledger"] += 1
        _write_if_changed(ledger, _dump_json(data))
    if rewrite_manifest is not None:
        data = json.loads(rewrite_manifest.read_text(encoding="utf-8"))
        for key in ("files", "hand_edited"):
            if key in data:
                counts["rewrite_manifest"] += sum(p in mapping for p in data[key])
                data[key] = _map_paths(data[key], mapping)
        _write_if_changed(rewrite_manifest, _dump_json(data))
    return counts


# --- Command line -------------------------------------------------------------------------


def _root(args: argparse.Namespace) -> Path:
    """Return the resolved repository root from `--root`."""
    return Path(args.root).resolve()


def _rel_to_root(path: Path, root: Path) -> str | None:
    """Return `path` relative to `root` as POSIX text, or None when it lies outside."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _allow_deeper(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the default deeper study folders plus any `--allow-deeper` values."""
    return (*DEFAULT_ALLOW_DEEPER, *(args.allow_deeper or ()))


def _cmd_check(args: argparse.Namespace) -> int:
    """Run `check`: print every problem and return 1 if there is any."""
    rows = load_moves(Path(args.moves))
    problems = check_moves(rows, set(reorg.git_tracked_files(_root(args))), _allow_deeper(args))
    for problem in problems:
        print(f"PROBLEM {problem}")
    if problems:
        print(f"FAIL {len(problems)} problems in {len(rows)} moves")
        return 1
    print(f"OK {len(rows)} moves")
    return 0


def _cmd_move(args: argparse.Namespace) -> int:
    """Run `move`: `git mv` every file, or print the plan with `--dry-run`."""
    rows = load_moves(Path(args.moves))
    try:
        pairs = apply_moves(rows, _root(args), args.dry_run, _allow_deeper(args))
    except MoveRefusedError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(f"OK {'planned' if args.dry_run else 'moved'} {len(pairs)} files")
    return 0


def _cmd_rewrite(args: argparse.Namespace) -> int:
    """Run `rewrite`: retarget references, write the report and the rewrite manifest."""
    root = _root(args)
    moves = Path(args.moves)
    rows = load_moves(moves)
    manifest = Path(args.manifest) if args.manifest else root / DEFAULT_MANIFEST
    exclude = [rel for rel in [_rel_to_root(moves, root)] if rel]
    result = run_rewrite(
        root,
        rows,
        args.dry_run,
        Path(args.report) if args.report else None,
        manifest=manifest,
        exclude=exclude,
    )
    if args.manifest_out and not args.dry_run and result["status"] == reorg.STATUS_OK:
        out = Path(args.manifest_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_dump_json(result["rewrite_manifest"]), encoding="utf-8")
    verb = "would change" if args.dry_run else "changed"
    print(
        f"{result['status'].upper()} {verb} {len(result['changed'])} files, "
        f"{len(result['unresolved'])} unresolved, {len(result['excluded'])} excluded"
    )
    if result["error"]:
        print(result["error"], file=sys.stderr)
    return 0 if result["status"] == reorg.STATUS_OK else 1


def _cmd_snapshot(args: argparse.Namespace) -> int:
    """Run `snapshot`: write the sha256 of every file about to move."""
    rows = load_moves(Path(args.moves))
    before = snapshot_moves(rows, _root(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dump_json(dict(sorted(before.items()))), encoding="utf-8")
    print(f"OK snapshot {len(before)} files")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Run `verify`: check the moved and rewritten tree, print each problem."""
    root = _root(args)
    rows = load_moves(Path(args.moves))
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    rewrite_files = reorg.load_rewrite_manifest(Path(args.rewrite_manifest))
    manifest = Path(args.manifest) if args.manifest else root / DEFAULT_MANIFEST
    frozen = _frozen_set(manifest, {r.old: r.new for r in rows})
    problems = verify_moved_tree(
        rows,
        before,
        rewrite_files,
        root,
        base=args.base,
        frozen=frozen,
        allow=args.allow or (),
    )
    for path in args.allow or ():
        if _is_frozen(path, frozen):
            print(f"NOTE allowed frozen path: {path}")
    for problem in problems:
        print(f"PROBLEM {problem}")
    if problems:
        print(f"FAIL {len(problems)} problems")
        return 1
    print(f"OK {len(rows)} moved files, {len(rewrite_files)} rewritten files")
    return 0


def _cmd_update_manifest(args: argparse.Namespace) -> int:
    """Run `update-manifest`: point the first-generation records at the new paths."""
    rows = load_moves(Path(args.moves))
    try:
        counts = update_manifest(
            rows,
            Path(args.manifest),
            Path(args.ledger) if args.ledger else None,
            Path(args.rewrite_manifest) if args.rewrite_manifest else None,
        )
    except ValueError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(f"OK updated {counts}")
    return 0


def _add_common(parser: argparse.ArgumentParser, moves: bool = True) -> None:
    """Add `--moves` (optional) and `--root` to a subcommand parser."""
    if moves:
        parser.add_argument("--moves", required=True, help="Moves CSV (old_path,new_path).")
    parser.add_argument(
        "--root", default=str(REPO_ROOT), help="Repository root (default: this repository)."
    )


def _add_deeper(parser: argparse.ArgumentParser) -> None:
    """Add the repeatable `--allow-deeper` option to a subcommand parser."""
    parser.add_argument(
        "--allow-deeper",
        action="append",
        metavar="PREFIX",
        help=(
            "Study folder (relative to research/; repeatable) whose files may sit one level "
            "deeper. Adds to the default: " + ", ".join(DEFAULT_ALLOW_DEEPER)
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Check a moves CSV against git and the layout rules.")
    _add_common(check)
    _add_deeper(check)
    check.set_defaults(func=_cmd_check)
    move = sub.add_parser("move", help="git mv every listed file (refuses on any problem).")
    _add_common(move)
    _add_deeper(move)
    move.add_argument("--dry-run", action="store_true", help="Print the plan; change nothing.")
    move.set_defaults(func=_cmd_move)
    rewrite = sub.add_parser("rewrite", help="Retarget references to the moved files.")
    _add_common(rewrite)
    rewrite.add_argument("--dry-run", action="store_true", help="Report only; write no file.")
    rewrite.add_argument("--report", help="Path of the JSON report to write (also on dry run).")
    rewrite.add_argument(
        "--manifest-out", help="Path of the rewrite-manifest JSON to write (not on dry run)."
    )
    rewrite.add_argument(
        "--manifest",
        help=f"First-generation manifest CSV that freezes files (default: {DEFAULT_MANIFEST}).",
    )
    rewrite.set_defaults(func=_cmd_rewrite)
    snapshot = sub.add_parser("snapshot", help="Record the sha256 of every file about to move.")
    _add_common(snapshot)
    snapshot.add_argument("--out", required=True, help="Path of the snapshot JSON to write.")
    snapshot.set_defaults(func=_cmd_snapshot)
    verify = sub.add_parser("verify", help="Check the moved and rewritten tree (exit 1 on any).")
    _add_common(verify)
    verify.add_argument("--before", required=True, help="Snapshot JSON from `snapshot`.")
    verify.add_argument("--base", default=DEFAULT_BASE, help="Revision to diff against.")
    verify.add_argument("--rewrite-manifest", required=True, help="Rewrite-manifest JSON.")
    verify.add_argument(
        "--manifest", help=f"First-generation manifest CSV (default: {DEFAULT_MANIFEST})."
    )
    verify.add_argument(
        "--allow",
        action="append",
        metavar="PATH",
        help="Extra path allowed to differ from the base (repeatable).",
    )
    verify.set_defaults(func=_cmd_verify)
    update = sub.add_parser(
        "update-manifest",
        help="Point the first manifest, ledger and rewrite manifest at new paths.",
    )
    _add_common(update)
    update.add_argument("--manifest", required=True, help="First-generation manifest CSV.")
    update.add_argument("--ledger", help="First-generation ledger JSON.")
    update.add_argument("--rewrite-manifest", help="First-generation rewrite manifest JSON.")
    update.set_defaults(func=_cmd_update_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
