"""Driver for the dead-stock fresh-window study: anchor check and run-once fresh window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ggTrader.utils.binding import binding, git_sha
from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import fp_dead_stock_first_test as fp  # noqa: E402  (found on sys.path through bootstrap)
import fresh_compare  # noqa: E402
import fresh_inputs  # noqa: E402

STUDY_DIR = study_dir(__file__)
CODE_DIR = Path(__file__).resolve().parent
STUDY_ID = "dead-stock-fresh-window"
BRIEF = "2026-11-24-dead-stock-fresh-window-brief.md"
UNIVERSE = fresh_inputs.UNIVERSE
DEFAULT_RESULTS_DIR = ROOT / "results" / "dead_stock_fresh_window"
DEFAULT_SAVED_HOLDOUT = STUDY_DIR.parent / "dead-stock-first-test" / "_holdout-A.json"
ANCHOR_FILENAME = f"_anchor-{UNIVERSE}.json"
FRESH_FILENAME = f"_fresh-{UNIVERSE}.json"
STUDY_CODE_REL = "research/stock-signals/dead-stock-fresh-window/code/"
STUDY_CODE_TESTS_REL = "research/stock-signals/dead-stock-fresh-window/code/tests/"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_STRATEGY_CELLS = 18
EXPECTED_SPY_CELLS = 6
CODE_CHANGE_PREFIXES: tuple[str, ...] = (
    "src/",
    "research/stock-signals/_lib/",
    STUDY_CODE_REL,
)
PROTECTED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "dead-stock-first-test",
        "dead-stock-payout-rerun",
        "dead-stock-data-integrity",
        "dead_stock_first_test",
        "dead_stock_payout_rerun",
        "dead_stock_data_integrity",
    }
)
PROTECTED_DIRS: tuple[Path, ...] = (
    STUDY_DIR.parent / "dead-stock-first-test",
    STUDY_DIR.parent / "dead-stock-payout-rerun",
    STUDY_DIR.parent / "dead-stock-data-integrity",
    ROOT / "results" / "dead_stock_first_test",
    ROOT / "results" / "dead_stock_payout_rerun",
    ROOT / "results" / "dead_stock_data_integrity",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (pinned window names, fixed output paths)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        choices=sorted(fresh_inputs.WINDOWS),
        required=True,
        help="registered window: anchor (2020-2024 check) or fresh (2000-2004, run once)",
    )
    parser.add_argument(
        "--release-id",
        type=int,
        required=True,
        help="published combined release id (pinned to 1)",
    )
    parser.add_argument(
        "--crsp-load",
        type=int,
        default=fp.PIN_CRSP_LOAD,
        help="raw CRSP load id for eligibility (pinned to 1)",
    )
    return parser


def _refuse(message: str) -> int:
    """Print a plain-word refusal to stderr and return exit code 2."""
    print(message, file=sys.stderr)
    return 2


def _sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _code_hashes(code_dir: Path = CODE_DIR) -> dict[str, str]:
    """Return SHA-256 digests of every non-test Python file under the study's ``code/`` folder."""
    hashes: dict[str, str] = {}
    for path in sorted(code_dir.rglob("*.py")):
        rel_parts = path.relative_to(code_dir).parts
        if "tests" in rel_parts or "__pycache__" in rel_parts:
            continue
        if path.name.startswith("test_"):
            continue
        hashes[path.relative_to(code_dir).as_posix()] = _sha256_bytes(path.read_bytes())
    return hashes


def brief_commit() -> str | None:
    """Return the commit hash that last touched the frozen brief (``None`` outside git)."""
    path = STUDY_DIR / BRIEF
    try:
        done = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _git_changed_paths(anchor_commit: str) -> list[str]:
    """Return repo-relative paths modified between ``anchor_commit`` and ``HEAD``."""
    done = subprocess.run(
        ["git", "diff", "--name-only", anchor_commit, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def _is_code_path(path: str) -> bool:
    """Return True when a repo-relative path lies under code bound by the anchor check."""
    normalized = path.replace("\\", "/").lstrip("./")
    return any(normalized.startswith(prefix) for prefix in CODE_CHANGE_PREFIXES)


def _git_ls_files(root: Path, *extra_args: str) -> list[str]:
    """Run ``git ls-files -z`` in ``root`` and return non-empty repo-relative POSIX paths."""
    done = subprocess.run(
        ["git", "ls-files", *extra_args, "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [item.replace("\\", "/") for item in done.stdout.split("\x00") if item]


def untracked_code_problems(root: Path = ROOT) -> list[str]:
    """Return plain-word problems for untracked Python files or non-tracked loaded modules.

    Performs two checks against ``root``:
    1. Every untracked or gitignored ``.py`` file under ``src/``, ``research/stock-signals/_lib/``,
       or this study's ``code/`` folder (skipping ``__pycache__`` and the study's ``code/tests/``).
    2. Every module in ``sys.modules`` whose ``__file__`` resolves inside ``root`` (excluding
       ``root/.venv``) must belong to git's tracked file set, catching untracked packages or
       modules that shadow a tracked sibling.

    Args:
        root: Repository checkout root to inspect.

    Returns:
        Sorted plain-word problem descriptions (empty when clean).
    """
    try:
        others = _git_ls_files(root, "--others", "--exclude-standard")
        ignored = _git_ls_files(root, "--others", "--ignored", "--exclude-standard")
        tracked_rel = _git_ls_files(root)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"cannot inspect git tracked/untracked files under {root}: {exc}"]

    problems: list[str] = []

    untracked_py: set[str] = set()
    for rel in (*others, *ignored):
        if not rel.endswith(".py"):
            continue
        if not _is_code_path(rel):
            continue
        if rel.startswith(STUDY_CODE_TESTS_REL):
            continue
        if "__pycache__" in Path(rel).parts:
            continue
        untracked_py.add(rel)
    for rel in sorted(untracked_py):
        problems.append(f"untracked Python file in code tree: {rel}")

    resolved_root = root.resolve(strict=False)
    venv_dir = resolved_root / ".venv"
    tracked_paths = {(resolved_root / rel).resolve(strict=False) for rel in tracked_rel}
    bad_modules: list[tuple[str, str]] = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if not isinstance(mod_file, str) or not mod_file:
            continue
        resolved_file = Path(mod_file).resolve(strict=False)
        if not resolved_file.is_relative_to(resolved_root):
            continue
        if resolved_file.is_relative_to(venv_dir):
            continue
        if resolved_file not in tracked_paths:
            rel_display = resolved_file.relative_to(resolved_root).as_posix()
            bad_modules.append((name, rel_display))
    for name, rel_display in sorted(bad_modules):
        problems.append(f"loaded module {name!r} resolves to non-tracked file: {rel_display}")

    return problems


def _lexical_abs(path: Path) -> Path:
    """Return the normalized lexical absolute path without following symlinks."""
    return Path(os.path.abspath(path))


def _resolved(path: Path) -> Path:
    """Return the fully resolved path (following symlinks and ``..``), allowing missing targets."""
    return path.resolve(strict=False)


def _is_inside(path: Path, folder: Path) -> bool:
    """Return True when ``path`` lies inside ``folder`` lexically or after resolution."""
    raw_folder = _lexical_abs(folder)
    resolved_folder = _resolved(folder)
    raw_path = _lexical_abs(path)
    resolved_path = _resolved(path)
    return (
        raw_path.is_relative_to(raw_folder)
        or raw_path.is_relative_to(resolved_folder)
        or resolved_path.is_relative_to(resolved_folder)
        or resolved_path.is_relative_to(raw_folder)
    )


def _validate_output_paths(outputs: Sequence[tuple[str, Path]]) -> str | None:
    """Refuse any output path that is a symlink or resolves inside a protected sibling study."""
    current_label = "output path"
    try:
        for label, path in outputs:
            current_label = label
            if path.is_symlink():
                return f"refused: {label} {path} is a symlink; refusing to write through a symlink"
            for protected in PROTECTED_DIRS:
                if _is_inside(path, protected):
                    return (
                        f"refused: {label} {path} resolves inside protected directory {protected}; "
                        "use this study's own folder and results directory"
                    )
            raw_parts = set(_lexical_abs(path).parts)
            resolved_parts = set(_resolved(path).parts)
            forbidden = sorted((raw_parts | resolved_parts) & PROTECTED_DIR_NAMES)
            if forbidden:
                return (
                    f"refused: {label} {path} resolves inside protected folder "
                    f"{', '.join(forbidden)}; use this study's own folder and results directory"
                )
    except (OSError, RuntimeError, ValueError) as exc:
        return f"refused: cannot validate {current_label}: {type(exc).__name__}"
    return None


def _read_saved_holdout(
    path: Path, *, prefix: str = "anchor refused"
) -> tuple[dict[str, Any], str] | tuple[None, str]:
    """Read and validate the saved first-test holdout JSON file for the anchor comparison."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"{prefix}: cannot read saved first-test file {path}: {exc}"
    digest = _sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{prefix}: saved first-test file {path} is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{prefix}: saved first-test file {path} must contain a JSON object"
    result_block = payload.get("result")
    if not isinstance(result_block, dict):
        return None, f"{prefix}: saved first-test file {path} has no result mapping"
    return payload, digest


def _is_exact_int(value: Any, expected: int) -> bool:
    """Return True when ``value`` is a non-boolean int equal to ``expected``."""
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _verify_anchor_for_fresh(
    anchor_path: Path,
    saved_holdout_path: Path,
    current_hashes: Mapping[str, str],
    changed_paths_fn: Callable[[str], list[str]],
) -> tuple[dict[str, Any], str] | tuple[None, str]:
    """Verify ``_anchor-A.json`` exists, matches the saved first-test file, and binds this code.

    Re-checks anchor metadata (study, window, dates, universe, pins), comparison summary, current
    SHA-256 of the saved first-test holdout file, and every cell of ``anchor["result"]`` against
    the saved holdout via ``fresh_compare.compare_to_saved``, in addition to code hashes and git
    diff since ``anchor_commit``.

    Scope: this defends against stale, damaged, wrong-window, or lazily hand-written anchor files.
    A deliberately forged complete anchor (a copy of the saved result block) cannot be told apart
    without a trusted receipt and is out of scope because the only user is the owner's own session.
    """
    if not anchor_path.exists():
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} is missing; "
            "run --window anchor --release-id 1 first",
        )
    try:
        raw = anchor_path.read_bytes()
    except OSError as exc:
        return (
            None,
            f"fresh run refused: cannot read anchor file {anchor_path}: {exc}; "
            "run --window anchor --release-id 1 first",
        )
    anchor_sha = _sha256_bytes(raw)
    try:
        anchor = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} is not valid JSON ({exc}); "
            "re-run --window anchor --release-id 1",
        )
    if not isinstance(anchor, dict):
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} must contain a JSON object; "
            "re-run --window anchor --release-id 1",
        )
    if anchor.get("anchor_ok") is not True:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} does not have anchor_ok=true; "
            "re-run --window anchor --release-id 1 until the check matches",
        )
    if anchor.get("study") != STUDY_ID:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has study={anchor.get('study')!r}, "
            f"expected {STUDY_ID!r}; re-run --window anchor --release-id 1",
        )
    if anchor.get("window") != "anchor":
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has window={anchor.get('window')!r}, "
            "expected 'anchor'; re-run --window anchor --release-id 1",
        )
    anchor_start, anchor_end = fresh_inputs.WINDOWS["anchor"]
    expected_dates = {"start": anchor_start, "end": anchor_end}
    if anchor.get("window_dates") != expected_dates:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has "
            f"window_dates={anchor.get('window_dates')!r}, expected {expected_dates!r}; "
            "re-run --window anchor --release-id 1",
        )
    if anchor.get("universe") != UNIVERSE:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has "
            f"universe={anchor.get('universe')!r}, expected {UNIVERSE!r}; "
            "re-run --window anchor --release-id 1",
        )
    if not _is_exact_int(anchor.get("release_id"), fp.PIN_RELEASE_ID) or not _is_exact_int(
        anchor.get("crsp_load"), fp.PIN_CRSP_LOAD
    ):
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} was run on release "
            f"{anchor.get('release_id')} / load {anchor.get('crsp_load')}, expected "
            f"{fp.PIN_RELEASE_ID} / {fp.PIN_CRSP_LOAD}; re-run --window anchor --release-id 1",
        )
    comparison = anchor.get("comparison")
    if not isinstance(comparison, Mapping):
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has no comparison mapping; "
            "re-run --window anchor --release-id 1",
        )
    if not isinstance(comparison.get("differences"), list) or comparison.get("differences") != []:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has non-empty "
            f"comparison.differences ({comparison.get('differences')!r}); "
            "re-run --window anchor --release-id 1",
        )
    if not _is_exact_int(comparison.get("n_strategy_cells"), EXPECTED_STRATEGY_CELLS):
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has "
            f"comparison.n_strategy_cells={comparison.get('n_strategy_cells')!r}, "
            f"expected {EXPECTED_STRATEGY_CELLS}; re-run --window anchor --release-id 1",
        )
    if not _is_exact_int(comparison.get("n_spy_cells"), EXPECTED_SPY_CELLS):
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has "
            f"comparison.n_spy_cells={comparison.get('n_spy_cells')!r}, "
            f"expected {EXPECTED_SPY_CELLS}; re-run --window anchor --release-id 1",
        )
    anchor_result = anchor.get("result")
    if not isinstance(anchor_result, Mapping):
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has no result mapping; "
            "re-run --window anchor --release-id 1",
        )
    saved_payload, saved_sha_or_err = _read_saved_holdout(
        saved_holdout_path, prefix="fresh run refused"
    )
    if saved_payload is None:
        return None, saved_sha_or_err
    if comparison.get("saved_file_sha256") != saved_sha_or_err:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has "
            f"comparison.saved_file_sha256={comparison.get('saved_file_sha256')!r}, "
            f"which does not match current saved first-test file {saved_holdout_path} "
            f"({saved_sha_or_err}); re-run --window anchor --release-id 1",
        )
    try:
        recheck_diffs = fresh_compare.compare_to_saved(dict(anchor_result), saved_payload)
    except (TypeError, ValueError) as exc:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has unreadable result values "
            f"({exc}); re-run --window anchor --release-id 1",
        )
    if recheck_diffs:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} result no longer matches saved "
            f"first-test file {saved_holdout_path} ({len(recheck_diffs)} difference(s), e.g. "
            f"{recheck_diffs[0]}); re-run --window anchor --release-id 1",
        )
    anchor_commit = anchor.get("anchor_commit")
    if not isinstance(anchor_commit, str) or not GIT_SHA_RE.match(anchor_commit):
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has no valid 40-character git commit "
            "hash; re-run --window anchor --release-id 1",
        )
    stored_hashes = anchor.get("code_hashes")
    if not isinstance(stored_hashes, dict) or not stored_hashes:
        return (
            None,
            f"fresh run refused: anchor file {anchor_path} has no code_hashes; "
            "re-run --window anchor --release-id 1",
        )
    if stored_hashes != dict(current_hashes):
        all_names = sorted(set(stored_hashes) | set(current_hashes))
        mismatched = [
            name for name in all_names if stored_hashes.get(name) != current_hashes.get(name)
        ]
        return (
            None,
            f"fresh run refused: study code changed since the anchor check "
            f"(hash mismatch for {', '.join(mismatched)}); "
            "re-run --window anchor --release-id 1 on the current code",
        )
    try:
        changed = changed_paths_fn(anchor_commit)
    except (OSError, subprocess.SubprocessError) as exc:
        return (
            None,
            f"fresh run refused: cannot verify git changes since anchor commit "
            f"{anchor_commit}: {exc}; re-run --window anchor --release-id 1",
        )
    code_changes = [path for path in changed if _is_code_path(path)]
    if code_changes:
        return (
            None,
            f"fresh run refused: code changed since anchor commit {anchor_commit} "
            f"({', '.join(code_changes[:5])}); "
            "re-run --window anchor --release-id 1 on the current commit",
        )
    return anchor, anchor_sha


def _build_payload(
    *,
    window: str,
    start: str,
    end: str,
    release_id: int,
    crsp_load: int,
    current_head: str,
    repo_binding: dict[str, Any],
    command: list[str],
    result: dict[str, Any],
    inputs: Mapping[str, Any],
    read_drifted: list[str],
    current_hashes: dict[str, str],
    extra_top: Mapping[str, Any] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the results JSON payload carrying registered summary fields and provenance."""
    payload: dict[str, Any] = {
        "study": STUDY_ID,
        "window": window,
        "window_dates": {"start": start, "end": end},
        "universe": UNIVERSE,
        "release_id": release_id,
        "crsp_load": crsp_load,
        "git_head": current_head,
        "binding": repo_binding,
        "command": command,
        "headline": fresh_compare.headline(result),
        "gap_table": fresh_compare.gap_table(result),
        "death_price_paths": fresh_compare.price_path_block(result),
        "result": result,
        "provenance": {
            "release_id": release_id,
            "crsp_load": crsp_load,
            "git_head": current_head,
            "brief_commit": brief_commit(),
            "window": {"start": start, "end": end},
            "window_name": window,
            "universe": UNIVERSE,
            "crsp_end": fp.CRSP_END,
            "command": command,
            "data": inputs.get("data_provenance"),
            "drift_symbols": read_drifted,
            "binding": repo_binding,
            "code_hashes": current_hashes,
            **(dict(extra_provenance) if extra_provenance else {}),
        },
    }
    if extra_top:
        payload.update(extra_top)
    return payload


def _format_anchor_mismatch(diffs: Sequence[str], saved_path: Path) -> str:
    """Format anchor reproduction mismatches in plain words."""
    lines = [
        f"Stopped: anchor check did not reproduce {saved_path} within half a cent "
        f"({len(diffs)} difference(s)):",
        *(f"  - {item}" for item in diffs),
        "Nothing was written; fix the mismatch before running the fresh window.",
    ]
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    loader: Callable[[str, int, int], dict[str, Any]] | None = None,
    results_dir: Path | None = None,
    study_dir_: Path | None = None,
    git_dirty: Callable[[], bool] | None = None,
    drift_preflight: Callable[[int], list[str]] | None = None,
    saved_holdout: Path | None = None,
    changed_paths: Callable[[str], list[str]] | None = None,
    code_problems: Callable[[], list[str]] | None = None,
) -> int:
    """Run the anchor reproduction check or the one-shot fresh window.

    Everything except ``argv`` is for tests only: output paths are fixed in normal use so
    neither the anchor check nor the fresh-window claim can be bypassed from the CLI.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        loader: Optional test replacement for ``fresh_inputs.read_inputs``.
        results_dir: Optional test directory for the trades side file and claim file.
        study_dir_: Optional test directory for ``_anchor-A.json`` and ``_fresh-A.json``.
        git_dirty: Optional test replacement for ``fp_dead_stock_first_test.git_tree_dirty``.
        drift_preflight: Optional test replacement for ``fresh_inputs.read_drift``.
        saved_holdout: Optional test path replacing the saved first-test ``_holdout-A.json``.
        changed_paths: Optional test replacement for ``git diff --name-only <anchor_commit> HEAD``.
        code_problems: Optional test replacement for ``untracked_code_problems``.

    Returns:
        Exit code: 0 on success, 2 on a refused precondition, 3 on an anchor mismatch.
    """
    args = build_parser().parse_args(argv)
    command = list(sys.argv if argv is None else ["run.py", *argv])

    r_dir = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    s_dir = STUDY_DIR if study_dir_ is None else study_dir_
    anchor_path = s_dir / ANCHOR_FILENAME
    fresh_path = s_dir / FRESH_FILENAME
    trades_path = fp.trades_file(args.window, UNIVERSE, r_dir)
    holdout_target, holdout_claim = fp.holdout_files(UNIVERSE, r_dir)
    saved_path = DEFAULT_SAVED_HOLDOUT if saved_holdout is None else saved_holdout

    # 1. Fixed output path protection before any read or claim.
    if args.window == "anchor":
        outputs = [
            ("anchor result file", anchor_path),
            ("trades side file", trades_path),
        ]
    else:
        outputs = [
            ("fresh result file", fresh_path),
            ("trades side file", trades_path),
            ("claim file", holdout_claim),
            ("holdout target path", holdout_target),
        ]
    path_problem = _validate_output_paths(outputs)
    if path_problem is not None:
        return _refuse(path_problem)

    # 2. Pinned release, raw CRSP load, and clean tracked git tree on both windows.
    is_dirty = (
        bool(git_dirty()) if git_dirty is not None else (loader is None and fp.git_tree_dirty(ROOT))
    )
    try:
        fp.check_holdout_pins(args.release_id, args.crsp_load, git_dirty=is_dirty)
    except SystemExit as exc:
        return _refuse(str(exc).replace("holdout refused:", f"{args.window} refused:", 1))

    # 3. Refuse untracked code or non-tracked loaded project modules on both windows.
    found_code_problems = (
        code_problems() if code_problems is not None else untracked_code_problems(ROOT)
    )
    if found_code_problems:
        return _refuse(
            f"{args.window} refused: untracked or non-tracked Python code found "
            f"({'; '.join(found_code_problems[:5])}); commit or clean up untracked code first"
        )

    current_head = git_sha(ROOT)
    if not current_head:
        return _refuse(
            f"{args.window} refused: could not read current git HEAD; run inside a git checkout"
        )
    current_hashes = _code_hashes(CODE_DIR)
    start, end = fresh_inputs.WINDOWS[args.window]
    preflight = (
        drift_preflight
        if drift_preflight is not None
        else (fresh_inputs.read_drift if loader is None else (lambda _rid: []))
    )

    if args.window == "anchor":
        saved_payload, saved_hash_or_err = _read_saved_holdout(saved_path)
        if saved_payload is None:
            return _refuse(saved_hash_or_err)

        preflight_drifted = preflight(args.release_id)
        if preflight_drifted:
            return _refuse(
                f"anchor refused: live Yahoo rows no longer match pinned release "
                f"{args.release_id} (drift in {len(preflight_drifted)} linked columns, e.g. "
                f"{', '.join(preflight_drifted[:5])})"
            )

        inputs = (loader or fresh_inputs.read_inputs)(args.window, args.release_id, args.crsp_load)
        read_drifted = fresh_inputs.drift_in(inputs)
        if read_drifted:
            return _refuse(
                f"anchor refused: close read reported Yahoo drift for "
                f"{', '.join(read_drifted[:5])}; nothing was written"
            )

        sink: dict[str, list] = {}
        result = fp.run_study(
            inputs["close"],
            inputs["spy_close"],
            inputs["eligible"],
            inputs["span_rows"],
            start=start,
            end=end,
            release_spans=inputs.get("release_spans"),
            trade_sink=sink,
        )
        result["false_death_audit"] = inputs.get("false_death_audit")

        diffs = fresh_compare.compare_to_saved(result, saved_payload)
        if diffs:
            print(_format_anchor_mismatch(diffs, saved_path), file=sys.stderr)
            return 3

        repo_binding = binding(ROOT)
        payload = _build_payload(
            window=args.window,
            start=start,
            end=end,
            release_id=args.release_id,
            crsp_load=args.crsp_load,
            current_head=current_head,
            repo_binding=repo_binding,
            command=command,
            result=result,
            inputs=inputs,
            read_drifted=read_drifted,
            current_hashes=current_hashes,
            extra_top={
                "anchor_ok": True,
                "anchor_commit": current_head,
                "code_hashes": current_hashes,
                "comparison": {
                    "saved_file": str(saved_path),
                    "saved_file_sha256": saved_hash_or_err,
                    "tolerance_dollars": fresh_compare.HALF_CENT,
                    "differences": diffs,
                    "n_strategy_cells": len(result["strategy"]),
                    "n_spy_cells": len(result["spy"]),
                },
            },
            extra_provenance={
                "saved_holdout_file": str(saved_path),
                "saved_holdout_sha256": saved_hash_or_err,
            },
        )
        side = fp.write_trades_csv(trades_path, sink)
        write_json(anchor_path, payload)
        print(f"wrote {side} and {anchor_path}")
        return 0

    # --- Fresh window: verify run-once files, anchor, code binding, preflight, then claim ---
    for existing in (fresh_path, holdout_claim, holdout_target, trades_path):
        if existing.exists():
            return _refuse(
                f"fresh run refused: a fresh-window run has already been started or finished "
                f"(found {existing}). The fresh window may be run only once; ask the owner how "
                "to proceed."
            )

    anchor_payload, anchor_sha_or_err = _verify_anchor_for_fresh(
        anchor_path,
        saved_path,
        current_hashes,
        changed_paths or _git_changed_paths,
    )
    if anchor_payload is None:
        return _refuse(anchor_sha_or_err)

    preflight_drifted = preflight(args.release_id)
    if preflight_drifted:
        return _refuse(
            f"fresh run refused: live Yahoo rows no longer match pinned release "
            f"{args.release_id} (drift in {len(preflight_drifted)} linked columns, e.g. "
            f"{', '.join(preflight_drifted[:5])}); nothing was claimed"
        )

    # Atomic exclusive claim BEFORE reading any fresh-window price data.
    try:
        claim_path = fp.claim_holdout(UNIVERSE, r_dir, git_head=current_head, command=command)
    except SystemExit:
        found = holdout_target if holdout_target.exists() else holdout_claim
        return _refuse(
            f"fresh run refused: a fresh-window run has already been started or finished "
            f"(found {found}). The fresh window may be run only once; ask the owner how to "
            "proceed."
        )

    inputs = (loader or fresh_inputs.read_inputs)(args.window, args.release_id, args.crsp_load)
    read_drifted = fresh_inputs.drift_in(inputs)
    if read_drifted:
        # Claim stays in place by design: the owner decides how to proceed.
        return _refuse(
            f"fresh run stopped after the read: Yahoo drift was reported for "
            f"{', '.join(read_drifted[:5])}; no result was written and the claim at {claim_path} "
            "remains in place. Ask the owner how to proceed."
        )

    sink = {}
    result = fp.run_study(
        inputs["close"],
        inputs["spy_close"],
        inputs["eligible"],
        inputs["span_rows"],
        start=start,
        end=end,
        release_spans=inputs.get("release_spans"),
        trade_sink=sink,
    )
    result["false_death_audit"] = inputs.get("false_death_audit")

    repo_binding = binding(ROOT)
    payload = _build_payload(
        window=args.window,
        start=start,
        end=end,
        release_id=args.release_id,
        crsp_load=args.crsp_load,
        current_head=current_head,
        repo_binding=repo_binding,
        command=command,
        result=result,
        inputs=inputs,
        read_drifted=read_drifted,
        current_hashes=current_hashes,
        extra_provenance={
            "anchor_file": str(anchor_path),
            "anchor_commit": anchor_payload["anchor_commit"],
            "anchor_sha256": anchor_sha_or_err,
            "claim_file": str(claim_path),
        },
    )
    # Write the trades side file before the result JSON so a result implies its trades exist.
    side = fp.write_trades_csv(trades_path, sink)
    write_json(fresh_path, payload)
    print(f"wrote {side} and {fresh_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
