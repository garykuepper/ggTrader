"""Driver for the dead-stock payout rerun: one window, universe A, pinned payouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ggTrader.utils.binding import binding, git_sha
from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import ft_bars_reader  # noqa: E402  (found on sys.path through bootstrap)
import ft_dead_stock_first_test as ft  # noqa: E402
import ft_universe_frames  # noqa: E402
import payout_rerun  # noqa: E402

STUDY_DIR = study_dir(__file__)
BRIEF = "2026-11-24-dead-stock-payout-rerun-brief.md"
UNIVERSE = "A"
SPY = "SPY"
PAYOUTS_SHA256 = "7c4a40db56db09d5c661a458e2505f6d8f8256910d6dabdd2dc1d6de99beef63"
DEFAULT_PAYOUTS = STUDY_DIR.parent / "dead-stock-data-integrity" / "_payouts_v2.csv"
DEFAULT_FIRST_TEST_DIR = STUDY_DIR.parent / "dead-stock-first-test"
DEFAULT_RESULTS_DIR = ROOT / "results" / "dead_stock_payout_rerun"
FIRST_TEST_RESULTS_DIR = ROOT / "results" / "dead_stock_first_test"
REQUIRED_PAYOUT_COLUMNS: frozenset[str] = frozenset(
    {"symbol", "suggested_loss", "match_confidence"}
)
DIAGNOSTIC_KEYS: tuple[str, ...] = ("anchor", "identity_check", "first_test_range")
CODE_FILES: tuple[str, ...] = (
    "run.py",
    "payout_prices.py",
    "payout_rerun.py",
    "ft_replay.py",
    "ft_ledger_run.py",
    "ft_bars_reader.py",
    "ft_universe_frames.py",
    "ft_dead_stock_first_test.py",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (pinned window names, no date or universe flags)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        choices=sorted(ft.WINDOWS),
        required=True,
        help="pinned evaluation window (dev or holdout)",
    )
    parser.add_argument(
        "--release-id",
        type=int,
        required=True,
        help="published combined release id (pinned)",
    )
    parser.add_argument(
        "--crsp-load",
        type=int,
        default=1,
        help="raw CRSP load id for eligibility (pinned, default 1)",
    )
    parser.add_argument(
        "--payouts",
        type=Path,
        default=DEFAULT_PAYOUTS,
        help="path to the pinned payout CSV table",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results JSON path (default: <study folder>/_rerun-<window>-A.json)",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="optional path to write diagnostics JSON if anchor or identity check fails",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing result, trades side file or diagnostics file",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow running with uncommitted tracked changes (recorded in provenance)",
    )
    return parser


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a timezone-naive DatetimeIndex."""
    return index.tz_convert(None) if index.tz is not None else index


def _sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _code_hashes() -> dict[str, str]:
    """Return SHA-256 digests of the driver, rerun modules and five copied engine files."""
    code_dir = Path(__file__).resolve().parent
    return {name: _sha256_bytes((code_dir / name).read_bytes()) for name in CODE_FILES}


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


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Read release spans, candidate closes, SPY and false-death audit for universe A.

    Args:
        args: Parsed CLI arguments providing ``window``, ``release_id`` and ``crsp_load``.

    Returns:
        Dictionary matching the first-test driver shape: ``close``, ``spy_close``, ``eligible``,
        ``span_rows``, ``release_spans``, ``false_death_audit`` and ``data_provenance``.
    """
    from ggTrader.data.bars import connect_readonly

    start, end = ft.WINDOWS[args.window]
    read_start = (pd.Timestamp(start) - pd.Timedelta(days=ft.WARMUP_CALENDAR_DAYS)).strftime(
        "%Y-%m-%d"
    )
    # Eligibility uses the previous calendar month-end, so read fields starting one month earlier.
    fields_start = (pd.Timestamp(start) - pd.DateOffset(months=1)).replace(day=1)
    conn = connect_readonly()
    try:
        release_spans = ft_bars_reader.read_release_spans(conn, args.release_id)
        fields = ft_universe_frames.month_end_fields(
            conn, args.crsp_load, fields_start.strftime("%Y-%m-%d"), end
        )
    finally:
        conn.close()
    days = pd.bdate_range(start, end)
    ever = ft_universe_frames.eligibility(
        fields, release_spans, universe=UNIVERSE, daily_index=days
    )
    candidates = sorted(ever.columns[ever.to_numpy().any(axis=0)])
    del ever
    provenance: dict[str, Any] = {}
    close, span_rows = ft_bars_reader.read_close_batched(
        candidates, read_start, end, release_id=args.release_id, provenance_out=provenance
    )
    spy_provenance: dict[str, Any] = {}
    spy_frame, _ = ft_bars_reader.read_close_batched(
        [SPY], read_start, end, release_id=args.release_id, provenance_out=spy_provenance
    )
    if SPY not in spy_frame.columns:
        raise SystemExit(f"SPY has no bars in release {args.release_id} for {read_start}..{end}")
    # Recompute eligibility on actual trading days for candidate columns only.
    wanted = release_spans[release_spans["symbol"].isin(candidates)]
    window_days = _naive(close.index)
    window_days = window_days[window_days >= pd.Timestamp(start)]
    eligible = ft_universe_frames.eligibility(
        fields, wanted, universe=UNIVERSE, daily_index=window_days
    ).reindex(columns=close.columns, fill_value=False)
    conn = connect_readonly()
    try:
        # Use full release spans so later spells outside universe A are still seen.
        audit = ft.audit_false_deaths(
            conn,
            release_spans,
            crsp_load=args.crsp_load,
            window_start=start,
            window_end=end,
            crsp_end=ft.CRSP_END,
        )
    finally:
        conn.close()
    return {
        "false_death_audit": audit,
        "close": close,
        "spy_close": spy_frame[SPY],
        "eligible": eligible,
        "span_rows": span_rows,
        "release_spans": release_spans,
        "data_provenance": {
            "candidates": provenance,
            "spy": spy_provenance,
            "n_candidate_columns": len(candidates),
        },
    }


def read_drift(args: argparse.Namespace) -> list[str]:
    """Read drifted linked Yahoo columns of the pinned release (metadata only)."""
    from ggTrader.data.bars import connect_readonly

    conn = connect_readonly()
    try:
        return ft_bars_reader.read_linked_drift(conn, args.release_id)
    finally:
        conn.close()


def drift_in(inputs: Mapping[str, Any]) -> list[str]:
    """Return sorted symbols reported as drifted by the close reads (empty when none)."""
    found: set[str] = set()
    for part in (inputs.get("data_provenance") or {}).values():
        if isinstance(part, Mapping) and part.get("drift"):
            found.update(part.get("drift_symbols") or ["(drift reported without symbols)"])
    return sorted(found)


def _refuse(message: str) -> int:
    """Print a plain-word precondition refusal to stderr and return exit code 2."""
    print(message, file=sys.stderr)
    return 2


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


def _is_claim_path(path: Path) -> bool:
    """Return True when the lexical or resolved final path component ends with ``.claim``."""
    return (
        path.name.endswith(".claim")
        or _lexical_abs(path).name.endswith(".claim")
        or _resolved(path).name.endswith(".claim")
    )


def _same_path(first: Path, second: Path) -> bool:
    """Return True when two paths denote the same file lexically or after resolution."""
    return _lexical_abs(first) == _lexical_abs(second) or _resolved(first) == _resolved(second)


def _protected_first_test_dirs(
    *,
    results_dir: Path | None,
    first_test_dir: Path | None,
) -> tuple[Path, ...]:
    """Return the read-only first-test study and results directories to protect."""
    dirs: list[Path] = [DEFAULT_FIRST_TEST_DIR, FIRST_TEST_RESULTS_DIR]
    if first_test_dir is not None and first_test_dir not in dirs:
        dirs.append(first_test_dir)
    if results_dir is not None:
        counterpart = results_dir.parent / "dead_stock_first_test"
        if counterpart not in dirs:
            dirs.append(counterpart)
    return tuple(dirs)


def _validate_paths(
    *,
    outputs: Sequence[tuple[str, Path]],
    inputs: Sequence[tuple[str, Path]],
    protected_dirs: Sequence[Path],
    force: bool,
) -> str | None:
    """Validate all input and output paths before any content is read or any loader runs."""
    current_label = "supplied path"
    try:
        # 1. Never read or write any .claim path (checked lexically and after resolution).
        for label, path in (*inputs, *outputs):
            current_label = label
            if _is_claim_path(path):
                return (
                    f"refused: {label} {path} is a .claim path; "
                    ".claim files are never read or written by the rerun"
                )

        # 2. Never write through a symlink or into any protected first-test folder.
        for label, path in outputs:
            current_label = label
            if path.is_symlink():
                return f"refused: {label} {path} is a symlink; refusing to write through a symlink"
            for protected in protected_dirs:
                if _is_inside(path, protected):
                    return (
                        f"refused: {label} {path} lies inside protected first-test directory "
                        f"{protected}; first-test files are read-only and never written by the "
                        "rerun"
                    )

        # 3. Refuse colliding outputs and any output that targets a consumed input file.
        for idx, (label_a, path_a) in enumerate(outputs):
            current_label = label_a
            for label_b, path_b in outputs[idx + 1 :]:
                if _same_path(path_a, path_b):
                    return (
                        f"refused: {label_a} {path_a} collides with {label_b} {path_b}; "
                        "output paths must be distinct files"
                    )
            for in_label, in_path in inputs:
                if _same_path(path_a, in_path):
                    return (
                        f"refused: {label_a} {path_a} collides with input {in_label} {in_path}; "
                        "refusing to overwrite a file the run reads"
                    )

        # 4. Refuse clobbering existing outputs unless --force was explicitly passed.
        for label, path in outputs:
            current_label = label
            if path.exists() and not force:
                return f"refused: {label} {path} already exists (pass --force to overwrite)"
    except (OSError, RuntimeError, ValueError) as exc:
        # Never leak raw OS/path tracebacks from symlink loops, long names, permissions or NULs.
        return f"refused: cannot validate {current_label}: {type(exc).__name__}"
    return None


def _read_payouts(
    path: Path, expected_sha256: str
) -> tuple[list[dict[str, str]], str] | tuple[None, str]:
    """Read raw payout bytes, verify SHA-256 first, then parse the same bytes as CSV rows.

    Returns:
        ``(rows, sha256)`` on success, or ``(None, error_message)`` on any refusal condition.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"refused: cannot read payout table {path}: {exc}"
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        return (
            None,
            f"refused: payout table SHA-256 mismatch for {path}: "
            f"got {actual}, expected {expected_sha256}",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"refused: payout table {path} is not valid UTF-8: {exc}"
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = set(reader.fieldnames or ())
    missing = sorted(REQUIRED_PAYOUT_COLUMNS - fieldnames)
    if missing:
        return (
            None,
            f"refused: payout table {path} is missing required column(s): {', '.join(missing)}",
        )
    try:
        rows = list(reader)
    except csv.Error as exc:
        return None, f"refused: cannot parse payout table {path}: {exc}"
    return rows, actual


def _read_first_test(path: Path) -> tuple[dict[str, Any], str] | tuple[None, str]:
    """Read and validate the saved first-test JSON file for this window.

    Returns:
        ``(result_block, sha256)`` on success, or ``(None, error_message)`` on refusal.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"refused: cannot read first-test result file {path}: {exc}"
    digest = _sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"refused: first-test result file {path} is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"refused: first-test result file {path} must contain a JSON object"
    result_block = payload.get("result")
    if not isinstance(result_block, dict) or not isinstance(result_block.get("strategy"), dict):
        return None, f"refused: first-test result file {path} has no result.strategy mapping"
    if not isinstance(result_block.get("spy"), dict):
        return None, f"refused: first-test result file {path} has no result.spy mapping"
    return result_block, digest


def _format_anchor_mismatches(anchor: Mapping[str, Any]) -> str:
    """Format reproduction-anchor mismatches in plain words."""
    lines = ["Stopped: reproduction check against the first test failed:"]
    for section in ("strategy", "spy"):
        for cell_name, entry in (anchor.get(section) or {}).items():
            if entry.get("matches"):
                continue
            for field, info in (entry.get("compared_fields") or {}).items():
                if info.get("matches"):
                    continue
                lines.append(
                    f"  - {section} {cell_name} ({field}): "
                    f"rerun {info['rerun_cents']:.2f} vs first test {info['first_test_cents']:.2f}"
                )
    lines.append("Nothing was written; the cause must be found before anything is published.")
    return "\n".join(lines)


def _format_identity_mismatches(identity_check: Mapping[str, Any]) -> str:
    """Format reading-0.0 identity-check mismatches in plain words."""
    mismatches = identity_check.get("mismatches") or []
    lines = [f"Stopped: reading-0.0 identity check failed on {len(mismatches)} cell(s):"]
    for item in mismatches:
        lines.append(
            f"  - {item['cell']} vs {item['first_test_cell']}: "
            f"final {item['rerun_after_tax_final']:.2f} vs "
            f"{item['first_test_after_tax_final']:.2f}, "
            f"gap {item['rerun_gap_cagr_after_tax_pp']:.4f} vs "
            f"{item['first_test_gap_cagr_after_tax_pp']:.4f} pp"
        )
    lines.append("Nothing was written; the cause must be found before anything is published.")
    return "\n".join(lines)


def _write_diagnostics(diag_path: Path | None, result: Mapping[str, Any]) -> None:
    """Write the optional diagnostics JSON containing whichever gate blocks exist."""
    if diag_path is None:
        return
    blocks = {key: result[key] for key in DIAGNOSTIC_KEYS if key in result}
    write_json(diag_path, blocks)


def main(
    argv: Sequence[str] | None = None,
    *,
    loader: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    git_dirty: Callable[[], bool] | None = None,
    drift_preflight: Callable[[argparse.Namespace], list[str]] | None = None,
    results_dir: Path | None = None,
    first_test_dir: Path | None = None,
    expected_payouts_sha256: str | None = None,
) -> int:
    """Run one pinned payout-rerun window (universe A) and write trades then result JSON.

    Everything except ``argv`` is for tests only.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        loader: Optional test replacement for ``read_inputs``.
        git_dirty: Optional test replacement for ``ft.git_tree_dirty(ROOT)``.
        drift_preflight: Optional test replacement for ``read_drift``.
        results_dir: Optional test directory for the trades side file.
        first_test_dir: Optional test directory containing ``_<window>-A.json``.
        expected_payouts_sha256: Optional test override for ``PAYOUTS_SHA256``.

    Returns:
        Exit code: 0 on success, 2 on a refused precondition, 3 on anchor or identity mismatch.
    """
    args = build_parser().parse_args(argv)
    command = list(sys.argv if argv is None else ["run.py", *argv])

    # 1. Argument, pin, clean-tree and output-path preconditions (exit 2).
    is_dirty = (
        bool(git_dirty()) if git_dirty is not None else (loader is None and ft.git_tree_dirty(ROOT))
    )
    try:
        ft.check_holdout_pins(
            args.release_id,
            args.crsp_load,
            git_dirty=bool(is_dirty and not args.allow_dirty),
        )
    except SystemExit as exc:
        return _refuse(str(exc))

    folder = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    ft_dir = DEFAULT_FIRST_TEST_DIR if first_test_dir is None else first_test_dir
    out_path = (
        args.out if args.out is not None else STUDY_DIR / f"_rerun-{args.window}-{UNIVERSE}.json"
    )
    trades_path = ft.trades_file(args.window, UNIVERSE, folder)
    ft_path = ft_dir / f"_{args.window}-{UNIVERSE}.json"

    try:
        path_problem = _validate_paths(
            outputs=[
                ("output file (--out)", out_path),
                ("trades side file", trades_path),
                *(
                    []
                    if args.diagnostics is None
                    else [("diagnostics file (--diagnostics)", args.diagnostics)]
                ),
            ],
            inputs=[
                ("payout table (--payouts)", args.payouts),
                ("first-test result file", ft_path),
            ],
            protected_dirs=_protected_first_test_dirs(
                results_dir=results_dir,
                first_test_dir=first_test_dir,
            ),
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _refuse(f"refused: cannot validate supplied path: {type(exc).__name__}")
    if path_problem is not None:
        return _refuse(path_problem)

    # 2. Read and hash the payout table bytes before parsing them.
    expected_hash = PAYOUTS_SHA256 if expected_payouts_sha256 is None else expected_payouts_sha256
    payout_rows, payouts_hash_or_err = _read_payouts(args.payouts, expected_hash)
    if payout_rows is None:
        return _refuse(payouts_hash_or_err)

    # 3. Read the saved first-test window result block.
    first_test_result, ft_hash_or_err = _read_first_test(ft_path)
    if first_test_result is None:
        return _refuse(ft_hash_or_err)

    # 4. Metadata drift preflight on both windows before the big price read.
    preflight = drift_preflight or (read_drift if loader is None else (lambda _a: []))
    preflight_drifted = preflight(args)
    if preflight_drifted:
        return _refuse(
            f"refused ({args.window}): live Yahoo rows no longer match pinned release "
            f"{args.release_id} (drift in {len(preflight_drifted)} linked columns, e.g. "
            f"{', '.join(preflight_drifted[:5])})"
        )

    # 5. Load inputs and check close-read drift on both windows.
    inputs = (loader or read_inputs)(args)
    read_drifted = drift_in(inputs)
    if read_drifted:
        return _refuse(
            f"refused ({args.window}): close read reported Yahoo drift for "
            f"{', '.join(read_drifted[:5])}; nothing was written"
        )

    # 6. Run the payout rerun (anchor runs first inside run_rerun).
    start, end = ft.WINDOWS[args.window]
    sink: dict[str, list] = {}
    result = payout_rerun.run_rerun(
        inputs["close"],
        inputs["spy_close"],
        inputs["eligible"],
        inputs["span_rows"],
        payout_rows,
        first_test_result,
        start=start,
        end=end,
        release_spans=inputs.get("release_spans"),
        trade_sink=sink,
    )

    # 7. Gate on anchor_ok and identity_ok (exit 3; write nothing except optional diagnostics).
    if not result.get("anchor_ok", False):
        print(_format_anchor_mismatches(result.get("anchor") or {}), file=sys.stderr)
        _write_diagnostics(args.diagnostics, result)
        return 3

    identity_check = result.get("identity_check") or {}
    if not identity_check.get("identity_ok", False):
        print(_format_identity_mismatches(identity_check), file=sys.stderr)
        _write_diagnostics(args.diagnostics, result)
        return 3

    # 8. Attach false-death audit and write trades side file BEFORE result JSON.
    result["false_death_audit"] = inputs.get("false_death_audit")
    payload = {
        "study": "dead-stock-payout-rerun",
        "result": result,
        "provenance": {
            "release_id": args.release_id,
            "crsp_load": args.crsp_load,
            "git_head": git_sha(ROOT),
            "git_dirty": bool(is_dirty),
            "allow_dirty": bool(args.allow_dirty),
            "brief_commit": brief_commit(),
            "window": {"start": start, "end": end},
            "window_name": args.window,
            "universe": UNIVERSE,
            "crsp_end": ft.CRSP_END,
            "command": command,
            "payouts_sha256": payouts_hash_or_err,
            "payouts_path": str(args.payouts),
            "first_test_file": str(ft_path),
            "first_test_file_sha256": ft_hash_or_err,
            "data": inputs.get("data_provenance"),
            "drift_symbols": read_drifted,
            "binding": binding(ROOT),
            "code_sha256": _code_hashes(),
        },
    }
    side = ft.write_trades_csv(trades_path, sink)
    write_json(out_path, payload)
    print(f"wrote {side} and {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
