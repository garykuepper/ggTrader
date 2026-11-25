"""Driver for Phase 3 of the dead-stock data-integrity study: open-case matching under rule v2."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ggTrader.utils.io import atomic_write_text, write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import open_cases  # noqa: E402  (found on sys.path through bootstrap)
import payout_table  # noqa: E402
import sec_filings  # noqa: E402
import sec_name_index  # noqa: E402
import sec_pilot  # noqa: E402
from dead_stock_data_integrity import git_provenance  # noqa: E402
from sec_client import (  # noqa: E402
    SecClient,
    SecFetchError,
    SecUserAgentMissing,
    load_user_agent,
    write_sha256sums,
)

STUDY_DIR = study_dir(__file__)
PHASE_NAME = "open_cases"
OUTPUT_CSV = "_payouts_v2.csv"
OUTPUT_JSON = "_payouts_v2.json"
EVENTS_STATUS_FETCH_ERROR = sec_pilot.EVENTS_STATUS_FETCH_ERROR
REASON_NO_NAME_AFTER_PROBE = "no_name_after_probe"

_STUDY_REL = "research/stock-signals/dead-stock-data-integrity/code/"
BRIEF_REL = (
    "research/stock-signals/dead-stock-data-integrity/phase-3/2026-11-24-open-cases-brief.md"
)
PHASE3_CODE_FILES: tuple[str, ...] = (
    f"{_STUDY_REL}sec_name_index.py",
    f"{_STUDY_REL}open_cases.py",
    f"{_STUDY_REL}open_cases_run.py",
)
CODE_FILES: tuple[str, ...] = sec_pilot.CODE_FILES + PHASE3_CODE_FILES


class PreconditionError(RuntimeError):
    """Raised when a Phase 3 precondition or zero-archive-download invariant is violated."""


def output_paths(out_dir: Path) -> dict[str, Path]:
    """The two Phase 3 result files, by file name, inside ``out_dir``."""
    return {name: out_dir / name for name in (OUTPUT_CSV, OUTPUT_JSON)}


def symlinked_outputs(out_dir: Path) -> list[Path]:
    """Return ``out_dir`` and any Phase 3 output file that is a symbolic link."""
    found = [out_dir] if out_dir.is_symlink() else []
    return found + [path for path in output_paths(out_dir).values() if path.is_symlink()]


def _refuse(message: str) -> int:
    """Print a plain refusal message to stderr and return exit code 2."""
    print(f"error: {message}", file=sys.stderr)
    return 2


def _read_payouts_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read ``_payouts.csv`` once and return ``(rows, sha256)`` of the exact bytes parsed."""
    if not path.is_file():
        raise PreconditionError(f"payouts file not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    except UnicodeDecodeError as exc:
        raise PreconditionError(f"payouts file {path} is not valid UTF-8: {exc}") from exc
    fieldnames = reader.fieldnames or []
    missing_cols = [col for col in payout_table.TABLE_COLUMNS if col not in fieldnames]
    if missing_cols:
        raise PreconditionError(
            f"payouts file {path} is missing required columns: {', '.join(missing_cols)}"
        )
    rows = [dict(row) for row in reader]
    return rows, digest


def _read_form25_json(path: Path) -> tuple[dict[str, Any], str]:
    """Read ``_form25_matches.json`` once and return ``(payload, sha256)`` of the parsed bytes."""
    if not path.is_file():
        raise PreconditionError(f"Form 25 file not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PreconditionError(f"Form 25 file {path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("stocks"), list):
        raise PreconditionError(f"Form 25 file {path} must be a JSON object with a 'stocks' list")
    return payload, digest


def _read_roster_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read ``_roster.csv`` once via ``sec_pilot.load_roster`` with a clean refusal on failure."""
    if not path.is_file():
        raise PreconditionError(f"roster file not found: {path}")
    try:
        return sec_pilot.load_roster(path)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PreconditionError(f"roster file {path} could not be read: {exc}") from exc


def _parse_positive_int(value: Any, label: str) -> int:
    """Parse an integer identifier (PERMNO or CIK) strictly, refusing booleans and blanks."""
    if isinstance(value, bool):
        raise PreconditionError(f"invalid {label}: {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise PreconditionError(f"invalid {label}: {value!r}")
    if parsed <= 0:
        raise PreconditionError(f"invalid {label}: must be positive, got {parsed}")
    return parsed


def _validate_inputs(
    rows: Sequence[Mapping[str, str]],
    form25: Mapping[str, Any],
) -> dict[str, int]:
    """Validate payouts rows and return ``{symbol: cik}`` for every Phase 2 ``no_8k_found`` row."""
    seen_symbols: set[str] = set()
    valid_classes = set(open_cases.PHASE3_CLASSES)
    phase2_no_8k_rows: list[Mapping[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        symbol = str(row.get("symbol", ""))
        if not symbol.strip():
            raise PreconditionError(f"payouts row {idx} is missing a symbol")
        if symbol in seen_symbols:
            raise PreconditionError(f"duplicate symbol in payouts file: {symbol!r}")
        seen_symbols.add(symbol)

        evidence_class = str(row.get("evidence_class", ""))
        if evidence_class not in valid_classes:
            raise PreconditionError(
                f"payouts row {symbol!r} has unknown evidence_class {evidence_class!r}"
            )
        if evidence_class == sec_filings.EVIDENCE_NONE:
            phase2_no_8k_rows.append(row)

    stocks = form25.get("stocks")
    if not isinstance(stocks, list):
        raise PreconditionError("Form 25 payload is missing a 'stocks' list")

    form25_by_permno: dict[int, Mapping[str, Any]] = {}
    for entry in stocks:
        if not isinstance(entry, Mapping) or "permno" not in entry:
            continue
        permno = _parse_positive_int(entry["permno"], "Form 25 permno")
        if permno in form25_by_permno:
            raise PreconditionError(f"Form 25 file has duplicate stock entry for permno {permno}")
        form25_by_permno[permno] = entry

    phase2_ciks: dict[str, int] = {}
    for row in phase2_no_8k_rows:
        symbol = str(row["symbol"])
        permno = _parse_positive_int(row.get("permno"), f"permno for {symbol}")
        entry = form25_by_permno.get(permno)
        if entry is None:
            raise PreconditionError(
                f"Form 25 file lacks a stock entry for no_8k_found row {symbol} (permno {permno})"
            )
        ciks = entry.get("ciks")
        if not isinstance(ciks, list) or len(ciks) != 1:
            raise PreconditionError(
                f"Form 25 entry for no_8k_found row {symbol} (permno {permno}) must have "
                f"exactly one CIK, found {ciks!r}"
            )
        phase2_ciks[symbol] = _parse_positive_int(ciks[0], f"Form 25 CIK for {symbol}")

    return phase2_ciks


def _passthrough_phase3_row(row: Mapping[str, str], *, reason: str) -> dict[str, str]:
    """Preserve all 21 original columns and stamp Phase 3 metadata columns."""
    out = {col: str(row.get(col, "")) for col in payout_table.TABLE_COLUMNS}
    out["phase"] = open_cases.PHASE_3
    out["name_rule"] = ""
    out["reason"] = reason
    out["still_filing_check"] = ""
    out["deal_form_accessions"] = ""
    return {col: out[col] for col in open_cases.OUTPUT_COLUMNS}


def rule_constants() -> dict[str, Any]:
    """Frozen rule v2 constants from ``open_cases`` recorded in provenance."""
    return {
        "rule_version": open_cases.RULE_VERSION_V2,
        "deal_forms": sorted(open_cases.DEAL_FORMS),
        "deal_days_before": open_cases.DEAL_DAYS_BEFORE,
        "deal_days_after": open_cases.DEAL_DAYS_AFTER,
        "candidate_days_before": open_cases.CANDIDATE_DAYS_BEFORE,
        "candidate_days_after": open_cases.CANDIDATE_DAYS_AFTER,
        "still_filing_first_day": open_cases.STILL_FILING_FIRST_DAY,
        "index_last_day": open_cases.INDEX_LAST_DAY.isoformat(),
        "periodic_forms": sorted(open_cases.PERIODIC_FORMS),
        "event_window_days": sec_filings.EVENT_WINDOW_DAYS,
    }


def phase3_provenance(
    *,
    git_info: Mapping[str, Any],
    git_dirty_ignores: list[str],
    payouts_sha256: str,
    roster_sha256: str,
    form25_sha256: str,
    brief_sha256: str,
    quarters: list[tuple[int, int]],
    run_date: str,
    allow_dirty: bool,
) -> dict[str, Any]:
    """Build the Phase 3 provenance block; never reads or records the SEC contact line."""
    start, end = sec_pilot.quarter_dates(quarters)
    return {
        "git_head": git_info["git_head"],
        "git_dirty": git_info["git_dirty"],
        "git_dirty_ignores": git_dirty_ignores,
        "phase": PHASE_NAME,
        "run_date": run_date,
        "code_sha256": {name: sec_pilot.sha256_file(ROOT / name) for name in CODE_FILES},
        "payouts_sha256": payouts_sha256,
        "roster_sha256": roster_sha256,
        "form25_sha256": form25_sha256,
        "brief_sha256": brief_sha256,
        "quarter_window": {
            "first": list(quarters[0]) if quarters else None,
            "last": list(quarters[-1]) if quarters else None,
            "start": start,
            "end": end,
            "needed": len(quarters),
        },
        "rule_constants": rule_constants(),
        "allow_dirty": allow_dirty,
    }


def run_open_cases(
    client: Any,
    archives_factory: Callable[[Any, Sequence[tuple[int, int]]], Iterable[tuple[str, bytes]]],
    rows: Sequence[Mapping[str, str]],
    form25: Mapping[str, Any],
    *,
    quarters: Sequence[tuple[int, int]],
    run_date: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Run the two-pass index scan, 8-K fetches, classification, merge, and coverage summary.

    Args:
        client: ``SecClient`` (or fake with ``get``, ``manifest``, and counter attributes).
        archives_factory: Callable returning a fresh iterable of ``(quarter_label, bytes)`` for
            each pass over the cached quarterly archives.
        rows: All rows of ``_payouts.csv`` in original order.
        form25: Parsed ``_form25_matches.json`` mapping.
        quarters: Ordered ``(year, quarter)`` tuples covering the cached quarterly indexes.
        run_date: ISO date string recorded in the payload.

    Returns:
        ``(merged_rows, payload)`` where ``merged_rows`` holds all 26-column text row dicts in
        original order and ``payload`` is ready for JSON serialization (before provenance is added).

    Raises:
        sec_name_index.MissingArchive: If any needed quarterly archive is absent or unreadable.
        PreconditionError: If row/Form 25 validation fails or archive scanning triggers a download.
    """
    phase2_ciks = _validate_inputs(rows, form25)

    rule1_rows: list[Mapping[str, str]] = []
    phase2_no_8k_rows: list[Mapping[str, str]] = []
    unnamed_rows: list[Mapping[str, str]] = []
    other_open_rows: list[Mapping[str, str]] = []

    for row in rows:
        evidence_class = str(row["evidence_class"])
        if evidence_class == payout_table.CLASS_NO_FILING_MATCH:
            rule1_rows.append(row)
        elif evidence_class == sec_filings.EVIDENCE_NONE:
            phase2_no_8k_rows.append(row)
        elif evidence_class == payout_table.CLASS_NAME_UNKNOWN:
            unnamed_rows.append(row)
        elif evidence_class not in payout_table.DOCUMENTED_CLASSES:
            other_open_rows.append(row)

    # Guard the frozen zero-archive-download rule across both streaming passes.
    fetches_before = getattr(client, "network_fetches", 0)
    matcher = open_cases.make_matcher(rule1_rows)
    pass1_hits = sec_name_index.find_filers(archives_factory(client, quarters), matcher)

    ciks_by_symbol: dict[str, set[int]] = {}
    for cik, matched_symbols in pass1_hits.items():
        for symbol in matched_symbols:
            ciks_by_symbol.setdefault(symbol, set()).add(cik)

    pass2_ciks = set(pass1_hits) | set(phase2_ciks.values())
    filings_by_cik = sec_name_index.filings_of(archives_factory(client, quarters), pass2_ciks)
    fetches_after = getattr(client, "network_fetches", 0)

    archive_downloads = fetches_after - fetches_before
    if archive_downloads != 0:
        raise PreconditionError(
            f"quarterly archive scan triggered {archive_downloads} network fetch(es); "
            "cached archives must never be downloaded"
        )

    phase3_rows: dict[str, dict[str, str]] = {}
    open_matches: dict[str, open_cases.OpenMatch] = {}
    events_status_counts: dict[str, int] = {}
    events_fetch_errors: list[dict[str, Any]] = []

    for row in rule1_rows:
        symbol = str(row["symbol"])
        candidate_filings = {
            cik: filings_by_cik.get(cik, []) for cik in ciks_by_symbol.get(symbol, set())
        }
        match = open_cases.match_open_case(row, candidate_filings)
        # phase3_coverage requires every rule 1 row's OpenMatch identity.
        open_matches[symbol] = match

        if match.status == open_cases.STATUS_MATCHED and match.cik is not None:
            exit_date = dt.date.fromisoformat(str(row["exit_date"]).strip())
            try:
                events, status = sec_filings.fetch_events_with_status(
                    client,
                    str(match.cik),
                    exit_date,
                    days=sec_filings.EVENT_WINDOW_DAYS,
                )
            except (SecFetchError, ValueError) as exc:
                events = []
                status = EVENTS_STATUS_FETCH_ERROR
                events_fetch_errors.append(
                    {
                        "symbol": symbol,
                        "permno": _parse_positive_int(row["permno"], f"permno for {symbol}"),
                        "cik": match.cik,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            events_status_counts[status] = events_status_counts.get(status, 0) + 1
            phase3_rows[symbol] = open_cases.classify_open_case(
                row,
                match,
                events,
                status,
                filings_by_cik.get(match.cik, []),
            )
        else:
            phase3_rows[symbol] = open_cases.classify_open_case(row, match, [], "", [])

    # Rule 2: Phase 2 no_8k_found rows keep their matched CIK and never refetch 8-K events.
    for row in phase2_no_8k_rows:
        symbol = str(row["symbol"])
        cik = phase2_ciks[symbol]
        phase3_rows[symbol] = open_cases.classify_phase2_no_8k(row, filings_by_cik.get(cik, []))

    # Rule 7: unnamed rows pass through; the DB name probe runs outside this driver.
    unnamed_symbols: list[str] = []
    for row in unnamed_rows:
        symbol = str(row["symbol"])
        unnamed_symbols.append(symbol)
        phase3_rows[symbol] = _passthrough_phase3_row(row, reason=REASON_NO_NAME_AFTER_PROBE)

    for row in other_open_rows:
        symbol = str(row["symbol"])
        phase3_rows[symbol] = _passthrough_phase3_row(row, reason="")

    merged_rows = open_cases.merge_open_cases(rows, phase3_rows)
    coverage = open_cases.phase3_coverage(rows, merged_rows, open_matches=open_matches)

    payload: dict[str, Any] = {
        "phase": PHASE_NAME,
        "run_date": run_date,
        "rows": merged_rows,
        "coverage": coverage,
        "events_status_counts": dict(sorted(events_status_counts.items())),
        "events_fetch_errors": events_fetch_errors,
        "unnamed_rows": unnamed_symbols,
        "request_count": {
            "archive_downloads": archive_downloads,
            "network_fetches": getattr(client, "network_fetches", 0),
            "cache_hits": getattr(client, "cache_hits", 0),
            "corrupt_cache_refetches": getattr(client, "corrupt_cache_refetches", 0),
        },
        "manifest": client.manifest(),
    }
    return merged_rows, payload


def _format_csv(rows: Sequence[Mapping[str, str]]) -> str:
    """Serialize Phase 3 output rows with exact ``OUTPUT_COLUMNS`` order and ``\\n`` line ends."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(open_cases.OUTPUT_COLUMNS),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _format_summary(payload: Mapping[str, Any]) -> str:
    """Plain one-paragraph summary of Phase 3 counts, v1.1-only, deal forms, and rejections."""
    cov = payload["coverage"]
    by_class = ", ".join(f"{name} {count}" for name, count in cov["after_by_class"].items())
    return (
        f"Phase 3 finished for {cov['roster_rows']} stocks: {by_class}; "
        f"v1.1-only matches {len(cov['v1_1_only_matches'])}; "
        f"deal_forms_only {cov['deal_forms_only']}; "
        f"rejected still filing {len(cov['rejected_still_filing'])}; "
        f"incomplete still-filing checks {cov['still_filing_incomplete']}."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Phase 3 open-cases driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="folder holding cached SEC files"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="study folder for _payouts_v2 outputs"
    )
    parser.add_argument(
        "--payouts",
        type=Path,
        default=STUDY_DIR / sec_pilot.PAYOUTS_CSV,
        help="Phase 2 _payouts.csv input",
    )
    parser.add_argument(
        "--form25",
        type=Path,
        default=STUDY_DIR / sec_pilot.OUTPUT_JSON,
        help="Phase 2 _form25_matches.json input",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=STUDY_DIR / sec_pilot.ROSTER_FILE,
        help="Phase 1 _roster.csv input (defines the quarterly index window)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing _payouts_v2 output files"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow running with uncommitted changes (recorded in provenance)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    client_factory: Callable[[str, Path], Any] = SecClient,
    archives_factory: Callable[
        [Any, Sequence[tuple[int, int]]], Iterable[tuple[str, bytes]]
    ] = sec_name_index.cached_archives,
) -> int:
    """Validate preconditions, run Phase 3 open-case matching, and write outputs atomically."""
    parser = build_parser()
    args = parser.parse_args(argv)

    links = symlinked_outputs(args.out)
    if links:
        return _refuse(
            "output path is a symbolic link, refusing to run: " + ", ".join(map(str, links))
        )

    csv_target = args.out / OUTPUT_CSV
    json_target = args.out / OUTPUT_JSON
    for target in (csv_target, json_target):
        if target.exists() and not args.force:
            return _refuse(f"{target} already exists; use --force to replace it")

    try:
        user_agent = load_user_agent(os.environ)
    except SecUserAgentMissing as exc:
        return _refuse(str(exc))

    own_outputs = [
        rel
        for path in output_paths(args.out).values()
        if (rel := sec_pilot.lexical_relative(path, ROOT)) is not None
    ]
    git_info = git_provenance(ROOT, ignore=tuple(own_outputs))
    if git_info.get("git_dirty") is not False and not args.allow_dirty:
        return _refuse(
            f"git tree is not clean (git_dirty={git_info.get('git_dirty')!r}); "
            "commit changes or pass --allow-dirty for a trial run"
        )

    brief_path = ROOT / BRIEF_REL
    if not brief_path.is_file():
        return _refuse(f"Phase 3 brief file not found: {brief_path}")

    try:
        payouts_rows, payouts_sha256 = _read_payouts_csv(args.payouts)
        form25, form25_sha256 = _read_form25_json(args.form25)
        roster_rows, roster_sha256 = _read_roster_csv(args.roster)
        _validate_inputs(payouts_rows, form25)
    except PreconditionError as exc:
        return _refuse(str(exc))

    brief_sha256 = sec_pilot.sha256_file(brief_path)
    quarters = sec_pilot.needed_quarters(roster_rows)
    run_date = dt.date.today().isoformat()
    provenance = phase3_provenance(
        git_info=git_info,
        git_dirty_ignores=own_outputs,
        payouts_sha256=payouts_sha256,
        roster_sha256=roster_sha256,
        form25_sha256=form25_sha256,
        brief_sha256=brief_sha256,
        quarters=quarters,
        run_date=run_date,
        allow_dirty=bool(args.allow_dirty),
    )

    client = client_factory(user_agent, args.cache_dir)
    try:
        merged_rows, payload = run_open_cases(
            client,
            archives_factory,
            payouts_rows,
            form25,
            quarters=quarters,
            run_date=run_date,
        )
    except (sec_name_index.MissingArchive, PreconditionError) as exc:
        return _refuse(str(exc))
    finally:
        write_sha256sums(args.cache_dir)

    payload["provenance"] = provenance
    atomic_write_text(csv_target, _format_csv(merged_rows))
    write_json(json_target, payload, strict=True)
    print(_format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
