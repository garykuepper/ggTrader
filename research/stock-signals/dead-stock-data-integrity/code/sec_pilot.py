"""Driver for the SEC phase of the dead-stock data-integrity study: Form 25 matching pilot."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ggTrader.utils.io import atomic_write_text, write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import payout_table  # noqa: E402  (found on sys.path through bootstrap)
import sec_filings  # noqa: E402
from dead_stock_data_integrity import git_provenance  # noqa: E402
from sec_client import (  # noqa: E402
    SecClient,
    SecFetchError,
    SecUserAgentMissing,
    load_user_agent,
    write_sha256sums,
)

STUDY_DIR = study_dir(__file__)
ROSTER_FILE = "_roster.csv"
OUTPUT_JSON = "_form25_matches.json"
PAYOUTS_CSV = "_payouts.csv"
PAYOUTS_JSON = "_payouts.json"
EVENTS_STATUS_FETCH_ERROR = "fetch_error"
DEFAULT_MAX_SKIPPED = 10
STATUS_NAME_UNKNOWN = "name_unknown"


_STUDY_REL = "research/stock-signals/dead-stock-data-integrity/code/"
# Every file that shapes a result, by repository-relative path: this study's SEC code, the two
# shared parsers of the downloaded indexes and notices, and the shared writers and path helpers.
CODE_FILES = (
    f"{_STUDY_REL}sec_pilot.py",
    f"{_STUDY_REL}sec_filings.py",
    f"{_STUDY_REL}sec_client.py",
    f"{_STUDY_REL}payout_table.py",
    f"{_STUDY_REL}dead_stock_data_integrity.py",
    "research/stock-signals/_lib/sec_master_index.py",
    "research/stock-signals/_lib/sec_form25.py",
    "src/ggTrader/utils/study.py",
    "src/ggTrader/utils/io.py",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TooManySkippedQuarters(RuntimeError):
    """Raised when more quarterly indexes failed than the run allows."""


def read_roster(path: Path) -> list[dict[str, str]]:
    """Read the roster file as text rows (empty cells stay empty strings)."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_named(row: Mapping[str, Any]) -> bool:
    """True when the roster row has a company name to match on."""
    name = row.get("stock_name")
    return row.get("name_source") != "none" and isinstance(name, str) and bool(name.strip())


def needed_quarters(rows: list[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Every quarter whose index could hold a named stock's Form 25 (exit -30 to +180 days)."""
    wanted: set[tuple[int, int]] = set()
    for row in rows:
        if not is_named(row):
            continue
        exit_date = dt.date.fromisoformat(str(row["exit_date"]))
        start = exit_date - dt.timedelta(days=sec_filings.BEFORE_DAYS)
        end = exit_date + dt.timedelta(days=sec_filings.AFTER_DAYS)
        wanted.update(sec_filings.quarters_covering(start, end))
    return sorted(wanted)


def matching_rules() -> dict[str, Any]:
    """The constants that decide a name match and the filing windows, with a hash of the word list.

    ``abbreviations_sha256`` fingerprints the fixed short-form table, so a changed table shows up
    in the result file even if nobody remembers editing it.
    """
    table = json.dumps(sorted(sec_filings.ABBREVIATIONS.items()), ensure_ascii=True)
    return {
        "before_days": sec_filings.BEFORE_DAYS,
        "after_days": sec_filings.AFTER_DAYS,
        "truncated_name_length": sec_filings.TRUNCATED_NAME_LENGTH,
        "event_window_days": sec_filings.EVENT_WINDOW_DAYS,
        "suffix_words": sorted(sec_filings.SUFFIX_WORDS),
        "abbreviations_sha256": hashlib.sha256(table.encode("ascii")).hexdigest(),
    }


def output_paths(out_dir: Path) -> dict[str, Path]:
    """The three result files of the SEC phases, by file name, inside ``out_dir``."""
    return {name: out_dir / name for name in (OUTPUT_JSON, PAYOUTS_CSV, PAYOUTS_JSON)}


def lexical_relative(path: Path, root: Path) -> str | None:
    """``path`` relative to ``root`` as plain text, or ``None`` if it is not inside ``root``.

    Both paths are only made absolute and tidied (``..`` folded away), never resolved: a symlink
    must not make an output name stand for some other file.
    """
    try:
        return Path(os.path.abspath(path)).relative_to(os.path.abspath(root)).as_posix()
    except ValueError:
        return None


def symlinked_outputs(out_dir: Path) -> list[Path]:
    """The output folder and result files that are symbolic links (all must be real files)."""
    found = [out_dir] if out_dir.is_symlink() else []
    return found + [p for p in output_paths(out_dir).values() if p.is_symlink()]


def refuse_symlinked_outputs(parser: argparse.ArgumentParser, out_dir: Path) -> None:
    """Stop (exit 2) when an output path is a symlink, so no exclusion can hide another file."""
    links = symlinked_outputs(out_dir)
    if links:
        parser.error(
            "output path is a symbolic link, refusing to run: " + ", ".join(map(str, links))
        )


def quarter_dates(quarters: list[tuple[int, int]]) -> tuple[str, str] | tuple[None, None]:
    """First day of the first quarter and last day of the last quarter, as ISO dates."""
    if not quarters:
        return None, None
    (y0, q0), (y1, q1) = quarters[0], quarters[-1]
    start = dt.date(y0, 3 * q0 - 2, 1)
    end = dt.date(y1 + (q1 == 4), 1 if q1 == 4 else 3 * q1 + 1, 1) - dt.timedelta(days=1)
    return start.isoformat(), end.isoformat()


def phase_provenance(
    phase: str,
    roster_path: Path,
    rows: list[dict[str, str]],
    *,
    out_dir: Path,
    run_date: str,
    limit_quarters: int | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """The provenance block both SEC phases write: what code, rules and roster made the file.

    The block holds the git commit and dirty flag (leaving out only the three result files of
    this phase pair, by their exact repository-relative names), the SHA-256 of each code file and
    of the roster file, the run date, the quarterly-index window and the matching-rule constants.
    It holds nothing about who asked the SEC: the contact line is never read here.

    Args:
        phase: ``"form25"`` or ``"payouts"``.
        roster_path: the roster file the phase read.
        rows: the roster rows (they set the quarter window).
        out_dir: the folder the result files are written to.
        run_date: ISO date of the run.
        limit_quarters: trial-run limit on quarters, or ``None``.
        root: repository root for git (default: this study's).
    """
    root = ROOT if root is None else root
    quarters = needed_quarters(rows)
    start, end = quarter_dates(quarters)
    own_outputs = [
        rel
        for p in output_paths(out_dir).values()
        if (rel := lexical_relative(p, root)) is not None
    ]
    return {
        **git_provenance(root, ignore=tuple(own_outputs)),
        "git_dirty_ignores": own_outputs,
        "phase": phase,
        "run_date": run_date,
        "roster_sha256": sha256_file(roster_path),
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "quarter_window": {
            "first": list(quarters[0]) if quarters else None,
            "last": list(quarters[-1]) if quarters else None,
            "start": start,
            "end": end,
            "needed": len(quarters),
            "limit_quarters": limit_quarters,
        },
        "matching_rules": matching_rules(),
    }


def _is_iso_date(value: Any) -> bool:
    """True for an ISO calendar-date string."""
    try:
        dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return isinstance(value, str)


def form25_provenance_problems(prov: Any, roster_sha256: str) -> list[str]:
    """Plain-words problems with a Form 25 result's provenance block (empty means it is usable).

    Checks the structure and the values: a real commit, a true or false dirty flag, a hash for
    every file in ``CODE_FILES``, the same roster file and matching rules as now, a dated quarter
    window and a full (not trial) run.

    Args:
        prov: the ``provenance`` value read from the Form 25 file.
        roster_sha256: SHA-256 of the roster file the payouts phase is about to read.
    """
    if not isinstance(prov, dict) or not prov:
        return ["there is no provenance block"]
    problems: list[str] = []
    if prov.get("phase") != "form25":
        problems.append("phase is not 'form25'")
    head = prov.get("git_head")
    if not isinstance(head, str) or not HEX40.match(head):
        problems.append("git_head is not a 40-character commit hash")
    if not isinstance(prov.get("git_dirty"), bool):
        problems.append("git_dirty is not true or false")
    if not _is_iso_date(prov.get("run_date")):
        problems.append("run_date is not a date")
    roster = prov.get("roster_sha256")
    if not isinstance(roster, str) or not HEX64.match(roster):
        problems.append("roster_sha256 is not a 64-character hash")
    elif roster != roster_sha256:
        problems.append("it was made from a different roster file than the one given now")
    hashes = prov.get("code_sha256")
    if not isinstance(hashes, dict) or not hashes:
        problems.append("code_sha256 is missing or empty")
    else:
        absent = [name for name in CODE_FILES if name not in hashes]
        if absent:
            problems.append("code_sha256 lacks: " + ", ".join(absent))
        if not all(isinstance(v, str) and HEX64.match(v) for v in hashes.values()):
            problems.append("code_sha256 holds a value that is not a 64-character hash")
    window = prov.get("quarter_window")
    if not isinstance(window, dict) or not (
        _is_iso_date(window.get("start")) and _is_iso_date(window.get("end"))
    ):
        problems.append("quarter_window has no start and end dates")
    elif window["start"] > window["end"]:
        problems.append("quarter_window starts after it ends")
    elif window.get("limit_quarters") is not None:
        problems.append("it is a trial run (limit_quarters is set), not a full one")
    rules = prov.get("matching_rules")
    current = matching_rules()
    if not isinstance(rules, dict) or not HEX64.match(str(rules.get("abbreviations_sha256"))):
        problems.append("matching_rules has no abbreviations_sha256")
    elif rules != current:
        problems.append("matching_rules differ from the rules in the code now")
    return problems


# Per-stock confirmation levels (the only values allowed in the output).
CONFIRMED_COMMON = "confirmed_common"
UNCONFIRMED_UNREADABLE = "unconfirmed_unreadable"
NON_COMMON_ONLY = "non_common_only"
REJECTED_ISSUER_MISMATCH = "rejected_issuer_mismatch"
NOT_CHECKED = "not_checked"
CONFIRMATIONS = (
    CONFIRMED_COMMON,
    UNCONFIRMED_UNREADABLE,
    NON_COMMON_ONLY,
    REJECTED_ISSUER_MISMATCH,
    NOT_CHECKED,
)


def check_cik(read: Callable[[str, str], dict], accessions: list[str], cik: str) -> dict:
    """Decide what one name-matched CIK is, by reading its Form 25 notices in filing order.

    A name match is only a lead. The CIK counts as the issuer only when a readable notice names
    it as the issuer; a notice that names another issuer rejects it; notices that cannot be read
    (an issuer-filed HTML Form 25, a 404) leave it unconfirmed.

    Returns:
        ``{"cik", "state", "notice"}``: ``state`` is ``confirmed_common`` (the first readable
        notice whose security is common stock), ``non_common`` (readable notices, none common),
        ``rejected`` (every filing was read and each names another issuer) or ``unreadable``
        (at least one filing could not be read and none was confirmed common). ``non_common`` and
        ``rejected`` need EVERY candidate filing read. ``notice`` is the deciding notice (or the
        first failure).
    """
    first_readable: dict | None = None
    first_mismatch: dict | None = None
    first_failure: dict | None = None
    for accession in accessions:
        notice = read(accession, cik)
        if notice["status"] == sec_filings.STATUS_READABLE:
            is_common = sec_filings.notice_is_common_stock(notice["notice"]["security_description"])
            notice = {**notice, "is_common_stock": is_common}
            if is_common:
                return {"cik": cik, "state": "confirmed_common", "notice": notice}
            first_readable = first_readable or notice
        elif notice["status"] == sec_filings.STATUS_ISSUER_MISMATCH:
            first_mismatch = first_mismatch or notice
        else:
            first_failure = first_failure or notice
    # One unreadable filing means we cannot say "non-common only" or "other issuer": the CIK
    # stays a candidate (unconfirmed) even when other filings of it were read.
    if first_failure is not None or (first_readable is None and first_mismatch is None):
        return {
            "cik": cik,
            "state": "unreadable",
            "notice": first_failure or {"status": "unreadable", "reason": "no filings"},
            "other_filings_read": first_readable is not None or first_mismatch is not None,
        }
    if first_readable is not None:
        return {"cik": cik, "state": "non_common", "notice": first_readable}
    return {"cik": cik, "state": "rejected", "notice": first_mismatch}


def resolve_stock(match: dict, read: Callable[[str, str], dict]) -> dict:
    """Turn a name match into the stock's final status, ciks, confirmation and notice.

    Only CIKs confirmed as the issuer of a common-stock notice, or unconfirmed because no
    notice could be read, stay. Two or more staying CIKs make the stock ``ambiguous`` (no pick).
    """
    if match["status"] == sec_filings.STATUS_NONE:
        return {**match, "reason": "no_name_match", "confirmation": NOT_CHECKED, "notice": None}
    checks = [check_cik(read, match["by_cik"][cik], cik) for cik in match["ciks"]]
    kept = [c for c in checks if c["state"] in ("confirmed_common", "unreadable")]
    dropped = [c for c in checks if c not in kept]
    result: dict[str, Any] = {
        "ciks": [c["cik"] for c in kept],
        "accessions": [a for c in kept for a in match["by_cik"][c["cik"]]],
        "rejected_ciks": [
            {
                "cik": c["cik"],
                "reason": "issuer_mismatch" if c["state"] == "rejected" else "non_common_only",
                "detail": c["notice"].get("reason")
                or c["notice"]["notice"]["security_description"],
            }
            for c in dropped
        ],
        "cik_checks": [{"cik": c["cik"], "state": c["state"]} for c in checks],
    }
    if len(kept) > 1:
        return {
            **result,
            "status": sec_filings.STATUS_AMBIGUOUS,
            "confirmation": NOT_CHECKED,
            "notice": None,
        }
    if not kept:
        non_common = any(c["state"] == "non_common" for c in dropped)
        return {
            **result,
            "status": sec_filings.STATUS_NONE,
            "reason": "non_common_only" if non_common else "issuer_mismatch",
            "confirmation": NON_COMMON_ONLY if non_common else REJECTED_ISSUER_MISMATCH,
            "notice": None,
        }
    only = kept[0]
    confirmed = only["state"] == "confirmed_common"
    return {
        **result,
        "status": sec_filings.STATUS_MATCHED,
        "confirmation": CONFIRMED_COMMON if confirmed else UNCONFIRMED_UNREADABLE,
        "notice": only["notice"],
    }


def summarize(stocks: list[dict]) -> dict[str, Any]:
    """Count the stocks by outcome; the counts add up to the roster size.

    ``matched`` (the headline) is ``confirmed_common + unconfirmed_unreadable``; both parts are
    reported. ``name_matched`` counts the stocks the name rule found before any notice was read.
    """
    by_confirmation = {name: 0 for name in CONFIRMATIONS}
    counts = {"ambiguous": 0, "none": 0, STATUS_NAME_UNKNOWN: 0}
    for stock in stocks:
        if stock["status"] == sec_filings.STATUS_MATCHED:
            by_confirmation[stock["confirmation"]] += 1
        elif stock["confirmation"] in (NON_COMMON_ONLY, REJECTED_ISSUER_MISMATCH):
            by_confirmation[stock["confirmation"]] += 1
        else:
            counts[stock["status"]] += 1
    matched = by_confirmation[CONFIRMED_COMMON] + by_confirmation[UNCONFIRMED_UNREADABLE]
    named = len(stocks) - counts[STATUS_NAME_UNKNOWN]
    return {
        "roster_rows": len(stocks),
        "named_stocks": named,
        "name_matched": named - counts["none"],
        CONFIRMED_COMMON: by_confirmation[CONFIRMED_COMMON],
        UNCONFIRMED_UNREADABLE: by_confirmation[UNCONFIRMED_UNREADABLE],
        NON_COMMON_ONLY: by_confirmation[NON_COMMON_ONLY],
        REJECTED_ISSUER_MISMATCH: by_confirmation[REJECTED_ISSUER_MISMATCH],
        "ambiguous": counts["ambiguous"],
        "none": counts["none"],
        STATUS_NAME_UNKNOWN: counts[STATUS_NAME_UNKNOWN],
        "matched": matched,
        "matched_rate_of_named": matched / named if named else None,
    }


def run_form25(
    client: Any,
    rows: list[dict[str, str]],
    *,
    limit_quarters: int | None = None,
    max_skipped: int = DEFAULT_MAX_SKIPPED,
    run_date: str = "",
) -> dict[str, Any]:
    """Download the needed quarterly indexes, match every roster row, read the notices.

    Args:
        client: a ``SecClient`` (or a fake with ``get`` and ``manifest``).
        rows: the roster rows.
        limit_quarters: use only the first N needed quarters (a trial run).
        max_skipped: stop with ``TooManySkippedQuarters`` when more quarters than this fail.
        run_date: ISO date recorded in the output.

    Returns:
        The JSON-ready result: per-stock results, summary counts, quarters downloaded and
        skipped, request counts and the client's manifest of cached files.
    """
    quarters = needed_quarters(rows)
    total_needed = len(quarters)
    if limit_quarters is not None:
        quarters = quarters[:limit_quarters]
    candidates: list[dict] = []
    downloaded: list[list[int]] = []
    skipped: list[dict] = []
    for year, quarter in quarters:
        try:
            candidates.extend(sec_filings.load_quarter(client, year, quarter))
            downloaded.append([year, quarter])
        except (SecFetchError, ValueError) as exc:
            skipped.append({"quarter": [year, quarter], "reason": f"{type(exc).__name__}: {exc}"})
            if len(skipped) > max_skipped:
                raise TooManySkippedQuarters(
                    f"{len(skipped)} quarterly indexes failed (limit {max_skipped}); last: "
                    f"{year} Q{quarter}: {exc}"
                ) from exc
    memo: dict[tuple[str, str], dict] = {}

    def read(accession: str, cik: str) -> dict:
        """Read each filing once, even when several roster stocks lead to it."""
        if (accession, cik) not in memo:
            memo[(accession, cik)] = sec_filings.read_notice(client, accession, cik)
        return memo[(accession, cik)]

    stocks = []
    for row in rows:
        result: dict[str, Any] = {
            "symbol": row["symbol"],
            "permno": row.get("permno"),
            "stock_name": row.get("stock_name"),
            "exit_date": row["exit_date"],
        }
        if not is_named(row):
            result.update(
                status=STATUS_NAME_UNKNOWN,
                ciks=[],
                accessions=[],
                confirmation=NOT_CHECKED,
                notice=None,
            )
        else:
            result.update(resolve_stock(sec_filings.match_roster_row(row, candidates), read))
        stocks.append(result)
    return {
        "phase": "form25",
        "run_date": run_date,
        "limit_quarters": limit_quarters,
        "roster_rows": len(rows),
        "quarters_needed": total_needed,
        "quarters_downloaded": downloaded,
        "quarters_skipped": skipped,
        "request_count": {
            "network_fetches": getattr(client, "network_fetches", 0),
            "cache_hits": getattr(client, "cache_hits", 0),
            "corrupt_cache_refetches": getattr(client, "corrupt_cache_refetches", 0),
        },
        "summary": summarize(stocks),
        "stocks": stocks,
        "manifest": client.manifest(),
    }


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes (the input files are small)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_payouts(
    client: Any,
    rows: list[dict[str, Any]],
    form25: Mapping[str, Any],
    *,
    run_date: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the 8-K events of every matched stock and build the payout table.

    Only a stock with exactly one matched issuer (confirmed or name-only) gets its submissions
    file fetched, around its exit date. A failed fetch is recorded as ``fetch_error`` and the run
    goes on; the stock then reads as ``no_8k_found`` with that status beside it, never silently.

    Args:
        client: a ``SecClient`` (or a fake with ``get`` and ``manifest``).
        rows: the roster rows (text values, as ``read_roster`` returns them).
        form25: the parsed ``_form25_matches.json``.
        run_date: ISO date recorded in the output.

    Returns:
        ``(table, payload)``: the payout table (one row per roster row) and the JSON-ready
        summary (coverage, status counts, events by stock, request counts, the client manifest).
    """
    matches = {int(stock["permno"]): stock for stock in form25["stocks"]}
    events: dict[int, list[dict]] = {}
    statuses: dict[int, str] = {}
    errors: list[dict[str, Any]] = []
    for row in rows:
        permno = int(row["permno"])
        match = matches.get(permno)
        if (
            not is_named(row)
            or match is None
            or match["status"] != sec_filings.STATUS_MATCHED
            or len(match["ciks"]) != 1
        ):
            continue
        around = dt.date.fromisoformat(str(row["exit_date"]))
        try:
            events[permno], statuses[permno] = sec_filings.fetch_events_with_status(
                client, match["ciks"][0], around, days=sec_filings.EVENT_WINDOW_DAYS
            )
        except (SecFetchError, ValueError) as exc:
            events[permno], statuses[permno] = [], EVENTS_STATUS_FETCH_ERROR
            errors.append({"permno": permno, "reason": f"{type(exc).__name__}: {exc}"})
    roster = pd.DataFrame(rows)
    roster["last_close"] = pd.to_numeric(roster["last_close"], errors="coerce")
    # Exchange codes arrive as text from the csv; the table only trusts real numbers.
    roster["exchcd"] = pd.to_numeric(roster["exchcd"], errors="coerce")
    table = payout_table.build_payout_table(roster, matches, {}, events, events_status=statuses)
    items_missing = sum(sec_filings.count_items_missing(found) for found in events.values())
    skipped = form25.get("quarters_skipped") or []
    status_counts: dict[str, int] = {}
    for status in statuses.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "phase": "payouts",
        "run_date": run_date,
        "roster_rows": len(rows),
        "coverage_summary": payout_table.coverage_summary(
            table, items_missing=items_missing, quarters_skipped=len(skipped)
        ),
        "events_status_counts": dict(sorted(status_counts.items())),
        "events_fetch_errors": errors,
        "items_missing": items_missing,
        "form25_quarters_skipped": skipped,
        "events": {str(permno): found for permno, found in sorted(events.items())},
        "literature_fills": {
            "-".join(map(str, codes)): fill for codes, fill in payout_table.LITERATURE_FILLS.items()
        },
        "request_count": {
            "network_fetches": getattr(client, "network_fetches", 0),
            "cache_hits": getattr(client, "cache_hits", 0),
            "corrupt_cache_refetches": getattr(client, "corrupt_cache_refetches", 0),
        },
        "manifest": client.manifest(),
    }
    return table, payload


def build_parser() -> argparse.ArgumentParser:
    """The command line: where the cache and the output go; no date window to default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["form25", "payouts"], required=True, help="which SEC phase"
    )
    parser.add_argument("--cache-dir", type=Path, required=True, help="folder for cached files")
    parser.add_argument("--out", type=Path, required=True, help="folder for the result file")
    parser.add_argument("--roster", type=Path, default=STUDY_DIR / ROSTER_FILE, help="roster csv")
    parser.add_argument("--limit-quarters", type=int, default=None, help="trial: first N quarters")
    parser.add_argument("--max-skipped", type=int, default=DEFAULT_MAX_SKIPPED, help="stop above")
    parser.add_argument("--force", action="store_true", help="overwrite existing result files")
    return parser


def main(
    argv: list[str] | None = None,
    client_factory: Callable[[str, Path], Any] = SecClient,
) -> int:
    """Check the arguments, run the chosen SEC phase, write the result and the cache checksums."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase == "payouts":
        return main_payouts(parser, args, client_factory)
    refuse_symlinked_outputs(parser, args.out)
    target = args.out / OUTPUT_JSON
    if target.exists() and not args.force:
        parser.error(f"{target} already exists; use --force to replace it")
    try:
        user_agent = load_user_agent(os.environ)
    except SecUserAgentMissing as exc:
        parser.error(str(exc))
    rows = read_roster(args.roster)
    client = client_factory(user_agent, args.cache_dir)
    run_date = dt.date.today().isoformat()
    # Taken before any download so the commit and the code hashes are those the run started from.
    provenance = phase_provenance(
        "form25",
        args.roster,
        rows,
        out_dir=args.out,
        run_date=run_date,
        limit_quarters=args.limit_quarters,
    )
    try:
        payload = run_form25(
            client,
            rows,
            limit_quarters=args.limit_quarters,
            max_skipped=args.max_skipped,
            run_date=run_date,
        )
    except TooManySkippedQuarters as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2
    finally:
        write_sha256sums(args.cache_dir)
    payload["provenance"] = provenance
    write_json(target, payload, strict=True)
    summary = payload["summary"]
    print(
        f"matched {summary['matched']} (confirmed common {summary['confirmed_common']}, "
        f"unconfirmed {summary['unconfirmed_unreadable']}); "
        f"name matched {summary['name_matched']}; "
        f"non-common only {summary['non_common_only']}, issuer mismatch "
        f"{summary['rejected_issuer_mismatch']}, ambiguous {summary['ambiguous']}, "
        f"none {summary['none']}, name unknown {summary['name_unknown']}; "
        f"{len(payload['quarters_skipped'])} quarters skipped"
    )
    return 0


def main_payouts(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    client_factory: Callable[[str, Path], Any],
) -> int:
    """The payouts phase: read the roster and the Form 25 result, fetch events, write the table."""
    refuse_symlinked_outputs(parser, args.out)
    csv_target, json_target = args.out / PAYOUTS_CSV, args.out / PAYOUTS_JSON
    for target in (csv_target, json_target):
        if target.exists() and not args.force:
            parser.error(f"{target} already exists; use --force to replace it")
    form25_file = args.out / OUTPUT_JSON
    if not form25_file.is_file():
        parser.error(f"{form25_file} is missing; run --phase form25 first")
    # One read: the bytes that are hashed are the bytes that are parsed and used.
    form25_bytes = form25_file.read_bytes()
    form25_sha256 = hashlib.sha256(form25_bytes).hexdigest()
    try:
        form25 = json.loads(form25_bytes.decode("utf-8"))
    except ValueError as exc:
        parser.error(f"{form25_file} is not readable JSON ({exc})")
    rows = read_roster(args.roster)
    form25_prov = form25.get("provenance") if isinstance(form25, dict) else None
    problems = form25_provenance_problems(form25_prov, sha256_file(args.roster))
    if problems:
        parser.error(
            f"{form25_file} cannot be used: "
            + "; ".join(problems)
            + ". Rerun --phase form25 so the file records what made it."
        )
    if {int(r["permno"]) for r in rows} != {int(s["permno"]) for s in form25["stocks"]}:
        parser.error(f"{form25_file} does not cover the same stocks as {args.roster}")
    try:
        user_agent = load_user_agent(os.environ)
    except SecUserAgentMissing as exc:
        parser.error(str(exc))
    client = client_factory(user_agent, args.cache_dir)
    run_date = dt.date.today().isoformat()
    provenance = phase_provenance("payouts", args.roster, rows, out_dir=args.out, run_date=run_date)
    try:
        table, payload = run_payouts(client, rows, form25, run_date=run_date)
    finally:
        write_sha256sums(args.cache_dir)
    payload["provenance"] = {
        **provenance,
        "form25_sha256": form25_sha256,
        "form25_git_head": form25_prov["git_head"],
    }
    atomic_write_text(csv_target, table.to_csv(index=False))
    write_json(json_target, payload, strict=True)
    cov = payload["coverage_summary"]
    print(
        f"{cov['roster_rows']} stocks; documented {cov['documented_evidence']}; "
        + ", ".join(f"{name} {n}" for name, n in cov["by_class"].items())
        + f"; confirmed {cov['by_match_confidence']['confirmed']}, "
        f"name only {cov['by_match_confidence']['name_only']}; "
        f"suggested loss for {cov['with_suggested_loss']}; conflicts {cov['conflicts']}; "
        f"events status {payload['events_status_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
