"""Driver for the SEC phase of the dead-stock data-integrity study: Form 25 matching pilot."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
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
    target = args.out / OUTPUT_JSON
    if target.exists() and not args.force:
        parser.error(f"{target} already exists; use --force to replace it")
    try:
        user_agent = load_user_agent(os.environ)
    except SecUserAgentMissing as exc:
        parser.error(str(exc))
    rows = read_roster(args.roster)
    client = client_factory(user_agent, args.cache_dir)
    try:
        payload = run_form25(
            client,
            rows,
            limit_quarters=args.limit_quarters,
            max_skipped=args.max_skipped,
            run_date=dt.date.today().isoformat(),
        )
    except TooManySkippedQuarters as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2
    finally:
        write_sha256sums(args.cache_dir)
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
    csv_target, json_target = args.out / PAYOUTS_CSV, args.out / PAYOUTS_JSON
    for target in (csv_target, json_target):
        if target.exists() and not args.force:
            parser.error(f"{target} already exists; use --force to replace it")
    form25_file = args.out / OUTPUT_JSON
    if not form25_file.is_file():
        parser.error(f"{form25_file} is missing; run --phase form25 first")
    form25 = json.loads(form25_file.read_text(encoding="utf-8"))
    rows = read_roster(args.roster)
    if {int(r["permno"]) for r in rows} != {int(s["permno"]) for s in form25["stocks"]}:
        parser.error(f"{form25_file} does not cover the same stocks as {args.roster}")
    try:
        user_agent = load_user_agent(os.environ)
    except SecUserAgentMissing as exc:
        parser.error(str(exc))
    client = client_factory(user_agent, args.cache_dir)
    try:
        table, payload = run_payouts(client, rows, form25, run_date=dt.date.today().isoformat())
    finally:
        write_sha256sums(args.cache_dir)
    payload["provenance"] = {
        **git_provenance(ROOT),
        "roster_sha256": sha256_file(args.roster),
        "form25_sha256": sha256_file(form25_file),
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
