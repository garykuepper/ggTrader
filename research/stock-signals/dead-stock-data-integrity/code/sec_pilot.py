"""Driver for the SEC phase of the dead-stock data-integrity study: Form 25 matching pilot."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import sec_filings  # noqa: E402  (found on sys.path through bootstrap)
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
        ``rejected`` (every readable notice names another issuer; none is about this CIK) or
        ``unreadable``. ``notice`` is the deciding notice (or the first failure).
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
    if first_readable is not None:
        return {"cik": cik, "state": "non_common", "notice": first_readable}
    if first_mismatch is not None:
        return {"cik": cik, "state": "rejected", "notice": first_mismatch}
    return {
        "cik": cik,
        "state": "unreadable",
        "notice": first_failure or {"status": "unreadable", "reason": "no filings"},
    }


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


def build_parser() -> argparse.ArgumentParser:
    """The command line: where the cache and the output go; no date window to default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["form25"], required=True, help="which SEC phase")
    parser.add_argument("--cache-dir", type=Path, required=True, help="folder for cached files")
    parser.add_argument("--out", type=Path, required=True, help="folder for the result file")
    parser.add_argument("--roster", type=Path, default=STUDY_DIR / ROSTER_FILE, help="roster csv")
    parser.add_argument("--limit-quarters", type=int, default=None, help="trial: first N quarters")
    parser.add_argument("--max-skipped", type=int, default=DEFAULT_MAX_SKIPPED, help="stop above")
    parser.add_argument("--force", action="store_true", help="overwrite an existing result file")
    return parser


def main(
    argv: list[str] | None = None,
    client_factory: Callable[[str, Path], Any] = SecClient,
) -> int:
    """Check the arguments, run the Form 25 phase, write the result and the cache checksums."""
    parser = build_parser()
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
