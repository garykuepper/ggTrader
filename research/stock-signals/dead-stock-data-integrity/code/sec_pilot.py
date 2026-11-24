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


def choose_notice(client: Any, accessions: list[str], cik: str) -> dict:
    """Read the stock's Form 25 filings in filing order and keep the first common-stock one.

    If none is common stock, keep the first readable one (flagged), else the first failure.
    """
    first_readable: dict | None = None
    first_failure: dict | None = None
    for accession in accessions:
        notice = sec_filings.read_notice(client, accession, cik)
        if notice["status"] == "readable":
            is_common = sec_filings.notice_is_common_stock(notice["notice"]["security_description"])
            notice = {**notice, "is_common_stock": is_common}
            if is_common:
                return notice
            first_readable = first_readable or notice
        else:
            first_failure = first_failure or notice
    return first_readable or first_failure or {"status": "unreadable", "reason": "no filings"}


def summarize(stocks: list[dict]) -> dict[str, Any]:
    """Count the match statuses and what happened to the matched stocks' notices."""
    counts = {"matched": 0, "ambiguous": 0, "none": 0, STATUS_NAME_UNKNOWN: 0}
    for stock in stocks:
        counts[stock["status"]] += 1
    notices = [s["notice"] for s in stocks if s["status"] == sec_filings.STATUS_MATCHED]
    readable = [n for n in notices if n["status"] == "readable"]
    named = len(stocks) - counts[STATUS_NAME_UNKNOWN]
    return {
        **counts,
        "named_stocks": named,
        "exact_match_rate_of_named": counts["matched"] / named if named else None,
        "notices_readable": len(readable),
        "notices_unreadable": len(notices) - len(readable),
        "notices_common_stock": sum(1 for n in readable if n.get("is_common_stock")),
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
    stocks = []
    for row in rows:
        result: dict[str, Any] = {
            "symbol": row["symbol"],
            "permno": row.get("permno"),
            "stock_name": row.get("stock_name"),
            "exit_date": row["exit_date"],
        }
        if not is_named(row):
            result.update(status=STATUS_NAME_UNKNOWN, ciks=[], accessions=[], notice=None)
        else:
            match = sec_filings.match_roster_row(row, candidates)
            result.update(match)
            if match["status"] == sec_filings.STATUS_MATCHED:
                result["notice"] = choose_notice(client, match["accessions"], match["ciks"][0])
            else:
                result["notice"] = None
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
        f"matched {summary['matched']}, ambiguous {summary['ambiguous']}, none {summary['none']}, "
        f"name unknown {summary['name_unknown']}; notices readable {summary['notices_readable']}, "
        f"unreadable {summary['notices_unreadable']}; "
        f"{len(payload['quarters_skipped'])} quarters skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
