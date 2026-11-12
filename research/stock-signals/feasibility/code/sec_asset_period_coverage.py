"""Count annual SEC asset-period metadata without calculating a signal.

SEC notes-archive period and fact dates are rounded to month-end. Counts and
four-month checks use those labels and do not validate exact-date eligibility.
"""

import argparse
import calendar
import csv
import io
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from zipfile import ZipFile

from ggTrader.utils.manifest import sha256_file


def _four_months(day: date) -> date:
    month = day.month + 4
    year, month = day.year + (month - 1) // 12, (month - 1) % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def summarize(reports: list[tuple], facts: dict, decision: date) -> dict:
    """Count period presence; this cannot establish actual public availability."""
    by_cik = defaultdict(list)
    for cik, accession, end, filed in reports:
        if end <= filed < decision and _four_months(end) <= decision:
            by_cik[cik].append((end, accession))
    counts = {
        "ciks_with_eligible_10k_in_archives": len(by_cik),
        "ciks_with_annual_period_pair_350_380_days": 0,
        "ciks_with_current_and_prior_year_assets": 0,
        "ciks_with_three_asset_periods": 0,
        "three_period_ciks_with_non_dec31_latest": 0,
        "ciks_with_ambiguous_prior_period": 0,
    }
    for rows in by_cik.values():
        latest = max(end for end, _ in rows)
        previous = {end for end, _ in rows if 350 <= (latest - end).days <= 380}
        if len(previous) != 1:
            counts["ciks_with_ambiguous_prior_period"] += len(previous) > 1
            continue
        prior = next(iter(previous))
        counts["ciks_with_annual_period_pair_350_380_days"] += 1
        if not any({latest, prior} <= facts.get(acc, set()) for end, acc in rows if end == latest):
            continue
        counts["ciks_with_current_and_prior_year_assets"] += 1
        if any(
            prior in facts.get(acc, set())
            and any(350 <= (prior - day).days <= 380 for day in facts.get(acc, set()))
            for end, acc in rows
            if end == prior
        ):
            counts["ciks_with_three_asset_periods"] += 1
            counts["three_period_ciks_with_non_dec31_latest"] += (latest.month, latest.day) != (
                12,
                31,
            )
    return counts


def _date(value: str) -> date | None:
    try:
        result = date.fromisoformat(value)
        return result if result.strftime("%Y%m%d") == value else None
    except ValueError:
        return None


def scan(archives: list[Path], decision: date) -> dict:
    """Scan annual filings and retain fact dates, not values."""
    csv.field_size_limit(sys.maxsize)
    reports, facts, seen = [], defaultdict(set), set()
    for path in archives:
        with ZipFile(path) as archive:
            with archive.open("sub.tsv") as raw:
                filings = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
                selected = set()
                for row in filings:
                    end, filed = _date(row["period"]), _date(row["filed"])
                    if row["form"] != "10-K" or end is None or filed is None:
                        continue
                    if not (end <= filed < decision and _four_months(end) <= decision):
                        continue
                    accession = row["adsh"]
                    if not accession or accession in seen:
                        raise ValueError(f"Duplicate or missing accession: {accession!r}")
                    seen.add(accession)
                    selected.add(accession)
                    reports.append((row["cik"], accession, end, filed))
            with archive.open("num.tsv") as raw:
                rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
                for row in rows:
                    if (
                        row["adsh"] in selected
                        and row["tag"] == "Assets"
                        and row["version"].startswith("us-gaap/")
                        and row["uom"] == "USD"
                        and row["qtrs"] == "0"
                        and row["dimn"] == "0"
                        and not row["coreg"]
                        and row["value"].strip()
                    ):
                        end = _date(row["ddate"])
                        if end is not None:
                            facts[row["adsh"]].add(end)
    return summarize(reports, facts, decision)


def main() -> None:
    """Write a new metadata report with source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decision = date.fromisoformat(args.date)
    if decision.isoformat() != args.date:
        raise ValueError("Invalid decision date")
    result = {
        "purpose": "Filing and asset-period presence only; not a validated training panel",
        "decision_date": args.date,
        "counts": scan(args.archive, decision),
        "archive_sha256": [sha256_file(path) for path in args.archive],
        "code_sha256": sha256_file(Path(__file__)),
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
