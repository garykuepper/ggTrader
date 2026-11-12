"""Compare SEC filing ticker labels with a saved snapshot, without returns."""

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZipFile

from ggTrader.utils.manifest import sha256_file


def _symbol(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def coverage(
    archive: Path,
    snapshot: Path,
    comparison_date: str,
    *,
    additional_archives: tuple[Path, ...] = (),
) -> dict:
    """Count label matches; filing dates do not establish public availability."""
    try:
        day = date.fromisoformat(comparison_date)
        if day.isoformat() != comparison_date:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid comparison date") from exc
    cutoff = (day - timedelta(days=1)).strftime("%Y%m%d")
    panel = {_symbol(row["symbol"]) for row in json.loads(snapshot.read_text())}
    csv.field_size_limit(sys.maxsize)
    by_symbol, by_issuer = defaultdict(set), defaultdict(set)
    facts, submission_count = 0, 0
    seen, archive_matches = set(), []
    for path in (archive, *additional_archives):
        current = set()
        with ZipFile(path) as z:
            with z.open("sub.tsv") as raw:
                rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
                submissions = {}
                for row in rows:
                    accession, filed = row["adsh"], row["filed"]
                    if not accession or accession in seen:
                        raise ValueError(f"Duplicate or missing accession: {accession!r}")
                    seen.add(accession)
                    try:
                        if len(filed) != 8 or date.fromisoformat(filed).strftime("%Y%m%d") != filed:
                            raise ValueError
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid filing date for {accession}") from exc
                    if filed <= cutoff and row["form"] in {"10-K", "10-Q", "8-K"}:
                        submissions[accession] = row["cik"]
                submission_count += len(submissions)
            with z.open("txt.tsv") as raw:
                rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
                for row in rows:
                    cik = submissions.get(row["adsh"])
                    symbol = _symbol(row["value"])
                    if cik and row["tag"] == "TradingSymbol" and symbol:
                        by_symbol[symbol].add(cik)
                        by_issuer[cik].add(symbol)
                        current.add(symbol)
                        facts += 1
        archive_matches.append(panel & current)
    matched = panel & by_symbol.keys()
    ambiguous = sorted(symbol for symbol in matched if len(by_symbol[symbol]) > 1)
    return {
        "counts": {
            "snapshot_distinct_normalized_symbols": len(panel),
            "qualifying_submissions": submission_count,
            "nonempty_symbol_facts": facts,
            "snapshot_matched_symbols": len(matched),
            "matched_single_cik": len(matched) - len(ambiguous),
            "matched_multiple_ciks": len(ambiguous),
            "snapshot_unmatched_symbols": len(panel - matched),
            "issuers_with_multiple_ticker_labels": sum(len(v) > 1 for v in by_issuer.values()),
        },
        "matched_multiple_cik_symbols": ambiguous,
        "per_archive_matched": [len(labels) for labels in archive_matches],
        "matched_in_every_archive": len(set.intersection(*archive_matches)),
    }


def main() -> None:
    """Write a new descriptive result with hashes of both inputs and this code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, action="append")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = coverage(
        args.archive[0], args.snapshot, args.date, additional_archives=args.archive[1:]
    )
    result.update(
        purpose="Label comparison only; not a verified identity, universe, or return study",
        comparison_date=args.date,
        archive_sha256=[sha256_file(path) for path in args.archive],
        snapshot_sha256=sha256_file(args.snapshot),
        code_sha256=sha256_file(Path(__file__)),
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
