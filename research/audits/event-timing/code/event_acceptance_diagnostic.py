"""Reproduce cached SEC acceptance-clock discrepancies without changing audit rules."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from ggTrader.lab.event_timing_sources import filing_rows, parse_filing_header

NEW_YORK = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "data/external/event_timing_audit/insider-predecessor-bnz1mn22"


def compare_acceptance_pair(
    header: str, api: str, header_filing_date: str, api_filing_date: str
) -> dict:
    """Compare candidate clocks against same-day Form 4 filing dates and EDGAR hours."""
    naive = dt.datetime.fromisoformat(header)
    api_utc = dt.datetime.fromisoformat(api.replace("Z", "+00:00"))
    if naive.tzinfo is not None or api_utc.utcoffset() != dt.timedelta(0):
        raise ValueError("Expected a naive header clock and an explicitly UTC API clock")
    local = naive.replace(tzinfo=NEW_YORK)
    api_local = api_utc.astimezone(NEW_YORK)
    filed = dt.date.fromisoformat(header_filing_date)
    dates_match = filed == dt.date.fromisoformat(api_filing_date)
    offset_hours = local.utcoffset().total_seconds() / 3600
    wall_difference = (api_utc.replace(tzinfo=None) - naive).total_seconds() / 3600
    header_hours = dt.time(6) <= local.time() <= dt.time(22)
    api_hours = dt.time(6) <= api_local.time() <= dt.time(22)
    return {
        "header_acceptance": header,
        "api_acceptance": api,
        "header_filing_date": header_filing_date,
        "api_filing_date": api_filing_date,
        "header_new_york": local.isoformat(),
        "api_read_as_utc_in_new_york": api_local.isoformat(),
        "new_york_utc_offset_hours": offset_hours,
        "wall_clock_difference_hours": wall_difference,
        "instant_difference_hours": (api_utc - local).total_seconds() / 3600,
        "twice_offset_pattern": wall_difference == -2 * offset_hours,
        "filing_dates_corroborate": dates_match,
        "header_in_filing_hours": header_hours,
        "api_in_filing_hours": api_hours,
        "header_matches_filing_date": local.date() == filed,
        "api_matches_filing_date": api_local.date() == filed,
        "header_consistent": dates_match and header_hours and local.date() == filed,
        "api_consistent": dates_match and api_hours and api_local.date() == filed,
    }


def diagnose_cache(cache_dir: Path) -> dict:
    """Join every cached Form 4 header to its issuer/accession API rows, checking hashes."""
    cache_dir = cache_dir.resolve()
    manifest_bytes = (cache_dir / "MANIFEST.json").read_bytes()
    entries = json.loads(manifest_bytes)["files"]
    api_rows: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    headers = []
    for entry in entries:
        url = entry["url"]
        is_api = url.startswith("https://data.sec.gov/submissions/")
        submission = re.search(r"/edgar/data/(\d+)/\d+/(\d{10}-\d{2}-\d{6})\.txt$", url)
        if not is_api and not submission:
            continue
        path = (cache_dir / entry["cache_file"]).resolve()
        if not path.is_relative_to(cache_dir):
            raise ValueError("Cache file escapes its manifest directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {entry['cache_file']}")
        if entry.get("content_encoding", "identity") != "identity":
            raise ValueError(f"Unsupported cached encoding: {entry['cache_file']}")
        reference = {key: entry[key] for key in ("url", "cache_file", "sha256")}
        if is_api:
            cik = str(int(re.search(r"CIK(\d+)", url)[1]))
            payload = json.loads(raw)
            block = payload["filings"]["recent"] if "filings" in payload else payload
            for row in filing_rows(block):
                api_rows.setdefault((cik, row["accession"]), []).append((row, reference))
        else:
            header = raw.decode("utf-8").split("</SEC-HEADER>", 1)[0]
            form = re.search(r"CONFORMED SUBMISSION TYPE:\s*(\S+)", header)
            if form and form[1] in {"4", "4/A"}:
                cik, accession = str(int(submission[1])), submission[2]
                parsed = parse_filing_header(header, accession, cik)
                if parsed.get("reason"):
                    raise ValueError(f"{accession}: {parsed['reason']}")
                headers.append((cik, accession, parsed, reference))
    pairs = []
    seen = set()
    for cik, accession, header, reference in sorted(headers, key=lambda item: item[:2]):
        if (cik, accession) in seen:
            raise ValueError(f"Duplicate cached header: {accession}")
        seen.add((cik, accession))
        matches = api_rows.get((cik, accession), [])
        values = {(row["acceptance"], row["filing_date"], row["form"]) for row, _ in matches}
        if len(values) != 1:
            raise ValueError(f"Missing or inconsistent cached API rows: {accession}")
        row = matches[0][0]
        if row["form"] not in {"4", "4/A"}:
            raise ValueError(f"API/header form mismatch: {accession}")
        if not all((header["header_acceptance"], header["header_filing_date"], row["acceptance"])):
            raise ValueError(f"Missing cached acceptance/date field: {accession}")
        pairs.append(
            {
                "issuer_cik": cik,
                "accession": accession,
                "header_source": reference,
                "api_sources": [ref for _, ref in matches],
                **compare_acceptance_pair(
                    header["header_acceptance"],
                    row["acceptance"],
                    header["header_filing_date"],
                    row["filing_date"],
                ),
            }
        )
    return {
        "schema": "event-acceptance-diagnostic-v1",
        "scope": "Cached Form 4/4-A pairs only; no source-rule change or annual qualification",
        "cache_directory": cache_dir.name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pair_count": len(pairs),
        "header_consistent_count": sum(pair["header_consistent"] for pair in pairs),
        "api_inconsistent_count": sum(not pair["api_consistent"] for pair in pairs),
        "twice_offset_pattern_count": sum(pair["twice_offset_pattern"] for pair in pairs),
        "pairs": pairs,
    }


def main() -> None:
    """Print a deterministic JSON diagnostic from an existing immutable cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    print(json.dumps(diagnose_cache(args.cache_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
