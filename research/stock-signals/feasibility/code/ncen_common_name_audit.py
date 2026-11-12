"""Cross-check historical screener labels against retrospective SEC fund metadata."""

import argparse
import json
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ggTrader.utils.manifest import sha256_file

TABLES = {
    "SUBMISSION.tsv": ["ACCESSION_NUMBER", "CIK", "FILING_DATE"],
    "REGISTRANT.tsv": ["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME", "INVESTMENT_COMPANY_TYPE"],
    "FUND_REPORTED_INFO.tsv": ["ACCESSION_NUMBER", "FUND_ID"],
    "SECURITY_RELATED_ITEM.tsv": [
        "FUND_ID",
        "TICKER_SYMBOL",
        "DESCRIPTION",
        "SECURITY_CLASS_TITLE",
        "EXCHANGE",
    ],
}


def compare(screener, submissions, registrants, funds, securities, cutoff):
    """Return exact-label candidate matches, not security classifications."""
    decision = date.fromisoformat(cutoff)
    if decision.isoformat() != cutoff:
        raise ValueError("Invalid cutoff")
    names = pd.DataFrame(screener)[["symbol", "name"]]
    if names.isna().any().any() or names["symbol"].duplicated().any():
        raise ValueError("Missing or duplicate screener labels")
    names = names[names["name"].str.contains("common stock", case=False, regex=False)]
    for frame, key in (
        (submissions, "ACCESSION_NUMBER"),
        (registrants, "ACCESSION_NUMBER"),
        (funds, "FUND_ID"),
    ):
        if frame[key].isna().any() or (frame[key] == "").any() or frame[key].duplicated().any():
            raise ValueError(f"Missing or duplicate {key}")
    dates = pd.to_datetime(submissions["FILING_DATE"], format="%d-%b-%Y", errors="raise")
    submissions = submissions.loc[dates.dt.date < decision]
    registrants = registrants[registrants["INVESTMENT_COMPANY_TYPE"] == "N-2"]
    rows = securities.merge(funds, on="FUND_ID", validate="many_to_one")
    rows = rows.merge(registrants, on="ACCESSION_NUMBER", validate="many_to_one")
    rows = rows.merge(
        submissions, on="ACCESSION_NUMBER", validate="many_to_one", suffixes=("", "_SUBMISSION")
    )
    rows = rows.merge(names, left_on="TICKER_SYMBOL", right_on="symbol", validate="many_to_one")
    if (rows["CIK"] != rows["CIK_SUBMISSION"]).any():
        raise ValueError("Mismatched registrant and submission CIK")
    rows = rows.sort_values(["TICKER_SYMBOL", "ACCESSION_NUMBER", "FUND_ID"])
    return {
        "screener_common_name_labels": len(names),
        "matched_rows": len(rows),
        "matched_unique_labels": int(rows["TICKER_SYMBOL"].nunique()),
        "labels_with_multiple_ciks": int(
            (rows.groupby("TICKER_SYMBOL")["CIK"].nunique() > 1).sum()
        ),
        "matches": rows.to_dict(orient="records"),
    }


def main():
    """Read pinned sources and write a new retrospective diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("ncen", "screener", "ncen-sha256", "screener-sha256", "cutoff", "out"):
        parser.add_argument(f"--{arg}", required=True)
    args = parser.parse_args()
    for path, expected in ((args.ncen, args.ncen_sha256), (args.screener, args.screener_sha256)):
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
    with open(args.screener, encoding="utf-8") as handle:
        screener = json.load(handle)
    with ZipFile(args.ncen) as archive:
        tables = []
        for filename, columns in TABLES.items():
            found = [info for info in archive.infolist() if info.filename == filename]
            if len(found) != 1 or found[0].file_size > 4_000_000:
                raise ValueError(f"Missing, duplicate, or oversized table: {filename}")
            with archive.open(found[0]) as handle:
                tables.append(
                    pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, usecols=columns)
                )
    result = {
        "purpose": "Retrospective exact-label leads only; never use as automatic exclusions.",
        "limitations": (
            "One quarterly archive; identity, share class, completeness and historical "
            "availability unverified."
        ),
        "cutoff": args.cutoff,
        "inputs": {"ncen_sha256": args.ncen_sha256, "screener_sha256": args.screener_sha256},
        "driver_sha256": sha256_file(Path(__file__)),
        **compare(screener, *tables, cutoff=args.cutoff),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(f"{result['matched_unique_labels']} matching labels in {result['matched_rows']} rows")


if __name__ == "__main__":
    main()
