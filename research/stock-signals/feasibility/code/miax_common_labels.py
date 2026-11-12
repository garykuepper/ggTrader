"""Compare exact common-stock labels from a frozen MIAX filing, not returns."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from ggTrader.utils.manifest import sha256_file

HEADER = "MIAX Pearl Equities as of 6/24/21"
ROW = re.compile(r"(\S+)\s+.+\s+CommonStock\s+Equity\s*")


def extract(text: str) -> set[str]:
    """Extract recognizable rows; absent labels do not establish noncoverage."""
    labels: set[str] = set()
    pages = [page for page in text.split("\f") if page.strip()]
    if not pages or any(HEADER not in page for page in pages):
        raise ValueError("Missing MIAX page header")
    for page in pages:
        for line in page.splitlines():
            match = ROW.fullmatch(line)
            if match:
                symbol = match.group(1)
                if symbol in labels:
                    raise ValueError(f"duplicate label: {symbol}")
                labels.add(symbol)
    return labels


def main() -> None:
    """Read hash-pinned metadata and write a new diagnostic record."""
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        "pdf",
        "pdf-sha256",
        "screener",
        "screener-sha256",
        "ncen",
        "ncen-sha256",
        "out",
    ):
        parser.add_argument(f"--{argument}", required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.pdf, args.pdf_sha256),
        (args.screener, args.screener_sha256),
        (args.ncen, args.ncen_sha256),
    ):
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
    text = subprocess.check_output(
        ["pdftotext", "-f", "160", "-l", "341", "-layout", args.pdf, "-"], text=True
    )
    labels = extract(text)
    with open(args.screener, encoding="utf-8") as handle:
        screener = json.load(handle)
    common = {row["symbol"] for row in screener if "common stock" in row["name"].lower()}
    with open(args.ncen, encoding="utf-8") as handle:
        ncen = json.load(handle)
    fund_leads = {row["TICKER_SYMBOL"] for row in ncen["matches"]}
    result = {
        "purpose": "Retrospective exact-label leads; not classification or exclusion.",
        "limitations": (
            "Recognized text rows only; no verified identity, completeness or "
            "point-in-time availability."
        ),
        "source_hashes": {
            "pdf": args.pdf_sha256,
            "screener": args.screener_sha256,
            "ncen": args.ncen_sha256,
        },
        "driver_sha256": sha256_file(Path(__file__)),
        "miax_recognized_labels": len(labels),
        "screener_common_name_labels": len(common),
        "screener_matching_labels": len(common & labels),
        "ncen_lead_labels": len(fund_leads),
        "ncen_matching_labels": sorted(fund_leads & labels),
    }
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(content)
    print(f"{len(fund_leads & labels)} of {len(fund_leads)} N-CEN leads match")


if __name__ == "__main__":
    main()
