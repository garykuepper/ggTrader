"""Audit exact SEC income-period presence without calculating returns or signals."""

import argparse
import json
from datetime import date
from pathlib import Path

from ggTrader.utils.manifest import sha256_file


def observations(document: dict, cutoff: str, periods: dict[str, tuple[str, str]]) -> dict:
    """List matching filed observations; filing dates are not availability times."""
    limit = date.fromisoformat(cutoff)
    if limit.isoformat() != cutoff:
        raise ValueError("Cutoff must be an ISO date")
    entries = document.get("facts", {}).get("us-gaap", {}).get("NetIncomeLoss", {})
    entries = entries.get("units", {}).get("USD", [])
    result = {}
    for label, (start, end) in periods.items():
        matching = []
        for row in entries:
            if row.get("start") != start or row.get("end") != end:
                continue
            if row.get("form") != {"annual": "10-K", "nine_month": "10-Q"}[label]:
                continue
            filed = date.fromisoformat(row["filed"])
            if filed.isoformat() != row["filed"]:
                raise ValueError("Invalid filing date")
            if filed <= limit:
                matching.append({"accession": row["accn"], "filed": row["filed"]})
        result[label] = sorted(matching, key=lambda item: (item["filed"], item["accession"]))
    return result


def main() -> None:
    """Verify a frozen manifest and write a new exact-period coverage artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    periods = {"annual": ("2023-01-01", "2023-12-31"), "nine_month": ("2023-01-01", "2023-09-30")}
    cutoff = "2024-05-01"
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = []
    for item in manifest["files"]:
        path = args.manifest.parent / item["cache_file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        result.append(
            {
                "cik": document["cik"],
                "issuer": document["entityName"],
                "sha256": item["sha256"],
                "observations": observations(document, cutoff, periods),
            }
        )
    output = {
        "purpose": "Period coverage only; not availability or identity verification",
        "manifest_sha256": sha256_file(args.manifest),
        "code_sha256": sha256_file(Path(__file__)),
        "periods": periods,
        "filing_cutoff": cutoff,
        "issuers": result,
    }
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as file:
        file.write(serialized)


if __name__ == "__main__":
    main()
