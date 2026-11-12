"""Count possible ticker-linked coverage without changing or validating bars."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ggTrader.utils.manifest import sha256_file

# Issuer announcements and the reason these are candidates are in the inventory.
ALIASES = {
    "MMC": ("MRSH", "2026-01-14"),
    "ABC": ("COR", "2023-08-30"),
    "ANTM": ("ELV", "2022-06-28"),
    "BLL": ("BALL", "2022-05-10"),
    "RE": ("EG", "2023-07-10"),
    "PKI": ("RVTY", "2023-05-16"),
    "CDAY": ("DAY", "2024-02-01"),
    "WLTW": ("WTW", "2022-01-10"),
    "FB": ("META", "2022-06-09"),
}


def count_candidates(rows: list[dict], bars: pd.DataFrame, aliases: dict) -> dict:
    """Count missing old symbols where a successor row exists before the change."""
    if bars[["timestamp", "symbol"]].isna().any().any():
        raise ValueError("Null bar dates or symbols")
    if bars.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Found duplicate bar dates")
    available = set(zip(bars.symbol, pd.to_datetime(bars.timestamp).dt.strftime("%Y-%m-%d")))
    result = {}
    for old, (new, effective) in aliases.items():
        dates = [row["date"] for row in rows if old in row["missing"] and row["date"] < effective]
        result[old] = {
            "successor": new,
            "eligible": len(dates),
            "present": sum((new, date) in available for date in dates),
        }
    return result


def main() -> None:
    """Check source identity and write a new coverage-only result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    tape_hash = sha256_file(args.tape)
    if tape_hash != coverage["tape_sha256"]:
        raise ValueError("Tape hash does not match coverage record")
    bars = pd.read_parquet(args.tape, columns=["timestamp", "symbol"])
    result = {
        "purpose": "Date presence only; not a price, security identity or adjustment validation",
        "tape_sha256": tape_hash,
        "coverage_sha256": sha256_file(args.coverage),
        "code_sha256": sha256_file(Path(__file__)),
        "candidates": count_candidates(coverage["month_ends"], bars, ALIASES),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
