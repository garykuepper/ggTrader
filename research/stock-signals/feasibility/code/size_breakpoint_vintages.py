"""Compare two frozen NYSE size-breakpoint vintages without testing returns."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ggTrader.utils.manifest import sha256_file
from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from nyse_size_breakpoints import parse  # noqa: E402


def compare(older: pd.DataFrame, newer: pd.DataFrame) -> dict:
    """Summarize overlap, using the older median as the relative-change denominator."""
    paired = pd.concat(
        [older["median_usd_millions"].rename("old"), newer["median_usd_millions"].rename("new")],
        axis=1,
        join="inner",
    )
    if paired.empty:
        raise ValueError("No overlap between vintages")
    if (
        paired.index.has_duplicates
        or not np.isfinite(paired.to_numpy()).all()
        or (paired <= 0).any().any()
    ):
        raise ValueError("Invalid breakpoint values")
    diff = (paired["new"] / paired["old"] - 1).abs()
    return {
        "months": len(paired),
        "first_month": paired.index.min().date().isoformat(),
        "last_month": paired.index.max().date().isoformat(),
        "different_medians": int((diff > 1e-12).sum()),
        "max_absolute_relative_difference": float(diff.max()),
        "max_difference_month": diff.idxmax().date().isoformat(),
    }


def load(path: Path, expected_hash: str) -> pd.DataFrame:
    """Verify archive bytes and parse its single CSV member."""
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Archive hash mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected exactly one CSV member")
        return parse(archive.read(members[0]).decode("latin-1"))


def main() -> None:
    """Write a new artifact recording the input, parser and driver hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older", type=Path, required=True)
    parser.add_argument("--older-sha256", required=True)
    parser.add_argument("--newer", type=Path, required=True)
    parser.add_argument("--newer-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old = load(args.older, args.older_sha256)
    new = load(args.newer, args.newer_sha256)
    result = {
        "purpose": "Descriptive differences in reconstructed NYSE medians, not stock returns",
        "older_sha256": args.older_sha256,
        "newer_sha256": args.newer_sha256,
        "driver_sha256": sha256_file(Path(__file__)),
        "parser_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / "_lib/nyse_size_breakpoints.py"
        ),
        "full_overlap": compare(old, new),
        "2021_2023": compare(old.loc["2021":"2023"], new.loc["2021":"2023"]),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as output:
        output.write(serialized)


if __name__ == "__main__":
    main()
