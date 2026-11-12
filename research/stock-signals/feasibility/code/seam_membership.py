"""Join price-discrepancy flags to historical membership snapshots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ggTrader.data.core.index_constituents import load_sp500_history, normalize_yf_ticker
from ggTrader.utils.manifest import sha256_file


def assess(flags: list[dict], history: pd.DataFrame, start: str, end: str) -> list[dict]:
    """Classify flags without changing the replay universe."""
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("invalid membership snapshot order")
    first, last = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if last < first:
        raise ValueError("end precedes start")
    rows = []
    seen = set()
    for flag in flags:
        date = pd.Timestamp(flag["date"], tz="UTC")
        symbol = normalize_yf_ticker(flag["symbol"])
        if not symbol or (symbol, date) in seen:
            raise ValueError("missing symbol or duplicate flag")
        seen.add((symbol, date))
        if not first <= date <= last:
            continue
        index = history.index.searchsorted(date, side="right") - 1
        if index < 0:
            membership, snapshot, age_days = "unknown", None, None
        else:
            members = {normalize_yf_ticker(s) for s in history.tickers.iloc[index]}
            if not members:
                raise ValueError("empty membership snapshot")
            membership = "member" if symbol in members else "no_exact_match"
            snapshot = str(history.index[index].date())
            age_days = (date - history.index[index]).days
        rows.append(
            {
                **flag,
                "normalized_symbol": symbol,
                "membership": membership,
                "membership_snapshot": snapshot,
                "snapshot_age_days": age_days,
            }
        )
    return rows


def main() -> None:
    """Write a fresh inventory from explicitly hash-pinned inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("flags", "membership"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    for name in ("flags", "membership"):
        if sha256_file(getattr(args, name)) != getattr(args, f"{name}_sha256"):
            raise ValueError(f"Input hash mismatch: {name}")
    source = json.loads(args.flags.read_text(encoding="utf-8"))
    rows = assess(source["flags"], load_sp500_history(args.membership), args.start, args.end)
    counts = Counter(row["membership"] for row in rows)
    result = {
        "purpose": "Retrospective discrepancy triage; does not select or exclude stocks",
        "warning": "Do not use this retrospective list to construct a trading universe; "
        "no exact match is not proof of nonmembership or a bad price.",
        "matching": "Normalized ticker only; no historical alias or security identity resolution",
        "flags_path": str(args.flags),
        "membership_path": str(args.membership),
        "flags_sha256": args.flags_sha256,
        "membership_sha256": args.membership_sha256,
        "code_sha256": sha256_file(Path(__file__)),
        "start": args.start,
        "end": args.end,
        "source_flag_count": len(source["flags"]),
        "outside_window_count": len(source["flags"]) - len(rows),
        "counts": {key: counts.get(key, 0) for key in ("member", "no_exact_match", "unknown")},
        "flags": rows,
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
