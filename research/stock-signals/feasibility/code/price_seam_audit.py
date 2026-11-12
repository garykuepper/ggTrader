"""Compare overlapping adjusted-price vintages for potential seams."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from ggTrader.utils.manifest import sha256_file

THRESHOLD = 0.01


def assess(tape: pd.DataFrame, reference: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:
    """Flag ratio changes across consecutive sessions, without assuming cause."""
    if not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("invalid session calendar")
    for frame in (tape, reference):
        if frame[["symbol", "date"]].isna().any().any():
            raise ValueError("missing symbol or date")
        if frame.duplicated(["symbol", "date"]).any():
            raise ValueError("duplicate symbol/date")
    both = tape.merge(reference, on=["symbol", "date"], suffixes=("_tape", "_ref"))
    both = both.sort_values(["symbol", "date"]).copy()
    both["session"] = sessions.get_indexer(both.date)
    if (both.session < 0).any():
        raise ValueError("matched date outside session calendar")
    valid = np.isfinite(both.price_tape) & np.isfinite(both.price_ref)
    valid &= (both.price_tape > 0) & (both.price_ref > 0)
    both["valid"] = valid
    both["ratio"] = (both.price_tape / both.price_ref).where(valid)
    previous = both.groupby("symbol", sort=False).shift()
    consecutive = both.session.eq(previous.session + 1) & both.valid & previous.valid.fillna(False)
    change = both.ratio / previous.ratio - 1
    flagged = both.loc[consecutive & (change.abs() >= THRESHOLD), ["symbol", "date"]].copy()
    flagged["relative_change"] = change.loc[flagged.index]
    flagged["date"] = flagged.date.dt.strftime("%Y-%m-%d")
    return {
        "matched_rows": len(both),
        "invalid_matched_rows": int((~valid).sum()),
        "matched_labels": int(both.symbol.nunique()),
        "matched_pairs": int(consecutive.sum()),
        "flagged_labels": int(flagged.symbol.nunique()),
        "flags": flagged.to_dict("records"),
    }


def main() -> None:
    """Compare exact ticker labels in two hash-pinned local inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tape", "reference"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    for name in ("tape", "reference"):
        if sha256_file(getattr(args, name)) != getattr(args, f"{name}_sha256"):
            raise ValueError(f"Input hash mismatch: {name}")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise ValueError("end precedes start")
    tape = pd.read_parquet(args.tape, columns=["timestamp", "symbol", "close"])
    tape = tape.rename(columns={"timestamp": "date", "close": "price"})
    reference = pd.read_parquet(args.reference, columns=["date", "ticker", "adj_close"])
    reference = reference.rename(columns={"ticker": "symbol", "adj_close": "price"})
    for frame in (tape, reference):
        frame["date"] = (
            pd.to_datetime(frame.date, errors="raise").dt.tz_localize(None).dt.normalize()
        )
    tape = tape[tape.date.between(start, end)]
    reference = reference[reference.date.between(start, end) & reference.symbol.isin(tape.symbol)]
    sessions = mcal.get_calendar("NYSE").valid_days(start, end).tz_localize(None)
    result = {
        "purpose": "Compare exact-label adjusted-price vintages; flags are not proof of faults",
        "tape_sha256": args.tape_sha256,
        "reference_sha256": args.reference_sha256,
        "code_sha256": sha256_file(Path(__file__)),
        "calendar_version": version("pandas-market-calendars"),
        "start": args.start,
        "end": args.end,
        "relative_change_threshold": THRESHOLD,
        **assess(tape, reference, sessions),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
