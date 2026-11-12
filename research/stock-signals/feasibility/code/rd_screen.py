"""Calculate the fixed, exploratory R&D portfolio comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from accruals_comparison import screen
from french_data import load_market_total_return, verify_manifest
from high52_screen import MARKET_FILE, MARKET_SHA256, PORTFOLIO_SHA256


def select_rd(frame: pd.DataFrame) -> pd.Series:
    """Select the provider's highest R&D-to-market long-only portfolio."""
    rows = frame[(frame.signalname == "RD") & (frame.port == "05")].copy()
    if rows.empty or rows[["date", "ret", "Nlong", "Nshort"]].isna().any().any():
        raise ValueError("Missing R&D portfolio observations")
    if (rows.Nlong <= 0).any() or (rows.Nshort != 0).any():
        raise ValueError("R&D portfolio is not long-only")
    dates = pd.to_datetime(rows.date).dt.to_period("M").dt.to_timestamp("M")
    if dates.duplicated().any():
        raise ValueError("Found duplicate R&D month")
    return pd.Series(rows.ret.to_numpy(dtype=float) / 100, index=dates).sort_index()


def evaluate(candidate: pd.Series, market: pd.Series) -> dict:
    """Apply the two fixed windows and cost assumptions."""
    result = {}
    for name, start in (("full", "1975-07-31"), ("since_1994", "1994-01-31")):
        idx = pd.date_range(start, "2025-12-31", freq="ME")
        for suffix, cost in (("", 0.01), ("_stress", 0.02)):
            result[name + suffix] = screen(candidate.reindex(idx), market.reindex(idx), cost)
    return result


def main() -> None:
    """Read hash-pinned files and write an exploratory result once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--french", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if hashlib.sha256(args.portfolio.read_bytes()).hexdigest() != PORTFOLIO_SHA256:
        raise ValueError("Portfolio hash mismatch")
    manifest = verify_manifest(args.french)
    if MARKET_FILE not in manifest or manifest[MARKET_FILE]["sha256"] != MARKET_SHA256:
        raise ValueError("Market archive hash mismatch")
    frame = pd.read_parquet(
        args.portfolio, columns=["signalname", "port", "date", "ret", "Nlong", "Nshort"]
    )
    result = {
        "status": "exploratory; source file and prior research already seen",
        "candidate": "RD portfolio 05, equal weighted",
        "benchmark": "French market, not SPY",
        "input_sha256": {"portfolio": PORTFOLIO_SHA256, "market": MARKET_SHA256},
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calculator_sha256": hashlib.sha256(
            Path(__file__).with_name("accruals_comparison.py").read_bytes()
        ).hexdigest(),
        "results": evaluate(select_rd(frame), load_market_total_return(args.french)["mkt"]),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
