"""Screen the provider's high earnings-consistency portfolio."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ggTrader.utils.study import bootstrap

bootstrap(__file__)

from french_data import load_market_total_return, verify_manifest
from high52_screen import MARKET_FILE, MARKET_SHA256, PORTFOLIO_SHA256, evaluate


def select_portfolio(frame: pd.DataFrame) -> pd.Series:
    """Return portfolio 05's long-only monthly returns in decimal form."""
    rows = frame[(frame.signalname == "EarningsConsistency") & (frame.port == "05")]
    if rows.empty or rows[["date", "ret", "Nlong", "Nshort"]].isna().any().any():
        raise ValueError("Missing earnings-consistency observations")
    if (rows.Nlong <= 0).any() or (rows.Nshort != 0).any():
        raise ValueError("Portfolio is not long-only")
    dates = pd.to_datetime(rows.date).dt.to_period("M").dt.to_timestamp("M")
    if dates.duplicated().any():
        raise ValueError("Found duplicate month")
    return pd.Series(rows.ret.to_numpy(dtype=float) / 100, index=dates).sort_index()


def main() -> None:
    """Evaluate fixed historical windows with hash-checked sources."""
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
        "status": "exploratory; historical data and related studies already seen",
        "candidate": "EarningsConsistency portfolio 05, equal weighted",
        "benchmark": "French market total return, not SPY",
        "input_sha256": {"portfolio": PORTFOLIO_SHA256, "market": MARKET_SHA256},
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": evaluate(select_portfolio(frame), load_market_total_return(args.french)["mkt"]),
    }
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
