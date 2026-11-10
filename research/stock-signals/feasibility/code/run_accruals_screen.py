"""Prepare the proposed low-accruals comparison using pinned local archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ggTrader.utils.study import bootstrap

ROOT = bootstrap(__file__)

from accruals_comparison import evaluate, monthly_returns, screen
from french_data import (
    _read_frozen_csv,
    load_market_total_return,
    parse_french_monthly,
    verify_manifest,
)

SIZE_FILE = "25_Portfolios_ME_AC_5x5_CSV.zip"
MARKET_FILE = "F-F_Research_Data_Factors_CSV.zip"
SPY_HASH = "3575a86f0d760a3ab46164ce43aefe792068caec5a265521160d7de998c24a44"
EXPECTED = {
    SIZE_FILE: "af9ea3f17ce817064775246393f0aa1c93c91d3c5359a9be004c6fba0b0aff8b",
    MARKET_FILE: "b03d39e4bb253dda0f14893db3b78e64b40b8d6549a0fc0fd4111dae84c98618",
}


def main() -> None:
    """Write a new result only for the exact locally frozen input pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Frozen French archive directory")
    parser.add_argument("--spy", required=True, type=Path, help="Pinned SPY adjusted-close CSV")
    parser.add_argument(
        "--out", required=True, type=Path, help="New JSON output; never overwritten"
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    manifest = verify_manifest(args.data)
    for name, digest in EXPECTED.items():
        if name not in manifest or manifest[name]["sha256"] != digest:
            raise ValueError(f"Unexpected frozen input: {name}")
    if hashlib.sha256(args.spy.read_bytes()).hexdigest() != SPY_HASH:
        raise ValueError("Unexpected SPY input hash")
    table = parse_french_monthly(
        _read_frozen_csv(SIZE_FILE, args.data), "Average Value Weighted Returns -- Monthly"
    )
    if "BIG LoAC" not in table:
        raise ValueError("Missing required BIG LoAC portfolio")
    market = load_market_total_return(args.data)
    spy_frame = pd.read_csv(args.spy)
    dates = (
        pd.to_datetime(spy_frame["date"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    spy_prices = pd.Series(spy_frame["Adj Close"].to_numpy(), index=dates).loc[
        "2021-04-30":"2026-04-30"
    ]
    if (
        spy_prices.empty
        or str(spy_prices.index[0].date()) != "2021-04-30"
        or str(spy_prices.index[-1].date()) != "2026-04-30"
    ):
        raise ValueError("SPY prices must cover both fixed endpoints")
    spy_returns = monthly_returns(spy_prices)
    candidate_overlap = table["BIG LoAC"].reindex(spy_returns.index)
    result = {
        "status": "research calculation; no verdict or authorization implied",
        "portfolio": "BIG LoAC value-weighted",
        "comparator": "French broad market (Mkt-RF + RF), not SPY",
        "limitations": (
            "Revised portfolio history; point-in-time holdings and taxes unverified; "
            "SPY daily exchange-calendar completeness and adjusted-close accuracy unverified; "
            "1% annual candidate cost is an assumption, not measured trading cost"
        ),
        "inputs_sha256": {**EXPECTED, "SPY.csv": SPY_HASH},
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calculation_sha256": hashlib.sha256(
            Path(__file__).with_name("accruals_comparison.py").read_bytes()
        ).hexdigest(),
        "results": evaluate(table["BIG LoAC"], market["mkt"]),
        "spy_overlap": {
            "window": "2021-05 through 2026-04; previously inspected",
            "comparison": screen(candidate_overlap, spy_returns, annual_cost=0.01),
        },
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
