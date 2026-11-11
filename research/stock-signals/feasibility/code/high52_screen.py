"""Run the fixed long-only High52 historical portfolio screen."""

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

PORTFOLIO_SHA256 = "76599d94beab94cc8973d8a126e8efbed98ee79f677738c55b2389f17387ba96"
MARKET_FILE = "F-F_Research_Data_Factors_CSV.zip"
MARKET_SHA256 = "b03d39e4bb253dda0f14893db3b78e64b40b8d6549a0fc0fd4111dae84c98618"


def select_long(frame: pd.DataFrame) -> pd.Series:
    """Extract the provider's signed high-rank long-only portfolio in decimal units."""
    selected = frame[(frame["signalname"] == "High52") & (frame["port"] == "03")].copy()
    if selected.empty:
        raise ValueError("Missing required High52 portfolio 03")
    if selected[["date", "ret", "Nlong", "Nshort"]].isna().any().any():
        raise ValueError("Missing required High52 observations")
    if (selected["Nlong"] <= 0).any() or (selected["Nshort"] != 0).any():
        raise ValueError("Portfolio is not long-only")
    dates = pd.to_datetime(selected["date"]).dt.to_period("M").dt.to_timestamp("M")
    if dates.duplicated().any():
        raise ValueError("Found duplicate High52 month")
    return pd.Series(selected["ret"].to_numpy(dtype=float) / 100, index=dates).sort_index()


def evaluate(candidate: pd.Series, market: pd.Series) -> dict:
    """Evaluate the fixed windows and costs without changing their gates."""
    result = {}
    for name, start in (("full", "1963-07-31"), ("since_1994", "1994-01-31")):
        index = pd.date_range(start, "2025-12-31", freq="ME")
        result[name] = screen(candidate.reindex(index), market.reindex(index), annual_cost=0.01)
        result[f"{name}_stress"] = screen(
            candidate.reindex(index), market.reindex(index), annual_cost=0.02
        )
    return result


def main() -> None:
    """Read the pinned archives and write a new research result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--french", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if hashlib.sha256(args.portfolio.read_bytes()).hexdigest() != PORTFOLIO_SHA256:
        raise ValueError("Unexpected High52 portfolio hash")
    manifest = verify_manifest(args.french)
    if MARKET_FILE not in manifest or manifest[MARKET_FILE]["sha256"] != MARKET_SHA256:
        raise ValueError("Unexpected French market archive")
    frame = pd.read_parquet(
        args.portfolio, columns=["signalname", "port", "date", "ret", "Nlong", "Nshort"]
    )
    candidate = select_long(frame)
    market = load_market_total_return(args.french)["mkt"]
    result = {
        "status": "historical portfolio research; no trading authorization",
        "source": "Open Source Asset Pricing 2026.10: High52 portfolio 03, long-only",
        "benchmark": "French market total return (Mkt-RF + RF), not SPY",
        "limitations": (
            "Different publication vintages; assumption-based costs; no taxes, "
            "stock-level execution or future evidence; existing momentum NO-GO preserved"
        ),
        "input_sha256": {"portfolio": PORTFOLIO_SHA256, "french_market": MARKET_SHA256},
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calculator_sha256": hashlib.sha256(
            Path(__file__).with_name("accruals_comparison.py").read_bytes()
        ).hexdigest(),
        "results": evaluate(candidate, market),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
