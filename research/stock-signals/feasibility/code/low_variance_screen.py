"""Proposed low-variance screen; the owner has not approved a historical run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ggTrader.utils.manifest import sha256_file
from ggTrader.utils.study import bootstrap

ROOT = bootstrap(__file__)

import accruals_comparison  # noqa: E402
import french_data  # noqa: E402

from ggTrader.utils import significance  # noqa: E402

PORTFOLIO_FILE = "25_Portfolios_ME_VAR_5x5_CSV.zip"
MARKET_FILE = "F-F_Research_Data_Factors_CSV.zip"
EXPECTED = {
    PORTFOLIO_FILE: "5aa05cbb2e17da21b7770ea819ba38b79181eb11b4c735abfe23d64ee4d57ba0",
    MARKET_FILE: "b03d39e4bb253dda0f14893db3b78e64b40b8d6549a0fc0fd4111dae84c98618",
}


def main() -> None:
    """Read the pinned archives locally and write a new descriptive result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Frozen input archive directory")
    parser.add_argument("--out", type=Path, help="New output file, never overwritten")
    parser.add_argument("--check-only", action="store_true", help="Verify input hashes only")
    args = parser.parse_args()
    if (args.out is None) == (not args.check_only):
        parser.error("choose exactly one of --out and --check-only")
    if args.out is not None and args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    manifest = french_data.verify_manifest(args.data)
    for filename, digest in EXPECTED.items():
        if filename not in manifest or manifest[filename]["sha256"] != digest:
            raise ValueError(f"Unexpected frozen input: {filename}")
    if args.check_only:
        print("Inputs verified; no return values examined and no authorization implied.")
        return
    table = french_data.parse_french_monthly(
        french_data._read_frozen_csv(PORTFOLIO_FILE, args.data), french_data.VW_SECTION
    )
    if "BIG LoVAR" not in table:
        raise ValueError("Missing required BIG LoVAR portfolio")
    market = french_data.load_market_total_return(args.data)["mkt"]
    result = {
        "status": "proposed study; output alone grants no approval or trading authority",
        "candidate": "BIG LoVAR, value weighted",
        "benchmark": "French broad market (Mkt-RF + RF), not SPY",
        "limitations": "Previously published data; 1% and 2% costs assumed, not measured; pretax",
        "input_sha256": EXPECTED,
        "code_sha256": {
            "driver": sha256_file(Path(__file__)),
            "calculator": sha256_file(Path(accruals_comparison.__file__)),
            "parser": sha256_file(Path(french_data.__file__)),
            "significance": sha256_file(Path(significance.__file__)),
            "lockfile": sha256_file(ROOT / "uv.lock"),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": accruals_comparison.evaluate(table["BIG LoVAR"], market),
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
