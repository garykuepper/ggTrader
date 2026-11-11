"""Proposed expected-growth screen; historical calculation requires owner approval."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from ggTrader.utils.manifest import sha256_file
from ggTrader.utils.study import bootstrap

ROOT = bootstrap(__file__)

import accruals_comparison  # noqa: E402
import global_q  # noqa: E402

from ggTrader.utils import significance  # noqa: E402

PORTFOLIO_FILE = "benportf_me_eg_monthly_2025.csv"
MARKET_FILE = "q5_factors_monthly_2025.csv"
BRIEF = Path("research/stock-signals/feasibility/2026-11-11-expected-growth-proposal.md")
EXPECTED = {
    PORTFOLIO_FILE: "932ff7d3ec330fc8dcb22f81f40a9bc98ed02ade4a53adb3c08b2baa559386b4",
    MARKET_FILE: "8b2e2f1364890f8f5876da20e6bfad0acd8d272425de3b65c725f54b51128f6a",
}


def _read_csv(raw: bytes) -> pd.DataFrame:
    """Reject duplicate headers before pandas can silently rename them."""
    source = io.StringIO(raw.decode("utf-8-sig"))
    header = next(csv.reader(source), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("Missing or duplicate CSV header")
    source.seek(0)
    return pd.read_csv(source)


def _hurdles(t: float | None, p: float | None, recent_gap: float) -> dict:
    """Apply the proposed two-hurdle rule without rounding the statistics."""
    full_pass = None if t is None or p is None else t >= 3 and p <= 0.00135
    recent_pass = recent_gap >= 0.01
    return {
        "full_pass": full_pass,
        "since_1994_growth_gap": recent_gap,
        "since_1994_pass": recent_pass,
        "overall_pass": None if full_pass is None else full_pass and recent_pass,
    }


def main() -> None:
    """Check frozen inputs or calculate the proposed comparison once approved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Frozen input directory")
    parser.add_argument("--out", type=Path, help="New output file, never overwritten")
    parser.add_argument(
        "--check-only", action="store_true", help="Verify hashes without parsing returns"
    )
    args = parser.parse_args()
    if (args.out is None) == (not args.check_only):
        parser.error("choose exactly one of --out and --check-only")
    if args.out is not None and args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    data = {}
    for filename, expected in EXPECTED.items():
        raw = (args.data / filename).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Input hash mismatch: {filename}")
        data[filename] = raw
    if args.check_only:
        print("Input hashes verified; no return values examined and no approval implied.")
        return
    candidate, market = global_q.extract_expected_growth(
        _read_csv(data[PORTFOLIO_FILE]), _read_csv(data[MARKET_FILE]), "1967-01", "2025-12"
    )
    results = {}
    for name, start in (("full", "1967-01-31"), ("since_1994", "1994-01-31")):
        for suffix, cost in (("", 0.01), ("_stress", 0.02)):
            results[name + suffix] = accruals_comparison.screen(
                candidate.loc[start:], market.loc[start:], annual_cost=cost
            )
    full = results["full"]
    t, p = full["mean_excess_t"], full["one_sided_p"]
    recent = results["since_1994"]["comparison"]
    recent_gap = recent["candidate"]["annualized_return"] - recent["benchmark"]["annualized_return"]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=10)
    )
    result = {
        "status": "proposed historical screen; output does not establish approval",
        "candidate": "Global-q size 2 / expected-growth 3, value-weighted total return",
        "benchmark": "Global-q broad market (R_MKT + R_F), not SPY",
        "limitations": "Published and previously examined research; costs assumed; pretax",
        "input_sha256": EXPECTED,
        "code_sha256": {
            "driver": sha256_file(Path(__file__)),
            "selector": sha256_file(Path(global_q.__file__)),
            "calculator": sha256_file(Path(accruals_comparison.__file__)),
            "significance": sha256_file(Path(significance.__file__)),
            "lockfile": sha256_file(ROOT / "uv.lock"),
            "brief": sha256_file(ROOT / BRIEF),
        },
        "git": {"commit": commit, "dirty": dirty},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "proposed_hurdles": _hurdles(t, p, recent_gap),
        "results": results,
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as out:
        out.write(encoded)


if __name__ == "__main__":
    main()
