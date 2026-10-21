"""Run the pre-registered Phase 1 factor screen on verified frozen French archives."""

import argparse
import json
from pathlib import Path

from ggTrader.lab.factor_screen import screen
from ggTrader.lab.french_data import (
    INVESTMENT_FILE,
    PROFITABILITY_FILE,
    VALUE_FILE,
    load_market_total_return,
    load_size_factor_vw,
    load_size_momentum_vw,
    verify_manifest,
)
from ggTrader.utils.paths import find_project_root

PROJECT_ROOT = find_project_root()
BRIEF = "docs/research/briefs/2026-10-20-factor-screen.md"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "research" / "_factor_screen_results.json"


def main() -> None:
    """Write the fixed screen and all diagnostics after validating every source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Screen results JSON path")
    args = parser.parse_args()
    manifest = verify_manifest()
    result = screen(
        load_size_factor_vw(VALUE_FILE),
        load_size_factor_vw(PROFITABILITY_FILE),
        load_size_factor_vw(INVESTMENT_FILE),
        load_size_momentum_vw(),
        load_market_total_return(),
    )
    result.update(provenance=manifest, brief=BRIEF)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
