"""Run the pre-registered net-issuance screen on verified frozen French archives."""

import argparse
import json
from pathlib import Path

from ggTrader.lab.french_data import FACTORS_FILE, load_market_total_return, verify_manifest
from ggTrader.lab.net_issuance_screen import (
    SIZE_NI_FILE,
    UNI_NI_FILE,
    load_net_issuance_vw,
    screen,
)
from ggTrader.utils.paths import find_project_root

PROJECT_ROOT = find_project_root()
BRIEF = "docs/research/briefs/2026-10-23-net-issuance-screen.md"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "research" / "_net_issuance_screen_results.json"


def main() -> None:
    """Write the fixed screen after verifying all three frozen sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Screen results JSON path")
    args = parser.parse_args()
    manifest = verify_manifest()
    required = (SIZE_NI_FILE, UNI_NI_FILE, FACTORS_FILE)
    missing = [filename for filename in required if filename not in manifest]
    if missing:
        raise ValueError(f"French archives absent from MANIFEST.json: {missing}")
    size, univariate = load_net_issuance_vw()
    result = screen(size, univariate, load_market_total_return())
    result["inputs"]["sha256"] = {filename: manifest[filename]["sha256"] for filename in required}
    result["brief"] = BRIEF
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
