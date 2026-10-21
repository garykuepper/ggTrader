"""Freeze French source files or run the pre-registered Phase 1 screen offline."""

import argparse
import json
import time
from pathlib import Path

from ggTrader.lab.french_data import (
    FACTORS_FILE,
    SIZE_MOM_FILE,
    download_and_freeze,
    load_market_total_return,
    load_size_momentum_vw,
    verify_manifest,
)
from ggTrader.lab.momentum_screen import screen
from ggTrader.utils.paths import find_project_root

PROJECT_ROOT = find_project_root()
BRIEF = "docs/research/briefs/2026-10-20-momentum-retest.md"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "research" / "_momentum_french_screen_results.json"


def main() -> None:
    """Download without overwriting, or write verified offline screen results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="Freeze both sources only; commit before screening"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Screen results JSON path")
    args = parser.parse_args()
    started = time.monotonic()
    if args.download:
        download_and_freeze([SIZE_MOM_FILE, FACTORS_FILE])
        print("Frozen both French archives and MANIFEST.json. Commit them before the offline run.")
        return
    manifest = verify_manifest()
    result = screen(load_size_momentum_vw(), load_market_total_return())
    result.update(provenance=manifest, brief=BRIEF, runtime_sec=time.monotonic() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
