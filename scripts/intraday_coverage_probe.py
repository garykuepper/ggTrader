"""Run the fixed research-only intraday Stage 0A metadata coverage probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    """Parse before loading the probe, so help needs no credentials or side effects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs/research/_intraday_probe.json",
        help="New JSON artifact path; existing results are never overwritten",
    )
    args = parser.parse_args()
    from ggTrader.lab.intraday_probe import run_probe

    try:
        result = run_probe(args.output)
    except FileExistsError:
        print("result_already_exists", file=sys.stderr)
        return 2
    except Exception:
        print("probe_start_failed", file=sys.stderr)
        return 2
    print(result["conclusion"])
    print(result["auction_conclusion"])
    if result.get("local_failure"):
        print("unexpected_local_failure; last valid checkpoint preserved", file=sys.stderr)
        return 2
    return 0 if result["conclusion"] == "sample_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
