"""Reproduce the first test's already-used 2020–2024 arm A cells."""

import argparse
import json
from pathlib import Path

from ggTrader.utils.study import bootstrap, study_dir

bootstrap(__file__)

import ed_dead_stock_first_test as study  # noqa: E402
import ed_inputs  # noqa: E402

START = "2020-01-02"
END = "2024-12-31"


def main() -> None:
    """Run the authorized regression and compare its saved cells exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["holdout"], required=True)
    parser.add_argument("--arm", choices=["A"], required=True)
    parser.add_argument("--eval-start", choices=[START], required=True)
    parser.add_argument("--eval-end", choices=[END], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.universe = "A"
    args.release_id = 1
    args.crsp_load = 1
    inputs = ed_inputs.read_inputs(args)
    result = study.run_study(
        inputs["close"],
        inputs["spy_close"],
        inputs["eligible"],
        inputs["span_rows"],
        start=START,
        end=END,
        release_spans=inputs["release_spans"],
    )
    saved = json.loads(
        (study_dir(__file__).parent / "dead-stock-first-test" / "_holdout-A.json").read_text()
    )
    keys = [study.cell_key(0.0, 0.00183, tax) for tax in ("low", "high")]
    actual = {key: result["strategy"][key] for key in keys}
    expected = {key: saved["result"]["strategy"][key] for key in keys}
    args.out.write_text(
        json.dumps(
            {"start": START, "end": END, "cell": actual, "matches": actual == expected}, indent=2
        )
        + "\n"
    )
    if actual != expected:
        raise SystemExit("Regression did not match the saved cell exactly")
    print("Regression matched exactly")


if __name__ == "__main__":
    main()
