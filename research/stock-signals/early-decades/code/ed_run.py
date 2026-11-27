"""Reproduce the first test's already-used 2020–2024 arm A cells."""

import argparse
import hashlib
import json
from pathlib import Path

from ggTrader.utils.binding import git_sha
from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import ed_dead_stock_first_test as study  # noqa: E402
import ed_inputs  # noqa: E402
from ed_comparison import after_tax_growth, comparison_returns  # noqa: E402
from ed_simulation import simulate_arm, validate_calendar  # noqa: E402

START = "2020-01-02"
END = "2024-12-31"
EARLY_START = "1969-02-03"
EARLY_END = "1998-07-31"


def rule_hash() -> str:
    """Hash the study implementation and frozen-copy record."""
    digest = hashlib.sha256()
    code = study_dir(__file__) / "code"
    for path in sorted([*code.glob("ed_*.py"), code / "frozen_copies.json"]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_early(args: argparse.Namespace) -> None:
    """Read the pinned early window and write its six cells with provenance."""
    if (args.eval_start, args.eval_end) != (EARLY_START, EARLY_END):
        raise SystemExit("early window dates must match the registration")
    if args.out.exists():
        raise SystemExit("refusing to overwrite an existing result")
    if args.regression_proof is None:
        raise SystemExit("the matching 2020–2024 regression output is required")
    proof = json.loads(args.regression_proof.read_text())
    saved = json.loads(
        (study_dir(__file__).parent / "dead-stock-first-test" / "_holdout-A.json").read_text()
    )
    keys = [study.cell_key(0.0, 0.00183, tax) for tax in ("low", "high")]
    expected = {key: saved["result"]["strategy"][key] for key in keys}
    if (proof.get("start"), proof.get("end"), proof.get("matches"), proof.get("cell")) != (
        START,
        END,
        True,
        expected,
    ):
        raise SystemExit("regression output does not exactly match the saved cells")
    claim = study_dir(__file__) / f"_early-{args.arm}.claim"
    try:
        with claim.open("x", encoding="utf-8") as handle:
            handle.write(f"output: {args.out}\n")
    except FileExistsError as exc:
        raise SystemExit(f"early arm {args.arm} already claimed: {claim}") from exc
    inputs = ed_inputs.read_early_inputs(EARLY_START, EARLY_END, 1, 1)
    comparison = comparison_returns(EARLY_START, EARLY_END)
    validate_calendar(inputs["close"].index, comparison, EARLY_START, EARLY_END)
    cells = simulate_arm(args.arm, inputs, EARLY_START, EARLY_END)
    first = comparison.loc[:"1993-01-29"]
    second = comparison.loc["1993-02-01":]
    result = {
        "eval_start": EARLY_START,
        "eval_end": EARLY_END,
        "release_id": 1,
        "crsp_load": 1,
        "rule_hash": rule_hash(),
        "code_commit": git_sha(ROOT),
        "arm": args.arm,
        "cells": cells,
        "data_provenance": inputs["provenance"],
        "comparison": {
            "early": {
                "start": EARLY_START,
                "end": "1993-01-29",
                "label": "CRSP market return (stand-in)",
                "growth_after_tax": after_tax_growth(first, 0.15),
            },
            "later": {
                "start": "1993-02-01",
                "end": EARLY_END,
                "label": "SPY",
                "growth_after_tax": after_tax_growth(second, 0.15),
            },
            "combined_growth_after_tax": after_tax_growth(comparison, 0.15),
        },
    }
    write_json(args.out, result)


def main() -> None:
    """Run the authorized regression and compare its saved cells exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["holdout", "early"], required=True)
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument("--eval-start", required=True)
    parser.add_argument("--eval-end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regression-proof", type=Path, default=None)
    args = parser.parse_args()
    if args.window == "early":
        run_early(args)
        return
    if (args.arm, args.eval_start, args.eval_end) != ("A", START, END):
        parser.error("holdout regression requires arm A and the registered 2020–2024 dates")
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
