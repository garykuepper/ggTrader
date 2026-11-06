"""Run the synthetic WFO grid; under five minutes, so no checkpoint is needed."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ggTrader.lab.wfo import WfoPolicy
from ggTrader.lab.wfo_policy_synthetic import run_grid

ROOT = Path(__file__).resolve().parents[1]
MUS = (0.0, 0.5, 1.0, 1.5)
P_GATES = (0.3, 0.7, 1.0)
POLICIES = {
    "legacy": WfoPolicy(),
    "nan_exclude": WfoPolicy(nan_wfe="exclude"),
    "anchor_exclude_from_breaker": WfoPolicy(anchor_fold_wfe="exclude_from_breaker"),
}


def main() -> None:
    """Run and save the preregistered grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research/programme/wfo-methods/_wfo_policy_synthetic_results.json",
    )
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    cells = run_grid(args.seed, args.reps, MUS, P_GATES, POLICIES)
    comparisons = {}
    for mu in MUS:
        for gate in P_GATES:
            base = cells[f"{mu}|{gate}|legacy"]
            comparisons[f"{mu}|{gate}"] = {
                name: {
                    "p_halt": cells[f"{mu}|{gate}|{name}"]["ever_halted"],
                    "absolute_p_halt_difference": abs(
                        cells[f"{mu}|{gate}|{name}"]["ever_halted"] - base["ever_halted"]
                    ),
                    "absolute_anchor_share_difference": abs(
                        cells[f"{mu}|{gate}|{name}"]["anchor_share"] - base["anchor_share"]
                    ),
                    "fraction_halts_active_at_end": cells[f"{mu}|{gate}|{name}"][
                        "fraction_halts_active_at_end"
                    ],
                }
                for name in POLICIES
            }
    output = {
        "seed": args.seed,
        "n_reps": args.reps,
        "model": {
            "n_folds": 17,
            "p_nan": 0.05,
            "is_bias": 0.5,
            "is_noise_sd": 1.0,
            "oos_noise_sd": 2.0,
            "anchor_mean_multiplier": 0.5,
            "mus": MUS,
            "p_gates": P_GATES,
            "policies": {k: asdict(v) for k, v in POLICIES.items()},
            "common_random_numbers": (
                "Identical per-fold gate, IS noise, OOS noise and NaN draws "
                "across policies within each replication."
            ),
        },
        "cells": cells,
        "comparisons": comparisons,
    }
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
