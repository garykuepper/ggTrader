"""Driver for the dead-stock-random-baseline study: run it with --help for the options."""

from __future__ import annotations

import argparse
from pathlib import Path

from ggTrader.utils.binding import binding
from ggTrader.utils.io import write_json
from ggTrader.utils.paths import EVAL_END_17, EVAL_START_17
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import dead_stock_random_baseline  # noqa: E402  (found on sys.path through bootstrap)


def build_parser() -> argparse.ArgumentParser:
    """The command line: the evaluation window is pinned, never "today"."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-start", default=EVAL_START_17, help="first evaluation date")
    parser.add_argument("--eval-end", default=EVAL_END_17, help="last evaluation date")
    parser.add_argument(
        "--out",
        type=Path,
        default=study_dir(__file__) / "_dead_stock_random_baseline_results.json",
        help="results JSON to write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the study and write its results JSON with a provenance block."""
    args = build_parser().parse_args(argv)
    result = dead_stock_random_baseline.run_study(args.eval_start, args.eval_end)
    payload = {
        "study": "dead-stock-random-baseline",
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "result": result,
        "provenance": binding(ROOT),
    }
    write_json(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
