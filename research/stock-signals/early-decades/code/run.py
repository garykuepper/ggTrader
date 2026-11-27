"""Compatibility entry point for the early-decades driver."""

import argparse

from ggTrader.utils.study import bootstrap

bootstrap(__file__)

import ed_run  # noqa: E402


def main() -> None:
    """Validate arguments and invoke the registered driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["holdout", "early"], required=True)
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument("--eval-start", required=True)
    parser.add_argument("--eval-end", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--regression-proof")
    parser.parse_args()
    ed_run.main()


if __name__ == "__main__":
    main()
