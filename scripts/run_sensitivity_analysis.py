"""Run a vectorized sensitivity analysis (grid search) using FastBacktest."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ggTrader.core.orchestrator import run_sensitivity_orchestrator
from ggTrader.utils.run_config import sensitivity_script_config


def main() -> None:
    """Run vectorized sensitivity analysis using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    args = parser.parse_args()

    params = {
        "adx_threshold": list(range(15, 45, 5)),
        "adx_length": list(range(5, 40, 5)),
        "atr_length": list(range(5, 40, 5)),
        "atr_multiplier": list(np.arange(0.1, 1.1, 0.1)),
        "sar_acceleration": [0.02],
        "sar_maximum": [0.2],
        "use_dmp_cross": [True, False],
    }

    show_progress = not args.no_progress and sys.stdout.isatty()

    run_sensitivity_orchestrator(
        config=sensitivity_script_config(),
        param_grid=params,
        save_results=True,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    main()
