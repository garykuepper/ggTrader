"""Full pipeline script wrapper. Real logic delegates to src/."""

from __future__ import annotations

import argparse
from ggTrader.utils.run_config import full_pipeline_config
from ggTrader.pipeline.pipeline_runner import execute_full_pipeline

def main() -> None:
    """Orchestrate the full 4-phase pipeline."""
    parser = argparse.ArgumentParser(description="Run Full Trading Strategy Pipeline")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (recommended for background runs)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save results")
    parser.add_argument("--dry-run", action="store_true", help="Quick test: 3 coins, 1 chunk, 2 folds")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        metavar="N",
        help="Override MAX_SYMBOLS (default: use CONSTANTS, currently 5 for debug)",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override SYMBOLS_FILE JSON (rank list). Use e.g. data/top_50_*.json when "
            "--max-symbols exceeds rows in the default top_25 file."
        ),
    )
    parser.add_argument(
        "--detailed-sensitivity",
        action="store_true",
        help=(
            "Use DETAILED_SENSITIVITY_PARAM_GRIDS for WFO (and for Phase 1 if --sensitivity); "
            "default is coarse grids"
        ),
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help=(
            "Run Phase 1 sensitivity screen + narrow grids for WFO; default skips Phase 1 "
            "and WFO uses the active book unchanged"
        ),
    )
    parser.add_argument(
        "--wfo-debug-metrics",
        action="store_true",
        help=(
            "Set WFO_DEBUG_METRICS in pipeline config: log per-fold train-metric len/finite "
            "counts and combined robustness stats during multi-strategy WFO (verbose)"
        ),
    )
    parser.add_argument(
        "--train-metric",
        type=str,
        default=None,
        choices=("sharpe", "sortino", "calmar", "composite"),
        metavar="NAME",
        help=(
            "Override TRAIN_METRIC for WFO (and Phase 1 if --sensitivity): "
            "sharpe | sortino | calmar | composite (default: from full_pipeline_config)"
        ),
    )
    parser.add_argument(
        "--exits",
        type=str,
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated EXIT_TOURNAMENT override, e.g. atr_trailing or fixed_sl_tp or "
            "atr_trailing,fixed_sl_tp (default: from full_pipeline_config)"
        ),
    )
    parser.add_argument(
        "--dual-exits",
        action="store_true",
        help="Use both atr_trailing and fixed_sl_tp in EXIT_TOURNAMENT (ignored if --exits is set)",
    )
    parser.add_argument(
        "--recent-validation-start",
        type=str,
        default=None,
        metavar="DATE",
        help="YYYY-MM-DD: after WFO, run frozen-params combined backtest on this window (Phase 3B)",
    )
    parser.add_argument(
        "--recent-validation-end",
        type=str,
        default=None,
        metavar="DATE",
        help="YYYY-MM-DD end for recent validation (default: now UTC)",
    )
    parser.add_argument(
        "--recent-validation-ccxt-tail",
        action="store_true",
        help="After last TimescaleDB bar, append Kraken OHLCV via CCXT through validation end",
    )
    args = parser.parse_args()

    # Pass the arguments directly to the unified SRC execution engine
    execute_full_pipeline(args, full_pipeline_config())


if __name__ == "__main__":
    main()
