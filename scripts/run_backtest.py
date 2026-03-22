"""Run a single ggTrader backtest using the vectorized FastBacktest engine."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy

from ggTrader.core.orchestrator import run_backtest_orchestrator
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.run_config import DEFAULT_PSAR_ADX_PARAMS, backtest_script_config, merge_run_config


def main() -> None:
    """Run a single backtest using the orchestrator."""
    parser = argparse.ArgumentParser(description="Run a single ggTrader backtest")
    parser.add_argument("--params", type=str, help="Path to params.json")
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        help="Override symbol list (disables SYMBOLS_FILE)",
    )
    parser.add_argument(
        "--movers",
        type=int,
        default=None,
        metavar="N",
        help="Daily top-N mover mask (0 disables); default from run config",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable VectorBT progress bar",
    )
    args = parser.parse_args()

    config = backtest_script_config()
    if args.symbols:
        config = merge_run_config(config, SYMBOLS=list(args.symbols), SYMBOLS_FILE=None)
    if args.movers is not None:
        config = merge_run_config(config, USE_MOVERS=args.movers)

    params = deepcopy(DEFAULT_PSAR_ADX_PARAMS)
    if args.params:
        rm_temp = ResultsManager("temp")
        params = rm_temp.load_params(args.params)

    show_progress = not args.no_progress and sys.stdout.isatty()

    results = run_backtest_orchestrator(
        config=config, params=params, save_results=True, show_progress=show_progress
    )
    pf = results["portfolio"]
    print("Global Portfolio Stats:")
    stats_df = pf.stats().to_frame(name="Portfolio Stats")
    print(stats_df)
    pf.plot(subplots=["drawdowns", "value", "cum_returns"]).show()


if __name__ == "__main__":
    main()
