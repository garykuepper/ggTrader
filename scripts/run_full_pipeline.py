"""Full pipeline: Sensitivity Analysis → Per-Coin Multi-Strategy WFO → Final Validation → Report."""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


class StatusLogger:
    """Writes timestamped status lines to both stdout and a status.txt file."""

    def __init__(self, status_path: Path) -> None:
        self.path = status_path
        self.pipeline_start = time.time()
        self.phase_start = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Clear/create the file
        with open(self.path, "w") as f:
            f.write(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Monitor this file for live progress.\n\n")

    def _elapsed_str(self, seconds: float) -> str:
        return str(timedelta(seconds=int(seconds)))

    def update(self, message: str, start_phase: bool = False) -> None:
        if start_phase:
            self.phase_start = time.time()
        total_elapsed = time.time() - self.pipeline_start
        line = f"[{self._elapsed_str(total_elapsed):>10}] {message}"
        print(line)
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def phase_done(self, phase_name: str) -> None:
        phase_elapsed = time.time() - self.phase_start
        total_elapsed = time.time() - self.pipeline_start
        line = (
            f"[{self._elapsed_str(total_elapsed):>10}] "
            f"{phase_name} COMPLETE ({self._elapsed_str(phase_elapsed)})"
        )
        print(line)
        print()
        with open(self.path, "a") as f:
            f.write(line + "\n\n")

from ggTrader.core.orchestrator import (
    analyze_sensitivity_results,
    run_multi_strategy_per_coin_wfo,
    run_sensitivity_orchestrator,
)
from ggTrader.indicators.strategies import ENTRY_REGISTRY
from ggTrader.utils.config import load_symbols_from_json
from ggTrader.utils.report_generator import generate_pipeline_report

# =====================================================================
# CONFIGURATION
# =====================================================================

CONSTANTS = {
    "SYMBOLS_FILE": "data/top_25_USD_2023-01-01_2025-12-31.json",
    "MAX_SYMBOLS": 20,
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    "FREQ": "4h",
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.004,
    "SLIPPAGE": 0.003,
    "N_SPLITS": 4,
    "TEST_RATIO": 2,
    "MIN_TRADES": 2,
    "CHUNK_SIZE": 500,
    "USE_VECTORIZED": True,
    "USE_MOVERS": 0,
}

# Wide parameter grids for sensitivity analysis (expanded ranges)
SENSITIVITY_PARAM_GRIDS = {
    "psar_adx": {
        "sar_acceleration": [0.01, 0.02, 0.03],
        "sar_maximum": [0.1, 0.2, 0.3],
        "adx_length": [10, 14, 20],
        "adx_threshold": [15, 20, 25, 30],
        "use_dmp_cross": [True, False],
    },
    "ema_cross": {
        "ema_fast": [5, 9, 12],
        "ema_slow": [21, 26, 34],
    },
    "rsi_reversal": {
        "rsi_length": [7, 14, 21],
        "rsi_oversold": [20, 30, 40],
    },
}

# Exit strategy parameter grid (shared for all entry strategies)
EXIT_STRATEGY_PARAMS = {
    "atr_length": [10, 14, 20],
    "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
}


def _get_extended_param_grid(entry_strategy_name: str) -> dict:
    """Combine entry strategy params with exit strategy params."""
    entry_params = SENSITIVITY_PARAM_GRIDS.get(entry_strategy_name, {})
    
    # Only include exit strategy params for PSAR strategy
    # EMA and RSI use default ATR settings, not a grid
    if entry_strategy_name == "psar_adx":
        exit_params = EXIT_STRATEGY_PARAMS
        return {**entry_params, **exit_params}
    else:
        # For EMA and RSI, use single fixed ATR params
        return {**entry_params}


def _prepare_config_and_symbols(config: dict) -> dict:
    """Load symbols and truncate to MAX_SYMBOLS, prepare config dict."""
    # Load symbols from file
    symbols = load_symbols_from_json(config["SYMBOLS_FILE"])
    if symbols is None:
        raise ValueError(f"Failed to load symbols from {config['SYMBOLS_FILE']}")

    # Truncate to MAX_SYMBOLS
    max_symbols = config.get("MAX_SYMBOLS", len(symbols))
    symbols = symbols[:max_symbols]
    print(f"Using top {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")

    # Update config to use direct symbol list
    config = dict(config)
    config["SYMBOLS"] = symbols
    config["SYMBOLS_FILE"] = None

    return config


def phase_1_sensitivity_analysis(
    config: dict,
    narrowed_grids: dict,
    show_progress: bool = True,
    dry_run: bool = False,
    logger: "StatusLogger | None" = None,
) -> dict:
    """Phase 1: Run sensitivity analysis for each entry strategy."""
    strategies = list(ENTRY_REGISTRY.keys())
    n_strategies = len(strategies)
    print("\n" + "=" * 100)
    print("PHASE 1: SENSITIVITY ANALYSIS PER STRATEGY")
    print("=" * 100)
    if logger:
        logger.update(f"Phase 1 started: sensitivity analysis for {n_strategies} strategies", start_phase=True)

    sensitivity_results = {}

    for s_idx, strategy_name in enumerate(strategies, 1):
        print(f"\n--- Sensitivity Analysis: {strategy_name} ({s_idx}/{n_strategies}) ---")

        # Get parameter grid for this strategy
        param_grid = _get_extended_param_grid(strategy_name)
        
        # In dry-run, reduce the grid size dramatically
        if dry_run:
            # For dry-run, take only first 2 values per parameter
            param_grid = {k: v[:2] if isinstance(v, list) else [v] for k, v in param_grid.items()}

        # Run sensitivity analysis
        config_with_strategy = {**config, "ENTRY_STRATEGY": strategy_name}

        result = run_sensitivity_orchestrator(
            config=config_with_strategy,
            param_grid=param_grid,
            save_results=True,
            show_progress=show_progress,
            logger=logger,
        )

        results_df = result.get("results_df", pd.DataFrame())
        sensitivity_results[strategy_name] = results_df

        # Analyze and narrow parameter ranges
        print(f"Analyzing parameter importance for {strategy_name}...")
        if not results_df.empty:
            narrowed_grid = analyze_sensitivity_results(results_df, param_grid, top_percentile=20)
            narrowed_grids[strategy_name] = narrowed_grid
            print(f"  Narrowed grid: {narrowed_grid}")
        else:
            narrowed_grids[strategy_name] = param_grid
            print(f"  No results; using original grid")

        if logger:
            logger.update(f"  Phase 1: {strategy_name} sensitivity done ({s_idx}/{n_strategies})")

    if logger:
        logger.phase_done("Phase 1 (Sensitivity Analysis)")
    return sensitivity_results


def phase_2_per_coin_multi_strategy_wfo(
    config: dict,
    narrowed_grids: dict,
    show_progress: bool = True,
    logger: "StatusLogger | None" = None,
) -> dict:
    """Phase 2: Run per-coin WFO across all strategies with narrowed param ranges."""
    n_coins = len(config.get("SYMBOLS", []))
    n_strategies = len(narrowed_grids)
    print("\n" + "=" * 100)
    print("PHASE 2: PER-COIN MULTI-STRATEGY WFO")
    print("=" * 100)
    if logger:
        logger.update(
            f"Phase 2 started: WFO for {n_coins} coins x {n_strategies} strategies",
            start_phase=True,
        )

    # Run multi-strategy per-coin WFO
    wfo_result = run_multi_strategy_per_coin_wfo(
        config=config,
        strategy_param_grids=narrowed_grids,
        save_results=True,
        show_progress=show_progress,
        logger=logger,
    )

    if logger:
        logger.phase_done("Phase 2 (Per-Coin WFO) + Phase 3 (Final Validation)")
    return wfo_result


def main() -> None:
    """Orchestrate the full 4-phase pipeline."""
    parser = argparse.ArgumentParser(description="Run Full Trading Strategy Pipeline")
    parser.add_argument("--progress", action="store_true", default=True, help="Show progress bars")
    parser.add_argument("--no-save", action="store_true", help="Do not save results")
    parser.add_argument("--dry-run", action="store_true", help="Quick test: 3 coins, 1 chunk, 2 folds")
    args = parser.parse_args()

    # Prepare config and symbols
    config = _prepare_config_and_symbols(CONSTANTS)
    
    # Apply dry-run adjustments
    if args.dry_run:
        print("Running in DRY-RUN mode: 3 coins, 1 chunk, 2 folds")
        config["SYMBOLS"] = config["SYMBOLS"][:3]  # Only top 3 coins
        config["MAX_SYMBOLS"] = 3
        config["CHUNK_SIZE"] = 50  # Only first 50 combos per chunk
        config["N_SPLITS"] = 2  # Only 2 folds
    
    save_results = not args.no_save

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_results_dir = Path(f"results/pipeline_{timestamp}")
    pipeline_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline results will be saved to: {pipeline_results_dir}")

    # Create status logger - write to status.txt for live monitoring
    logger = StatusLogger(pipeline_results_dir / "status.txt")
    print(f"Monitor progress: Get-Content {pipeline_results_dir / 'status.txt'}")
    print()

    n_coins = len(config["SYMBOLS"])
    n_strategies = len(ENTRY_REGISTRY)
    n_folds = config.get("N_SPLITS", 4)
    logger.update(
        f"Starting pipeline: {n_coins} coins, {n_strategies} strategies, {n_folds} WFO folds"
    )

    # Track results from each phase
    narrowed_grids = {}
    sensitivity_results = {}
    wfo_results = {}
    final_backtest_results = {}

    try:
        # Phase 1: Sensitivity Analysis
        sensitivity_results = phase_1_sensitivity_analysis(
            config, narrowed_grids, args.progress, args.dry_run, logger
        )

        # Phase 2: Per-Coin Multi-Strategy WFO (includes Phase 3: Final Validation)
        wfo_results = phase_2_per_coin_multi_strategy_wfo(config, narrowed_grids, args.progress, logger)

        # Extract final backtest results
        final_backtest_results = {
            "final_portfolio": wfo_results.get("final_portfolio"),
            "per_coin_results": wfo_results.get("per_coin_results", {}),
            "final_stats": wfo_results.get("final_stats", {}),
        }

        # Phase 4: Report Generation
        print("\n" + "=" * 100)
        print("PHASE 4: REPORT GENERATION")
        print("=" * 100)

        generate_pipeline_report(
            sensitivity_results=sensitivity_results,
            wfo_results=wfo_results,
            final_backtest_results=final_backtest_results,
            output_dir=str(pipeline_results_dir),
        )

        total_elapsed = time.time() - logger.pipeline_start
        logger.update(
            f"Pipeline COMPLETE in {str(timedelta(seconds=int(total_elapsed)))} "
            f"| Results: {pipeline_results_dir}"
        )
        print(f"\n>>> Pipeline complete! Results saved to: {pipeline_results_dir}")
        print(f"  - pipeline_report.md: Comprehensive analysis report")
        print(f"  - status.txt: Full timestamped execution log")
        print(f"  - sensitivity_*.csv: Parameter sensitivity analysis")
        print(f"  - wfo_*.csv: Walk-forward optimization results")

    except Exception as e:
        logger.update(f"FAILED: {e}")
        print(f"\nX Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
