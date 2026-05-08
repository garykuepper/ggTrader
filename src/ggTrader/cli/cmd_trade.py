"""CLI Command: Trade (Live Execution)."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional


def register_trade_parser(subparsers: argparse._SubParsersAction):
    """Register the trade subcommand."""
    parser = subparsers.add_parser("trade", help="Start the live execution engine")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to run_results.json (default: auto-detect latest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate signals and sizing without placing real orders",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use exchange paper trading mode (if supported)",
    )
    sizing_group = parser.add_mutually_exclusive_group()
    sizing_group.add_argument(
        "--weighted-sizing",
        action="store_true",
        help="Use research-derived portfolio weights (default for crypto). "
             "Each entry sizes to portfolio × allocation_weight[symbol]; coins "
             "with 0% research weight are skipped.",
    )
    sizing_group.add_argument(
        "--adaptive-sizing",
        action="store_true",
        help="Use volatility-normalized position sizing (1%% target risk per trade). "
             "Ignores research allocation weights.",
    )
    parser.add_argument(
        "--min-trailing-stop-pct",
        type=float,
        default=4.0,
        help="Floor for trailing stop percentage (default: 4.0)",
    )
    parser.add_argument(
        "--min-atr-trailing-pct",
        type=float,
        default=4.0,
        help="Floor for ATR-derived stop percentage (default: 4.0)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Fixed capital per trade in USD (overridden by adaptive sizing)",
    )
    parser.add_argument(
        "--target-risk-pct",
        type=float,
        default=0.01,
        help="Adaptive sizing: portfolio fraction risked at stop (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.15,
        help="Adaptive sizing: cap on position as fraction of portfolio (default: 0.15)",
    )
    parser.add_argument(
        "--min-position-usd",
        type=float,
        default=15.0,
        help="Adaptive sizing: skip entry if sized position falls below this USD value (default: 15.0)",
    )
    parser.add_argument(
        "--dry-run-sizing",
        action="store_true",
        help="Calculate and print sizes for current signals, then exit",
    )
    parser.add_argument(
        "--portfolio-usd",
        type=float,
        default=None,
        help=(
            "Override portfolio USD value used by --dry-run-sizing "
            "(default: query exchange, fall back to START_CASH)"
        ),
    )
def run_trade(args: argparse.Namespace):
    """Executes the live trading engine."""
    from dotenv import load_dotenv
    load_dotenv()

    from ggTrader.core.crypto_execution_engine import CryptoExecutionEngine
    from ggTrader.utils.state_manager import get_latest_research_run

    if args.results:
        results_source: object = args.results
    else:
        latest = get_latest_research_run()
        if latest:
            results_source = latest
            print(
                f"Auto-detected latest research run: "
                f"{latest.run_id} (run_dir={latest.run_dir})"
            )
        else:
            print("Error: No research run found. Run `ggt research` first, or pass --results PATH.")
            sys.exit(1)

    from ggTrader.utils.run_config import full_pipeline_config, merge_run_config
    base_config = full_pipeline_config()

    # Sizing mode: weighted (default) > adaptive > fixed.
    weighted_flag = bool(getattr(args, "weighted_sizing", False))
    adaptive_flag = bool(args.adaptive_sizing)
    if not weighted_flag and not adaptive_flag and args.capital is None:
        weighted_flag = True

    config = merge_run_config(
        base_config,
        DRY_RUN=args.dry_run,
        PAPER=args.paper,
        WEIGHTED_SIZING=weighted_flag,
        ADAPTIVE_SIZING=adaptive_flag,
        MIN_TRAILING_STOP_PCT=args.min_trailing_stop_pct,
        MIN_ATR_TRAILING_PCT=args.min_atr_trailing_pct,
        CAPITAL_PER_TRADE=args.capital,
        TARGET_RISK_PCT=args.target_risk_pct,
        MAX_POSITION_PCT=args.max_position_pct,
        MIN_POSITION_USD=args.min_position_usd,
    )

    from ggTrader.utils.result_db_manager import ResultDBManager
    rm = ResultDBManager()

    engine = CryptoExecutionEngine(
        config,
        results_path=results_source,
        db_manager=rm,
        run_id="LIVE",
    )

    if args.dry_run_sizing:
        _dry_run_sizing(engine, override_portfolio_usd=args.portfolio_usd)
    else:
        engine.run_event_loop()


def _dry_run_sizing(engine: Any, override_portfolio_usd: Optional[float] = None) -> None:
    """Calculates and prints position sizes for current signals, then exits."""
    engine.logger.info("--- Sizing Dry Run ---")

    # Fetch latest data
    df = engine._fetch_latest_data()
    if df.empty:
        engine.logger.error("No data available for sizing dry run.")
        return

    # Compute signals
    signals = engine._compute_latest_signals(df)
    regime = engine._compute_live_regime_allowance(df)

    # Determine portfolio value
    portfolio_usd = override_portfolio_usd
    if portfolio_usd is None:
        portfolio_usd = engine._get_total_portfolio_usd()
    if portfolio_usd is None:
        portfolio_usd = float(engine.config.get("START_CASH", 1000.0))

    engine.logger.info(f"Using portfolio valuation: ${portfolio_usd:,.2f}")

    # Process each symbol
    entries = []
    for symbol, sig in signals.items():
        if sig["entry"]:
            allowed = regime.get(symbol, True)
            if not allowed:
                engine.logger.info(f"  [Regime] {symbol}: Blocked entry (bear regime)")
                continue

            if engine.config.get("ADAPTIVE_SIZING"):
                size_usd = engine._compute_adaptive_position_usd(symbol, sig, portfolio_usd)
            else:
                size_usd = engine.config.get("CAPITAL_PER_TRADE", 100.0)

            if size_usd:
                entries.append((symbol, size_usd, sig["exit_name"]))

    if not entries:
        engine.logger.info("No entry signals found for the current bar.")
    else:
        engine.logger.info(f"Found {len(entries)} entry signals:")
        for sym, size, exit_n in entries:
            engine.logger.info(f"  - {sym:10s} | Size: ${size:7.2f} | Exit: {exit_n}")

    engine.logger.info("Sizing dry run complete.")
