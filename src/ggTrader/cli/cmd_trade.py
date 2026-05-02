"""CLI Command: Trade (Live Execution)."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional


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
    parser.add_argument(
        "--adaptive-sizing",
        action="store_true",
        help="Use volatility-normalized position sizing",
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
    parser.add_argument(
        "--asset-class",
        type=str,
        default="crypto",
        choices=["crypto", "stocks"],
        help="Asset class to trade (default: crypto)",
    )


def run_trade(args: argparse.Namespace):
    """Executes the live trading engine."""
    from dotenv import load_dotenv
    load_dotenv()

    asset_class = getattr(args, "asset_class", "crypto")

    if asset_class == "stocks":
        from ggTrader.core.stock_execution_engine import StockExecutionEngine
        EngineClass = StockExecutionEngine
    else:
        from ggTrader.core.crypto_execution_engine import CryptoExecutionEngine
        EngineClass = CryptoExecutionEngine

    # Priority: Command line argument -> Auto-detect latest research run
    results_path = args.results
    if not results_path:
        from ggTrader.utils.state_manager import get_latest_research_run
        latest = get_latest_research_run()
        if latest:
            results_path = str(latest)
            print(f"Auto-detected latest research run: {results_path}")

    from ggTrader.utils.run_config import full_pipeline_config, stock_pipeline_config, merge_run_config
    base_config = stock_pipeline_config() if asset_class == "stocks" else full_pipeline_config()
    
    config = merge_run_config(
        base_config,
        DRY_RUN=args.dry_run,
        PAPER=args.paper,
        ADAPTIVE_SIZING=args.adaptive_sizing,
        MIN_TRAILING_STOP_PCT=args.min_trailing_stop_pct,
        MIN_ATR_TRAILING_PCT=args.min_atr_trailing_pct,
        CAPITAL_PER_TRADE=args.capital,
    )

    from ggTrader.utils.result_db_manager import ResultDBManager
    rm = ResultDBManager()

    engine = EngineClass(
        config,
        results_path=results_path,
        db_manager=rm,
        run_id="LIVE" if asset_class == "crypto" else "LIVE_STOCKS",
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
