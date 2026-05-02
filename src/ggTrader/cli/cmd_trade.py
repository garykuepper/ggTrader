"""CLI Command: Trade (Execution Engine)"""

import argparse
import sys

from ggTrader.utils.state_manager import get_latest_research_run


def register_trade_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'trade' subcommand."""
    parser = subparsers.add_parser("trade", help="Start the live execution engine")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Path to run_results.json for strategy params "
            "(default: auto-detect latest research run)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without placing real orders on the exchange"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=25.0,
        help="Fixed capital per trade when ADAPTIVE_SIZING is off (default: 25.0)",
    )
    parser.add_argument(
        "--interval", type=str, default=None, help="Override polling interval (e.g. 1h, 4h)"
    )
    parser.add_argument(
        "--adaptive-sizing",
        action="store_true",
        help=(
            "Size each entry to risk TARGET_RISK_PCT of portfolio at the stop "
            "(volatility-normalized). Overrides weight-based sizing when enabled."
        ),
    )
    parser.add_argument(
        "--target-risk-pct",
        type=float,
        default=0.01,
        help="Adaptive sizing: fraction of portfolio to risk per trade (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.15,
        help="Adaptive sizing: cap on single-position allocation (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--min-position-usd",
        type=float,
        default=15.0,
        help="Adaptive sizing: skip entry if sized below this USD floor (default: 15.0)",
    )
    parser.add_argument(
        "--min-trailing-stop-pct",
        type=float,
        default=4.0,
        help=(
            "Floor for trailing_stop exit's stop distance, in percent. "
            "WFO param is clamped upward to this. Default: 4.0"
        ),
    )
    parser.add_argument(
        "--min-atr-trailing-pct",
        type=float,
        default=4.0,
        help=(
            "Floor for atr_trailing exit's stop distance, in percent. "
            "WFO-derived ATR stop is clamped upward to this. Default: 4.0"
        ),
    )
    parser.add_argument(
        "--dry-run-sizing",
        action="store_true",
        help=(
            "Print what each symbol's adaptive position size would be at the "
            "current bar/ATR, then exit. Does not place orders or run the loop."
        ),
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

    # Import inside to prevent DB connections from initializing early
    from ggTrader.core.execution_engine import ExecutionEngine
    from ggTrader.utils.results_manager import ResultsManager
    from ggTrader.utils.run_config import full_pipeline_config, merge_run_config

    results_path = args.results
    if not results_path:
        print("Searching for latest Master Research results for parameters...")
        latest_res = get_latest_research_run()
        if not latest_res:
            print("Error: Could not automatically detect a valid run_results.json in results/")
            sys.exit(1)
        results_path = str(latest_res)
        print(f"Auto-detected latest research params: {results_path}")

    rm = ResultsManager("live_trader", pipeline_stage="trade")
    print(f"\n[{rm.run_id}] Initializing live trader...")

    config = full_pipeline_config()
    config = merge_run_config(
        config,
        DRY_RUN=args.dry_run,
        CAPITAL_PER_TRADE=args.capital,
        INTERVAL=args.interval,
        ADAPTIVE_SIZING=args.adaptive_sizing,
        TARGET_RISK_PCT=args.target_risk_pct,
        MAX_POSITION_PCT=args.max_position_pct,
        MIN_POSITION_USD=args.min_position_usd,
        MIN_TRAILING_STOP_PCT=args.min_trailing_stop_pct,
        MIN_ATR_TRAILING_PCT=args.min_atr_trailing_pct,
    )

    rm.save_run_results(params={}, metrics={}, metadata=config)

    if args.dry_run:
        print("!!! DRY RUN MODE ENABLED - No real orders will be placed !!!")

    try:
        engine = ExecutionEngine(
            config,
            results_path=results_path,
            db_manager=rm.db_manager,
            run_id="LIVE",
        )

        if args.dry_run_sizing:
            _dry_run_sizing(engine, override_portfolio_usd=args.portfolio_usd)
            return

        engine.run_event_loop()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"CRITICAL ERROR in event loop: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _dry_run_sizing(engine, override_portfolio_usd: float | None) -> None:
    """Print adaptive position-size projections for every symbol, then exit.

    Exercises the same `_estimate_stop_pct_for_sizing` and
    `_compute_adaptive_position_usd` paths the live trader uses, so the table
    reflects exactly what would be sized live at the current bar.
    """
    if override_portfolio_usd is not None:
        portfolio_usd = float(override_portfolio_usd)
        portfolio_src = "override"
    else:
        portfolio_usd = engine._get_total_portfolio_usd()
        portfolio_src = "exchange"
        if portfolio_usd is None:
            portfolio_usd = float(engine.config.get("START_CASH", 1000.0))
            portfolio_src = "START_CASH (exchange unavailable)"

    target_risk_pct = float(engine.config.get("TARGET_RISK_PCT", 0.01))
    max_position_pct = float(engine.config.get("MAX_POSITION_PCT", 0.15))
    min_position_usd = float(engine.config.get("MIN_POSITION_USD", 15.0))
    risk_usd = portfolio_usd * target_risk_pct
    cap_usd = portfolio_usd * max_position_pct

    print()
    print(f"Portfolio:       ${portfolio_usd:,.2f}  ({portfolio_src})")
    print(f"Risk per trade:  {target_risk_pct * 100:.2f}%  =  ${risk_usd:.2f}")
    print(f"Position cap:    {max_position_pct * 100:.1f}%  =  ${cap_usd:.2f}")
    print(f"Min entry:       ${min_position_usd:.2f}")
    print()

    print("Fetching latest market data...")
    ohlcv_df = engine._fetch_latest_data()
    signals = engine._compute_latest_signals(ohlcv_df)

    header = f"{'symbol':<14}{'exit':<14}{'stop_pct':>10}{'raw_$':>10}{'sized_$':>10}  status"
    print()
    print(header)
    print("-" * len(header))

    skipped = 0
    capped = 0
    sized = 0
    for symbol in sorted(engine.symbols):
        sig = signals.get(symbol)
        if sig is None:
            print(f"{symbol:<14}{'<no data>':<14}")
            continue
        # Force entry=True so the sizing path is exercised regardless of the
        # current strategy signal.
        sig = {**sig, "entry": True}
        stop_pct = engine._estimate_stop_pct_for_sizing(symbol, sig)
        raw = risk_usd / (stop_pct / 100.0)
        sized_usd = min(raw, cap_usd)
        if sized_usd < min_position_usd:
            status = "SKIP (below MIN_POSITION_USD)"
            skipped += 1
        elif raw > cap_usd:
            status = "CAPPED at MAX_POSITION_PCT"
            capped += 1
        else:
            status = "ok"
            sized += 1
        print(
            f"{symbol:<14}{sig.get('exit_name', '?'):<14}"
            f"{stop_pct:>9.2f}%{raw:>10.2f}{sized_usd:>10.2f}  {status}"
        )

    print()
    print(f"Summary:  ok={sized}  capped={capped}  skipped={skipped}")
