"""Phased execution helpers for the full strategy optimization pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ggTrader.core.orchestrator import (
    analyze_sensitivity_results,
    run_frozen_params_combined_backtest,
    run_multi_strategy_per_coin_wfo,
    run_sensitivity_orchestrator,
)
from ggTrader.indicators.strategies import ENTRY_REGISTRY, EXIT_REGISTRY
from ggTrader.pipeline.exit_tournament import parse_exit_tournament
from ggTrader.pipeline.param_grids import COARSE_ENTRY_PARAM_GRIDS, EXIT_AXIS_GRIDS
from ggTrader.utils.config import load_symbols_from_json

if TYPE_CHECKING:
    from ggTrader.utils.pipeline_status_logger import StatusLogger


def sensitivity_combo_count(grids: dict) -> int:
    """Total Cartesian products across all strategies (for logging)."""
    total = 0
    for _, grid in grids.items():
        n = 1
        for v in grid.values():
            n *= len(v) if isinstance(v, list) else 1
        total += n
    return total


def prepare_config_and_symbols(config: dict) -> dict:
    """Load symbols and truncate to MAX_SYMBOLS, prepare config dict."""
    symbols = load_symbols_from_json(config["SYMBOLS_FILE"])
    if symbols is None:
        raise ValueError(f"Failed to load symbols from {config['SYMBOLS_FILE']}")

    max_symbols = config.get("MAX_SYMBOLS", len(symbols))
    n_available = len(symbols)
    if n_available < max_symbols:
        print(
            f"Warning: SYMBOLS_FILE lists only {n_available} "
            f"symbols but MAX_SYMBOLS={max_symbols}. "
            f"Using all {n_available}. Use a larger rank JSON "
            "(e.g. data/top_50_USD_2023-01-01_2025-12-31.json) "
            "or lower --max-symbols."
        )
    symbols = symbols[:max_symbols]
    print(f"Using top {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")

    config = dict(config)
    config["SYMBOLS"] = symbols
    config["SYMBOLS_FILE"] = None

    return config


def phase_0_sensitivity_analysis(
    config: dict,
    narrowed_grids: dict,
    show_progress: bool = True,
    dry_run: bool = False,
    logger: StatusLogger | None = None,
    entry_book: dict | None = None,
    exit_book: dict | None = None,
    exit_tournament: list[str] | None = None,
    save_results: bool = True,
) -> dict:
    """Phase 0: Run sensitivity analysis for each entry strategy.

    Uses a single sensitivity exit (SENSITIVITY_EXIT_STRATEGY, default atr_trailing)
    to avoid multiplying Phase 1 cost across the full exit tournament. WFO still
    tries all exits in EXIT_TOURNAMENT on the narrowed entry params.
    """
    if entry_book is None:
        entry_book = COARSE_ENTRY_PARAM_GRIDS
    if exit_book is None:
        exit_book = EXIT_AXIS_GRIDS
    if exit_tournament is None:
        exit_tournament = config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys()))

    sensitivity_exit = config.get("SENSITIVITY_EXIT_STRATEGY", "atr_trailing")
    sens_exit_axes = dict(exit_book.get(sensitivity_exit, {}))

    strategies = list(ENTRY_REGISTRY.keys())
    n_strategies = len(strategies)
    print("\n" + "=" * 100)
    print("PHASE 1: SENSITIVITY ANALYSIS PER STRATEGY")
    print("=" * 100)
    merged_grids = {name: {**entry_book.get(name, {}), **sens_exit_axes} for name in entry_book}
    n_combos = sensitivity_combo_count(merged_grids)
    print(
        f"Active sensitivity book: {n_combos} total grid points across strategies "
        f"(coarse by default; use --detailed-sensitivity for the wide research grids)."
    )
    print(f"  Sensitivity exit: {sensitivity_exit} (exit tournament: {exit_tournament})")
    if logger:
        logger.update(
            f"Phase 1 started: sensitivity analysis for {n_strategies} strategies "
            f"({n_combos} grid points, exit={sensitivity_exit})"
        )

    sensitivity_results: dict = {}

    for s_idx, strategy_name in enumerate(strategies, 1):
        print(f"\n--- Sensitivity Analysis: {strategy_name} ({s_idx}/{n_strategies}) ---")

        param_grid = dict(entry_book.get(strategy_name, {}))
        param_grid.update(sens_exit_axes)

        if dry_run:
            param_grid = {k: v[:2] if isinstance(v, list) else [v] for k, v in param_grid.items()}

        config_with_strategy = {
            **config,
            "ENTRY_STRATEGY": strategy_name,
            "EXIT_STRATEGY": sensitivity_exit,
        }

        result = run_sensitivity_orchestrator(
            config=config_with_strategy,
            param_grid=param_grid,
            save_results=save_results,
            show_progress=show_progress,
            logger=logger,
        )

        results_df = result.get("results_df", pd.DataFrame())
        sensitivity_results[strategy_name] = results_df

        print(f"Analyzing parameter importance for {strategy_name}...")
        if not results_df.empty:
            narrowed_entry = analyze_sensitivity_results(results_df, param_grid, top_percentile=50)
            narrowed_grids[strategy_name] = narrowed_entry
            print(f"  Narrowed grid: {narrowed_entry}")
        else:
            narrowed_grids[strategy_name] = param_grid
            print("  No results; using original grid")

        if logger:
            logger.update(f"  Phase 0: {strategy_name} sensitivity done ({s_idx}/{n_strategies})")

    if logger:
        logger.phase_done("Phase 0 (Sensitivity Analysis)")
    return sensitivity_results


def phase_1_per_coin_multi_strategy_wfo(
    config: dict,
    narrowed_grids: dict,
    show_progress: bool = True,
    logger: StatusLogger | None = None,
    save_results: bool = True,
) -> dict:
    """Phase 1: Run per-coin WFO exit tournament across all (entry, exit) combinations."""
    n_coins = len(config.get("SYMBOLS", []))
    n_strategies = len(narrowed_grids)
    exit_tournament = config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys()))
    print("\n" + "=" * 100)
    print("PHASE 1: PER-COIN MULTI-STRATEGY WFO (EXIT TOURNAMENT)")
    print("=" * 100)
    if logger:
        logger.update(
            f"Phase 1 started: WFO for {n_coins} coins × {n_strategies} entries "
            f"× {len(exit_tournament)} exits",
            start_phase=True,
        )

    wfo_result = run_multi_strategy_per_coin_wfo(
        config=config,
        strategy_param_grids=narrowed_grids,
        save_results=save_results,
        show_progress=show_progress,
        logger=logger,
    )

    if logger:
        logger.phase_done("Phase 1 (Per-Coin WFO)")
    return wfo_result


def phase_2_full_data_validation(
    config: dict,
    wfo_results: dict,
    logger: StatusLogger | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> None:
    """Phase 2: Run best WFO parameters on the entire data range and compare to B&H."""
    from ggTrader.core.orchestrator import run_frozen_params_combined_backtest

    print("\n" + "=" * 100)
    print("PHASE 2: RESULT VALIDATION (FULL RANGE)")
    print("=" * 100)
    if logger:
        logger.update("Phase 2: Full-range final validation backtest", start_phase=True)

    exit_tournament = parse_exit_tournament(
        config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys())),
        EXIT_REGISTRY,
    )

    if ohlcv is None:
        # Pre-load EMA_WARMUP_BARS before START_DATE so the regime filter EMA is fully
        # warm from bar 0 of the actual backtest window — same approach as Phase 3.
        # Without this, EMA50/EMA200 start cold and regime signals are unreliable for
        # the first ~EMA_WARMUP_BARS × interval (default ~33 days at 4h).
        from ggTrader.utils.setup import load_hybrid_validation_ohlcv

        interval_str = config.get("INTERVAL", "4h")
        try:
            interval_hours = int(interval_str.rstrip("h"))
        except ValueError:
            interval_hours = 4
        n_warmup = int(config.get("EMA_WARMUP_BARS", 200))
        warmup_td = pd.Timedelta(hours=interval_hours * n_warmup)
        start_ts = pd.Timestamp(config["START_DATE"]).tz_localize("UTC")
        end_ts = pd.Timestamp(config["END_DATE"]).tz_localize("UTC")
        load_start = start_ts - warmup_td
        print(
            f"  [Phase 2] Loading warmup data from {load_start.date()} "
            f"({n_warmup} bars before {start_ts.date()}) — regime EMA warm from bar 0."
        )
        ohlcv = load_hybrid_validation_ohlcv(config, load_start, end_ts)

    # Pass PHASE3_STATS_CUTOFF so the portfolio is trimmed back to START_DATE after
    # indicators are computed on the warmup-extended data.  The key name is reused
    # since run_frozen_params_combined_backtest handles it generically.
    config_phase2 = {**config, "PHASE3_STATS_CUTOFF": config["START_DATE"]}

    out = run_frozen_params_combined_backtest(
        ohlcv,
        wfo_results["per_coin_results"],
        config_phase2,
        exit_tournament=exit_tournament,
        save_results=False,
        phase_title="PHASE 2: RESULT VALIDATION (FULL TRAINING/TEST RANGE)",
        combined_portfolio_label="Phase 2 - full training/test range combined portfolio vs Buy & Hold",
        logger=logger,
    )
    wfo_results["phase_2_stats"] = out["final_stats"]
    wfo_results["phase_2_per_coin_final_stats"] = out["per_coin_final_stats"]
    if logger:
        logger.phase_done("Phase 2 (Full-range validation)")


def phase_3_recent_performance(
    config: dict,
    wfo_results: dict,
    logger: StatusLogger | None = None,
) -> None:
    """Phase 3: Run best WFO parameters on the YTD window and compare to B&H."""
    import pandas as pd

    from ggTrader.utils.setup import load_hybrid_validation_ohlcv

    # Dynamically calculate window, or use config if provided
    start_raw = config.get("RECENT_VALIDATION_START_DATE")
    if start_raw:
        ts = pd.to_datetime(start_raw)
        start = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    else:
        start = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=1)

    end_raw = config.get("RECENT_VALIDATION_END_DATE")
    if end_raw:
        te = pd.to_datetime(end_raw)
        end = te.tz_localize("UTC") if te.tzinfo is None else te.tz_convert("UTC")
    else:
        end = pd.Timestamp.now(tz="UTC")

    use_ccxt = bool(config.get("RECENT_VALIDATION_USE_CCXT_TAIL", False))

    print(
        f"  Window: {start.strftime('%Y-%m-%d')} .. {end.strftime('%Y-%m-%d')} "
        f"| CCXT tail: {use_ccxt}"
    )
    if logger:
        logger.update(
            f"Phase 3: YTD performance {start.date()} .. {end.date()} (CCXT tail={use_ccxt})",
            start_phase=True,
        )

    # Load extra warmup bars before `start` so strategy indicators (e.g. EMA(200))
    # are fully warm from the first bar of the actual YTD window.  Without this,
    # strategies with slow EMAs suppress all entries for the first ~warmup_bars × interval
    # (e.g. 200 × 4h = 33 days) because the indicator starts from NaN.
    interval_str = config.get("INTERVAL", "4h")
    try:
        interval_hours = int(interval_str.rstrip("h"))
    except ValueError:
        interval_hours = 4
    n_warmup = int(config.get("EMA_WARMUP_BARS", 200))
    warmup_td = pd.Timedelta(hours=interval_hours * n_warmup)
    load_start = start - warmup_td

    # Pass the full warmup-extended window to the backtest so indicators are warm
    # by bar 0 of the actual YTD window.  The backtest trims its own equity curve
    # but the OHLCV needs to start early for correct indicator values.
    ohlcv_r = load_hybrid_validation_ohlcv(
        config,
        load_start,
        end,
        use_ccxt_tail=use_ccxt,
    )

    exit_tournament = parse_exit_tournament(
        config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys())),
        EXIT_REGISTRY,
    )
    # Pass the warmup cutoff so the portfolio is trimmed back to the intended YTD
    # start after indicators have been computed on the warmup-extended data.
    config_phase3 = {**config, "PHASE3_STATS_CUTOFF": start.isoformat()}
    out = run_frozen_params_combined_backtest(
        ohlcv_r,
        wfo_results["per_coin_results"],
        config_phase3,
        exit_tournament=exit_tournament,
        save_results=False,
        phase_title="PHASE 3: YEAR-TO-DATE PERFORMANCE",
        combined_portfolio_label="Phase 3 - YTD combined portfolio vs Buy & Hold",
        logger=logger,
    )
    wfo_results["phase_3_stats"] = out["final_stats"]
    wfo_results["phase_3_per_coin_final_stats"] = out["per_coin_final_stats"]

    # Save YTD dashboard plot alongside the full-range one
    rm = wfo_results.get("results_manager")
    ytd_pf = out.get("final_portfolio")
    if rm is not None and ytd_pf is not None:
        rm.save_vbt_dashboard(ytd_pf, "combined_portfolio_ytd_dashboard")

    if logger:
        logger.phase_done("Phase 3 (YTD performance)")
