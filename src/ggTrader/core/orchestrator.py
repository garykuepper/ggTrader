"""Centralized orchestration logic for backtesting, sensitivity analysis, and WFO."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.benchmarking import _enrich_final_stats_with_cagr_and_benchmark
from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.metrics import _apply_sensitivity_train_gates, _train_metric_series
from ggTrader.core.orchestrator_utils import (
    _as_optional_float,
    _coerce_metric_float,
    _coerce_strategy_params_for_engine,
    _default_params_from_grid,
    _eta_str,
    _extract_params,
    _first_grid_value,
    _format_robustness_metric,
    _is_bad_engine_param,
    _is_better_robustness,
    _log_memory_usage,
    _safe,
    _to_native,
    _wall_clock_eta,
    _wfo_per_coin_fallback_triple,
)
from ggTrader.core.regime_filtering import (
    _compute_altcoin_index_mask,
    _compute_btc_correlations,
    _compute_btc_regime_mask,
)
from ggTrader.core.sensitivity import (
    _cleanup_after_heavy_vectorized_run,
    _combo_index_keys,
    _execute_sensitivity_grid,
    _execute_sensitivity_vectorized,
    _metric_series_from_vectorized_pf,
    _process_sensitivity_chunk,
    _save_sensitivity_results,
    _vectorized_grid_metrics,
    analyze_sensitivity_results,
    run_sensitivity_orchestrator,
)
from ggTrader.core.wfo import (
    _calculate_oos_robustness,
    _calculate_robustness,
    _calculate_wfo_bounds,
    _execute_wfo_loop,
    _param_cv_series,
    _process_wfo_fold,
    _save_wfo_results,
    _weighted_robustness_series,
    _wfo_train_metric_row_key,
    run_wfo_orchestrator,
    run_wfo_per_coin_orchestrator,
)
from ggTrader.pipeline.exit_tournament import parse_exit_tournament
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers

# Re-export benchmarking helpers that may be used by external callers
from ggTrader.core.benchmarking import (
    _btc_buy_hold_portfolio_stats,
    _cagr_percent,
    _sp500_buy_hold_portfolio_stats,
    _years_from_price_index,
)

# Re-export metrics helpers
from ggTrader.core.metrics import (
    _align_grouped_combo_series,
    _calmar_ratio_series,
    _max_drawdown_for_train_gate,
    _open_position_count_end_for_gate,
    _print_wfo_fold_all_rejected_diagnostics,
    _profit_factor_series,
    _trade_counts_for_train_gate,
    _zscore_normalize_series,
)


def run_backtest_orchestrator(
    config: Dict[str, Any],
    params: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate a single backtest run.

    Args:
        config: Portfolio and data configuration.
        params: Strategy parameters.
        save_results: Whether to persist results via ResultsManager.

    Returns:
        Dictionary containing 'portfolio', 'stats', and 'rm' (if saved).
    """
    rm = (
        ResultsManager(
            "run_backtest",
            explicit_run_dir=config.get("EXPLICIT_RUN_DIR"),
            pipeline_stage=config.get("PIPELINE_STAGE"),
        )
        if save_results
        else None
    )

    ohlcv, mover_mask = load_data_with_movers(config)

    print("Running backtest...")
    engine = FastBacktest(ohlcv, params, config=config, mover_mask=mover_mask)
    pf = engine.run(show_progress=show_progress)
    stats = engine.get_stats()

    if save_results and rm:
        rm.save_run_results(params=params, metrics=stats, metadata=config)
        rm.save_vbt_dashboard(pf, "dashboard")
        rm.print_summary(stats)
        print(f"Results saved to: {rm.run_dir}")

    return {"portfolio": pf, "stats": stats, "results_manager": rm}


def run_frozen_params_combined_backtest(
    ohlcv: pd.DataFrame,
    per_coin_results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    *,
    exit_tournament: List[str],
    save_results: bool = False,
    results_manager: Any = None,
    phase_title: str = "PHASE 3: FINAL VALIDATION BACKTEST (FULL 3-YEAR RANGE)",
    combined_portfolio_label: str = "Final Combined Portfolio (Full 3-Year Range)",
    logger: Any = None,
) -> Dict[str, Any]:
    """Replay WFO-selected (entry, exit, params) on ``ohlcv`` and build one combined portfolio.

    Used for full-history Phase 3 and for optional recent-window validation (same frozen book).
    """
    print("\n" + "=" * 100)
    print(phase_title)
    print("=" * 100)

    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    combined_entries_list = []
    combined_exits_list = []
    combined_close_list = []
    combined_oos_scores: List[float] = []  # OOS robustness score per included symbol
    per_coin_final_stats: Dict[str, Any] = {}

    n_sym_total = len(symbols)
    for sym_idx, symbol in enumerate(symbols, start=1):
        if symbol not in per_coin_results:
            print(f"\nSkipping {symbol}: no WFO selection (not in per_coin_results).")
            continue

        strategy_name = per_coin_results[symbol]["best_strategy"]
        exit_name = per_coin_results[symbol]["best_exit"]
        best_params = per_coin_results[symbol]["best_params"]
        robustness_score = per_coin_results[symbol]["robustness_score"]
        selection_reason = per_coin_results[symbol].get("selection_reason", "wfo_robustness")
        best_label = f"{strategy_name}+{exit_name}"
        if selection_reason != "wfo_robustness":
            print(
                f"\nGenerating signals for {symbol} with {best_label} "
                f"(selection={selection_reason})..."
            )
        else:
            print(f"\nGenerating signals for {symbol} with {best_label}...")

        symbol_ohlcv = ohlcv[[symbol]]

        config_for_final = {
            **config,
            "ENTRY_STRATEGY": strategy_name,
            "EXIT_STRATEGY": exit_name,
            "USE_VECTORIZED": False,  # single-combo run; vectorized gives no benefit here
        }
        try:
            engine = FastBacktest(symbol_ohlcv, best_params, config=config_for_final)
            engine.run(show_progress=False)

            close = symbol_ohlcv.xs("close", axis=1, level=1, drop_level=True)
            entries, exits, _ = engine._generate_signals(show_progress=False)

            combined_entries_list.append(entries)
            combined_exits_list.append(exits)
            combined_close_list.append(close)
            oos_rob = per_coin_results[symbol].get("oos_robustness_score")
            combined_oos_scores.append(float(oos_rob) if oos_rob is not None and np.isfinite(float(oos_rob)) else 0.0)

            stats = engine.get_stats()
            per_coin_final_stats[symbol] = {
                "strategy": strategy_name,
                "exit": exit_name,
                "params": _to_native(best_params),
                "selection_reason": selection_reason,
                **stats,
            }
            rs_disp = _format_robustness_metric(robustness_score)
            sel_note = "" if selection_reason == "wfo_robustness" else f" | sel={selection_reason}"
            phase3_msg = (
                f"  > {symbol} ({sym_idx}/{n_sym_total}): Best={best_label} | "
                f"Robustness={rs_disp}{sel_note} | Win Rate={stats['win_rate']:.2f}% | "
                f"Trades={stats['total_trades']}"
            )
            print(phase3_msg)
            if logger:
                logger.update(phase3_msg)
        except Exception as sym_exc:
            print(f"  [WARN] Skipping {symbol} in combined backtest: {sym_exc!r}")

    if not combined_entries_list:
        raise ValueError(
            "Frozen-params backtest produced no symbols "
            "(empty OHLCV or no matching per_coin_results)."
        )

    combined_entries = pd.concat(combined_entries_list, axis=1)
    combined_exits = pd.concat(combined_exits_list, axis=1)
    combined_close = pd.concat(combined_close_list, axis=1)

    # Regime filters — three tiers based on BTC return correlation:
    #   corr >= BTC_REGIME_FILTER_MIN_CORRELATION        → BTC EMA filter
    #   corr in [ALTCOIN_REGIME_FILTER_CORR_MIN, btc_min) → Altcoin index EMA filter
    #   corr < ALTCOIN_REGIME_FILTER_CORR_MIN             → no filter (trade freely)
    if config.get("BTC_REGIME_FILTER", False):
        btc_regime = _compute_btc_regime_mask(ohlcv, config)
        alt_regime = (
            _compute_altcoin_index_mask(ohlcv, config)
            if config.get("ALTCOIN_REGIME_FILTER", False)
            else None
        )
        btc_corrs = _compute_btc_correlations(ohlcv, config)
        btc_min_corr = float(config.get("BTC_REGIME_FILTER_MIN_CORRELATION", 0.5))
        alt_min_corr = float(config.get("ALTCOIN_REGIME_FILTER_CORR_MIN", 0.3))
        n_warmup = int(config.get("EMA_WARMUP_BARS", 200))

        if btc_regime is None:
            bench_symbol = config.get("BENCHMARK_SYMBOL", "BTC-USD")
            print(f"\n  [Regime Filter] WARNING: {bench_symbol} data unavailable — filter skipped.")
        else:
            btc_aligned = btc_regime.reindex(combined_entries.index, fill_value=False)
            alt_aligned = (
                alt_regime.reindex(combined_entries.index, fill_value=False)
                if alt_regime is not None else None
            )
            mask_cols = {}
            btc_filtered, alt_filtered, exempt_coins = [], [], []
            for col in combined_entries.columns:
                sym = col[0] if isinstance(col, tuple) else col
                corr = btc_corrs.get(sym, 1.0)
                if corr >= btc_min_corr:
                    mask_cols[col] = btc_aligned.values
                    btc_filtered.append(f"{sym}({corr:.2f})")
                elif corr >= alt_min_corr and alt_aligned is not None:
                    mask_cols[col] = alt_aligned.values
                    alt_filtered.append(f"{sym}({corr:.2f})")
                else:
                    mask_cols[col] = np.ones(len(combined_entries), dtype=bool)
                    exempt_coins.append(f"{sym}({corr:.2f})")
            regime_mask = pd.DataFrame(mask_cols, index=combined_entries.index)
            n_blocked = int((combined_entries & ~regime_mask).values.sum())
            combined_entries = combined_entries & regime_mask
            print(f"\n  [Regime Filter] EMA({n_warmup}) applied — blocked {n_blocked} signals.")
            if btc_filtered:
                print(f"    BTC filter (corr>={btc_min_corr}):        {', '.join(btc_filtered)}")
            if alt_filtered:
                print(f"    Altcoin filter (corr>={alt_min_corr}):     {', '.join(alt_filtered)}")
            if exempt_coins:
                print(f"    Exempt (corr<{alt_min_corr}):              {', '.join(exempt_coins)}")

    # Compute OOS-robustness-proportional allocation weights.
    # Coins with higher OOS robustness receive a larger share of capital.
    # Falls back to equal weight if all OOS scores are zero/negative.
    raw_w = np.array([max(0.0, s) for s in combined_oos_scores], dtype=float)
    total_w = raw_w.sum()
    if total_w <= 0.0:
        raw_w = np.ones(len(combined_oos_scores), dtype=float)
        total_w = raw_w.sum()
    alloc_weights = raw_w / total_w  # normalized to sum 1.0

    # Iterative cap: no single coin exceeds MAX_COIN_ALLOCATION.
    max_alloc = float(config.get("MAX_COIN_ALLOCATION", 0.25))
    for _ in range(len(alloc_weights)):
        over = alloc_weights > max_alloc
        if not over.any():
            break
        excess = (alloc_weights[over] - max_alloc).sum()
        alloc_weights[over] = max_alloc
        under = ~over
        if not under.any():
            break
        alloc_weights[under] += excess * (alloc_weights[under] / alloc_weights[under].sum())

    print(
        f"\n  [Portfolio weights] OOS-weighted allocation "
        f"(max_alloc={max_alloc:.0%}, equal_fallback={total_w <= 0.0}):"
    )
    for sym, w in zip(combined_close.columns, alloc_weights):
        print(f"    {sym}: {w:.1%}")

    final_pf = vbt.Portfolio.from_signals(
        close=combined_close,
        entries=combined_entries,
        exits=combined_exits,
        init_cash=float(config["START_CASH"]),
        fees=float(config["FEES"]),
        slippage=float(config["SLIPPAGE"]),
        freq=config["FREQ"],
        size=alloc_weights,
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(combined_entries.shape[1], 0),
    ).copy()

    final_stats = {
        "total_value": _safe(final_pf.final_value().sum()),
        "total_profit": _safe(final_pf.total_profit().sum()),
        "profit_pct": _safe((final_pf.total_profit().sum() / float(config["START_CASH"])) * 100),
        "total_trades": int(final_pf.trades.count().sum()),
        "win_rate": _safe(final_pf.trades.win_rate().mean()) * 100,
        "sharpe": _safe(final_pf.sharpe_ratio().mean()),
        "sortino": _safe(final_pf.sortino_ratio().mean()),
        "max_drawdown": _safe(final_pf.max_drawdown().min()) * 100,
    }
    _enrich_final_stats_with_cagr_and_benchmark(final_stats, combined_close, config)

    print(f"\n{combined_portfolio_label}:")
    print(f"  Total Return: {final_stats['profit_pct']:.2f}%")
    cagr_s = f"{final_stats['cagr_pct']:.2f}%" if final_stats.get("cagr_pct") is not None else "n/a"
    print(f"  CAGR: {cagr_s}")

    bench_sym = final_stats.get("benchmark_label", "BTC-USD").split(" ")[0]
    b_cagr = final_stats.get("benchmark_cagr_pct")
    b_cagr_s = f"{b_cagr:.2f}%" if b_cagr is not None else "n/a"
    print(f"  Benchmark CAGR ({bench_sym} B&H): {b_cagr_s}")

    spy_cagr = final_stats.get("spy_cagr_pct")
    if spy_cagr is not None:
        spy_ret = final_stats.get("spy_profit_pct", 0.0)
        print(f"  Benchmark Return (S&P 500 B&H): {spy_ret:.2f}% | CAGR: {spy_cagr:.2f}%")

    print(f"  Sharpe Ratio: {final_stats['sharpe']:.4f}")
    print(f"  Max Drawdown: {final_stats['max_drawdown']:.2f}%")
    print(f"  Total Trades: {final_stats['total_trades']}")
    print(f"  Win Rate: {final_stats['win_rate']:.2f}%")

    if save_results and results_manager:
        strategy_summary = []
        for symbol, results in per_coin_results.items():
            strategy_summary.append(
                {
                    "symbol": symbol,
                    "strategy": results["best_strategy"],
                    "exit": results.get("best_exit", "atr_trailing"),
                    "robustness_score": results["robustness_score"],
                    "selection_reason": results.get("selection_reason", "wfo_robustness"),
                }
            )

        strategy_df = pd.DataFrame(strategy_summary)
        results_manager.save_metrics(strategy_df, "per_coin_strategy_selection.csv")

        final_stats_df = pd.DataFrame(
            [{"symbol": sym, **stats} for sym, stats in per_coin_final_stats.items()]
        )
        results_manager.save_metrics(final_stats_df, "per_coin_final_stats.csv")

        metadata = {
            **config,
            "exit_tournament": exit_tournament,
            "per_coin_results": _to_native(per_coin_results),
            "per_coin_final_stats": _to_native(per_coin_final_stats),
        }

        results_manager.save_run_results(
            params={"per_coin": _to_native(per_coin_results)},
            metrics=final_stats,
            metadata=metadata,
        )
        results_manager.save_vbt_dashboard(final_pf, "combined_portfolio_final_dashboard")
        print(f"\nMulti-Strategy WFO Results saved to: {results_manager.run_dir}")

    return {
        "final_portfolio": final_pf,
        "per_coin_final_stats": per_coin_final_stats,
        "final_stats": final_stats,
    }


def run_multi_strategy_per_coin_wfo(
    config: Dict[str, Any],
    strategy_param_grids: Dict[str, Dict],
    save_results: bool = True,
    show_progress: bool = True,
    logger: Any = None,
) -> Dict[str, Any]:
    """Run WFO for each symbol across multiple (entry, exit) combinations.

    For each symbol the tournament tests every registered entry strategy against
    every exit in ``config["EXIT_TOURNAMENT"]`` (default from ``full_pipeline_config``
    is typically ``["atr_trailing"]`` only; use ``--dual-exits`` or two names for both).
    The winner is picked by robustness score.  Phase 3 replays the full range with the
    winning (entry, exit, params) triple.

    Args:
        config: Portfolio and data configuration.  Supports optional key
            ``EXIT_TOURNAMENT`` (list of exit names); parsed against ``EXIT_REGISTRY``.
        strategy_param_grids: Dict mapping entry strategy name → merged param grid
            (entry-only keys already merged with exit-axis keys by the caller via
            ``build_param_grid``).
        save_results: Whether to save results.
        show_progress: Whether to show progress bars.

    Returns:
        Dict with per_coin_results, final_portfolio, final_stats, results_manager.
    """
    from ggTrader.indicators.strategies import EXIT_REGISTRY

    exit_tournament = parse_exit_tournament(
        config.get("EXIT_TOURNAMENT", list(EXIT_REGISTRY.keys())),
        EXIT_REGISTRY,
    )

    rm = (
        ResultsManager(
            "run_wfo_per_coin_multi_strategy",
            explicit_run_dir=config.get("EXPLICIT_RUN_DIR"),
            pipeline_stage=config.get("PIPELINE_STAGE"),
        )
        if save_results
        else None
    )
    ohlcv, mover_mask = load_data_with_movers(config)

    # Pre-compute regime masks once for all WFO folds.
    # Three tiers by BTC correlation: BTC filter, altcoin index filter, or no filter.
    wfo_btc_mask: Optional[pd.Series] = None
    wfo_alt_mask: Optional[pd.Series] = None
    wfo_btc_corrs: Dict[str, float] = {}
    if config.get("BTC_REGIME_FILTER", False):
        wfo_btc_mask = _compute_btc_regime_mask(ohlcv, config)
        wfo_btc_corrs = _compute_btc_correlations(ohlcv, config)
        if config.get("ALTCOIN_REGIME_FILTER", False):
            wfo_alt_mask = _compute_altcoin_index_mask(ohlcv, config)
        if wfo_btc_mask is not None:
            n_wu = int(config.get("EMA_WARMUP_BARS", 200))
            btc_min = float(config.get("BTC_REGIME_FILTER_MIN_CORRELATION", 0.5))
            alt_min = float(config.get("ALTCOIN_REGIME_FILTER_CORR_MIN", 0.3))
            alt_status = "altcoin index" if wfo_alt_mask is not None else "disabled"
            print(
                f"  [BTC Regime] EMA({n_wu}) masks pre-computed for WFO folds — "
                f"BTC(corr>={btc_min}), {alt_status}(corr>={alt_min}), exempt(<{alt_min})."
            )
        else:
            print("  [BTC Regime] WARNING: mask unavailable — WFO folds will run unfiltered.")

    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config.get("TEST_RATIO", 3.0)

    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    n_symbols = len(symbols)
    n_combos = len(strategy_param_grids) * len(exit_tournament)
    print(
        f"\nStarting Multi-Strategy Per-Coin WFO "
        f"({n_symbols} symbols, {n_splits} splits, "
        f"{len(strategy_param_grids)} entries × "
        f"{len(exit_tournament)} exits = {n_combos} combos)..."
    )
    print(f"  Exit tournament: {exit_tournament}")

    # Dictionary to store best (entry, exit, params) per symbol.
    per_coin_results: Dict[str, Dict[str, Any]] = {}
    t0_wfo_coins = time.time()

    # Phase 2: Per-coin WFO loop — test all (entry, exit) combos for each coin.
    for coin_idx, symbol in enumerate(symbols):
        coin_start = time.time()
        print(f"\n--- Optimizing {symbol} ({coin_idx + 1}/{n_symbols}) ---")
        try:
            symbol_ohlcv = ohlcv[[symbol]]
            symbol_mover_mask = mover_mask[[symbol]] if mover_mask is not None else None

            best_strategy: Optional[str] = None
            best_exit: Optional[str] = None
            best_robust_score = float("-inf")
            best_params_for_coin: Dict[str, Any] = {}
            best_wfo_stats: List[Dict] = []
            best_robust_top_5: List[Dict] = []
            best_is_robustness_score: float = float("-inf")
            best_oos_robustness_score: float = float("nan")
            best_fold_consistency: float = float("nan")
            debug_wfo = bool(config.get("WFO_DEBUG_METRICS", False))

            total_combos = len(strategy_param_grids) * len(exit_tournament)
            combo_idx = 0
            for strategy_name, param_grid in strategy_param_grids.items():
                for exit_name in exit_tournament:
                    combo_idx += 1
                    label = f"{strategy_name}+{exit_name}"
                    print(f"  Testing: {label} ({combo_idx}/{total_combos})")

                    config_combo = {
                        **config,
                        "ENTRY_STRATEGY": strategy_name,
                        "EXIT_STRATEGY": exit_name,
                    }

                    btc_min_corr = float(config.get("BTC_REGIME_FILTER_MIN_CORRELATION", 0.5))
                    alt_min_corr = float(config.get("ALTCOIN_REGIME_FILTER_CORR_MIN", 0.3))
                    coin_corr = wfo_btc_corrs.get(symbol, 1.0)
                    if coin_corr >= btc_min_corr:
                        coin_regime_mask = wfo_btc_mask
                    elif coin_corr >= alt_min_corr and wfo_alt_mask is not None:
                        coin_regime_mask = wfo_alt_mask
                    else:
                        coin_regime_mask = None

                    wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
                        symbol_ohlcv,
                        symbol_mover_mask,
                        param_grid,
                        config_combo,
                        list(param_grid.keys()),
                        n_splits,
                        test_ratio,
                        show_progress,
                        logger,
                        btc_regime_mask=coin_regime_mask,
                    )

                    oos_metrics_by_fold = {
                        fold_idx: stats["oos_sharpe"] for fold_idx, stats in enumerate(wfo_stats, 1)
                    }

                    robust_top_5, best_robust_params = _calculate_robustness(
                        is_metrics_by_fold,
                        list(param_grid.keys()),
                        param_grid,
                        oos_metrics_by_fold,
                        debug_metrics=debug_wfo,
                        config=config,
                    )

                    if robust_top_5:
                        robustness_score = float(robust_top_5[0]["robustness_score"])
                    else:
                        robustness_score = float("-inf")

                    # OOS-direct robustness: recency-weighted mean of per-fold OOS Sharpe.
                    oos_rob_combo, fold_cons_combo = _calculate_oos_robustness(
                        oos_metrics_by_fold, config=config
                    )
                    oos_blend_alpha = float(config.get("OOS_ROBUSTNESS_BLEND_ALPHA", 0.5))
                    if np.isfinite(oos_rob_combo) and np.isfinite(robustness_score):
                        is_oos_blend = (
                            1.0 - oos_blend_alpha
                        ) * robustness_score + oos_blend_alpha * oos_rob_combo
                    elif np.isfinite(oos_rob_combo):
                        is_oos_blend = oos_rob_combo
                    else:
                        is_oos_blend = robustness_score

                    # Fold consistency soft multiplier: strategies inconsistent across folds
                    # are penalized. floor=0.5 means the worst case halves the gate score.
                    use_fc_gate = bool(config.get("FOLD_CONSISTENCY_IN_GATE", True))
                    fc_floor = float(config.get("FOLD_CONSISTENCY_GATE_FLOOR", 0.5))
                    if use_fc_gate and np.isfinite(fold_cons_combo):
                        fc_factor = fc_floor + (1.0 - fc_floor) * fold_cons_combo
                        gate_score = is_oos_blend * fc_factor
                    else:
                        gate_score = is_oos_blend

                    print(
                        f"    {label} robustness: IS={_format_robustness_metric(robustness_score)} "
                        f"OOS={_format_robustness_metric(oos_rob_combo)} "
                        f"gate={_format_robustness_metric(gate_score)} "
                        f"consistency={fold_cons_combo:.0%}"
                    )

                    if _is_better_robustness(gate_score, best_robust_score):
                        best_robust_score = gate_score
                        best_strategy = strategy_name
                        best_exit = exit_name
                        best_params_for_coin = best_robust_params
                        best_wfo_stats = wfo_stats
                        best_robust_top_5 = robust_top_5
                        best_is_robustness_score = robustness_score
                        best_oos_robustness_score = oos_rob_combo
                        best_fold_consistency = fold_cons_combo

            selection_reason = "wfo_robustness"
            if best_strategy is None or not np.isfinite(best_robust_score):
                fb_s, fb_e, fb_p = _wfo_per_coin_fallback_triple(
                    strategy_param_grids, exit_tournament
                )
                best_strategy = fb_s
                best_exit = fb_e
                best_params_for_coin = fb_p
                best_robust_score = float("-inf")
                best_robust_top_5 = []
                selection_reason = "fallback_no_finite_robustness"
                print(
                    f"  WARNING: {symbol} — no finite WFO robustness winner; "
                    f"using fallback {best_strategy}+{best_exit} with first grid values."
                )

            per_coin_results[symbol] = {
                "best_strategy": best_strategy,
                "best_exit": best_exit,
                "best_params": best_params_for_coin,
                "robustness_score": best_robust_score,  # blended gate score
                "is_robustness_score": best_is_robustness_score,
                "oos_robustness_score": best_oos_robustness_score,
                "fold_consistency": best_fold_consistency,
                "wfo_stats": best_wfo_stats,
                "robust_top_5": best_robust_top_5,
                "selection_reason": selection_reason,
            }

            coin_elapsed = time.time() - coin_start
            total_elapsed = time.time() - t0_wfo_coins
            avg_per_coin = total_elapsed / (coin_idx + 1)
            eta = (n_symbols - coin_idx - 1) * avg_per_coin
            status_msg = (
                f"  > {symbol} ({coin_idx + 1}/{n_symbols}): WFO complete in {coin_elapsed:.0f}s | "
                f"ETA {_eta_str(eta)} (est. {_wall_clock_eta(eta)}) "
                f"(full-range Best / Win Rate printed in Phase 3)"
            )
            print(status_msg)
            if logger:
                logger.update(status_msg)
        except Exception as coin_exc:
            print(
                f"  [ERROR] {symbol} failed and will be skipped: {coin_exc!r}\n"
                f"  Continuing with remaining coins..."
            )

    # Filter out coins that failed to meet the minimum robustness threshold.
    # Set MIN_ROBUSTNESS_SCORE=None in config to disable.
    min_robust_cfg = config.get("MIN_ROBUSTNESS_SCORE")
    if min_robust_cfg is not None:
        min_robust = float(min_robust_cfg)
        skipped = [
            sym
            for sym, r in per_coin_results.items()
            if not np.isfinite(r["robustness_score"]) or r["robustness_score"] < min_robust
        ]
        if skipped:
            print(
                f"\n  [Robustness gate] Dropping {len(skipped)} coin(s) below "
                f"MIN_ROBUSTNESS_SCORE={min_robust}: {skipped}"
            )
            per_coin_results = {sym: r for sym, r in per_coin_results.items() if sym not in skipped}
        if not per_coin_results:
            print(
                "  WARNING: All coins dropped by robustness gate — lowering threshold or "
                "setting MIN_ROBUSTNESS_SCORE=None is recommended."
            )

    # Filter out coins with insufficient OOS fold consistency.
    # Set MIN_FOLD_CONSISTENCY=None in config to disable.
    min_consistency_cfg = config.get("MIN_FOLD_CONSISTENCY")
    if min_consistency_cfg is not None:
        min_consistency = float(min_consistency_cfg)
        skipped_fc = [
            sym
            for sym, r in per_coin_results.items()
            if (r.get("fold_consistency") is None
                or not np.isfinite(float(r["fold_consistency"]))
                or float(r["fold_consistency"]) < min_consistency)
        ]
        if skipped_fc:
            print(
                f"\n  [Consistency gate] Dropping {len(skipped_fc)} coin(s) below "
                f"MIN_FOLD_CONSISTENCY={min_consistency:.0%}: {skipped_fc}"
            )
            per_coin_results = {sym: r for sym, r in per_coin_results.items() if sym not in skipped_fc}
        if not per_coin_results:
            print(
                "  WARNING: All coins dropped by consistency gate — lowering threshold or "
                "setting MIN_FOLD_CONSISTENCY=None is recommended."
            )

    # Strategy diversity cap: limit how many coins can use the same entry strategy.
    # Prevents a single strategy from dominating the portfolio (correlation risk).
    max_per_strat_cfg = config.get("MAX_COINS_PER_STRATEGY")
    if max_per_strat_cfg is not None:
        max_per_strat = int(max_per_strat_cfg)
        from collections import defaultdict
        strat_groups: dict = defaultdict(list)
        for sym, r in per_coin_results.items():
            strat_groups[r["best_strategy"]].append((sym, r))
        diversity_dropped = []
        for strat, group in strat_groups.items():
            if len(group) <= max_per_strat:
                continue
            # Sort by OOS robustness descending; keep top N
            group_sorted = sorted(
                group,
                key=lambda x: float(x[1].get("oos_robustness_score") or float("-inf")),
                reverse=True,
            )
            to_drop = [sym for sym, _ in group_sorted[max_per_strat:]]
            diversity_dropped.extend(to_drop)
        if diversity_dropped:
            print(
                f"\n  [Diversity cap] Dropping {len(diversity_dropped)} coin(s) over "
                f"MAX_COINS_PER_STRATEGY={max_per_strat}: {diversity_dropped}"
            )
            per_coin_results = {sym: r for sym, r in per_coin_results.items() if sym not in diversity_dropped}

    phase3_out = run_frozen_params_combined_backtest(
        ohlcv,
        per_coin_results,
        config,
        exit_tournament=exit_tournament,
        save_results=save_results,
        results_manager=rm,
        logger=logger,
    )
    final_pf = phase3_out["final_portfolio"]
    per_coin_final_stats = phase3_out["per_coin_final_stats"]
    final_stats = phase3_out["final_stats"]

    return {
        "final_portfolio": final_pf,
        "per_coin_results": per_coin_results,
        "per_coin_final_stats": per_coin_final_stats,
        "final_stats": final_stats,
        "results_manager": rm,
    }
