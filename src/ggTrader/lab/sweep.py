"""Parameter sweep: grid generation, vectorized orchestration, results display."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Type

STOP_PARAMS: frozenset = frozenset({"ts_stop", "atr_period", "atr_mult"})
VOL_PARAMS: frozenset = frozenset({"vol_target", "vol_lookback"})
OVERLAY_PARAMS: frozenset = STOP_PARAMS | VOL_PARAMS


def split_params(combo: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a combo dict into (signal_params, overlay_params).

    Overlay params include stop-loss and vol-targeting settings; they are
    passed to the simulation config rather than signal generation.
    """
    signal = {k: v for k, v in combo.items() if k not in OVERLAY_PARAMS}
    overlay = {k: v for k, v in combo.items() if k in OVERLAY_PARAMS}
    return signal, overlay


def _is_valid_combo(params: Dict[str, Any]) -> bool:
    """Filter invalid combos: fast >= slow, or both ts_stop and atr_mult."""
    if "ts_stop" in params and "atr_mult" in params:
        return False
    fast_keys = sorted(k for k in params if "fast" in k)
    slow_keys = sorted(k for k in params if "slow" in k)
    for fk, sk in zip(fast_keys, slow_keys):
        if params[fk] >= params[sk]:
            return False
    return True


def build_grid(
    strategy_cls: Type,
    overrides: Optional[Dict[str, list]] = None,
) -> List[Dict[str, Any]]:
    """Cartesian product of sweep params, filtering invalid combos."""
    raw = strategy_cls.sweep_params()
    if overrides:
        raw = {**raw, **overrides}
    keys = sorted(raw.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*(raw[k] for k in keys))]
    return [c for c in combos if _is_valid_combo(c)]


def combo_name(strategy_name: str, params: Dict[str, Any]) -> str:
    """Deterministic label from strategy name + sorted param key-value pairs."""
    parts = [f"{k}{v}" for k, v in sorted(params.items())]
    return strategy_name + "__" + "_".join(parts)


def group_by_stop_config(
    grid: List[Dict[str, Any]],
) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group combos by their stop/overlay config for batched simulation."""
    from collections import defaultdict

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for combo in grid:
        _, stop_p = split_params(combo)
        groups[tuple(sorted(stop_p.items()))].append(combo)
    return groups


def sweep_signal_group(
    strategy_name: str,
    strat_instance: Any,
    stop_key: tuple,
    group_combos: List[Dict[str, Any]],
    symbols: List[str],
    ohlcv: Any,
    prices: Any,
    base_config: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Run signal generation + simulation for one stop-config group.

    Returns (eq_dict, diags_dict) mapping combo keys to equity Series and diagnostics.
    """
    from ggTrader.lab.simulate import simulate_signals
    from ggTrader.lab.strategy import SignalTargets

    stop_config = dict(stop_key)
    signal_combos = [split_params(c)[0] for c in group_combos]
    seen: set = set()
    unique_signal: List[Dict[str, Any]] = []
    for sc in signal_combos:
        k = tuple(sorted(sc.items()))
        if k not in seen:
            seen.add(k)
            unique_signal.append(sc)
    targets = strat_instance.sweep_signals(unique_signal, symbols, ohlcv)
    group_targets: Dict[str, SignalTargets] = {}
    for combo in group_combos:
        signal_p, _ = split_params(combo)
        signal_key = combo_name(strategy_name, signal_p)
        full_key = combo_name(strategy_name, combo)
        group_targets[full_key] = targets[signal_key]

    sim_config = {**base_config, **stop_config}
    ohlcv_arg = ohlcv if "atr_mult" in stop_config else None
    _rets, eq, diag = simulate_signals(group_targets, prices, sim_config, ohlcv=ohlcv_arg)
    eq_dict = {key: eq[key] for key in group_targets}
    return eq_dict, {key: diag.get(key, {}) for key in group_targets}


def format_results_table(
    rows: List[Dict[str, Any]],
    strategy_name: str,
    n_combos: int,
    eval_start: str,
    eval_end: str,
    sweep_id: str,
    spy_metrics: Dict[str, Any],
) -> str:
    """Ranked results table sorted by Sharpe descending."""

    def _safe_sharpe(r: Dict[str, Any]) -> float:
        v = r.get("sharpe", float("-inf"))
        return v if v == v else float("-inf")  # NaN != NaN

    sorted_rows = sorted(rows, key=_safe_sharpe, reverse=True)
    lines = [
        f"Sweep complete: {strategy_name} | {n_combos} combos | {eval_start} → {eval_end}",
        f"sweep_id: {sweep_id}",
        "",
        f"{'Rank':<6}{'Combo':<40}{'Sharpe':>8}{'CAGR%':>8}{'MaxDD%':>8}"
        f"{'Sortino':>9}{'TotRet%':>9}",
        "─" * 88,
    ]
    for i, r in enumerate(sorted_rows, 1):
        lines.append(
            f"{i:<6}{r['combo']:<40}{r['sharpe']:>8.2f}{r['cagr_pct']:>7.1f}%"
            f"{r['max_drawdown_pct']:>7.1f}%{r['sortino']:>9.2f}"
            f"{r['total_return_pct']:>8.1f}%"
        )
    lines.append("")
    lines.append(
        f"SPY baseline: CAGR {spy_metrics['cagr_pct']:.1f}%"
        f" | Sharpe {spy_metrics['sharpe']:.2f}"
        f" | MaxDD {spy_metrics['max_drawdown_pct']:.1f}%"
    )
    return "\n".join(lines)


def run_sweep(
    strategy_name: str,
    strategy_cls: Type,
    cfg: "LabConfig",
    ohlcv: "pd.DataFrame",
    spy_close: "pd.Series",
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    universe: str = "sp500",
) -> str:
    """Run a full parameter sweep: vectorized signals -> batched vbt sim -> persist + print."""
    import pandas as pd  # noqa: F811

    from ggTrader.lab import persist
    from ggTrader.lab.data import eligible_at, rebalance_dates
    from ggTrader.lab.metrics import curve_stats
    from ggTrader.lab.simulate import simulate_weights
    from ggTrader.lab.strategies.indicators import extract_close
    from ggTrader.lab.strategy import LabConfig

    persist.init_schema()
    param_grid: Dict[str, Any] = {}
    for combo in grid:
        for k, v in combo.items():
            param_grid.setdefault(k, set()).add(v)
    param_grid = {k: sorted(v) for k, v in param_grid.items()}
    sweep_id = persist.start_sweep(strategy_name, market, param_grid, len(grid))

    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    prices = extract_close(ohlcv, symbols)

    # Generate signals and simulate — grouped by stop config
    strat_instance = strategy_cls(cfg)
    all_eq: Dict[str, "pd.Series"] = {}
    all_diags: Dict[str, Dict[str, Any]] = {}

    # Batched simulation setup
    start_cash = float(base_config["START_CASH"])

    # Compute trade_start: first bar at or after eval_start (strip warmup/cash prefix)
    eval_start_ts = pd.Timestamp(eval_start, tz="UTC")
    trade_window = ohlcv.index[ohlcv.index >= eval_start_ts]
    trade_start = trade_window[0] if len(trade_window) else ohlcv.index[0]

    # Score SPY over the eval window only
    spy_eval = spy_close.loc[trade_start:].dropna()
    spy_stats = curve_stats(start_cash * (spy_eval / spy_eval.iloc[0]))

    if hasattr(strat_instance, "sweep_signals"):
        for stop_key, group_combos in group_by_stop_config(grid).items():
            eq_dict, diag_dict = sweep_signal_group(
                strategy_name,
                strat_instance,
                stop_key,
                group_combos,
                symbols,
                ohlcv,
                prices,
                base_config,
            )
            all_eq.update(eq_dict)
            all_diags.update(diag_dict)
    else:
        # Weight strategies: build per-combo, simulate together
        all_targets: Dict[str, Any] = {}
        start_ts = pd.Timestamp(eval_start, tz="UTC")
        end_ts = pd.Timestamp(eval_end, tz="UTC")
        dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
        for combo_params in grid:
            merged = {
                **{"top_n": cfg.top_n, "lookback": cfg.lookback, "skip": cfg.skip},
                **combo_params,
            }
            combo_cfg = LabConfig(
                top_n=int(merged.get("top_n", cfg.top_n)),
                lookback=int(merged.get("lookback", cfg.lookback)),
                skip=int(merged.get("skip", cfg.skip)),
                min_history_bars=cfg.min_history_bars,
                max_stocks=cfg.max_stocks,
            )
            strat = strategy_cls(combo_cfg)
            plans: Dict[str, Any] = {}
            for asof in dates:
                past = ohlcv.loc[:asof]
                elig = eligible_at(asof, past, combo_cfg, universe=universe)[0]
                plans[asof] = strat.select(asof, past, elig)
            target = strat.to_targets(plans, ohlcv)
            key = combo_name(strategy_name, combo_params)
            all_targets[key] = target

        _rets_df, eq_df, wdiags = simulate_weights(all_targets, prices, base_config)
        for key in all_targets:
            all_eq[key] = eq_df[key]
            all_diags[key] = wdiags.get(key, {})

    # Convert all_eq to a DataFrame for consistent scoring
    eq_df = pd.DataFrame(all_eq)

    # Score each combo over the eval window only (after warmup) and persist
    result_rows: List[Dict[str, Any]] = []
    for key in all_eq:
        eq = eq_df[key].loc[trade_start:].dropna()
        if len(eq) < 2:
            continue
        metrics = curve_stats(eq)
        combo_params = next(c for c in grid if combo_name(strategy_name, c) == key)
        persist.write_sweep_combo(
            sweep_id, key, combo_params, metrics, spy_stats, all_diags.get(key, {})
        )
        result_rows.append({"combo": key, **metrics})

    persist.finish_sweep(sweep_id)

    table = format_results_table(
        result_rows, strategy_name, len(grid), eval_start, eval_end, sweep_id, spy_stats
    )
    print(table)
    return sweep_id
