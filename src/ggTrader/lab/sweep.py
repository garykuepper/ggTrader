"""Parameter sweep: grid generation, vectorized orchestration, results display."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Type


def _is_valid_combo(params: Dict[str, Any]) -> bool:
    """Filter combos where a 'fast' param is >= a corresponding 'slow' param."""
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

    sorted_rows = sorted(rows, key=lambda r: r.get("sharpe", float("-inf")), reverse=True)
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
) -> str:
    """Run a full parameter sweep: vectorized signals -> batched vbt sim -> persist + print."""
    import pandas as pd  # noqa: F811

    from ggTrader.lab import persist
    from ggTrader.lab.data import eligible_at, rebalance_dates
    from ggTrader.lab.metrics import curve_stats
    from ggTrader.lab.simulate import simulate_signals, simulate_weights
    from ggTrader.lab.strategy import LabConfig, SignalTargets

    persist.init_schema()
    param_grid = strategy_cls.sweep_params()
    sweep_id = persist.start_sweep(strategy_name, market, param_grid, len(grid))

    prices = pd.concat(
        {s: ohlcv[s]["close"] for s in ohlcv.columns.get_level_values(0).unique()},
        axis=1,
    )

    # Determine symbols from the universe (all available in ohlcv)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())

    # Generate signals for all combos
    strat_instance = strategy_cls(cfg)
    if hasattr(strat_instance, "sweep_signals"):
        all_targets = strat_instance.sweep_signals(grid, symbols, ohlcv)
    else:
        # Weight strategies: build per-combo, simulate together
        all_targets = {}
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
            start_ts = pd.Timestamp(eval_start, tz="UTC")
            end_ts = pd.Timestamp(eval_end, tz="UTC")
            dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
            strat = strategy_cls(combo_cfg)
            plans: Dict[str, Any] = {}
            for asof in dates:
                past = ohlcv.loc[:asof]
                elig = eligible_at(asof, past, combo_cfg)[0]
                plans[asof] = strat.select(asof, past, elig)
            targets = strat.to_targets(plans, ohlcv)
            key = combo_name(strategy_name, combo_params)
            all_targets[key] = targets

    # Batched simulation — one vbt call
    start_cash = float(base_config["START_CASH"])
    spy_stats = curve_stats(start_cash * (spy_close / spy_close.dropna().iloc[0]))

    if isinstance(next(iter(all_targets.values())), SignalTargets):
        rets_df, eq_df, diags = simulate_signals(all_targets, prices, base_config)
    else:
        rets_df, eq_df, diags = simulate_weights(all_targets, prices, base_config)

    # Score each combo and persist
    result_rows: List[Dict[str, Any]] = []
    for key in all_targets:
        eq = eq_df[key].dropna()
        if len(eq) < 2:
            continue
        metrics = curve_stats(eq)
        combo_params = next(c for c in grid if combo_name(strategy_name, c) == key)
        persist.write_sweep_combo(
            sweep_id, key, combo_params, metrics, spy_stats, diags.get(key, {})
        )
        result_rows.append({"combo": key, **metrics})

    persist.finish_sweep(sweep_id)

    table = format_results_table(
        result_rows, strategy_name, len(grid), eval_start, eval_end, sweep_id, spy_stats
    )
    print(table)
    return sweep_id
