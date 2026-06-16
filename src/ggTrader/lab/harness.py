"""The lab walk-forward harness: plan -> vectorized simulate -> persist -> score."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ggTrader.lab import persist
from ggTrader.lab.data import rebalance_dates
from ggTrader.lab.metrics import benchmark
from ggTrader.lab.simulate import simulate_signals, simulate_weights
from ggTrader.lab.strategy import Plan, SignalTargets, Strategy

UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]


def leak_check(
    strategy: Strategy, ohlcv: pd.DataFrame, asof: pd.Timestamp, eligible: List[str]
) -> bool:
    """select at asof must be identical with and without post-asof rows present.

    The ``unmasked`` call passes the full frame so a strategy that reads data
    before self-truncating (e.g. at module/closure level) is caught here.
    """
    full = strategy.select(asof, ohlcv.loc[:asof], eligible)
    truncated = strategy.select(asof, ohlcv.loc[:asof].copy(deep=True), eligible)
    unmasked = strategy.select(asof, ohlcv, eligible)
    return (
        json.dumps(full, sort_keys=True, default=str)
        == json.dumps(truncated, sort_keys=True, default=str)
        == json.dumps(unmasked, sort_keys=True, default=str)
    )


def walkforward(
    strategies: List[Strategy],
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    freq: str,
    universe_fn: UniverseFn,
    base_config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> str:
    """Run one or more strategies (weight or signal) over [eval_start, eval_end).

    Weight strategies (target_kind='weights') are simulated via from_orders.
    Signal strategies (target_kind='signals') are simulated via from_signals.
    Both groups run in a single grouped vbt call each; results are merged.
    """
    start_ts = pd.Timestamp(eval_start, tz="UTC")
    end_ts = pd.Timestamp(eval_end, tz="UTC")
    dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
    if not dates:
        raise RuntimeError("No rebalance dates in the eval span.")

    prices = pd.concat(
        {s: ohlcv[s]["close"] for s in ohlcv.columns.get_level_values(0).unique()}, axis=1
    )

    persist.init_schema()  # must precede start_run on a fresh database
    if run_id is None:
        run_name = strategies[0].name if len(strategies) == 1 else "multi"
        run_id = persist.start_run(
            run_name, market, freq, eval_start, eval_end, params=dict(base_config)
        )

    # Phase 1: plan phase (point-in-time select, resumable)
    weight_targets: Dict[str, pd.DataFrame] = {}
    signal_targets: Dict[str, SignalTargets] = {}

    for strat in strategies:
        plans: Dict[pd.Timestamp, Plan] = {}
        for asof in dates:
            if persist.plan_done(run_id, strat.name, asof):
                plans[asof] = persist.read_plan(run_id, strat.name, asof)
                continue
            past = ohlcv.loc[:asof]
            eligible = universe_fn(asof, past)
            plan = strat.select(asof, past, eligible)
            persist.write_plan(
                run_id,
                strat.name,
                asof,
                plan,
                eligible_count=len(eligible),
                coverage={"n_eligible": len(eligible)},
            )
            plans[asof] = plan

        targets = strat.to_targets(plans, ohlcv)
        if strat.target_kind == "signals":
            signal_targets[strat.name] = targets  # type: ignore[assignment]
        else:
            weight_targets[strat.name] = targets  # type: ignore[assignment]

    # Phase 2: vectorized simulation (one grouped vbt call per family)
    all_returns: Dict[str, pd.Series] = {}
    all_equity: Dict[str, pd.Series] = {}
    all_diags: Dict[str, Dict[str, Any]] = {}

    if weight_targets:
        w_rets, w_eq, w_diags = simulate_weights(weight_targets, prices, base_config)
        for name in weight_targets:
            all_returns[name] = w_rets[name]
            all_equity[name] = w_eq[name]
            all_diags[name] = w_diags[name]

    if signal_targets:
        s_rets, s_eq, s_diags = simulate_signals(signal_targets, prices, base_config)
        for name in signal_targets:
            all_returns[name] = s_rets[name]
            all_equity[name] = s_eq[name]
            all_diags[name] = s_diags[name]

    # The data window includes a warmup prefix (history for eligibility/lookback)
    # during which the portfolio is pure cash. Score only the traded span — from
    # the first bar after the first rebalance — so the cash prefix doesn't deflate
    # Sharpe or stretch the SPY benchmark across untraded years.
    forward = ohlcv.index[ohlcv.index > dates[0]]
    trade_start = forward[0] if len(forward) else dates[0]

    for strat in strategies:
        name = strat.name
        eq = all_equity[name].loc[trade_start:].dropna()
        rets = all_returns[name].loc[trade_start:]
        rep = benchmark(eq, spy_close, float(base_config["START_CASH"]))
        spy = spy_close.reindex(eq.index).ffill()
        bench_curve = float(base_config["START_CASH"]) * (spy / spy.dropna().iloc[0])
        persist.write_returns_equity(run_id, name, rets, eq, bench_curve)
        persist.write_summary(
            run_id,
            name,
            rep["strategy"],
            rep["spy"],
            {
                **all_diags[name],
                "monthly_hit_rate_vs_spy": rep["monthly_hit_rate_vs_spy"],
                "n_months": rep["n_months"],
            },
        )

    persist.finish_run(run_id)
    return run_id
