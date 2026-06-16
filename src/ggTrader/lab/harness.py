"""The lab walk-forward harness: plan -> vectorized simulate -> persist -> score."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ggTrader.lab import persist
from ggTrader.lab.data import rebalance_dates
from ggTrader.lab.metrics import benchmark
from ggTrader.lab.simulate import simulate_weights
from ggTrader.lab.strategy import Plan, Strategy

UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]


def leak_check(
    strategy: Strategy, ohlcv: pd.DataFrame, asof: pd.Timestamp, eligible: List[str]
) -> bool:
    """select at asof must be identical with and without post-asof rows present."""
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
    """Run one or more weight-based strategies over [eval_start, eval_end)."""
    start_ts = pd.Timestamp(eval_start, tz="UTC")
    end_ts = pd.Timestamp(eval_end, tz="UTC")
    dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
    if not dates:
        raise RuntimeError("No rebalance dates in the eval span.")

    prices = pd.concat(
        {s: ohlcv[s]["close"] for s in ohlcv.columns.get_level_values(0).unique()}, axis=1
    )

    if run_id is None:
        run_id = persist.start_run(
            strategies[0].name, market, freq, eval_start, eval_end, params=dict(base_config)
        )
    persist.init_schema()

    targets_by_strategy: Dict[str, pd.DataFrame] = {}
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
        targets_by_strategy[strat.name] = strat.to_targets(plans, ohlcv)

    returns, equity, diags = simulate_weights(targets_by_strategy, prices, base_config)

    for strat in strategies:
        name = strat.name
        eq = equity[name].dropna()
        rep = benchmark(eq, spy_close, float(base_config["START_CASH"]))
        spy = spy_close.reindex(eq.index).ffill()
        bench_curve = float(base_config["START_CASH"]) * (spy / spy.dropna().iloc[0])
        persist.write_returns_equity(run_id, name, returns[name], eq, bench_curve)
        persist.write_summary(
            run_id,
            name,
            rep["strategy"],
            rep["spy"],
            {
                **diags[name],
                "monthly_hit_rate_vs_spy": rep["monthly_hit_rate_vs_spy"],
                "n_months": rep["n_months"],
            },
        )

    persist.finish_run(run_id)
    return run_id
