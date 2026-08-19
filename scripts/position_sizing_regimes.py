"""Research script: compare position-sizing regimes for the 3-sleeve blend.

Regime P (status quo): vbt.Portfolio.from_signals(size_type="percent") --
each entry sizes to SIGNAL_POSITION_SIZE of REMAINING CASH, so order sizes
decay geometrically and a sleeve asymptotically never fully deploys. This is
the exact call `ggTrader.lab.simulate.simulate_signals` makes; reused
unmodified here.

Regime F (live-equivalent): vbt.Portfolio.from_signals(size_type="targetpercent")
-- each admitted entry targets a FIXED position_pct of *current total
portfolio value* (not remaining cash), with a per-sleeve cap on concurrent
open positions enforced by a sequential admission filter (vbt has no native
"max concurrent positions" knob; entries beyond the cap on a given bar are
simply dropped for that bar, mirroring how RiskGuard.max_new_positions works
live -- a blocked signal is skipped, not queued/retried).

Both regimes reuse the SAME sleeve entries/exits (built once via each
strategy's normal to_targets(), which includes the PIT eligibility_mask
added in commit 2eafab1) and the SAME `combine_sleeves` inverse-vol /
target-vol blend + `curve_stats` used everywhere else in the lab.

Method note (labeled per the report's limitation, see docs/research/
2026-08-19-position-sizing-regimes.md Sec 1): this is a SINGLE long backtest
window over fixed, default EnsembleSignal params -- not a full gated WFO with
per-fold combo selection. A full 3-sleeve WFO (48 combos x ~17 folds x 3
sleeves, each fold also recomputing the leak-fixed expanding-window anchor)
was estimated at multiple hours per the 2026-08-18 anchor report and is well
outside this task's budget. The single-window run uses the identical
STOCK_BASE_CONFIG, LabConfig() defaults, and blend machinery as the
validated run, so the two regimes are apples-to-apples with each other; it
is NOT expected to exactly reproduce the WFO-selected-combo 1.14 Sharpe
figure, and the report states what this run actually measures for Regime P
as the anchor instead.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.lab.allocation import combine_sleeves
from ggTrader.lab.data import (
    STOCK_BASE_CONFIG,
    eligible_at,
    equity_universe_between,
    load_ohlcv,
    rebalance_dates,
)
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig, SignalTargets

SLEEVE_UNIVERSES: tuple[str, ...] = ("sp500", "midcap400", "nasdaq100")
POSITION_PCT = 0.033  # matches live RiskGuard.position_pct (risk.py)
SLOT_CAPS = (8, 12, 16, 20, 25, 30)


@dataclass
class SleeveData:
    label: str
    targets: SignalTargets
    prices: pd.DataFrame


def build_sleeve_targets(
    universe: str, cfg: LabConfig, ohlcv: pd.DataFrame, es: pd.Timestamp, ee: pd.Timestamp
) -> SignalTargets:
    """PIT-correct entries/exits for one sleeve: select() at each monthly
    rebalance date (universe membership as-of that date via eligible_at),
    then to_targets() -- which ANDs in eligibility_mask() so a symbol can't
    trade before it was actually a PIT member. Same construction the
    production `walkforward()` harness and `run_blend()` use per sleeve,
    minus the WFO grid search (single default-param instance)."""
    dates = rebalance_dates(ohlcv.index, es, ee)
    strat = EnsembleSignal(cfg)
    plans: dict[pd.Timestamp, list] = {}
    for asof in dates:
        past = ohlcv.loc[:asof]
        eligible, _ = eligible_at(asof, past, cfg, universe=universe)
        plans[asof] = strat.select(asof, past, eligible)
    return strat.to_targets(plans, ohlcv)


def admission_filter(entries: np.ndarray, exits: np.ndarray, cap: int) -> np.ndarray:
    """Sequential concurrent-position admission control.

    At each bar: close out symbols with an exit signal, then admit up to
    (cap - currently_open) of that bar's entry signals (deterministic order
    = column/symbol order), dropping the rest for that bar only -- a denied
    signal is not queued, matching how RiskGuard.max_new_positions works
    live (a blocked entry is simply skipped that day).
    """
    t_n, s_n = entries.shape
    open_mask = np.zeros(s_n, dtype=bool)
    admitted = np.zeros((t_n, s_n), dtype=bool)
    for t in range(t_n):
        open_mask &= ~exits[t]
        want = entries[t] & ~open_mask
        capacity = cap - int(open_mask.sum())
        if capacity <= 0:
            continue
        idx = np.flatnonzero(want)
        if idx.size > capacity:
            idx = idx[:capacity]
        sel = np.zeros(s_n, dtype=bool)
        sel[idx] = True
        admitted[t] = sel
        open_mask |= sel
    return admitted


def _group(n: int, label: str) -> pd.Index:
    return pd.Index([label] * n, name="strategy")


def simulate_percent(
    label: str, targets: SignalTargets, prices: pd.DataFrame, base_config: dict
) -> tuple[pd.Series, "vbt.Portfolio"]:
    """Regime P: unmodified simulate_signals (size_type='percent', decaying)."""
    _rets, _eq, _diags, pf = simulate_signals(
        {label: targets}, prices, dict(base_config), return_pf=True
    )
    return pf.value()[label] if isinstance(pf.value(), pd.DataFrame) else pf.value(), pf


def simulate_targetpercent(
    label: str,
    targets: SignalTargets,
    prices: pd.DataFrame,
    base_config: dict,
    slot_cap: int,
) -> tuple[pd.Series, "vbt.Portfolio"]:
    """Regime F: fixed position_pct of current total value, slot-capped.

    vbt's from_signals only supports SizeType.{Amount,Value,Percent} (Percent
    being the remaining-cash-based Regime P semantics) -- TargetPercent is
    not implemented there. TargetPercent orders that rebalance a position to
    a fixed fraction of *current total group value* ARE supported by
    from_orders (the same call `simulate_weights` uses for weight-based
    strategies), so entries/exits are converted here into a sparse order-size
    matrix: +POSITION_PCT on an admitted entry bar, 0.0 (flat) on an exit
    bar, NaN (no order) everywhere else.
    """
    entries, exits = targets.entries, targets.exits
    close = prices[entries.columns].reindex(entries.index).ffill()
    admitted = admission_filter(entries.to_numpy(), exits.to_numpy(), slot_cap)
    size = pd.DataFrame(np.nan, index=entries.index, columns=entries.columns)
    size[admitted] = POSITION_PCT
    size = size.mask(exits.to_numpy(), 0.0)
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="targetpercent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=_group(entries.shape[1], label),
        call_seq="auto",
    ).copy()
    value = pf.value()
    if isinstance(value, pd.DataFrame):
        value = value[label]
    return value, pf


def deployment_stats(pf: "vbt.Portfolio", label: str) -> dict:
    """Gross deployment (% of capital invested) and concurrent-position stats."""
    cash = pf.cash()
    value = pf.value()
    if isinstance(cash, pd.DataFrame):
        cash = cash[label]
    if isinstance(value, pd.DataFrame):
        value = value[label]
    deployed_pct = (1.0 - cash / value) * 100.0
    assets = pf.assets()
    if isinstance(assets.columns, pd.MultiIndex):
        assets = assets[label]
    n_open = (assets.abs() > 1e-9).sum(axis=1)
    return {
        "mean_deployment_pct": float(deployed_pct.mean()),
        "max_deployment_pct": float(deployed_pct.max()),
        "mean_concurrent": float(n_open.mean()),
        "max_concurrent": float(n_open.max()),
    }


def run(eval_start: str, eval_end: str, out_path: str) -> None:
    t0 = time.time()
    cfg = LabConfig()
    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    base_config = dict(STOCK_BASE_CONFIG)
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    members = {u: equity_universe_between(es, ee, universe=u) for u in SLEEVE_UNIVERSES}
    all_symbols = sorted({s for ms in members.values() for s in ms} | {"SPY"})
    print(f"[data] {len(all_symbols)} unique symbols across {SLEEVE_UNIVERSES}", flush=True)
    ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)
    available = set(ohlcv.columns.get_level_values(0))
    spy_close = ohlcv["SPY"]["close"].dropna()
    print(f"[data] loaded in {time.time() - t0:.1f}s", flush=True)

    sleeves: dict[str, SleeveData] = {}
    for universe in SLEEVE_UNIVERSES:
        label = f"ensemble@{universe}"
        syms = [x for x in members[universe] if x in available]
        t1 = time.time()
        targets = build_sleeve_targets(universe, cfg, ohlcv[syms], es, ee)
        prices = pd.concat({s: ohlcv[s]["close"] for s in syms}, axis=1)
        sleeves[label] = SleeveData(label, targets, prices)
        print(
            f"[sleeve] {label}: {targets.entries.shape[1]} symbols, "
            f"{int(targets.entries.to_numpy().sum())} entry signals, "
            f"{time.time() - t1:.1f}s",
            flush=True,
        )

    results: dict = {"regime_P": {}, "regime_F": {}}

    # --- Regime P ---
    sleeve_curves_p: dict[str, pd.Series] = {}
    for label, sd in sleeves.items():
        t1 = time.time()
        value, pf = simulate_percent(label, sd.targets, sd.prices, base_config)
        sleeve_curves_p[label] = value
        stats = curve_stats(value)
        dep = deployment_stats(pf, label)
        results["regime_P"][label] = {**stats, **dep}
        print(f"[P] {label}: Sharpe {stats['sharpe']:.2f} ({time.time() - t1:.1f}s)", flush=True)

    blend_eq_p, returns_df_p, diag_p = blend(sleeve_curves_p, base_config)
    results["regime_P"]["blend"] = {
        **curve_stats(blend_eq_p),
        "avg_leverage": float(diag_p["scale"].mean()),
        "max_leverage": float(diag_p["scale"].max()),
    }

    # --- Regime F, swept over slot caps ---
    for cap in SLOT_CAPS:
        sleeve_curves_f: dict[str, pd.Series] = {}
        cap_stats: dict = {}
        for label, sd in sleeves.items():
            t1 = time.time()
            value, pf = simulate_targetpercent(label, sd.targets, sd.prices, base_config, cap)
            sleeve_curves_f[label] = value
            stats = curve_stats(value)
            dep = deployment_stats(pf, label)
            cap_stats[label] = {**stats, **dep}
            dep_str = f"{dep['mean_deployment_pct']:.1f}/{dep['max_deployment_pct']:.1f}%"
            print(
                f"[F cap={cap}] {label}: Sharpe {stats['sharpe']:.2f} "
                f"deploy(mean/max)={dep_str} ({time.time() - t1:.1f}s)",
                flush=True,
            )
        blend_eq_f, returns_df_f, diag_f = blend(sleeve_curves_f, base_config)
        cap_stats["blend"] = {
            **curve_stats(blend_eq_f),
            "avg_leverage": float(diag_f["scale"].mean()),
            "max_leverage": float(diag_f["scale"].max()),
        }
        results["regime_F"][str(cap)] = cap_stats
        print(f"[F cap={cap}] blend Sharpe {cap_stats['blend']['sharpe']:.2f}", flush=True)

    # --- SPY benchmark over the common blend window ---
    common = returns_df_p.index
    spy_common = spy_close.reindex(common).ffill().dropna()
    results["spy"] = curve_stats(spy_common)

    results["meta"] = {
        "eval_start": eval_start,
        "eval_end": eval_end,
        "elapsed_sec": time.time() - t0,
        "n_symbols": len(all_symbols),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[done] {time.time() - t0:.1f}s total, wrote {out_path}", flush=True)


def blend(curves: dict[str, pd.Series], base_config: dict):
    from functools import reduce

    common = reduce(lambda a, b: a.intersection(b), (c.index for c in curves.values()))
    returns_df = pd.DataFrame(
        {label: curves[label].reindex(common).pct_change() for label in curves}
    ).dropna()
    blended, diag = combine_sleeves(returns_df, target_vol=0.068, window=60, max_leverage=1.0)
    start_cash = float(base_config["START_CASH"])
    blend_equity = (1.0 + blended).cumprod() * start_cash
    return blend_equity, returns_df, diag


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval-start", default="2021-01-31")
    p.add_argument("--eval-end", default="2026-04-30")
    p.add_argument("--out", default="scripts/_position_sizing_regimes_results.json")
    args = p.parse_args()
    run(args.eval_start, args.eval_end, args.out)
