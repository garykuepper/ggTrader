"""Walk-forward optimization: rolling train/test folds with composite scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Type

import pandas as pd

from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategy import LabConfig, SignalTargets
from ggTrader.lab.sweep import combo_name, split_params

TRAIN_MONTHS = 12
TEST_MONTHS = 3


class Fold(NamedTuple):
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> List[Fold]:
    """Rolling fixed-width folds. Slides forward by test_months each step."""
    folds: List[Fold] = []
    cursor = eval_start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > eval_end:
            break
        folds.append(Fold(cursor, train_end, train_end, test_end))
        cursor += pd.DateOffset(months=test_months)
    return folds


def _min_max_normalize(values: List[float]) -> List[float]:
    """Min-max scale to [0, 1]. Returns all 0.0 if min == max."""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def composite_score(metrics_list: List[Dict[str, float]]) -> List[float]:
    """Composite rank: 0.5*norm(sharpe) + 0.3*norm(sortino) - 0.2*norm(|maxdd|).

    NaN values are replaced with the worst value in each metric's range.
    """
    sharpes: List[float] = []
    sortinos: List[float] = []
    drawdowns: List[float] = []
    for m in metrics_list:
        sharpes.append(m.get("sharpe", float("nan")))
        sortinos.append(m.get("sortino", float("nan")))
        drawdowns.append(abs(m.get("max_drawdown_pct", 0.0)))

    def _floor_nan(vals: List[float]) -> List[float]:
        finite = [v for v in vals if not math.isnan(v)]
        floor = min(finite) if finite else 0.0
        return [floor if math.isnan(v) else v for v in vals]

    sharpes = _floor_nan(sharpes)
    sortinos = _floor_nan(sortinos)
    drawdowns = _floor_nan(drawdowns)

    ns = _min_max_normalize(sharpes)
    no = _min_max_normalize(sortinos)
    nd = _min_max_normalize(drawdowns)

    return [0.5 * ns[i] + 0.3 * no[i] - 0.2 * nd[i] for i in range(len(metrics_list))]


def _sweep_fold(
    strategy_name: str,
    strat_instance: Any,
    ohlcv: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run all combos on a single time window and return per-combo metrics.

    Signal generation uses all data up to window_end (for EMA warmup).
    Scoring uses only [window_start, window_end).
    Returns list of dicts with keys: 'combo', 'params', and all curve_stats keys.
    """
    ohlcv_window = ohlcv.loc[:window_end]
    symbols = sorted(ohlcv_window.columns.get_level_values(0).unique())
    prices = pd.concat(
        {s: ohlcv_window[s]["close"] for s in symbols},
        axis=1,
    )

    start_cash = float(base_config["START_CASH"])

    # Group by stop config (same logic as sweep.py)
    stop_groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for combo in grid:
        _, stop_p = split_params(combo)
        stop_key = tuple(sorted(stop_p.items()))
        stop_groups[stop_key].append(combo)

    all_eq: Dict[str, pd.Series] = {}
    for stop_key, group_combos in stop_groups.items():
        stop_config = dict(stop_key)
        signal_combos = [split_params(c)[0] for c in group_combos]
        seen: set = set()
        unique_signal: List[Dict[str, Any]] = []
        for sc in signal_combos:
            k = tuple(sorted(sc.items()))
            if k not in seen:
                seen.add(k)
                unique_signal.append(sc)
        targets = strat_instance.sweep_signals(unique_signal, symbols, ohlcv_window)
        group_targets: Dict[str, SignalTargets] = {}
        for combo in group_combos:
            signal_p, _ = split_params(combo)
            signal_key = combo_name(strategy_name, signal_p)
            full_key = combo_name(strategy_name, combo)
            group_targets[full_key] = targets[signal_key]

        sim_config = {**base_config, **stop_config}
        ohlcv_arg = ohlcv_window if "atr_mult" in stop_config else None
        _rets, eq, _diag = simulate_signals(group_targets, prices, sim_config, ohlcv=ohlcv_arg)
        for key in group_targets:
            all_eq[key] = eq[key]

    # Score each combo over the scoring window only
    results: List[Dict[str, Any]] = []
    for key, eq_series in all_eq.items():
        eq_window = eq_series.loc[window_start:window_end].dropna()
        if len(eq_window) < 2:
            continue
        # Rescale to start at start_cash for consistent metrics
        eq_scaled = start_cash * (eq_window / eq_window.iloc[0])
        metrics = curve_stats(eq_scaled)
        combo_params = next(c for c in grid if combo_name(strategy_name, c) == key)
        results.append({"combo": key, "params": combo_params, **metrics})
    return results


def run_wfo(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
) -> str:
    """Main WFO entry point: fold, train, test, concatenate, report."""
    eval_start_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    folds = generate_folds(eval_start_ts, eval_end_ts)
    if not folds:
        return (
            f"WFO: {strategy_name} | no valid folds (need >= {TRAIN_MONTHS + TEST_MONTHS} months)"
        )

    strat_instance = strategy_cls(cfg)
    start_cash = float(base_config["START_CASH"])
    fold_results: List[Dict[str, Any]] = []
    oos_curves: List[pd.Series] = []
    oos_running_value = start_cash
    fold_winners: List[Dict[str, Any]] = []

    for i, fold in enumerate(folds):
        # Train: sweep all combos on train window
        train_metrics = _sweep_fold(
            strategy_name,
            strat_instance,
            ohlcv,
            fold.train_start,
            fold.train_end,
            base_config,
            grid,
        )
        if not train_metrics:
            continue

        # Pick winner by composite score
        scores = composite_score(train_metrics)
        best_idx = max(range(len(scores)), key=lambda j: scores[j])
        winner = train_metrics[best_idx]

        # Test: simulate winner only on data up to test_end, score test window
        winner_grid = [winner["params"]]
        test_metrics = _sweep_fold(
            strategy_name,
            strat_instance,
            ohlcv,
            fold.test_start,
            fold.test_end,
            base_config,
            winner_grid,
        )
        oos_score = 0.0
        if test_metrics:
            oos_score = composite_score(test_metrics)[0]

            # Build continuous OOS equity curve
            test_ohlcv = ohlcv.loc[: fold.test_end]
            symbols = sorted(test_ohlcv.columns.get_level_values(0).unique())
            prices = pd.concat({s: test_ohlcv[s]["close"] for s in symbols}, axis=1)
            signal_combos = [split_params(winner["params"])[0]]
            _, stop_p = split_params(winner["params"])
            targets = strat_instance.sweep_signals(signal_combos, symbols, test_ohlcv)
            key = combo_name(strategy_name, signal_combos[0])
            full_key = combo_name(strategy_name, winner["params"])
            sim_config = {**base_config, **stop_p}
            ohlcv_arg = test_ohlcv if "atr_mult" in stop_p else None
            _r, eq, _d = simulate_signals(
                {full_key: targets[key]}, prices, sim_config, ohlcv=ohlcv_arg
            )
            eq_test = eq[full_key].loc[fold.test_start : fold.test_end].dropna()
            if len(eq_test) > 0:
                normalized = oos_running_value * (eq_test / eq_test.iloc[0])
                oos_curves.append(normalized)
                oos_running_value = float(normalized.iloc[-1])

        fold_results.append(
            {
                "fold_num": i + 1,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "winner_combo": winner["combo"],
                "winner_params": winner["params"],
                "train_score": scores[best_idx],
                "oos_score": oos_score,
            }
        )
        fold_winners.append(winner)

    # Concatenate OOS curves and score
    if oos_curves:
        oos_equity = pd.concat(oos_curves)
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
        oos_metrics = curve_stats(oos_equity)
        spy_oos = spy_close.reindex(oos_equity.index).ffill().dropna()
        if len(spy_oos) > 1:
            spy_curve = start_cash * (spy_oos / spy_oos.iloc[0])
            spy_metrics = curve_stats(spy_curve)
        else:
            spy_metrics = {
                "sharpe": float("nan"),
                "cagr_pct": float("nan"),
                "max_drawdown_pct": float("nan"),
            }
    else:
        oos_metrics = {
            "sharpe": float("nan"),
            "cagr_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
        }
        spy_metrics = oos_metrics.copy()

    # Recommended live params
    live = select_live_params(
        strategy_name,
        strategy_cls,
        cfg,
        ohlcv,
        eval_end,
        base_config,
        grid,
        fold_winners,
    )

    table = format_wfo_table(
        fold_results,
        oos_metrics,
        spy_metrics,
        live,
        strategy_name,
        len(grid),
        len(folds),
    )
    print(table)
    return table


def select_live_params(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    eval_end: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    fold_winners: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Train on the most recent TRAIN_MONTHS window and pick the composite winner."""
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    live_train_start = eval_end_ts - pd.DateOffset(months=TRAIN_MONTHS)
    strat_instance = strategy_cls(cfg)

    train_metrics = _sweep_fold(
        strategy_name,
        strat_instance,
        ohlcv,
        live_train_start,
        eval_end_ts,
        base_config,
        grid,
    )
    if not train_metrics:
        return {"combo": "none", "params": {}, "train_metrics": {}, "stability": 0}

    scores = composite_score(train_metrics)
    best_idx = max(range(len(scores)), key=lambda j: scores[j])
    winner = train_metrics[best_idx]

    # Stability: count how many WFO folds selected the same combo
    stability = sum(1 for fw in fold_winners if fw["combo"] == winner["combo"])

    return {
        "combo": winner["combo"],
        "params": winner["params"],
        "train_metrics": {
            k: winner[k] for k in ("sharpe", "cagr_pct", "max_drawdown_pct") if k in winner
        },
        "stability": stability,
        "train_start": live_train_start,
        "train_end": eval_end_ts,
    }


def format_wfo_table(
    fold_results: List[Dict[str, Any]],
    oos_metrics: Dict[str, float],
    spy_metrics: Dict[str, float],
    live_params: Dict[str, Any],
    strategy_name: str,
    n_combos: int,
    n_folds: int,
) -> str:
    """Render per-fold table + OOS aggregate + recommended live params."""
    lines = [
        f"WFO: {strategy_name} | {n_combos} combos x {n_folds} folds"
        f" | rolling {TRAIN_MONTHS}mo/{TEST_MONTHS}mo",
        "",
        f"{'Fold':<6}{'Train Window':<20}{'Test Window':<20}{'Winner':<36}{'Train':>7}{'OOS':>7}",
        "─" * 96,
    ]
    for r in fold_results:
        ts = r["train_start"].strftime("%Y-%m")
        te = r["train_end"].strftime("%Y-%m")
        os_ = r["test_start"].strftime("%Y-%m")
        oe = r["test_end"].strftime("%Y-%m")
        # Shorten winner name: strip strategy prefix
        short = r["winner_combo"].replace(f"{strategy_name}__", "")
        if len(short) > 34:
            short = short[:31] + "..."
        lines.append(
            f"{r['fold_num']:<6}{ts} → {te:<13}{os_} → {oe:<13}"
            f"{short:<36}{r['train_score']:>7.2f}{r['oos_score']:>7.2f}"
        )

    lines.append("")
    lines.append(
        f"OOS Aggregate: Sharpe {oos_metrics.get('sharpe', float('nan')):.2f}"
        f" | CAGR {oos_metrics.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {oos_metrics.get('max_drawdown_pct', float('nan')):.1f}%"
    )
    lines.append(
        f"SPY baseline:  Sharpe {spy_metrics.get('sharpe', float('nan')):.2f}"
        f" | CAGR {spy_metrics.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {spy_metrics.get('max_drawdown_pct', float('nan')):.1f}%"
    )

    # Recommended live params
    lines.append("")
    lines.append("── Recommended Live Params " + "─" * 71)
    ts = live_params.get("train_start")
    te = live_params.get("train_end")
    ts_str = ts.strftime("%Y-%m") if ts else "?"
    te_str = te.strftime("%Y-%m") if te else "?"
    lines.append(f"Train window: {ts_str} → {te_str}")
    lines.append(f"Winner:       {live_params.get('combo', 'none')}")
    tm = live_params.get("train_metrics", {})
    lines.append(
        f"Train Sharpe: {tm.get('sharpe', float('nan')):.2f}"
        f" | CAGR {tm.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {tm.get('max_drawdown_pct', float('nan')):.1f}%"
    )
    lines.append(
        f"Stability:    selected in {live_params.get('stability', 0)}/{len(fold_results)} folds"
    )

    return "\n".join(lines)
