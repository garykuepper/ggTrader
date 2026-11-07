"""WFO result container, the printed table (format unchanged) and the JSON payload."""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Tuple

import pandas as pd

from ggTrader.engine import stats
from ggTrader.lab.wfo.anchor import AnchorSet
from ggTrader.lab.wfo.folds import TEST_MONTHS, TRAIN_MONTHS

NAN = float("nan")


class WfoResult(NamedTuple):
    """Structured result of a walk-forward run (the table is also printed).

    The first five fields are the legacy ``lab.wfo.WfoResult``; ``diagnostics`` is new and
    carries simulation counts, timings, peak memory, the per-fold OD12 comparison and the
    the engine and the statistic definitions (vectorbt's) the run used.
    """

    oos_equity: pd.Series
    fold_results: List[Dict[str, Any]]
    live_params: Dict[str, Any]
    table: str
    skipped_folds: Tuple[Dict[str, Any], ...] = ()
    diagnostics: Dict[str, Any] = {}


def curve_summary(curve: pd.Series) -> Dict[str, float]:
    """``curve_stats``-style summary of an equity curve through ``engine.stats`` (percent units)."""
    curve = curve.dropna()
    empty = {
        "total_return_pct": NAN,
        "cagr_pct": NAN,
        "sharpe": NAN,
        "sortino": NAN,
        "ann_vol_pct": NAN,
        "max_drawdown_pct": NAN,
    }
    if len(curve) < 2:
        return empty
    rets = curve.pct_change(fill_method=None)
    # a measurement of a deployed book, not a selection window: only the flat-curve guard
    table = stats.window_stats(rets, 0, len(curve), min_losing_days=0).iloc[:, 0]
    window = rets.iloc[1:]
    return {
        "total_return_pct": float(stats.total_return(window)) * 100,
        "cagr_pct": float(table["cagr"]) * 100,
        "sharpe": float(table["sharpe"]),
        "sortino": float(table["sortino"]),
        "ann_vol_pct": float(table["volatility"]) * 100,
        "max_drawdown_pct": float(table["max_drawdown"]) * 100,
    }


def anchor_history(fold_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One record per fold: the anchor fit up to that fold's ``train_end``, what was deployed
    and why the anchor was used (``None`` when the winner was deployed)."""
    return [
        {
            "fold": r["fold_num"],
            "anchor_combo": r.get("anchor_combo"),
            "anchor_params": r.get("anchor_params"),
            "deployed_combo": r.get("deployed_combo"),
            "reason": r.get("anchor_reason"),
        }
        for r in fold_results
    ]


def _varying_keys(param_sets: List[Dict[str, Any]]) -> List[str]:
    """Parameter names whose value is not the same in every set (the full set is in the JSON)."""
    keys = sorted({k for p in param_sets for k in p})
    return [k for k in keys if len({repr(p.get(k)) for p in param_sets}) > 1]


def format_wfo_table(
    fold_results: List[Dict[str, Any]],
    oos_metrics: Dict[str, float],
    spy_metrics: Dict[str, float],
    live_params: Dict[str, Any],
    strategy_name: str,
    n_combos: int,
    n_folds: int,
    halted: bool = False,
    anchor: AnchorSet | None = None,
    anchor_history: List[Dict[str, Any]] | None = None,
) -> str:
    """Render per-fold table + OOS aggregate + recommended live params.

    ``anchor_history`` (optional) appends a separate "Anchor Per Fold" section after the legacy
    sections; without it the text is byte-for-byte the legacy table, and the section's lines do
    not start with a digit, so callers that parse the fold rows are unaffected.
    """
    lines = [
        f"WFO: {strategy_name} | {n_combos} combos x {n_folds} folds"
        f" | rolling {TRAIN_MONTHS}mo/{TEST_MONTHS}mo",
        "",
        f"{'Fold':<6}{'Train Window':<20}{'Test Window':<20}"
        f"{'Winner':<26}{'IS_SR':>6}{'OOS_SR':>6}  {'Gate':<4} {'WFE':>5}",
        "─" * 96,
    ]
    for r in fold_results:
        ts = r["train_start"].strftime("%Y-%m")
        te = r["train_end"].strftime("%Y-%m")
        os_ = r["test_start"].strftime("%Y-%m")
        oe = r["test_end"].strftime("%Y-%m")
        short = r["winner_combo"].replace(f"{strategy_name}__", "")
        if len(short) > 24:
            short = short[:21] + "..."
        gate_str = "PASS" if r.get("gates_passed", False) else "FAIL"
        wfe_str = f"{r['wfe']:.2f}" if r.get("wfe") is not None else " n/a"
        halt_str = " [H]" if r.get("halted") else ""
        anchor_str = " [A]" if r.get("used_anchor") else ""
        # Both columns are Sharpe so the adjacent WFE (= OOS/IS Sharpe) can be verified by eye.
        oos_val = r.get("oos_sharpe", NAN)
        is_val = r.get("is_sharpe", NAN)
        oos_cell = f"{oos_val:>6.2f}" if oos_val == oos_val else f"{'n/a':>6}"
        is_cell = f"{is_val:>6.2f}" if is_val == is_val else f"{'n/a':>6}"
        lines.append(
            f"{r['fold_num']:<6}{ts} → {te:<13}{os_} → {oe:<13}"
            f"{short:<26}{is_cell}{oos_cell}"
            f"  {gate_str:<4} {wfe_str:>5}{halt_str}{anchor_str}"
        )

    lines.append("")
    lines.append(
        f"OOS Aggregate: Sharpe {oos_metrics.get('sharpe', NAN):.2f}"
        f" | CAGR {oos_metrics.get('cagr_pct', NAN):.1f}%"
        f" | MaxDD {oos_metrics.get('max_drawdown_pct', NAN):.1f}%"
    )
    lines.append(
        f"SPY baseline:  Sharpe {spy_metrics.get('sharpe', NAN):.2f}"
        f" | CAGR {spy_metrics.get('cagr_pct', NAN):.1f}%"
        f" | MaxDD {spy_metrics.get('max_drawdown_pct', NAN):.1f}%"
    )

    valid_wfes = [r["wfe"] for r in fold_results if r.get("wfe") is not None]
    avg_wfe = sum(valid_wfes) / len(valid_wfes) if valid_wfes else NAN
    lines.append(f"Aggregate WFE: {avg_wfe:.2f} (target >= 0.50)")
    if halted:
        lines.append("!! REGIME HALT ACTIVE -- trading on anchor params")

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
        f"Train Sharpe: {tm.get('sharpe', NAN):.2f}"
        f" | CAGR {tm.get('cagr_pct', NAN):.1f}%"
        f" | MaxDD {tm.get('max_drawdown_pct', NAN):.1f}%"
    )
    lines.append(
        f"Stability:    selected in {live_params.get('stability', 0)}/{len(fold_results)} folds"
    )

    if anchor is not None:
        lines.append("")
        lines.append("── Anchor Set (Defensive Fallback) " + "─" * 63)
        lines.append(f"Combo:    {anchor.combo}")
        lines.append(
            f"MaxDD:    {anchor.max_drawdown_pct:.1f}%"
            f" | CAGR {anchor.cagr_pct:.1f}%"
            f" | Sharpe {anchor.sharpe:.2f}"
        )

    if anchor_history:
        lines.append("")
        lines.append("── Anchor Per Fold (fit on data up to each train_end) " + "─" * 44)
        varying = _varying_keys([rec.get("anchor_params") or {} for rec in anchor_history])
        for rec in anchor_history:
            params = rec.get("anchor_params") or {}
            shown = " ".join(f"{k}={params.get(k)}" for k in varying) or "(same set every fold)"
            reason = rec["reason"] or "winner deployed"
            lines.append(f"  fold {rec['fold']:<3} [{reason}] {shown}")

    return "\n".join(lines)


def result_payload(result: WfoResult, **extra: Any) -> Dict[str, Any]:
    """JSON-able run record (everything but provenance, which ``write_result_json`` stamps)."""
    rows = list(result.fold_results)
    gates = sum(1 for r in rows if r["gates_passed"])
    return {
        "n_folds": len(rows),
        "gates": f"{gates}/{len(rows)}",
        "anchor_folds": sum(1 for r in rows if r["used_anchor"]),
        "winners": [r["winner_combo"] for r in rows],
        "fold_table": rows,
        "anchor_history": anchor_history(rows),
        "live_params": result.live_params,
        "skipped_folds": list(result.skipped_folds),
        "table": result.table,
        "oos_bars": int(len(result.oos_equity)),
        "oos_first": None if result.oos_equity.empty else result.oos_equity.index[0],
        "oos_last": None if result.oos_equity.empty else result.oos_equity.index[-1],
        "diagnostics": result.diagnostics,
        **extra,
    }
