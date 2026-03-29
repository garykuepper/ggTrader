"""Metric computation and gating for train windows."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _align_grouped_combo_series(agg: pd.Series, sh_index: pd.Index) -> pd.Series:
    """Map ``groupby`` on column MultiIndex levels onto metric index (e.g. 0 -> (0,))."""
    if agg.index.equals(sh_index):
        return agg
    if isinstance(sh_index, pd.MultiIndex):
        vals: list[float] = []
        for k in sh_index:
            v = float("nan")
            if k in agg.index:
                v = float(agg.loc[k])
            elif isinstance(k, tuple) and len(k) >= 1 and k[0] in agg.index:
                v = float(agg.loc[k[0]])
            vals.append(v if np.isfinite(v) else float("nan"))
        return pd.Series(vals, index=sh_index, dtype=float)
    return agg.reindex(sh_index)


def _trade_counts_for_train_gate(pf: Any, sharpe_series: Any) -> pd.Series:
    """Closed-trade counts indexed like ``sharpe_series`` (sum per param combo if needed).

    VectorBT usually aligns ``trades.count()`` with ``sharpe_ratio()`` for ``group_by``
    portfolios. If the raw count Series is per underlying column (MultiIndex), aggregate
    by summing across the symbol level so gating matches portfolio-level Sharpe.
    """
    sh = sharpe_series if isinstance(sharpe_series, pd.Series) else pd.Series([sharpe_series])
    raw = pf.trades.count()
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])

    # If lengths match exactly, assume positional alignment (common in vectorized group runs)
    if len(raw) == len(sh):
        return pd.Series(np.asarray(raw, dtype=float).ravel(), index=sh.index)

    if raw.index.equals(sh.index):
        return raw.astype(float)

    cols = pf.wrapper.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2 and len(raw) == len(cols):
        per_col = pd.Series(np.asarray(raw, dtype=float).ravel(), index=cols)
        agg = per_col.groupby(level=list(range(cols.nlevels - 1))).sum()
        aligned = _align_grouped_combo_series(agg, sh.index)
        return aligned.fillna(0.0).astype(float)
    out = raw.reindex(sh.index)
    return out.fillna(0.0).astype(float)


def _open_position_count_end_for_gate(pf: Any, sharpe_series: Any) -> pd.Series:
    """Open-position counts at last bar, aligned to ``sharpe_series.index`` (max per combo)."""
    sh = sharpe_series if isinstance(sharpe_series, pd.Series) else pd.Series([sharpe_series])
    try:
        raw = pf.positions.open.count()
    except Exception:
        return pd.Series(0.0, index=sh.index, dtype=float)
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])

    # If lengths match exactly, assume positional alignment
    if len(raw) == len(sh):
        return pd.Series(np.asarray(raw, dtype=float).ravel(), index=sh.index)

    if raw.index.equals(sh.index):
        return raw.astype(float)

    cols = pf.wrapper.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2 and len(raw) == len(cols):
        per_col = pd.Series(np.asarray(raw, dtype=float).ravel(), index=cols)
        agg = per_col.groupby(level=list(range(cols.nlevels - 1))).max()
        aligned = _align_grouped_combo_series(agg, sh.index)
        return aligned.fillna(0.0).astype(float)
    out = raw.reindex(sh.index)
    return out.fillna(0.0).astype(float)


def _calmar_ratio_series(pf_train: Any) -> pd.Series:
    """Per-combo Calmar-like ratio: total_return / abs(max_drawdown)."""
    tr = pf_train.total_return()
    mdd = pf_train.max_drawdown()
    if isinstance(tr, pd.Series) and isinstance(mdd, pd.Series):
        denom = mdd.abs().replace(0, np.nan)
        m = tr / denom
    else:
        tr_f = float(tr) if np.isfinite(float(tr)) else float("nan")
        mdd_f = float(mdd) if np.isfinite(float(mdd)) else float("nan")
        m = tr_f / abs(mdd_f) if mdd_f != 0 else float("nan")
        m = pd.Series([m])
    if not isinstance(m, pd.Series):
        m = pd.Series([m])
    return m


def _profit_factor_series(pf_train: Any) -> pd.Series:
    """Per-combo profit factor, mean-centred: (gross_profit/gross_loss - 1), clipped [-3, 3].

    0.0 = breakeven, 1.0 = 2:1 reward ratio, inf (all-win) clipped to 3.0.
    NaN (no trades) propagates — the outer ``m.where(sh_f.notna(), nan)`` gate then
    removes those combos, consistent with the Sharpe NaN convention.
    """
    raw = pf_train.trades.profit_factor()
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])
    return (raw - 1.0).clip(lower=-3.0, upper=3.0).astype(float)


def _zscore_normalize_series(s: pd.Series, eps: float = 1e-8) -> pd.Series:
    """Z-score normalize a Series across param combos; clip to [-3, 3].

    Guards: returns raw values unchanged when fewer than 2 finite elements or
    std < eps (all combos have identical metric — normalization meaningless).
    Non-finite inputs (NaN, inf, -inf) are excluded from mean/std and mapped
    to NaN in the output — using s.notna() instead of np.isfinite would let
    inf values contaminate the mean and collapse all z-scores to NaN.
    """
    finite_mask = pd.Series(np.isfinite(s.values), index=s.index)
    if int(finite_mask.sum()) < 2:
        return s
    vals = s[finite_mask]
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1))
    if sigma < eps:
        return s
    z = (s - mu).divide(sigma).clip(lower=-3.0, upper=3.0)
    # Non-finite inputs (NaN/inf) must stay NaN — do not clip inf to ±3.
    return z.where(finite_mask, other=float("nan"))


def _train_metric_series(pf_train: Any, config: Dict[str, Any]) -> pd.Series:
    """In-sample metric Series used to pick best params on the train window."""
    name = str(config.get("TRAIN_METRIC", "sharpe")).lower().strip()
    if name == "sortino":
        m = pf_train.sortino_ratio()
    elif name == "calmar":
        m = _calmar_ratio_series(pf_train)
    elif name == "composite":
        raw_w = config.get("TRAIN_METRIC_COMPOSITE_WEIGHTS") or {}
        ws = float(raw_w.get("sharpe", 1.0 / 3.0))
        wso = float(raw_w.get("sortino", 1.0 / 3.0))
        wc = float(raw_w.get("calmar", 1.0 / 3.0))
        wpf = float(raw_w.get("profit_factor", 0.0))
        s = ws + wso + wc + wpf
        if s <= 0:
            ws, wso, wc, wpf = 0.25, 0.25, 0.25, 0.25
        else:
            ws, wso, wc, wpf = ws / s, wso / s, wc / s, wpf / s
        sh = pf_train.sharpe_ratio()
        so = pf_train.sortino_ratio()
        ca = _calmar_ratio_series(pf_train)
        if not isinstance(sh, pd.Series):
            sh = pd.Series([float(sh)])
        if not isinstance(so, pd.Series):
            so = pd.Series([float(so)])
        so = so.reindex(sh.index)
        ca = ca.reindex(sh.index)
        # Clip calmar to [-5, 5] to prevent extreme values (near-zero drawdown) from
        # dominating the composite and artificially inflating a strategy's score.
        ca_clipped = ca.clip(lower=-5.0, upper=5.0)
        # When calmar is NaN but sharpe is valid (e.g. fold with zero drawdown),
        # treat calmar as 0 rather than propagating NaN to the whole composite.
        sh_f = sh.astype(float)
        so_f = so.astype(float)
        ca_fin = ca_clipped.where(ca_clipped.notna(), other=0.0)
        # profit_factor: (GP/GL - 1), clipped [-3, 3]; NaN → 0 (same as calmar convention).
        pf_s = _profit_factor_series(pf_train).reindex(sh.index)
        pf_fin = pf_s.where(pf_s.notna(), other=0.0)
        # Z-score each component to put Calmar, ProfitFactor, Sharpe on the same
        # scale before blending — prevents high-magnitude Calmar values from dominating.
        if config.get("TRAIN_METRIC_NORMALIZE_ZSCORE", True):
            sh_f   = _zscore_normalize_series(sh_f)
            so_f   = _zscore_normalize_series(so_f)
            ca_fin = _zscore_normalize_series(ca_fin)
            pf_fin = _zscore_normalize_series(pf_fin)
        m = ws * sh_f + wso * so_f + wc * ca_fin + wpf * pf_fin
        # Preserve NaN for combos where sharpe itself is NaN (no trades / gated out).
        m = m.where(sh_f.notna(), other=float("nan"))
    else:
        m = pf_train.sharpe_ratio()
    if not isinstance(m, pd.Series):
        m = pd.Series([m])
    return m


def _max_drawdown_for_train_gate(pf_train: Any, sharpe_series: Any) -> pd.Series:
    """Max drawdown per combo, aligned to sharpe index (more negative = deeper DD)."""
    sh = sharpe_series if isinstance(sharpe_series, pd.Series) else pd.Series([sharpe_series])
    raw = pf_train.max_drawdown()
    if not isinstance(raw, pd.Series):
        raw = pd.Series([float(raw)])
    if raw.index.equals(sh.index):
        return raw.astype(float)
    cols = pf_train.wrapper.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2 and len(raw) == len(cols):
        per_col = pd.Series(np.asarray(raw, dtype=float).ravel(), index=cols)
        agg = per_col.groupby(level=list(range(cols.nlevels - 1))).min()
        return _align_grouped_combo_series(agg, sh.index)
    return raw.reindex(sh.index)


def _apply_sensitivity_train_gates(
    sharpe_series: pd.Series,
    trade_for_gate: pd.Series,
    pf: Any,
    config: Dict[str, Any],
) -> pd.Series:
    """NaN Sharpe for combos that fail closed-trade / open-position gates."""
    out = sharpe_series.copy()
    min_closed = config.get("MIN_CLOSED_TRADES_TRAIN", 1)
    if min_closed > 0:
        incomplete_mask = trade_for_gate < min_closed
        if incomplete_mask.any():
            out = out.copy()
            out[incomplete_mask] = np.nan

    reject_open_lt = config.get("REJECT_OPEN_END_IF_CLOSED_LT", 0)
    if reject_open_lt > 0:
        try:
            open_end = _open_position_count_end_for_gate(pf, out)
            hold_mask = (open_end > 0) & (trade_for_gate < reject_open_lt)
            if hold_mask.any():
                out = out.copy()
                out[hold_mask] = np.nan
        except Exception:
            pass
    return out


def _print_wfo_fold_all_rejected_diagnostics(
    fold_idx: int,
    train_ohlcv: pd.DataFrame,
    train_metrics_before_gates: pd.Series,
    trade_for_gate: pd.Series,
    config: Dict[str, Any],
    pf_train: Any,
) -> None:
    """Log structured diagnostics when every train combo is NaN after gating."""
    n_bars = len(train_ohlcv)
    t0, t1 = train_ohlcv.index[0], train_ohlcv.index[-1]
    n_combo = len(train_metrics_before_gates)
    min_closed = int(config.get("MIN_CLOSED_TRADES_TRAIN", 1) or 0)

    tfg_alg = trade_for_gate.reindex(train_metrics_before_gates.index).fillna(0.0)
    tfg_arr = np.asarray(tfg_alg, dtype=float).ravel()
    raw_arr = np.asarray(train_metrics_before_gates, dtype=float).ravel()

    if min_closed > 0:
        pass_trade = tfg_arr >= float(min_closed)
    else:
        pass_trade = np.ones(len(tfg_arr), dtype=bool)
    n_pass_trade = int(np.sum(pass_trade))

    n_fin_on_pass = int(np.sum(np.isfinite(raw_arr[pass_trade])))
    raw_fin_total = int(np.sum(np.isfinite(raw_arr)))
    tfg_min = float(np.nanmin(tfg_arr)) if tfg_arr.size else 0.0
    tfg_max = float(np.nanmax(tfg_arr)) if tfg_arr.size else 0.0

    raw_tc = pf_train.trades.count()
    idx_match: Optional[bool] = None
    align_note = "trade_count_index=n/a"
    if isinstance(raw_tc, pd.Series) and isinstance(train_metrics_before_gates, pd.Series):
        idx_match = bool(raw_tc.index.equals(train_metrics_before_gates.index))
        align_note = f"trade_count_index_equals_metric_index={idx_match}"
        if not idx_match:
            align_note += " (_trade_counts_for_train_gate uses groupby/reindex)"

    if min_closed > 0 and n_pass_trade == 0:
        hypothesis = "all_combos_below_MIN_CLOSED_TRADES_TRAIN"
    elif n_pass_trade > 0 and n_fin_on_pass == 0:
        hypothesis = "TRAIN_METRIC_non_finite_despite_enough_closed_trades"
    elif n_pass_trade > 0 and n_fin_on_pass > 0:
        dd_on = config.get("MAX_TRAIN_DRAWDOWN_PCT") is not None
        open_on = int(config.get("REJECT_OPEN_END_IF_CLOSED_LT", 0) or 0) > 0
        if dd_on or open_on:
            hypothesis = "likely_drawdown_or_open_position_gate"
        else:
            hypothesis = "unexpected_all_nan_check_alignment"
    else:
        hypothesis = "unknown"

    print(
        f"  WARNING: Fold {fold_idx} - All param combos rejected after train gates. "
        f"Diagnostics: train_bars={n_bars} [{t0}..{t1}] combos={n_combo} | "
        f"closed_trades min={tfg_min:.0f} max={tfg_max:.0f} "
        f"count_ge_min_closed={n_pass_trade} (min_closed={min_closed}) | "
        f"raw_{config.get('TRAIN_METRIC', 'sharpe')}_finite_total={raw_fin_total} "
        f"finite_on_trade_ok={n_fin_on_pass} | {align_note} | hypothesis={hypothesis}"
    )
