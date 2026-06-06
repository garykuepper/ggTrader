"""Metric computation and gating for train windows."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
import vectorbt.returns.nb as _returns_nb


def _ann_factor_for(pf_train: Any) -> float:
    """vbt annualization factor = year_freq / freq (matches ReturnsAccessor.ann_factor)."""
    year_freq = vbt.settings.returns["year_freq"]
    return pd.Timedelta(year_freq) / pd.Timedelta(pf_train.wrapper.freq)


def _returns_based_metrics(pf_train: Any) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Per-combo (Sortino, total-return, max-drawdown) from ONE returns extraction.

    Replaces three independent vbt accessor calls — ``pf.sortino_ratio()``,
    ``pf.total_return()``, ``pf.max_drawdown()`` — each of which rebuilt the returns
    accessor and re-ran vbt's config/type machinery (~58% of WFO runtime; see
    docs/profiling_report_2026-06-05.md). Here ``pf.returns()`` is extracted once and fed
    to vbt's **own** numba kernels, so results are bit-identical to the accessors
    (verified incl. inf/NaN edge cases) while collapsing ~5 returns-accessor builds per
    fold to one. ~36x faster on fresh per-fold portfolios.

    Returns Series indexed by the portfolio's column labels (one entry per param combo).
    """
    ret = pf_train.returns()
    if isinstance(ret, pd.Series):
        ret = ret.to_frame()
    cols = ret.columns
    arr = np.asarray(ret.values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    ann = _ann_factor_for(pf_train)
    sortino = pd.Series(_returns_nb.sortino_ratio_nb(arr, ann), index=cols)
    max_dd = pd.Series(_returns_nb.max_drawdown_nb(arr), index=cols)
    total_ret = pd.Series(np.prod(1.0 + arr, axis=0) - 1.0, index=cols)
    return sortino, total_ret, max_dd


def _fold_stats_metrics(pf: Any) -> Dict[str, pd.Series]:
    """Sharpe + Sortino + total-return + max-drawdown (+ the returns frame) from ONE
    ``pf.returns()`` extraction, via vbt's own numba kernels.

    For the per-fold OOS/train diagnostic block in ``wfo._process_wfo_fold``, which
    previously made ~9 separate vbt accessor calls per fold (each rebuilding the returns
    accessor). Bit-identical to ``pf.sharpe_ratio()`` / ``sortino_ratio()`` /
    ``total_return()`` / ``max_drawdown()`` (verified incl. inf/NaN). The returned
    ``returns`` frame is the same object callers store as ``oos_returns`` (no re-call).
    """
    ret = pf.returns()
    ret_df = ret.to_frame() if isinstance(ret, pd.Series) else ret
    cols = ret_df.columns
    arr = np.asarray(ret_df.values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    ann = _ann_factor_for(pf)
    return {
        "sharpe": pd.Series(_returns_nb.sharpe_ratio_nb(arr, ann), index=cols),
        "sortino": pd.Series(_returns_nb.sortino_ratio_nb(arr, ann), index=cols),
        "total_return": pd.Series(np.prod(1.0 + arr, axis=0) - 1.0, index=cols),
        "max_drawdown": pd.Series(_returns_nb.max_drawdown_nb(arr), index=cols),
        "returns": ret,
    }


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


def _calmar_ratio_series(
    pf_train: Any,
    tr: Optional[pd.Series] = None,
    mdd: Optional[pd.Series] = None,
) -> pd.Series:
    """Per-combo Calmar-like ratio: total_return / abs(max_drawdown).

    ``tr`` / ``mdd`` may be supplied (from a shared ``_returns_based_metrics`` extraction)
    to avoid re-deriving them via separate vbt accessor calls; otherwise they are computed.
    """
    if tr is None or mdd is None:
        _, tr, mdd = _returns_based_metrics(pf_train)
    denom = mdd.abs().replace(0, np.nan)
    return tr / denom


def _profit_factor_series(pf_train: Any) -> pd.Series:
    """Per-combo profit factor, mean-centred: (gross_profit/gross_loss - 1), clipped [-3, 3].

    0.0 = breakeven, 1.0 = 2:1 reward ratio, inf (all-win) clipped to 3.0.
    NaN (no trades) propagates — the outer ``m.where(so.notna(), nan)`` gate then
    removes those combos, consistent with the Sortino NaN convention.
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


def _rank_composite_score(
    sortino: pd.Series, calmar: pd.Series, profit_factor: pd.Series
) -> pd.Series:
    """Rank-based composite (Sortino + Calmar + PF). No Sharpe — redundant with Sortino
    (Sortino is a strict refinement, only counts downside vol).

    Rank cells by each of the three ratios descending (rank 1 = best). Average rank on
    ties (standard statistical convention). Average the ranks across the three axes.
    Score = -mean_rank so "max wins" downstream is preserved. Cells with NaN Sortino
    (no trades / gated out) get NaN score, which propagates to selection drop.
    """
    # Clip to keep extreme values from distorting ranks of ties at the edges (rank itself
    # is scale-invariant, but clipping defends against +inf / -inf making ranks unstable).
    ca_clipped = calmar.clip(lower=-5.0, upper=5.0)
    pf_clipped = profit_factor.clip(lower=-3.0, upper=3.0)

    # rank(ascending=False) → highest value = rank 1, lowest = rank N.
    # method='average' gives tied cells the mean of their tied position range.
    # NaN values produce NaN ranks (skipped by mean below).
    r_so = sortino.rank(ascending=False, method="average")
    r_ca = ca_clipped.rank(ascending=False, method="average")
    r_pf = pf_clipped.rank(ascending=False, method="average")

    # Mean rank across the 3 axes. NaN in any axis -> NaN final score.
    mean_rank = pd.concat([r_so, r_ca, r_pf], axis=1).mean(axis=1, skipna=False)
    m = -mean_rank  # negate so higher score = better, matches downstream "max wins"
    # Force NaN where Sortino itself is NaN (no trades / gated out).
    return m.where(sortino.notna(), other=float("nan"))


def _train_metric_series(pf_train: Any, config: Dict[str, Any]) -> pd.Series:
    """In-sample metric Series used to pick best params on the train window."""
    name = str(config.get("TRAIN_METRIC", "sharpe")).lower().strip()
    # Sortino/Calmar/composite all derive from the same returns series — extract it once
    # and reuse (vbt's per-accessor overhead is ~58% of WFO runtime; see profiling report).
    if name in ("sortino", "calmar", "composite"):
        sortino, total_ret, max_dd = _returns_based_metrics(pf_train)
    if name == "sortino":
        m = sortino
    elif name == "calmar":
        m = _calmar_ratio_series(pf_train, tr=total_ret, mdd=max_dd)
    elif name == "composite":
        so = sortino
        ca = _calmar_ratio_series(pf_train, tr=total_ret, mdd=max_dd).reindex(so.index)
        pf_s = _profit_factor_series(pf_train).reindex(so.index)
        m = _rank_composite_score(so, ca, pf_s)
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
