"""Walk-Forward Optimization fold processing and orchestration."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.metrics import (
    _max_drawdown_for_train_gate,
    _open_position_count_end_for_gate,
    _print_wfo_fold_all_rejected_diagnostics,
)
from ggTrader.core.orchestrator_utils import (
    _coerce_metric_float,
    _default_params_from_grid,
    _eta_str,
    _extract_params,
    _to_native,
    _wall_clock_eta,
)
from ggTrader.core.sensitivity import _vectorized_grid_metrics
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.setup import load_data_with_movers


def _process_wfo_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ohlcv: pd.DataFrame,
    mover_mask: Optional[pd.DataFrame],
    param_grid: Dict[str, Any],
    config: Dict[str, Any],
    show_progress: bool,
    param_names: List[str],
    btc_regime_mask: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Helper to process a single WFO fold (Train & Test).

    Train window uses the vectorized registry path so ENTRY_STRATEGY /
    EXIT_STRATEGY in ``config`` are honoured.  Metric alignment relies on
    ``_vectorized_grid_metrics``.  Falls back to the legacy non-vectorized
    SignalFactory path if the vectorized pass raises.
    """
    train_ohlcv = ohlcv.loc[train_idx]
    test_ohlcv = ohlcv.loc[test_idx]

    # Skip folds where the coin has too few real price bars (e.g. newly listed coins).
    # Newly listed coins have NaN close prices for dates before they existed, while
    # illiquid-but-real coins still have non-NaN prices. Volume is not used because
    # low-volume coins on illiquid venues (e.g. Kraken) may have legitimate zero-volume
    # bars and would be wrongly skipped.
    _min_fold_bars = int(config.get("MIN_FOLD_BARS", 50))
    _close_col = next((c for c in ("Close", "close") if c in train_ohlcv.columns), None)
    if _close_col is not None:
        _active_bars = int(train_ohlcv[_close_col].notna().sum())
    else:
        _active_bars = len(train_ohlcv)
    if _active_bars < _min_fold_bars:
        return {
            "fold_idx": fold_idx,
            "best_params": {},
            "oos_sharpe": float("nan"),
            "oos_return": float("nan"),
            "oos_is_bear": False,
            "profit": float("nan"),
            "start_capital": float("nan"),
            "end_capital": float("nan"),
            "return_pct": float("nan"),
            "train_metrics": pd.Series(dtype=float),
            "oos_returns": pd.DataFrame(),
            "_skipped_insufficient_bars": True,
        }

    train_mask = mover_mask.loc[train_idx] if mover_mask is not None else None
    test_mask = mover_mask.loc[test_idx] if mover_mask is not None else None
    train_regime = (
        btc_regime_mask.reindex(train_idx, fill_value=False)
        if btc_regime_mask is not None else None
    )
    test_regime = (
        btc_regime_mask.reindex(test_idx, fill_value=False) if btc_regime_mask is not None else None
    )

    # Try vectorized train path first (honours ENTRY/EXIT_STRATEGY).
    wfo_train_cfg = {**config, "USE_VECTORIZED": True}
    train_metrics: pd.Series
    pf_train: Any = None
    try:
        train_engine = FastBacktest(
            train_ohlcv, param_grid, config=wfo_train_cfg, mover_mask=train_mask,
            regime_mask=train_regime,
        )
        pf_train = train_engine.run(show_progress=show_progress)
        train_metrics, trade_for_gate = _vectorized_grid_metrics(
            pf_train, train_engine, param_names, config
        )
        train_metrics_before_gates = train_metrics.copy()
    except Exception as vec_exc:
        # When the vectorized path fails, skip this fold rather than running the slow
        # SignalFactory fallback (which takes 100x longer and still produces 0 trades
        # for data-quality issues like newly listed coins or shape mismatches).
        print(
            f"  WFO fold {fold_idx}: vectorized train failed ({vec_exc!r}), skipping fold."
        )
        return {
            "fold_idx": fold_idx,
            "best_params": {},
            "oos_sharpe": float("nan"),
            "oos_return": float("nan"),
            "oos_is_bear": False,
            "profit": float("nan"),
            "start_capital": float("nan"),
            "end_capital": float("nan"),
            "return_pct": float("nan"),
            "train_metrics": pd.Series(dtype=float),
            "oos_returns": pd.DataFrame(),
            "_skipped_vectorized_failure": True,
        }

    # Bear Market check: if Buy & Hold return of the fold is negative.
    try:
        train_close = train_ohlcv.xs("close", axis=1, level=1)
        bnh_return = float((train_close.iloc[-1].mean() / train_close.iloc[0].mean()) - 1.0)
        is_bear_market = bnh_return < 0.0
    except Exception:
        is_bear_market = False

    # Gate: require MIN_CLOSED_TRADES_TRAIN completed round-trips (per combo).
    # Build a single rejection mask to avoid repeated defensive .copy() calls.
    nan_mask = pd.Series(False, index=train_metrics.index)
    zero_mask = pd.Series(False, index=train_metrics.index)

    min_closed = config.get("MIN_CLOSED_TRADES_TRAIN", 1)
    if min_closed > 0:
        incomplete_mask = trade_for_gate < min_closed

        # If it's a bear market, forgive combos with exactly 0 trades (stayed out).
        if is_bear_market:
            zero_trades_mask = trade_for_gate == 0
            incomplete_mask = incomplete_mask & ~zero_trades_mask
            zero_mask |= zero_trades_mask

        nan_mask |= incomplete_mask

    # Optional: reject combos with train drawdown deeper than -MAX_TRAIN_DRAWDOWN_PCT%.
    dd_limit = config.get("MAX_TRAIN_DRAWDOWN_PCT")
    if dd_limit is not None:
        try:
            mdd = _max_drawdown_for_train_gate(pf_train, train_metrics)
            nan_mask |= mdd < -(float(dd_limit) / 100.0)
        except Exception:
            pass

    # Optional stricter tier (REJECT_OPEN_END_IF_CLOSED_LT, default 0 = off).
    reject_open_lt = config.get("REJECT_OPEN_END_IF_CLOSED_LT", 0)
    if reject_open_lt > 0:
        try:
            open_end = _open_position_count_end_for_gate(pf_train, train_metrics)
            nan_mask |= (open_end > 0) & (trade_for_gate < reject_open_lt)
        except Exception:
            pass

    # Apply all gates in one copy
    if nan_mask.any() or zero_mask.any():
        train_metrics = train_metrics.copy()
        if zero_mask.any():
            train_metrics[zero_mask] = 0.0
        if nan_mask.any():
            train_metrics[nan_mask] = np.nan

    if train_metrics.isnull().all():
        _print_wfo_fold_all_rejected_diagnostics(
            fold_idx,
            train_ohlcv,
            train_metrics_before_gates,
            trade_for_gate,
            config,
            pf_train,
        )
        best_param_idx = train_metrics.index[0]
    else:
        best_param_idx = train_metrics.idxmax()

    fold_best_params = _extract_params(best_param_idx, train_metrics, param_names, param_grid)

    # Test (single winner): kept for downstream stats reporting (oos_sharpe, sortino, profit, etc.).
    # USE_VECTORIZED False is fine here (scalar params).
    wfo_test_cfg = {**config, "USE_VECTORIZED": False}
    test_engine = FastBacktest(
        test_ohlcv, fold_best_params, config=wfo_test_cfg, mover_mask=test_mask,
        regime_mask=test_regime,
    )
    pf_test = test_engine.run(show_progress=show_progress)

    # Bear-market detection on the OOS test window — used by fold-consistency
    # scoring downstream so a no-fire fold during a bear market doesn't count
    # against the strategy (correct sit-out), but a no-fire fold during a bull
    # market does (missed opportunity).
    try:
        test_close = test_ohlcv.xs("close", axis=1, level=1)
        oos_bnh_return = float((test_close.iloc[-1].mean() / test_close.iloc[0].mean()) - 1.0)
        oos_is_bear = oos_bnh_return < 0.0
    except Exception:
        oos_is_bear = False

    return {
        "fold": fold_idx,
        "train_start": str(train_ohlcv.index[0]),
        "test_start": str(test_ohlcv.index[0]),
        "test_end": str(test_ohlcv.index[-1]),
        "params": _to_native(fold_best_params),
        "is_sharpe": _to_native(train_metrics.max()),
        "oos_sharpe": _to_native(pf_test.sharpe_ratio().mean()),
        "oos_is_bear": bool(oos_is_bear),
        "sortino": _to_native(pf_test.sortino_ratio().mean()),
        "profit": _to_native(pf_test.total_profit().sum()),
        "start_capital": _to_native(pf_test.init_cash.sum()),
        "end_capital": _to_native(pf_test.value().iloc[-1].sum()),
        "return_pct": _to_native(pf_test.total_return().mean() * 100),
        "train_metrics": train_metrics,
        "oos_returns": pf_test.returns(),
    }


def _wfo_train_metric_row_key(idx: Any) -> tuple[Any, ...]:
    """Hashable row label for train-metric Series (flatten MultiIndex tuples)."""
    if isinstance(idx, tuple):
        return tuple(idx)
    return (idx,)


def _weighted_robustness_series(
    is_metrics_by_fold: Dict[int, pd.Series],
    weights: Dict[int, float],
    config: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """Fold-weighted mean per param combo keyed by flattened tuple rows.

    Avoids ``pd.DataFrame({fold: Series})`` on mixed MultiIndex depths (pandas can
    assert). Vectorized vs legacy WFO paths may return different row counts; we
    take the **union** of tuple keys and average each fold's value with fold
    weights, skipping NaNs (denominator = sum of weights for folds with finite
    values at that key).
    """
    fold_keys = sorted(weights.keys())
    if not fold_keys:
        raise ValueError("weights must be non-empty")

    flat: Dict[int, pd.Series] = {}
    for fk in fold_keys:
        s = is_metrics_by_fold[fk]
        keys = [_wfo_train_metric_row_key(i) for i in s.index]
        vals = np.asarray(s, dtype=float).ravel()
        if len(keys) != len(vals):
            raise ValueError(f"WFO fold {fk}: index/values length mismatch")
        flat[fk] = pd.Series(vals, index=pd.Index(keys, dtype=object))

    all_keys: set = set()
    for s in flat.values():
        all_keys.update(s.index.tolist())
    if not all_keys:
        return pd.Series(dtype=float)

    union_list = sorted(all_keys)
    union_idx = pd.Index(union_list, dtype=object)

    wvec = np.array([float(weights[fk]) for fk in fold_keys], dtype=float)
    if not np.any(wvec > 0):
        wvec = np.ones(len(fold_keys), dtype=float)

    # Build a (n_keys × n_folds) matrix via reindex; NaN where a key is absent in a fold.
    # Pass union_idx (dtype=object) instead of the raw list — reindexing against a plain
    # list of tuples triggers pandas MultiIndex.from_tuples(), which crashes on 1-element
    # tuples (single-param grids like donchian_breakout).
    n_keys = len(union_list)
    n_folds = len(fold_keys)
    mat = np.full((n_keys, n_folds), np.nan, dtype=float)
    for j, fk in enumerate(fold_keys):
        s = flat[fk]
        aligned = s.reindex(union_idx)  # NaN for missing keys
        mat[:, j] = aligned.to_numpy(dtype=float, copy=False)

    # Vectorized weighted mean ignoring NaNs
    finite_mask = np.isfinite(mat)                          # (n_keys, n_folds)
    weighted_vals = np.where(finite_mask, mat * wvec, 0.0)  # zero out non-finite
    weighted_wts  = np.where(finite_mask, wvec, 0.0)        # zero out missing weights
    den = weighted_wts.sum(axis=1)
    num = weighted_vals.sum(axis=1)
    combined = np.where(den > 0.0, num / den, np.nan)

    # Per-fold z-rank blend (Step 1.5 of WFO overfitting work).
    # alpha=0 reproduces the raw weighted-mean behavior above. alpha>0 mixes in a
    # weighted mean of per-fold z-scores so cells that rank consistently high across
    # folds beat cells that spike in one fold and average elsewhere.
    alpha = float((config or {}).get("PARAM_ZRANK_WEIGHT", 0.0))
    if alpha > 0.0:
        z_mat = np.full_like(mat, np.nan)
        for j in range(n_folds):
            col = mat[:, j]
            col_mask = np.isfinite(col)
            if col_mask.sum() < 2:
                continue  # need >= 2 cells for std
            col_finite = col[col_mask]
            col_std = col_finite.std()
            if col_std == 0.0:
                continue  # degenerate fold: all cells equal, leave z's NaN
            col_mean = col_finite.mean()
            z_mat[col_mask, j] = (col_finite - col_mean) / col_std
        z_finite = np.isfinite(z_mat)
        z_weighted_vals = np.where(z_finite, z_mat * wvec, 0.0)
        z_weighted_wts = np.where(z_finite, wvec, 0.0)
        z_den = z_weighted_wts.sum(axis=1)
        z_num = z_weighted_vals.sum(axis=1)
        zrank = np.where(z_den > 0.0, z_num / z_den, np.nan)
        # Blend; if one side is NaN at a cell, fall back to the finite side.
        both_finite = np.isfinite(combined) & np.isfinite(zrank)
        combined = np.where(
            both_finite,
            (1.0 - alpha) * combined + alpha * zrank,
            np.where(np.isfinite(combined), combined, zrank),
        )

    return pd.Series(combined, index=union_idx)


def _calculate_oos_robustness(
    oos_metrics_by_fold: Dict[int, float],
    config: Optional[Dict[str, Any]] = None,
    oos_bear_by_fold: Optional[Dict[int, bool]] = None,
) -> Tuple[float, float]:
    """Recency-weighted mean OOS Sharpe and bear-aware fold consistency.

    Fold consistency semantics (with ``oos_bear_by_fold`` provided):
        - Fold fired (Sharpe is finite): counts in denominator; positive Sharpe = profitable.
        - Fold didn't fire AND that fold was a bear market: forgiven (excluded from denominator).
          Correct behaviour is "stay out during a downtrend"; not firing is correct, not penalised.
        - Fold didn't fire AND that fold was a bull market: missed opportunity, counts against
          consistency (in denominator, not profitable). Without this branch a sparse-fire
          strategy can score 100% consistency from a few lucky bull entries.

    When ``oos_bear_by_fold`` is None, falls back to the legacy "firing folds only" denominator
    (no-fire folds excluded regardless of regime).

    Args:
        oos_metrics_by_fold: Dict mapping fold index → OOS Sharpe for the winning params.
        config: Optional run config; reads OOS_STABILITY_WEIGHT (default 0.3).
        oos_bear_by_fold: Dict mapping fold index → True if that fold's OOS test window
            was a bear market (B&H return < 0). Enables the bear-aware semantics above.

    Returns:
        (oos_robustness_score, fold_consistency)
    """
    if not oos_metrics_by_fold:
        return float("nan"), float("nan")
    fold_indices = sorted(oos_metrics_by_fold.keys())
    oos_vals = np.array(
        [
            float(oos_metrics_by_fold[f]) if oos_metrics_by_fold[f] is not None else float("nan")
            for f in fold_indices
        ],
        dtype=float,
    )
    weights = np.array([float(f) for f in fold_indices], dtype=float)
    mask = np.isfinite(oos_vals)
    if not mask.any():
        return float("nan"), 0.0
    w_sum = float(weights[mask].sum())
    weighted_mean = (
        float(np.dot(oos_vals[mask], weights[mask]) / w_sum) if w_sum > 0 else float("nan")
    )

    # Bear-aware fold-consistency denominator.
    if oos_bear_by_fold is not None:
        bear_arr = np.array(
            [bool(oos_bear_by_fold.get(f, False)) for f in fold_indices], dtype=bool
        )
        # Eligible folds: fired OR (didn't fire AND not bear).
        eligible_mask = mask | (~mask & ~bear_arr)
        n_eligible = int(eligible_mask.sum())
        n_profitable = int(np.sum((oos_vals > 0) & mask))
        fold_cons = float(n_profitable / n_eligible) if n_eligible > 0 else 0.0
    else:
        fold_cons = float(np.sum(oos_vals[mask] > 0) / int(mask.sum()))

    # OOS stability blend: tempers a single outlier fold from inflating the weighted mean.
    # oos_stability = mean / (std + 0.5) — a Sharpe-of-Sharpes measure across folds.
    # The +0.5 damper prevents blow-up when OOS returns are nearly identical across folds.
    _cfg = config or {}
    oos_stability_weight = float(_cfg.get("OOS_STABILITY_WEIGHT", 0.3))
    if oos_stability_weight > 0.0 and int(mask.sum()) >= 2:
        oos_mean_plain = float(np.mean(oos_vals[mask]))
        oos_std_plain = float(np.std(oos_vals[mask], ddof=1))
        oos_stability = oos_mean_plain / (oos_std_plain + 0.5)
        oos_rob = (
            (1.0 - oos_stability_weight) * weighted_mean
            + oos_stability_weight * oos_stability
        )
    else:
        oos_rob = weighted_mean

    return oos_rob, fold_cons


def _param_cv_series(
    is_metrics_by_fold: Dict[int, pd.Series],
) -> pd.Series:
    """Coefficient of Variation (std / |mean|) per param combo across folds.

    Higher CV = more fold-sensitive = overfitting-prone. Returns NaN for combos
    with fewer than 2 finite fold values (no penalty applied for those).
    Uses the same flattened-tuple key scheme as _weighted_robustness_series.
    """
    fold_keys = sorted(is_metrics_by_fold.keys())
    if not fold_keys:
        return pd.Series(dtype=float)

    flat: Dict[int, pd.Series] = {}
    for fk in fold_keys:
        s = is_metrics_by_fold[fk]
        keys = [_wfo_train_metric_row_key(i) for i in s.index]
        vals = np.asarray(s, dtype=float).ravel()
        flat[fk] = pd.Series(vals, index=pd.Index(keys, dtype=object))

    all_keys: set = set()
    for s in flat.values():
        all_keys.update(s.index.tolist())
    if not all_keys:
        return pd.Series(dtype=float)

    union_list = sorted(all_keys)
    union_idx = pd.Index(union_list, dtype=object)

    # Build (n_keys × n_folds) matrix; NaN where key absent in a fold.
    # Use union_idx (dtype=object) to avoid pandas triggering MultiIndex.from_tuples
    # on 1-element tuples (single-param grids), which raises IndexError.
    n_keys = len(union_list)
    n_folds = len(fold_keys)
    mat = np.full((n_keys, n_folds), np.nan, dtype=float)
    for j, fk in enumerate(fold_keys):
        aligned = flat[fk].reindex(union_idx)
        mat[:, j] = aligned.to_numpy(dtype=float, copy=False)

    # Vectorized CV: std(ddof=1) / (|mean| + eps) per row, NaN if < 2 finite values.
    finite_counts = np.sum(np.isfinite(mat), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_mean = np.nanmean(mat, axis=1)
        row_std  = np.nanstd(mat, axis=1, ddof=1)
    cv_vals = np.where(
        finite_counts >= 2,
        row_std / (np.abs(row_mean) + 1e-8),
        np.nan,
    )

    return pd.Series(cv_vals, index=union_idx)


def _calculate_robustness(
    is_metrics_by_fold: Dict[int, pd.Series],
    param_names: List[str],
    param_grid: Dict[str, Any],
    oos_metrics_by_fold: Optional[Dict[int, float]] = None,
    debug_metrics: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Calculates parameter robustness validation across folds.

    If oos_metrics_by_fold is provided, uses OOS Sharpe to weight each fold's
    contribution to robustness (penalizes parameter sets that don't generalize).
    Otherwise falls back to in-sample Sharpe with recency weighting.

    Args:
        is_metrics_by_fold: Dict mapping fold idx to in-sample Sharpe Series (all param combos)
        param_names: List of parameter names
        param_grid: Parameter grid definition
        oos_metrics_by_fold: Optional dict mapping fold idx to OOS Sharpe (fold-level consistency)

    Returns:
        (robust_top_5, best_robust_params) tuples
    """
    # If OOS metrics provided, use them to measure generalization
    if oos_metrics_by_fold:
        fold_indices = sorted(oos_metrics_by_fold.keys())
        oos_values = np.array(
            [_coerce_metric_float(oos_metrics_by_fold.get(f)) for f in fold_indices],
            dtype=float,
        )
        # Min-max normalize OOS Sharpes to [0.1, 1.0] so folds with better OOS get
        # proportionally higher weight even when *all* OOS values are negative.
        # This prevents the old ratio-to-mean collapsing to a flat 0.5 weight when
        # oos_mean <= 0 (which made OOS weighting useless in poor-OOS regimes).
        oos_finite = oos_values[np.isfinite(oos_values)]
        oos_min_val = float(np.min(oos_finite)) if oos_finite.size > 0 else 0.0
        oos_max_val = float(np.max(oos_finite)) if oos_finite.size > 0 else 0.0
        oos_range_val = oos_max_val - oos_min_val

        weights = {}
        for f in fold_indices:
            oos_sharpe = _coerce_metric_float(oos_metrics_by_fold.get(f))
            recency_weight = float(f)
            if np.isfinite(oos_sharpe) and oos_range_val > 1e-8:
                consistency_weight = 0.1 + 0.9 * (oos_sharpe - oos_min_val) / oos_range_val
            else:
                consistency_weight = 0.5
            weights[f] = recency_weight * consistency_weight

        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)
    else:
        # Original recency-weighted IS Sharpe (for backwards compatibility)
        weights = {f: float(f) for f in is_metrics_by_fold.keys()}
        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)

    _cfg = config or {}

    # Parameter stability penalty: penalize combos with high fold-to-fold CV.
    # CV = std / |mean| measures how much a combo's IS metric varies across folds —
    # high CV indicates the combo is curve-fitting to specific folds rather than
    # generalizing. The multiplier is 1/(1 + w*CV), so a stable combo (CV≈0) keeps
    # its full score while an unstable combo (CV=2, w=0.3) loses ~37%.
    param_stability_weight = float(_cfg.get("PARAM_STABILITY_WEIGHT", 0.3))
    if param_stability_weight > 0.0 and len(is_metrics_by_fold) >= 2:
        cv_series = _param_cv_series(is_metrics_by_fold)
        cv_aligned = cv_series.reindex(robustness_scores.index).fillna(0.0)
        stability_multiplier = 1.0 / (1.0 + param_stability_weight * cv_aligned)
        robustness_scores = robustness_scores * stability_multiplier

    if debug_metrics:
        for fk, s in sorted(is_metrics_by_fold.items()):
            arr = np.asarray(s, dtype=float).ravel()
            n_fin = int(np.sum(np.isfinite(arr)))
            print(f"      [WFO_DEBUG] fold {fk}: train_metric len={len(s)} finite={n_fin}")
        if robustness_scores.size:
            rs_arr = np.asarray(robustness_scores, dtype=float).ravel()
            n_c = int(np.sum(np.isfinite(rs_arr)))
            print(
                f"      [WFO_DEBUG] combined robustness: len={len(robustness_scores)} finite={n_c}"
            )
        else:
            print("      [WFO_DEBUG] combined robustness: empty")

    if robustness_scores.size == 0:
        return [], {}

    top_robust_idx = robustness_scores.nlargest(5)
    robust_top_5 = []

    for idx, score in top_robust_idx.items():
        extracted = _extract_params(idx, robustness_scores, param_names, param_grid)
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = float("nan")
        if not np.isfinite(score_f):
            score_f = float("nan")
        robust_top_5.append({"params": _to_native(extracted), "robustness_score": score_f})

    if not robust_top_5:
        return [], {}

    best_robust_params = robust_top_5[0]["params"]
    return robust_top_5, best_robust_params


def _calculate_wfo_bounds(
    total_len: int, n_splits: int, test_ratio: float
) -> List[Tuple[int, int, int, int]]:
    """Calculates exact integer index boundaries for WFO splits."""
    test_len = int(total_len / (test_ratio + n_splits))
    train_len = int(test_len * test_ratio)
    bounds = []

    for i in range(n_splits):
        start_idx = i * test_len
        train_end_idx = start_idx + train_len
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_len

        if i == n_splits - 1:
            test_end_idx = total_len

        bounds.append((start_idx, train_end_idx, test_start_idx, test_end_idx))

    return bounds


def _execute_wfo_loop(
    ohlcv: pd.DataFrame,
    mover_mask: Optional[pd.DataFrame],
    param_grid: Dict[str, Any],
    config: Dict[str, Any],
    param_names: List[str],
    n_splits: int,
    test_ratio: float,
    show_progress: bool,
    logger: Any = None,
    btc_regime_mask: Optional[pd.Series] = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, pd.Series], List[pd.Series]]:
    """Iterates through the dataset and processes each WFO fold."""
    wfo_stats = []
    is_metrics_by_fold = {}
    oos_returns_list = []

    bounds = _calculate_wfo_bounds(len(ohlcv), n_splits, test_ratio)
    t0_wfo = time.time()

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(bounds):
        fold_idx = i + 1
        fold_start = time.time()

        train_idx = ohlcv.index[tr_start:tr_end]
        test_idx = ohlcv.index[te_start:te_end]

        fold_result = _process_wfo_fold(
            fold_idx,
            train_idx,
            test_idx,
            ohlcv,
            mover_mask,
            param_grid,
            config,
            show_progress,
            param_names,
            btc_regime_mask=btc_regime_mask,
        )

        is_metrics_by_fold[fold_idx] = fold_result.pop("train_metrics")
        oos_returns_list.append(fold_result.pop("oos_returns"))
        wfo_stats.append(fold_result)

        fold_elapsed = time.time() - fold_start
        total_elapsed = time.time() - t0_wfo
        avg_per_fold = total_elapsed / fold_idx
        eta = (n_splits - fold_idx) * avg_per_fold
        fold_msg = (
            f"Fold {fold_idx}/{n_splits} done in {fold_elapsed:.1f}s "
            f"| total {_eta_str(total_elapsed)} | ETA {_eta_str(eta)} (est. {_wall_clock_eta(eta)})"
        )
        print(f"  > {fold_msg}")
        if logger:
            logger.update(f"  {fold_msg}")

    skipped = sum(1 for f in wfo_stats if f.get("_skipped_vectorized_failure"))
    if skipped > 0:
        print(
            f"  [WFO WARN] {skipped}/{n_splits} fold(s) skipped due to vectorized-train failure"
            f" — robustness built from {n_splits - skipped} fold(s) only"
        )

    return wfo_stats, is_metrics_by_fold, oos_returns_list


def _save_wfo_results(
    rm: Any,
    final_pf: Any,
    final_stats: Dict[str, Any],
    best_robust_params: Dict[str, Any],
    best_recent_params: Dict[str, Any],
    robust_top_5: List[Dict[str, Any]],
    wfo_stats: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """Handles persistence of metrics, parameters, and VectorBT dashboards."""
    # Exclude skipped folds (missing test_start/params) from the saved CSV/DB
    complete_stats = [
        s for s in wfo_stats
        if not s.get("_skipped_vectorized_failure") and not s.get("_skipped_insufficient_bars")
    ]
    rm.save_metrics(
        pd.DataFrame(complete_stats) if complete_stats else pd.DataFrame(), "wfo_results.csv"
    )

    metadata = {
        **config,
        "wfo_fold_results": wfo_stats,
        "robustness_summary": {
            "top_robust": robust_top_5,
            "recent_vs_robust": {
                "recent": best_recent_params,
                "robust": best_robust_params,
                "is_equal": best_recent_params == best_robust_params,
            },
        },
    }

    rm.save_run_results(
        params=best_robust_params,
        metrics=_to_native(final_stats),
        metadata=_to_native(metadata),
    )
    rm.save_vbt_dashboard(final_pf, "final_robust_model_dashboard")
    print(f"WFO Results saved to: {rm.run_dir}")


def run_wfo_orchestrator(
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Central orchestration function for Walk-Forward Optimization."""
    rm = (
        ResultsManager(
            "run_wfo",
            explicit_run_dir=config.get("EXPLICIT_RUN_DIR"),
            pipeline_stage=config.get("PIPELINE_STAGE"),
        )
        if save_results
        else None
    )
    from ggTrader.utils.plotting import plot_wfo_splits

    ohlcv, mover_mask = load_data_with_movers(config)
    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config.get("TEST_RATIO", 3.0)
    param_names = list(param_grid.keys())

    print(f"Starting WFO Loop ({n_splits} splits, Ratio: {test_ratio}:1)...")

    plot_wfo_splits(ohlcv, n_splits, test_ratio, results_manager=rm)

    wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
        ohlcv,
        mover_mask,
        param_grid,
        config,
        param_names,
        n_splits,
        test_ratio,
        show_progress,
        None,
    )

    # Extract OOS Sharpe ratios from wfo_stats to measure generalization
    {
        fold_idx: stats["oos_sharpe"] for fold_idx, stats in enumerate(wfo_stats, 1)
    }

    dbg = bool(config.get("WFO_DEBUG_METRICS", False))
    robust_top_5, best_robust_params = _calculate_robustness(
        is_metrics_by_fold,
        param_names,
        param_grid,
        None,  # Do not punish entire folds based on OOS performance of the single IS winner
        debug_metrics=dbg,
        config=config,
    )
    if not best_robust_params:
        best_robust_params = _default_params_from_grid(param_grid)
    last_fold = wfo_stats[-1]
    best_recent_params = (
        last_fold.get("params") or last_fold.get("best_params") or best_robust_params
    )

    final_engine = FastBacktest(ohlcv, best_robust_params, config=config, mover_mask=mover_mask)
    final_pf = final_engine.run(show_progress=show_progress)
    final_stats = final_engine.get_stats()

    if save_results and rm:
        _save_wfo_results(
            rm,
            final_pf,
            final_stats,
            best_robust_params,
            best_recent_params,
            robust_top_5,
            wfo_stats,
            config,
        )

    return {
        "final_portfolio": final_pf,
        "wfo_stats": wfo_stats,
        "robust_top_5": robust_top_5,
        "best_robust_params": best_robust_params,
        "best_recent_params": best_recent_params,
        "results_manager": rm,
    }


def run_wfo_per_coin_orchestrator(
    config: Dict[str, Any],
    param_grid: Dict[str, Any],
    save_results: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Walk-Forward Optimization with per-coin independent optimization.

    Each symbol is optimized independently, then combined into a single portfolio.

    Args:
        config: Portfolio and data configuration.
        param_grid: Parameter grid for strategy.
        save_results: Whether to save results.
        show_progress: Whether to show progress bars.

    Returns:
        Dictionary with results including per-coin params and combined portfolio.
    """
    from ggTrader.core.benchmarking import _enrich_final_stats_with_cagr_and_benchmark
    from ggTrader.core.orchestrator_utils import _safe
    from ggTrader.utils.plotting import plot_wfo_splits

    rm = (
        ResultsManager(
            "run_wfo_per_coin",
            explicit_run_dir=config.get("EXPLICIT_RUN_DIR"),
            pipeline_stage=config.get("PIPELINE_STAGE"),
        )
        if save_results
        else None
    )
    ohlcv, mover_mask = load_data_with_movers(config)
    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config.get("TEST_RATIO", 3.0)
    param_names = list(param_grid.keys())

    symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    print(f"\nStarting Per-Coin WFO ({len(symbols)} symbols, {n_splits} splits)...")

    plot_wfo_splits(ohlcv, n_splits, test_ratio, results_manager=rm)

    # Dictionary to store best params per symbol
    per_coin_results: Dict[str, Dict[str, Any]] = {}

    per_coin_config = {
        **config,
        "PORTFOLIO_SHARE": 1.0,
        "USE_CASH_SHARING": False,
        "group_by": False,
    }

    # Optimize each symbol independently
    for symbol in symbols:
        print(f"\n--- Optimizing {symbol} ---")

        # Extract single-symbol data
        symbol_ohlcv = ohlcv[[symbol]]
        symbol_mover_mask = mover_mask[[symbol]] if mover_mask is not None else None

        wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
            symbol_ohlcv,
            symbol_mover_mask,
            param_grid,
            per_coin_config,
            param_names,
            n_splits,
            test_ratio,
            show_progress,
            None,
        )

        # Extract OOS Sharpe ratios from wfo_stats to measure generalization
        {
            fold_idx: stats["oos_sharpe"] for fold_idx, stats in enumerate(wfo_stats, 1)
        }

        dbg = bool(config.get("WFO_DEBUG_METRICS", False))
        robust_top_5, best_robust_params = _calculate_robustness(
            is_metrics_by_fold,
            param_names,
            param_grid,
            None,  # Do not punish entire folds based on OOS performance of the single IS winner
            debug_metrics=dbg,
            config=config,
        )
        if not best_robust_params:
            best_robust_params = _default_params_from_grid(param_grid)

        per_coin_results[symbol] = {
            "best_params": best_robust_params,
            "wfo_stats": wfo_stats,
            "robust_top_5": robust_top_5,
        }

        print(f"  > {symbol} Best Params: {best_robust_params}")

    # Build combined portfolio with per-coin params
    print("\nBuilding combined portfolio with per-coin parameters...")

    combined_entries_list = []
    combined_exits_list = []
    combined_close_list = []
    all_cols = []

    for symbol in symbols:
        best_params = per_coin_results[symbol]["best_params"]
        symbol_ohlcv = ohlcv[[symbol]]

        engine = FastBacktest(symbol_ohlcv, best_params, config=per_coin_config)
        engine.run(show_progress=False)

        # Extract signals
        close = symbol_ohlcv.xs("close", axis=1, level=1, drop_level=True)
        entries = engine.entries
        exits = engine.exits

        combined_entries_list.append(entries)
        combined_exits_list.append(exits)
        combined_close_list.append(close)
        all_cols.append(symbol)

    # Concatenate all symbols
    combined_entries = pd.concat(combined_entries_list, axis=1)
    combined_exits = pd.concat(combined_exits_list, axis=1)
    combined_close = pd.concat(combined_close_list, axis=1)

    # Create final portfolio
    final_pf = vbt.Portfolio.from_signals(
        close=combined_close,
        entries=combined_entries,
        exits=combined_exits,
        init_cash=float(config.get("START_CASH", 10000.0)),
        fees=float(config.get("FEES", 0.001)),
        slippage=float(config.get("SLIPPAGE", 0.0005)),
        freq=config.get("FREQ", "4h"),
        size=float(config.get("PORTFOLIO_SHARE", 1.0)),
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(combined_entries.shape[1], 0),
    ).copy()

    final_stats = {
        "total_value": _safe(final_pf.final_value().sum()),
        "total_profit": _safe(final_pf.total_profit().sum()),
        "profit_pct": _safe(
            (final_pf.total_profit().sum() / float(config.get("START_CASH", 10000.0))) * 100
        ),
        "total_trades": int(final_pf.trades.count().sum()),
        "win_rate": _safe(final_pf.trades.win_rate().mean()) * 100,
        "sharpe": _safe(final_pf.sharpe_ratio().mean()),
        "sortino": _safe(final_pf.sortino_ratio().mean()),
        "max_drawdown": _safe(final_pf.max_drawdown().min()) * 100,
    }
    _enrich_final_stats_with_cagr_and_benchmark(final_stats, combined_close, config)

    if save_results and rm:
        # Save per-coin results
        per_coin_params_list = []
        for symbol, results in per_coin_results.items():
            params_dict = {**results["best_params"], "symbol": symbol}
            per_coin_params_list.append(params_dict)

        per_coin_df = pd.DataFrame(per_coin_params_list)
        rm.save_metrics(per_coin_df, "per_coin_params.csv")

        # Save overall results
        metadata = {
            **config,
            "per_coin_results": _to_native(per_coin_results),
        }

        rm.save_run_results(
            params={"per_coin": _to_native(per_coin_results)},
            metrics=final_stats,
            metadata=metadata,
        )
        rm.save_vbt_dashboard(final_pf, "combined_portfolio_dashboard")
        print(f"\nPer-Coin WFO Results saved to: {rm.run_dir}")

    return {
        "final_portfolio": final_pf,
        "per_coin_results": per_coin_results,
        "final_stats": final_stats,
        "results_manager": rm,
    }
