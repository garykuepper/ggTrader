"""Aggregate post-WFO metrics and the 4 PASS/FAIL gates.

After the 10-fold WFO completes for a (coin, entry, exit) combo, four metrics
characterize the combo across folds:
- WFE: walk-forward efficiency = mean(test_ann_ret) / mean(train_ann_ret)
- % profitable folds: fraction with test_ann_ret > 0
- Parameter CV: per-axis CV of the 10 chosen-per-fold params, MAX across axes
- DD ratio: mean(|test_max_dd|) / mean(|train_max_dd|)

Four gates are applied as pure PASS/FAIL filters (Pardo convention):
- WFE >= 0.5
- profitable >= 0.6
- param CV <= 0.3
- DD ratio <= 2.0

Gates are NOT selection criteria. A combo passes (proceeds to per-coin
selection in Task 7) or fails (excluded from candidate set).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def infer_bars_per_year(
    index: pd.DatetimeIndex,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Infer bars-per-year from the OHLCV DatetimeIndex frequency.

    Looks at the median spacing between consecutive bars and computes:
        bars_per_year = (365.25 * 24 * 3600) / median_spacing_seconds

    Examples (with no config override):
      4h bars  -> 365.25 * 24 / 4   = 2191.5
      1h bars  -> 365.25 * 24       = 8766.0
      1d bars  -> 365.25            = 365.25
      15m bars -> 365.25 * 24 * 4   = 35064.0

    If config sets ``WFO_BARS_PER_YEAR`` to a positive number, that value
    overrides inference. If inference fails (too few bars, non-uniform
    spacing), falls back to the config override or 2191.5 (4h default).
    """
    cfg = config or {}
    override = cfg.get("WFO_BARS_PER_YEAR")
    if override is not None:
        try:
            override_f = float(override)
            if override_f > 0:
                return override_f
        except (TypeError, ValueError):
            pass
    # Default fallback for 4h bars (project convention).
    default_4h = 365.25 * 24.0 / 4.0  # 2191.5
    if index is None or len(index) < 2:
        return default_4h
    try:
        deltas = pd.to_datetime(pd.Index(index)).to_series().diff().dropna()
        if len(deltas) == 0:
            return default_4h
        median_sec = float(deltas.median().total_seconds())
        if median_sec <= 0:
            return default_4h
        return (365.25 * 24.0 * 3600.0) / median_sec
    except Exception:
        return default_4h


def compute_wfe(train_returns: List[float], test_returns: List[float]) -> float:
    """Walk-Forward Efficiency = mean(test_ann_ret) / mean(train_ann_ret).

    Returns NaN if mean(train) is too close to zero (avoids division explosion).
    Spec convention from Pardo: WFE >= 0.5 means OOS performance is at least
    50% of in-sample.
    """
    train_arr = np.asarray([float(x) for x in train_returns if x is not None and np.isfinite(x)])
    test_arr = np.asarray([float(x) for x in test_returns if x is not None and np.isfinite(x)])
    if len(train_arr) == 0 or len(test_arr) == 0:
        return float("nan")
    train_mean = float(train_arr.mean())
    test_mean = float(test_arr.mean())
    if abs(train_mean) < 1e-6:
        return float("nan")
    return test_mean / train_mean


def fraction_profitable_folds(test_returns: List[float]) -> float:
    """Fraction of folds with test_ann_ret > 0.

    NaN/None values are excluded from both numerator and denominator.
    """
    finite = [float(x) for x in test_returns if x is not None and np.isfinite(x)]
    if len(finite) == 0:
        return float("nan")
    n_profitable = sum(1 for x in finite if x > 0)
    return n_profitable / len(finite)


def parameter_cv(fold_params: List[Dict[str, Any]]) -> float:
    """Per-axis CV across the chosen-per-fold params; report the MAX axis CV.

    For each param key, compute std(values) / |mean(values)| across folds.
    Axes with mean ~= 0 or constant value contribute 0 (no variation).
    Non-numeric param values are skipped.

    Returns the MAXIMUM CV across numeric axes (not the mean). Reasoning:
    a coin's stability is bottlenecked by its least stable axis. Averaging
    lets one wild axis hide behind several stable ones; max forces every
    axis to satisfy the CV gate.

    Returns 0.0 when all axes are constant. NaN when no numeric axes exist.
    """
    if not fold_params:
        return float("nan")
    keys = set()
    for p in fold_params:
        keys.update(p.keys())
    cvs: List[float] = []
    for k in keys:
        vals: List[float] = []
        for p in fold_params:
            v = p.get(k)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if len(vals) < 2:
            continue
        arr = np.asarray(vals)
        mean_v = float(arr.mean())
        std_v = float(arr.std(ddof=0))
        if abs(mean_v) < 1e-9:
            cvs.append(0.0 if std_v < 1e-9 else float("inf"))
        else:
            cvs.append(std_v / abs(mean_v))
    if not cvs:
        return float("nan")
    finite_cvs = [c for c in cvs if np.isfinite(c)]
    if not finite_cvs:
        return float("inf")
    return float(max(finite_cvs))


def dd_ratio(train_dds: List[float], test_dds: List[float]) -> float:
    """Test/train max-drawdown ratio = mean(|test_dd|) / mean(|train_dd|).

    Drawdowns are stored as negative numbers (worse = more negative). The
    ratio uses absolute values so it's always >= 0.
    """
    train_arr = np.asarray([abs(float(x)) for x in train_dds if x is not None and np.isfinite(x)])
    test_arr = np.asarray([abs(float(x)) for x in test_dds if x is not None and np.isfinite(x)])
    if len(train_arr) == 0 or len(test_arr) == 0:
        return float("nan")
    train_mean = float(train_arr.mean())
    if train_mean < 1e-9:
        return float("inf")
    return float(test_arr.mean()) / train_mean


def apply_gates(
    wfe: float,
    profitable_fraction: float,
    param_cv: float,
    dd_ratio_val: float,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Apply the 4 PASS/FAIL gates. Returns a dict with 'passed' and 'failures'.

    A combo passes if ALL four gates pass. NaN values fail their gate.
    """
    failures: List[str] = []

    if not np.isfinite(wfe) or wfe < thresholds["wfe_min"]:
        failures.append("wfe")

    if not np.isfinite(profitable_fraction) or profitable_fraction < thresholds["profitable_min"]:
        failures.append("profitable_fraction")

    if not np.isfinite(param_cv) or param_cv > thresholds["cv_max"]:
        failures.append("param_cv")

    if not np.isfinite(dd_ratio_val) or dd_ratio_val > thresholds["dd_max"]:
        failures.append("dd_ratio")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "metrics": {
            "wfe": float(wfe) if np.isfinite(wfe) else None,
            "profitable_fraction": (
                float(profitable_fraction) if np.isfinite(profitable_fraction) else None
            ),
            "param_cv": float(param_cv) if np.isfinite(param_cv) else None,
            "dd_ratio": float(dd_ratio_val) if np.isfinite(dd_ratio_val) else None,
        },
    }
