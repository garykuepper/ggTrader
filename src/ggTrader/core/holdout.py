"""Final holdout reservation, smoke testing, and warning generation.

Step 0 of the WFO textbook reset. The most recent HOLDOUT_FRACTION of bars is
reserved before the 10-fold WFO runs. After WFO + the 4 aggregate gates pass,
the median per-fold winners are evaluated on the holdout exactly once. A
warning is raised (but no auto-drop) when holdout annualized return is
negative OR holdout max drawdown exceeds 1.5x the worst test-fold drawdown
observed during WFO.

The holdout is NOT a gate. Numbers are always reported regardless of
warnings; the human decides whether to deploy.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Tuple

import pandas as pd


def split_train_holdout(
    ohlcv: pd.DataFrame,
    holdout_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split OHLCV into a leading train block and a trailing holdout block.

    The most recent ``holdout_fraction`` of bars goes to the holdout. The rest
    is used for the 10-fold WFO. Holdout starts immediately after train ends
    (no gap, no overlap).

    holdout_fraction <= 0 returns the full input as train and an empty holdout.
    """
    if holdout_fraction <= 0:
        return ohlcv, ohlcv.iloc[0:0]
    n = len(ohlcv)
    n_holdout = int(round(n * holdout_fraction))
    if n_holdout <= 0:
        return ohlcv, ohlcv.iloc[0:0]
    train = ohlcv.iloc[: n - n_holdout]
    holdout = ohlcv.iloc[n - n_holdout :]
    return train, holdout


def holdout_warning_flags(
    holdout_ann_return: float,
    holdout_max_dd: float,
    worst_wfo_test_dd: float,
    dd_multiple: float = 1.5,
) -> List[str]:
    """Determine which warning flags fire for a holdout evaluation result.

    Returns:
        List of warning names. Empty list means no warnings.
        Possible values: "negative_return", "max_dd_exceeds_threshold".

    The dd_multiple parameter is 1.5 by spec — holdout max_dd worse than
    1.5x the worst WFO test-fold max_dd is the threshold for the DD warning.
    """
    flags: List[str] = []
    if holdout_ann_return is not None and holdout_ann_return < 0:
        flags.append("negative_return")
    # max_dd is stored as a negative number (worse = more negative). Threshold
    # for "exceeds" is when |holdout_dd| > dd_multiple * |worst_wfo_test_dd|.
    if holdout_max_dd is not None and worst_wfo_test_dd is not None:
        # Guard against degenerate WFO baselines:
        # - 0.0: the 1.5x threshold becomes 0 and would fire on any non-zero
        #   holdout DD. A coin with zero WFO drawdowns has no scale to judge
        #   the holdout against; skip the flag.
        # - NaN: comparisons silently return False, masking the check. Skip
        #   explicitly so the absence of WFO baseline data is obvious in the
        #   flag list (which will not include max_dd_exceeds_threshold).
        if (
            worst_wfo_test_dd != 0.0
            and not (isinstance(worst_wfo_test_dd, float) and math.isnan(worst_wfo_test_dd))
            and abs(holdout_max_dd) > dd_multiple * abs(worst_wfo_test_dd)
        ):
            flags.append("max_dd_exceeds_threshold")
    return flags


def median_params_snap_to_grid(
    fold_winners: list,
    grid: Dict[str, list],
) -> Dict[str, Any]:
    """Per axis: take median across fold winners, snap to nearest grid value.

    Numeric axes use statistics.median. When the median falls exactly between
    two grid values (tie), the lower value is chosen (more conservative).
    Non-numeric axes (bool, str) use mode (most common value).

    Args:
        fold_winners: List of per-fold winner-params dicts (one per fold).
        grid: The parameter grid the strategy was optimized over.
              Each axis's value list provides the snap targets.

    Returns:
        Dict mapping axis name to the snapped median value.
    """
    if not fold_winners:
        return {}
    result: Dict[str, Any] = {}
    axes: set = set()
    for w in fold_winners:
        axes.update(w.keys())
    for axis in axes:
        values = [w.get(axis) for w in fold_winners if axis in w]
        if not values:
            continue
        # Try numeric path first — but skip if values are booleans or strings
        # (booleans subclass int in Python, so float(True) == 1.0, losing the type).
        all_bool = all(isinstance(v, bool) for v in values)
        all_str = all(isinstance(v, str) for v in values)
        try:
            if all_bool or all_str:
                raise TypeError("non-numeric axis; use mode path")
            numeric_vals = [float(v) for v in values]
            med = statistics.median(numeric_vals)
            grid_vals = [float(g) for g in grid.get(axis, [med])]
            if not grid_vals:
                result[axis] = med
                continue
            # Snap: find grid value with smallest absolute distance. Ties -> lower.
            grid_vals_sorted = sorted(grid_vals)
            best = grid_vals_sorted[0]
            best_dist = abs(best - med)
            for v in grid_vals_sorted[1:]:
                d = abs(v - med)
                # Strict-less means earlier tied values win (lower-is-conservative).
                if d < best_dist:
                    best = v
                    best_dist = d
            # Preserve the original numeric type from the grid if possible (int vs float).
            original_first = grid.get(axis, [best])[0]
            if isinstance(original_first, int) and best == int(best):
                result[axis] = int(best)
            else:
                result[axis] = best
        except (TypeError, ValueError):
            # Non-numeric axis: use mode.
            try:
                result[axis] = statistics.mode(values)
            except statistics.StatisticsError:
                # No unique mode — fall back to first.
                result[axis] = values[0]
    return result
