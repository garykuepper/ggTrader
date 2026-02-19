"""VectorBT Systemic Patches (ReadOnly Fix)."""

import numpy as np
from vectorbt.records.mapped_array import MappedArray
from vectorbt.portfolio.trades import Trades
import vectorbt.base.reshape_fns as reshape_fns
import vectorbt.portfolio.trades as vbt_trades
import vectorbt.portfolio.base as vbt_base

try:
    from vectorbt.portfolio.trades import EntryTrades, ExitTrades
except ImportError:
    EntryTrades = ExitTrades = None

try:
    from vectorbt.portfolio.positions import Positions, EntryPositions, ExitPositions
except ImportError:
    try:
        from vectorbt.portfolio.base import Positions, EntryPositions, ExitPositions
    except ImportError:
        Positions = EntryPositions = ExitPositions = None


def _ensure_writable(arr):
    """Helper to ensure arrays/Series are writable."""
    if arr is None:
        return arr
    if hasattr(arr, "values"):
        if not arr.values.flags.writeable:
            return arr.copy()
    elif isinstance(arr, np.ndarray):
        if not arr.flags.writeable:
            return arr.copy()
    return arr


def _patched_to_1d_array(arg, *args, **kwargs):
    """Patch for reshape_fns.to_1d_array."""
    # We need to capture the original function *before* we patch it ideally,
    # but here we are defining the patch module.
    # To avoid circular imports or recursion if we just called reshape_fns.to_1d_array,
    # we need to be careful.
    # The original implementations in fast_backtest.py captured `orig_to_1d_array` globally.
    # We will assume that when we apply this patch, we swap it out.
    # Ideally, we call the *original* implementation.
    # Since we can't easily get the "original" inside this function without storing it,
    # we'll store it in a module-level variable during the `apply_patches` call.
    if hasattr(reshape_fns, "_orig_to_1d_array"):
        res = reshape_fns._orig_to_1d_array(arg, *args, **kwargs)
    else:
        # Fallback if not patched yet? Or maybe this IS the original?
        # If we replaced it, reshape_fns.to_1d_array refers to THIS function.
        # So we definitely need the stash.
        raise RuntimeError("VectorBT Patch Error: Original to_1d_array not found.")
    return _ensure_writable(res)


def _patch_mapped_array_agg(method_name):
    if not hasattr(MappedArray, method_name):
        return
    orig = getattr(MappedArray, method_name)

    def new_method(self, *args, **kwargs):
        return _ensure_writable(orig(self, *args, **kwargs))

    setattr(MappedArray, method_name, new_method)


def _patch_vbt_metric(cls, name):
    if cls is None or not hasattr(cls, name):
        return
    attr = getattr(cls, name)
    if hasattr(attr, "_is_vbt_patch"):
        return
    if isinstance(attr, property):
        fget = attr.fget

        def new_fget(self):
            return _ensure_writable(fget(self))

        new_fget._is_vbt_patch = True
        setattr(cls, name, property(new_fget))
    elif callable(attr):

        def new_method(self, *args, **kwargs):
            return _ensure_writable(attr(self, *args, **kwargs))

        new_method._is_vbt_patch = True
        setattr(cls, name, new_method)


def _robust_profit_factor(self, group_by=None, wrap_kwargs=None):
    try:
        total_win = self.winning.pnl.sum(group_by=group_by)
        total_loss = self.losing.pnl.sum(group_by=group_by)

        # Ensure they are writable and at least 1D arrays for mask assignment
        total_win_arr = _ensure_writable(np.atleast_1d(np.asarray(total_win)))
        total_loss_arr = _ensure_writable(np.atleast_1d(np.asarray(total_loss)))

        has_values = self.count(group_by=group_by)
        hva = np.atleast_1d(np.asarray(has_values))
        mask = hva > 0

        total_win_arr[np.isnan(total_win_arr) & mask] = 0.0
        total_loss_arr[np.isnan(total_loss_arr) & mask] = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            result = total_win_arr / np.abs(total_loss_arr)

        # If original sum was a scalar, return a scalar
        if np.isscalar(total_win) and result.size == 1:
            return result[0]

        return self.wrapper.wrap_reduced(result, group_by=group_by, **(wrap_kwargs or {}))
    except Exception:
        # Fallback to manual aggregation if winning/losing properties fail (e.g. ExitTrades)
        try:
            pnl_series = self.pnl
            pnl = pnl_series.values if hasattr(pnl_series, "values") else np.asarray(pnl_series)
            col_map = self.col_mapper.col_map
            if isinstance(col_map, tuple):
                col_map = col_map[0]
            col_map = np.atleast_1d(np.asarray(col_map))
            group_labels = self.wrapper.resolve(group_by=group_by)
            group_labels_arr = np.atleast_1d(np.asarray(group_labels))

            unique_labels = []
            seen = set()
            for x in group_labels_arr:
                key = x if isinstance(x, (str, int, float, bool, tuple, type(None))) else str(x)
                if key not in seen:
                    unique_labels.append(x)
                    seen.add(key)
            n_groups = len(unique_labels)
            label_to_idx = {
                (str(l) if not isinstance(l, (str, int, float, bool, tuple, type(None))) else l): i
                for i, l in enumerate(unique_labels)
            }
            # Handle potential scalar group_labels_arr content access safely
            # (Though group_labels_arr is 1d, good to be safe)
            trade_group_idx = np.array(
                [
                    label_to_idx[
                        (
                            str(group_labels_arr[c])
                            if not isinstance(
                                group_labels_arr[c], (str, int, float, bool, tuple, type(None))
                            )
                            else group_labels_arr[c]
                        )
                    ]
                    for c in col_map
                ]
            )

            total_win = np.zeros(n_groups, dtype=np.float64)
            total_loss = np.zeros(n_groups, dtype=np.float64)
            np.add.at(total_win, trade_group_idx, np.where(pnl > 0, pnl, 0))
            np.add.at(total_loss, trade_group_idx, np.where(pnl < 0, pnl, 0))

            with np.errstate(divide="ignore", invalid="ignore"):
                result = total_win / np.abs(total_loss)
            return self.wrapper.wrap_reduced(result, group_by=group_by, **(wrap_kwargs or {}))
        except Exception:
            # Final fallback to base implementation which is now hopefully patched
            orig = getattr(self.__class__, "_orig_profit_factor", None)
            if orig:
                return orig(self, group_by=group_by, wrap_kwargs=wrap_kwargs)
            return np.nan


def apply_vbt_patches():
    """Apply all vectorbt patches."""
    # 1. Patch to_1d_array
    if not hasattr(reshape_fns, "_orig_to_1d_array"):
        reshape_fns._orig_to_1d_array = reshape_fns.to_1d_array
        reshape_fns.to_1d_array = _patched_to_1d_array

    # Patch shadow references
    vbt_trades.to_1d_array = _patched_to_1d_array
    vbt_base.to_1d_array = _patched_to_1d_array

    # 2. Patch MappedArray aggregation methods
    for agg in ["sum", "mean", "std", "min", "max", "count", "reduce"]:
        _patch_mapped_array_agg(agg)

    # 3. Patch Trades/Positions metrics
    metrics_to_patch = [
        "total_profit",
        "total_loss",
        "avg_win",
        "avg_loss",
        "win_rate",
        "expectancy",
        "profit_factor",
        "max_win",
        "max_loss",
        "total_win_count",
        "total_loss_count",
        "win_count",
        "loss_count",
    ]
    for cls in [Trades, EntryTrades, ExitTrades, Positions, EntryPositions, ExitPositions]:
        if cls is not None:
            for metric in metrics_to_patch:
                _patch_vbt_metric(cls, metric)

    # 4. Robust multi-class Profit Factor override
    for cls in [Trades, EntryTrades, ExitTrades]:
        if cls is not None:
            if not hasattr(cls, "_orig_profit_factor"):
                cls._orig_profit_factor = cls.profit_factor
            cls.profit_factor = _robust_profit_factor
