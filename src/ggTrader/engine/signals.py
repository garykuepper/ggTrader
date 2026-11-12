"""Signal helpers: vectorbt crossovers and a time stop, with the Rust dtype traps removed.

The Rust engine accepts only float64 arrays: ``frame.vbt.crossed_below(30)`` (an int
threshold) and ``bool_frame.vbt.fshift(n)`` both raise under ``engine="rust"``
("Rust engine requires float64 arrays and ... cannot be safely cast"). These wrappers coerce
inputs to float64 so strategies get the fast engine without thinking about dtypes.

Crossover semantics are vectorbt's (strict inequality with a stateful "was below" flag; a
tie or a NaN resets it). They differ from the legacy ``prev <= b & now > b`` only on exact
ties; on 150 real symbols x 1,968 bars there were 0 differing cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt  # noqa: F401  (registers the .vbt accessor)

Frame = pd.DataFrame | pd.Series


def _as_float(value: Any) -> Any:
    """Coerce a scalar, array or pandas object to float64 (an int raises on Rust)."""
    if isinstance(value, pd.DataFrame | pd.Series):
        return value.astype("float64")
    if np.isscalar(value):
        return float(value)
    return np.asarray(value, dtype="float64")


def cross_above(a: Frame, b: Any) -> Frame:
    """True on the bar where ``a`` goes strictly above ``b`` after having been below it.

    Args:
        a: series or (time x symbol) frame.
        b: float threshold, or an array/frame of the same shape as ``a``.
    """
    return _as_float(a).vbt.crossed_above(_as_float(b))


def cross_below(a: Frame, b: Any) -> Frame:
    """True on the bar where ``a`` goes strictly below ``b`` after having been above it."""
    return _as_float(a).vbt.crossed_below(_as_float(b))


def time_stop(entries: Frame, n: int) -> Frame:
    """Boolean frame true ``n`` bars after each entry signal.

    This shifts signals, not filled entries: a stale exit can affect a later trade.
    Legacy equivalent: ``entries.shift(n, fill_value=False)``.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, int | np.integer) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    shifted = entries.astype("float64").vbt.fshift(int(n), fill_value=0.0)
    return shifted > 0.5
