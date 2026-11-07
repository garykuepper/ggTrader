"""Return helpers that keep pandas 2 semantics under pandas 3."""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def padded_pct_change(prices: PandasT) -> PandasT:
    """Percent change that forward-fills NaN gaps first (pandas 2's default behaviour).

    pandas 2 filled non-leading NaN with the previous value before differencing
    (`fill_method="pad"`), so a stale or delisted symbol produced 0.0 returns. pandas 3
    removed that default and returns NaN instead, which changes cross-sectional ranks and
    rolling statistics on gappy price panels. Use this on price panels; equity curves that
    are already gap-free can call `pct_change(fill_method=None)` directly.

    Args:
        prices: price Series or (time x symbol) DataFrame.

    Returns:
        One-period returns; leading NaN stay NaN, interior and trailing gaps become 0.0.
    """
    return prices.ffill().pct_change(fill_method=None)
