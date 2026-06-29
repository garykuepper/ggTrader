"""Portfolio-of-sleeves blend: run sleeves through the gated WFO and combine
their OOS curves with the validated inverse-vol / target-vol overlay.

blend_curves is the pure math (no I/O); run_blend orchestrates data load, WFO,
blend, and persistence.
"""

from __future__ import annotations

from functools import reduce

import pandas as pd

from ggTrader.lab.allocation import combine_sleeves
from ggTrader.lab.data import STOCK_BASE_CONFIG


def blend_curves(
    curves: dict[str, pd.Series],
    *,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 2.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Align sleeve OOS equity curves on common dates, blend to a target vol.

    Returns (blend_equity, returns_df, diag). blend_equity is a cumprod curve
    starting at START_CASH; returns_df is the aligned per-sleeve daily returns.
    """
    common = reduce(lambda a, b: a.intersection(b), (c.index for c in curves.values()))
    returns_df = pd.DataFrame(
        {label: curves[label].reindex(common).pct_change() for label in curves}
    ).dropna()
    blended, diag = combine_sleeves(
        returns_df, target_vol=target_vol, window=window, max_leverage=max_leverage
    )
    start_cash = float(STOCK_BASE_CONFIG["START_CASH"])
    blend_equity = (1.0 + blended).cumprod() * start_cash
    return blend_equity, returns_df, diag
