"""Out-of-sample portfolio overlay math for the multi-sleeve research harness.

Pure functions only — no I/O, no DB, no WFO. All volatility estimates use
trailing data so a value at date t never depends on t's own future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trailing_realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling annualized realized volatility from daily returns.

    Returns NaN for the first ``window - 1`` observations (warmup).
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]:
    """Risk-parity weights: w_i = (1/vol_i) / sum(1/vol_j), summing to 1.0.

    Sleeves with non-positive or NaN vol are dropped. If none are valid,
    fall back to equal weights across the original keys.
    """
    valid = {k: v for k, v in vols.items() if v is not None and np.isfinite(v) and v > 0}
    if not valid:
        n = len(vols)
        return {k: 1.0 / n for k in vols}
    inv = {k: 1.0 / v for k, v in valid.items()}
    total = sum(inv.values())
    return {k: x / total for k, x in inv.items()}
