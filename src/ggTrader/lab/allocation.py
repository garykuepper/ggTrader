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
