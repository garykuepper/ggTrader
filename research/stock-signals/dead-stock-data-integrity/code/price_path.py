"""Classify each dead stock's final month of closes into descriptive price-path labels."""

from __future__ import annotations

import pandas as pd

# Trading days in the lookback window; 21 daily simple returns require 22 valid closes.
WINDOW = 21
# Descriptive thresholds carried over from the first report; they never set a payout.
CALM_STD = 0.01
COLLAPSE_BELOW = 0.5
DRIFT_BELOW = 0.9
CALM_BAND = (0.9, 1.1)


def classify_path(close: pd.Series) -> dict:
    """Label a stock's last trading month from its closing prices alone.

    NaN closes are dropped before counting so gaps in the trading calendar do not shorten
    the 21-return lookback (the "own valid closes" rule from the first test). Labels are
    descriptive only and never determine a delisting payout.

    Args:
        close: Closing-price series ordered chronologically up to the stock's final bar.

    Returns:
        A dict with keys ``ratio`` (last close divided by the close 21 valid closes earlier,
        or ``None`` when fewer than 22 valid closes exist or the baseline close is non-positive),
        ``daily_std`` (sample standard deviation of the last 21 daily simple returns, or
        ``None`` when undefined), and ``label`` (one of ``"too_short"``, ``"collapse_like"``,
        ``"drifting_down"``, ``"calm_like_deal"``, or ``"unclear"``).
    """
    valid = close.dropna().astype(float)
    if len(valid) < WINDOW + 1:
        return {"ratio": None, "daily_std": None, "label": "too_short"}

    tail = valid.iloc[-(WINDOW + 1) :]
    prior_close = float(tail.iloc[0])
    last_close = float(tail.iloc[-1])
    if prior_close <= 0.0:
        # Guard against corrupted zero/negative prices so ratio cannot divide by zero.
        return {"ratio": None, "daily_std": None, "label": "unclear"}

    ratio = last_close / prior_close
    daily_std = float(tail.pct_change(fill_method=None).dropna().std())

    if ratio < COLLAPSE_BELOW:
        label = "collapse_like"
    elif ratio < DRIFT_BELOW:
        label = "drifting_down"
    elif CALM_BAND[0] <= ratio <= CALM_BAND[1] and daily_std <= CALM_STD:
        label = "calm_like_deal"
    else:
        label = "unclear"

    return {"ratio": ratio, "daily_std": daily_std, "label": label}
