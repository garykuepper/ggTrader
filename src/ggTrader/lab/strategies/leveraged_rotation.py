"""Breadth-driven leveraged/inverse index rotation (weight-based, monthly).

Rotates between a universe's leveraged-long ETF, its inverse ETF, and cash,
driven by the breadth of the existing validated EnsembleSignal across that
universe's own constituent stocks -- a repurposing of a stock-picking signal
as an index-timing feature. See
docs/superpowers/specs/2026-07-14-leveraged-index-rotation-design.md.
"""

from __future__ import annotations

import pandas as pd


def compute_breadth(entries: pd.DataFrame) -> pd.Series:
    """Fraction of columns with an active (True) entry signal per row.

    Fixed denominator = entries.shape[1] (the full breadth-universe size),
    not the count of symbols currently past warmup -- see spec for why.
    """
    if entries.shape[1] == 0:
        return pd.Series(0.0, index=entries.index)
    return entries.sum(axis=1) / entries.shape[1]


def rotate_positions(
    breadth: pd.Series,
    upper_threshold: float,
    lower_threshold: float,
    min_hold_months: int,
) -> pd.Series:
    """Per-date state in {"long", "inverse", "cash"} with hysteresis.

    A state change only takes effect once the new raw signal (breadth vs.
    thresholds) has held for min_hold_months consecutive dates in the input
    series. The first date's state takes effect immediately.
    """
    raw = pd.Series("cash", index=breadth.index, dtype=object)
    raw[breadth > upper_threshold] = "long"
    raw[breadth < lower_threshold] = "inverse"

    states: list[str] = []
    current: str | None = None
    streak_value: str | None = None
    streak_len = 0
    for val in raw:
        if val == streak_value:
            streak_len += 1
        else:
            streak_value = val
            streak_len = 1
        if current is None:
            current = val
        elif val != current and streak_len >= min_hold_months:
            current = val
        states.append(current)
    return pd.Series(states, index=breadth.index)
