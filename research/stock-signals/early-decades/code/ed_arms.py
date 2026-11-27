"""Selection helpers for the yearly and jump-exclusion arms."""

import pandas as pd


def arm_b_schedule(rebalance_dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Select the first available rebalance date in each calendar year."""
    dates = sorted(pd.Timestamp(date) for date in rebalance_dates)
    return [date for i, date in enumerate(dates) if i == 0 or date.year != dates[i - 1].year]


def jump_excluded(
    returns: pd.DataFrame, asof: pd.Timestamp, threshold: float = 0.20, lookback: int = 252
) -> set[int]:
    """Find stocks with a jump in the prior trading rows, excluding the pick date."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("return dates must be unique and ordered")
    prior = returns.loc[returns.index < pd.Timestamp(asof)].tail(lookback)
    return set(prior.columns[(prior >= threshold).any(axis=0)])
