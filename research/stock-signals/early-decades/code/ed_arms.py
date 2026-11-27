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


def plan_arm_trades(
    arm: str,
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    top_n: int,
    lookback: int,
    skip: int,
    start: str,
    end: str,
) -> list:
    """Build targets with the first-test planner and the selected arm's restrictions."""
    from ed_replay import plan_trades

    if arm not in {"A", "B", "C"}:
        raise ValueError("unknown arm")
    allowed = eligible
    if arm == "C":
        # Missing observations never fill across gaps when calculating daily changes.
        daily = close.pct_change(fill_method=None)
        jumped = daily.ge(0.20).rolling(252, min_periods=1).max().shift(1).fillna(0).astype(bool)
        allowed = eligible.reindex(index=close.index, columns=close.columns, fill_value=False)
        allowed = allowed.fillna(False) & ~jumped
    steps = plan_trades(
        close,
        allowed,
        spans,
        top_n=top_n,
        lookback=lookback,
        skip=skip,
        dead_loss=0.0,
        start=start,
        end=end,
    )
    if arm == "B":
        schedule = set(arm_b_schedule([step.decision_date for step in steps]))
        steps = [step for step in steps if step.decision_date in schedule]
    return steps
