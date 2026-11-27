"""Selection helpers for the yearly and jump-exclusion arms."""

import pandas as pd


def arm_b_schedule(rebalance_dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Select the first date strictly after the previous date's anniversary."""
    selected: list[pd.Timestamp] = []
    for date in sorted(pd.Timestamp(value) for value in rebalance_dates):
        if not selected or date > selected[-1] + pd.DateOffset(years=1):
            selected.append(date)
    return selected


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
        previous = close.ffill(limit=10).shift(1)
        daily = close.div(previous).sub(1)
        jumped = daily.ge(0.20).rolling(252, min_periods=1).max().shift(1).fillna(0).astype(bool)
        aligned = eligible.copy()
        if close.index.tz is not None and aligned.index.tz is None:
            aligned.index = aligned.index.tz_localize(close.index.tz)
        elif close.index.tz is None and aligned.index.tz is not None:
            aligned.index = aligned.index.tz_convert(None)
        allowed = aligned.reindex(index=close.index, columns=close.columns, fill_value=False)
        allowed = allowed.fillna(False)
        if not allowed.to_numpy().any():
            raise ValueError("no eligible stocks after aligning dates")
        allowed = allowed & ~jumped
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
