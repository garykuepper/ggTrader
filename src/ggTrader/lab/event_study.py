"""Event-study supplementary test for sparse/event-driven strategies.

Motivation (2026-07-20, prompted by an audit of the project's high
strategy-rejection rate): the WFO/NDH/DSR gate framework's Sharpe-based
statistics assume something close to a continuously-active equity curve.
For a strategy that's in cash ~95%+ of the time and trades on a handful
of days per fold (fomc_drift, index_deletion_fade), a single trade
dominates a fold's annualized Sharpe -- observed directly on fomc_drift's
54-fold WFO, where per-fold OOS Sharpe swung from -3.59 to +3.35. That's
not evidence the underlying edge is unstable; it's evidence Sharpe is a
poor statistic for this shape of data.

This module offers the same test the source papers themselves typically
run: is the mean return on "event days" (when the strategy would hold a
position) statistically distinguishable from an ordinary day on the same
instruments? A two-sample Welch's t-test (unequal variance, appropriate
since event-day volatility often differs from baseline) rather than an
annualized Sharpe comparison.

This is a SUPPLEMENT to the WFO gate, not a replacement -- a candidate
still needs to clear the usual walk-forward discipline before being
called a GO. It exists specifically to give sparse/event-driven
candidates a fairer, less noisy hearing than fold-level Sharpe alone.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from scipy import stats


def in_position_mask(entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    """Bool (time x symbol): True from an entry bar through its matching
    exit bar, inclusive. A diagnostic approximation of vbt's own fill
    timing (not bit-perfect with next-bar execution mechanics) -- good
    enough for "was a position notionally held on this bar," which is all
    this supplementary test needs. Assumes well-formed, non-overlapping
    entry/exit pairs per symbol, as every strategy in this lab produces.
    """
    # The exit bar itself still counts as held (inclusive) -- the position
    # only actually closes starting the bar after the exit signal, so the
    # closing decrement is shifted forward by one bar before accumulating.
    exit_close = exits.astype(int).shift(1).fillna(0).astype(int)
    signal = entries.astype(int) - exit_close
    state = signal.cumsum()
    return state.clip(lower=0, upper=1).astype(bool)


def split_event_returns(returns: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Flatten a (time x symbol) returns frame into (event, non-event)
    long-format series using a matching boolean mask."""
    mask = mask.reindex_like(returns).fillna(False)
    event_returns = returns.where(mask).stack()
    nonevent_returns = returns.where(~mask).stack()
    return event_returns, nonevent_returns


class EventStudyResult(NamedTuple):
    n_event: int
    n_nonevent: int
    mean_event_return: float
    mean_nonevent_return: float
    mean_diff: float
    t_stat: float
    p_value: float
    significant: bool


def event_study_test(
    event_returns: pd.Series, nonevent_returns: pd.Series, alpha: float = 0.05
) -> EventStudyResult:
    """Welch's t-test: is the mean event-day return distinguishable from
    the mean ordinary-day return on the same instruments?"""
    event_returns = pd.Series(event_returns).dropna()
    nonevent_returns = pd.Series(nonevent_returns).dropna()
    if len(event_returns) < 2 or len(nonevent_returns) < 2:
        raise ValueError(
            "event_study_test requires at least 2 observations in each sample "
            f"(got {len(event_returns)} event, {len(nonevent_returns)} non-event)"
        )

    mean_event = float(event_returns.mean())
    mean_nonevent = float(nonevent_returns.mean())
    t_stat, p_value = stats.ttest_ind(event_returns, nonevent_returns, equal_var=False)

    return EventStudyResult(
        n_event=len(event_returns),
        n_nonevent=len(nonevent_returns),
        mean_event_return=mean_event,
        mean_nonevent_return=mean_nonevent,
        mean_diff=mean_event - mean_nonevent,
        t_stat=float(t_stat),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
    )


def run_event_study_from_signals(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    alpha: float = 0.05,
) -> EventStudyResult:
    """Convenience wrapper: derive daily returns and the in-position mask
    directly from a strategy's own entries/exits (the same SignalTargets
    used by the actual WFO simulation), then run the event-study test."""
    returns = close.pct_change()
    mask = in_position_mask(entries, exits)
    event_returns, nonevent_returns = split_event_returns(returns, mask)
    return event_study_test(event_returns, nonevent_returns, alpha=alpha)
