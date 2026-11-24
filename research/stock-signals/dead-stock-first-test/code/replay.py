"""Monthly replay planner: which stocks to hold after each month-end, and when stocks die.

``plan_trades`` is a pure function of prices, eligibility and span dates: no cash, no cost, no
tax (the ledger of the next task sizes and executes). ``dead_exit_dates`` and ``exit_close``
tell that ledger when a held stock ends and at what price it is sold.

Weight convention, identical to ``ggTrader.lab.strategies.xs_momentum.XsMomentum.build``
(``test_weights_follow_xs_momentum_convention`` compares the two):

* The score is ``momentum_scores(close, lookback, skip)`` read on the decision day. It is
  computed on each stock's own valid closes and forward-filled, so a stock with no print on the
  decision day (halted) keeps its last print for scoring.
* Candidates are the stocks eligible on the decision day with a finite score.
* Candidates are sorted by score, highest first, with a stable sort, so equal scores keep the
  column order of ``close``. The first ``top_n`` are chosen.
* Each chosen stock gets ``1 / n_chosen`` (not ``1 / top_n``): when fewer than ``top_n``
  qualify, the weight is split over those that do; when none qualify the target is empty.

Differences from the engine path, all deliberate:

* Decision days are the last trading day of each calendar month in ``[start, end]``. The engine
  also decides on the first day of its window; this study does not (the brief fixes it).
* A decision acts on the next trading day in ``close.index``. A decision with no next trading
  day, or whose next trading day falls after ``end``, produces no step.
* A stock is never chosen for an execution day on or after its last bar (the lab's
  ``entry_ok`` mask), because a stale score would otherwise keep a dead stock on the list.
* Stocks whose series ends with a successor (bought out or merged) are removed from new
  entries the same way, but ``exit_close`` does not write their last price down.

``dead_loss`` is validated here for symmetry with the ledger call; the plan itself does not
depend on it (the loss only sets the sale price, through ``exit_close``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ggTrader.lab.dead_stocks import forced_exit_frames, simulation_close
from ggTrader.lab.strategies.xs_momentum import momentum_scores

REBALANCE_REASON = "rebalance"
DEAD_EXIT_REASON = "dead_exit"


@dataclass(frozen=True)
class Trade:
    """One execution: ``qty`` > 0 buys, < 0 sells, at ``price`` before any cost."""

    date: pd.Timestamp
    symbol: str
    qty: float
    price: float
    reason: Literal["rebalance", "dead_exit"]


@dataclass(frozen=True)
class TargetStep:
    """Equal-weight targets chosen on ``decision_date`` and applied on ``date``.

    Attributes:
        decision_date: the month-end trading day whose prices chose the targets.
        date: the execution date, the next trading day in ``close.index``.
        targets: symbol to weight; weights sum to 1 (or the dict is empty).
    """

    decision_date: pd.Timestamp
    date: pd.Timestamp
    targets: dict[str, float]


def _as_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(None) if index.tz is not None else index


def _decision_days(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """Last trading day of each calendar month that lies inside ``[start, end]``."""
    stamps = pd.Series(index, index=index)
    last = stamps.groupby(_as_naive(index).to_period("M")).max()
    lo, hi = _localize(pd.Timestamp(start), index), _localize(pd.Timestamp(end), index)
    return [d for d in last if lo <= d <= hi]


def _localize(stamp: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """The timestamp in the index's time zone (naive dates are read as that zone)."""
    if index.tz is not None and stamp.tzinfo is None:
        return stamp.tz_localize(index.tz)
    return stamp


def _aligned_eligibility(
    eligible: pd.DataFrame, close: pd.DataFrame, days: list[pd.Timestamp]
) -> np.ndarray:
    """Boolean rows for ``days`` x ``close.columns``; a missing day or stock is not eligible."""
    frame = eligible.copy()
    if close.index.tz is not None and frame.index.tz is None:
        frame.index = frame.index.tz_localize(close.index.tz)
    elif close.index.tz is None and frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)
    frame = frame.reindex(index=pd.DatetimeIndex(days), columns=close.columns, fill_value=False)
    return frame.fillna(False).to_numpy(dtype=bool)


def plan_trades(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    top_n: int,
    lookback: int,
    skip: int,
    dead_loss: float,
    start: str,
    end: str,
) -> list[TargetStep]:
    """Equal-weight top-``top_n`` momentum targets for every month-end in the window.

    Args:
        close: Daily closes, trading days x symbols, with enough history before ``start`` for
            the look-back.
        eligible: Boolean frame (days x symbols) from ``universe_frames.eligibility``.
        spans: Release spans (``symbol``, ``last_date``, optional ``has_successor``).
        top_n: Number of stocks to hold.
        lookback: Trading days back for the older price of the score.
        skip: Trading days back for the newer price of the score.
        dead_loss: Share of its last price a dying stock loses, 0 to 1 (checked, not used).
        start: First day a decision may be taken, ``YYYY-MM-DD``.
        end: Last day an execution may happen, ``YYYY-MM-DD``.

    Returns:
        One ``TargetStep`` per decision day, in date order.

    Raises:
        ValueError: on a bad ``top_n``, ``dead_loss`` or window.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if not 0.0 <= dead_loss <= 1.0:
        raise ValueError("dead_loss must be between 0 and 1")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start must not be after end")
    index = close.index
    days = _decision_days(index, start, end)
    if not days:
        return []
    scores = momentum_scores(close, lookback, skip).reindex(pd.DatetimeIndex(days)).to_numpy()
    allowed = _aligned_eligibility(eligible, close, days)
    entry_ok = forced_exit_frames(close, spans).entry_ok
    symbols = np.array(close.columns)
    last_exec = _localize(pd.Timestamp(end), index)
    steps: list[TargetStep] = []
    for i, decision in enumerate(days):
        pos = index.get_loc(decision) + 1
        if pos >= len(index) or index[pos] > last_exec:
            continue
        date = index[pos]
        usable = np.flatnonzero(allowed[i] & np.isfinite(scores[i]) & entry_ok.iloc[pos].to_numpy())
        order = np.argsort(-scores[i][usable], kind="stable")
        top = usable[order[:top_n]]
        targets = {str(s): 1.0 / len(top) for s in symbols[top]} if len(top) else {}
        steps.append(TargetStep(decision_date=decision, date=date, targets=targets))
    return steps


def span_end_dates(close: pd.DataFrame, spans: pd.DataFrame) -> pd.Series:
    """Last bar date of every series that ends inside the window, successor or not.

    Args:
        close: Daily closes, trading days x symbols.
        spans: Release spans (``symbol``, ``last_date``).

    Returns:
        Symbol to the last bar on or before its span end, only for spans that end within
        ``close.index``.
    """
    exits = forced_exit_frames(close, spans).exits
    flags = exits.to_numpy()
    ended = flags.any(axis=0)
    rows = flags.argmax(axis=0)[ended]
    return pd.Series(close.index[rows], index=close.columns[ended], name="last_bar")


def dead_exit_dates(close: pd.DataFrame, spans: pd.DataFrame) -> pd.Series:
    """Last bar date of the series that end inside the window and have no successor.

    A stock bought out or merged (``has_successor``) is not a dead stock: it keeps its price
    path and is handled by the ledger as an ordinary end of series.

    Args:
        close: Daily closes, trading days x symbols.
        spans: Release spans (``symbol``, ``last_date``, optional ``has_successor``).

    Returns:
        Symbol to its last bar date.
    """
    ends = span_end_dates(close, spans)
    if "has_successor" not in spans.columns:
        return ends
    successor = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
    flagged = successor.reindex(ends.index).fillna(False).astype(bool)
    return ends[~flagged]


def exit_close(close: pd.DataFrame, spans: pd.DataFrame, loss: float) -> pd.DataFrame:
    """Closes with each dying stock's last bar replaced by its sale price.

    A thin wrapper over the lab's ``simulation_close``: a stock without a successor sells at
    last price times (1 minus ``loss``), never exactly zero (``SIM_PRICE_FLOOR``); a stock with
    a successor keeps its last price.
    """
    return simulation_close(close, spans, loss)
