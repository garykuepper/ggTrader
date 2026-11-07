"""Calendar folds and their bar positions for the vectorbt-engine walk-forward (spec 5.4).

``generate_folds`` is the legacy recurrence unchanged (12-month train, 3-month test, 3-month
step; ``train_end == test_start``, both ends inclusive, so adjacent windows share one bar).
:func:`fold_windows` maps each calendar window to bar positions with
``engine.windows.slice_positions`` so that ``index[i0:i1]`` equals the pandas label slice
``index[start:end]`` even when ``start`` is a Sunday (2021-01-31).
"""

from __future__ import annotations

from typing import List, NamedTuple

import pandas as pd

from ggTrader.engine import windows

TRAIN_MONTHS = 12
TEST_MONTHS = 3


class Fold(NamedTuple):
    """One walk-forward fold as calendar dates."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class FoldWindows(NamedTuple):
    """Bar positions ``(start, end)`` (end exclusive) of one fold's windows.

    ``anchor`` runs from the first loaded bar to ``train_end``: the anchor is fit on everything
    available as of the fold's ``train_end`` and nothing later (expanding window, causality).
    """

    train: tuple[int, int]
    test: tuple[int, int]
    anchor: tuple[int, int]


def generate_folds(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> List[Fold]:
    """Rolling fixed-width folds. Slides forward by ``test_months`` each step."""
    folds: List[Fold] = []
    cursor = eval_start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > eval_end:
            break
        folds.append(Fold(cursor, train_end, train_end, test_end))
        cursor += pd.DateOffset(months=test_months)
    return folds


def fold_windows(index: pd.DatetimeIndex, fold: Fold) -> FoldWindows:
    """Bar positions of a fold's train, test and anchor windows on ``index``."""
    train = windows.slice_positions(index, fold.train_start, fold.train_end)
    test = windows.slice_positions(index, fold.test_start, fold.test_end)
    return FoldWindows(train=train, test=test, anchor=(0, train[1]))
