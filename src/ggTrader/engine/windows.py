"""Calendar windows as bar positions, and a thin wrapper over vectorbt's ``range_split``.

ggTrader's walk-forward folds are calendar dates (12-month train, 3-month test, 3-month
step; ``lab.wfo.generate_folds`` stays in lab). vectorbt's splitters count bars and drop the
date index, so every consumer works on bar positions: :func:`slice_positions` maps a
calendar window to ``(i0, i1)`` with ``searchsorted`` so that ``index[i0:i1]`` equals the
pandas label slice ``index[start:end]`` even when ``start`` is a Sunday (2021-01-31).

The signal path simulates the whole history once and slices ``iloc[i0:i1]``; only the weight
path stacks windows with a fresh start per (combination, fold) via :func:`stack_windows`
(spec 5.4).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import vectorbt as vbt  # noqa: F401  (registers the .vbt accessor)

Closed = Literal["both", "left", "right", "neither"]
FOLD_FIELDS = ("train_start", "train_end", "test_start", "test_end")


def slice_positions(
    index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, *, closed: Closed = "both"
) -> tuple[int, int]:
    """Bar positions ``(i0, i1)`` such that ``index[i0:i1]`` lies between two calendar dates.

    Args:
        index: sorted bar dates.
        start: window start; need not be a trading day.
        end: window end; need not be a trading day.
        closed: which ends are included, as in pandas ("both" equals ``.loc[start:end]``).

    Raises:
        ValueError: if the index is not sorted or ``end`` is before ``start``.
    """
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    if end < start:
        raise ValueError(f"window end {end} is before start {start}")
    i0 = int(index.searchsorted(start, side="left" if closed in ("both", "left") else "right"))
    i1 = int(index.searchsorted(end, side="right" if closed in ("both", "right") else "left"))
    return i0, max(i0, i1)


def fold_positions(
    index: pd.DatetimeIndex, folds: Iterable[Any], *, closed: Closed = "both"
) -> pd.DataFrame:
    """Bar-position windows for each fold.

    Args:
        index: sorted bar dates.
        folds: objects with ``train_start``, ``train_end``, ``test_start``, ``test_end``
            (``lab.wfo.Fold`` fits; the engine reads them by attribute).
        closed: passed to :func:`slice_positions`.

    Returns:
        One row per fold; columns ``train_start``, ``train_end``, ``test_start``,
        ``test_end`` hold *positions* where ``train`` is ``index[train_start:train_end]``
        (end exclusive) and likewise for ``test``.
    """
    rows = []
    for fold in folds:
        train = slice_positions(index, fold.train_start, fold.train_end, closed=closed)
        test = slice_positions(index, fold.test_start, fold.test_end, closed=closed)
        rows.append([*train, *test])
    return pd.DataFrame(rows, columns=list(FOLD_FIELDS), dtype="int64")


@dataclass(frozen=True, eq=False)
class StackedWindows:
    """Windows stacked as columns: ``frame`` has columns ``(window, original column)``.

    ``indexes[k]`` holds the dates of window ``k``; shorter windows are NaN-padded at the tail.
    """

    frame: pd.DataFrame
    indexes: tuple[pd.DatetimeIndex, ...]

    def restore(self, values: pd.Series, window: int) -> pd.Series:
        """Put the dates back on one stacked column of window ``window``, dropping padding."""
        dates = self.indexes[window]
        return pd.Series(values.to_numpy()[: len(dates)], index=dates, name=values.name)


def stack_windows(frame: pd.DataFrame, windows: Sequence[tuple[int, int]]) -> StackedWindows:
    """Stack ``frame.iloc[a:b]`` windows side by side with ``range_split``.

    Args:
        frame: (time x column) data.
        windows: ``(start, end)`` bar positions, end exclusive.

    Raises:
        ValueError: on an empty list, an empty window or one outside the frame.
    """
    if not windows:
        raise ValueError("windows is empty")
    n_bars = len(frame)
    for start, end in windows:
        if not (0 <= start < end <= n_bars):
            raise ValueError(f"window ({start}, {end}) is empty or outside 0..{n_bars}")
    starts = np.array([w[0] for w in windows])
    inclusive_ends = np.array([w[1] - 1 for w in windows])  # range_split's end is inclusive
    stacked, indexes = frame.vbt.range_split(start_idxs=starts, end_idxs=inclusive_ends)
    return StackedWindows(frame=stacked, indexes=tuple(indexes))
