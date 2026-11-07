"""Lookahead checks for the engine contract (generalises ``lab.harness.leak_check``).

Two complementary tests, both usable on real or synthetic bars:

* **Truncation** (signals, spec 7.2): the signal row at bar ``t`` computed from bars up to
  ``t`` must equal the row at ``t`` computed from the full tape.
* **Perturbation** (signals and weights): replace every bar after ``t`` with different values
  (same index); anything the strategy emits for ``t`` must not move. For weights the row of
  interest is the one on the first bar after the decision date, which exists in both runs.

vectorbt label functions (``vectorbt.labels``) look forward by design and are banned by the
boundary test; this module catches any other lookahead a strategy might introduce.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ggTrader.data.bars import BAR_FIELDS, Bars
from ggTrader.lab.strategies.contract import ParamGrid, next_bar_positions


def truncate_bars(bars: Bars, last: pd.Timestamp) -> Bars:
    """Bars with every row after ``last`` removed."""
    cut = {name: bars.frame(name).loc[:last] for name in BAR_FIELDS}
    return replace(bars, **cut)


def perturb_after(bars: Bars, last: pd.Timestamp, seed: int = 0) -> Bars:
    """Bars whose rows after ``last`` are scaled by random factors (same index and shape)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in BAR_FIELDS:
        frame = bars.frame(name).copy()
        later = frame.index > last
        factor = rng.uniform(0.4, 2.5, size=(int(later.sum()), frame.shape[1]))
        frame.loc[later] = frame.loc[later].to_numpy() * factor
        out[name] = frame
    return replace(bars, **out)


def _same(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def signal_leak_check(
    strategy: Any,
    bars: Bars,
    grid: ParamGrid,
    dates: Sequence[pd.Timestamp],
    *,
    perturb: bool = True,
) -> list[str]:
    """Dates (with the field that moved) where a signal strategy's row depends on later bars.

    Args:
        strategy: object with ``build(bars, grid) -> SignalOut``.
        dates: bar dates to test.
        perturb: also run the perturbation test (truncation always runs).

    Returns:
        Empty when no lookahead was found, else messages like ``"2024-01-05 entries (truncate)"``.
    """
    full = strategy.build(bars, grid)
    problems: list[str] = []
    for t in dates:
        variants = {"truncate": truncate_bars(bars, t)}
        if perturb:
            variants["perturb"] = perturb_after(bars, t)
        for label, variant in variants.items():
            other = strategy.build(variant, grid)
            row = bars.close.index.get_loc(t)
            for field in ("entries", "exits"):
                a = getattr(full, field).iloc[row]
                b = getattr(other, field).iloc[row]
                if not _same(a, b):
                    problems.append(f"{t.date()} {field} ({label})")
            if isinstance(full.size, pd.DataFrame) and not _same(
                full.size.iloc[row], other.size.iloc[row]
            ):
                problems.append(f"{t.date()} size ({label})")
    return problems


def weight_leak_check(
    strategy: Any,
    bars: Bars,
    grid: ParamGrid,
    decision_dates: Sequence[pd.Timestamp],
    eligible: pd.DataFrame | None = None,
) -> list[str]:
    """Decision dates whose target weights change when every later bar is replaced.

    The weights land on the first bar after the decision date (``next_bar_positions``); that
    bar exists in the perturbed run too, so the rows are directly comparable.
    """
    problems: list[str] = []
    for asof in decision_dates:
        pos = next_bar_positions(bars.close.index, [asof])[0]
        if pos < 0:
            continue
        rebal = pd.DatetimeIndex([asof])
        base = strategy.build(bars, grid, rebal, eligible)
        other = strategy.build(perturb_after(bars, asof), grid, rebal, eligible)
        if not _same(base.iloc[pos], other.iloc[pos]):
            problems.append(f"{asof.date()} weights")
    return problems
