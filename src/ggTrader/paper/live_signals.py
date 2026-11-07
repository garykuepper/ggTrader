"""The live ensemble signal frames: one place that turns ``Bars`` into entries and exits.

``signal_runner`` (today's buys, sells and last exits) and ``overlay`` (the trailing sleeve
curve) both need the deployed ensemble's signals. They share this module so the two can never
describe different strategies (the "Invariant 1" the overlay tests guard): the validated
``Ensemble.build()`` with the one-combination grid of the live defaults (spec OD14).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from ggTrader.data.bars import BAR_FIELDS, Bars
from ggTrader.lab.strategies.ensemble import Ensemble

#: A symbol needs this many valid closes before it may trade (the legacy ``LabConfig`` value the
#: live path used; recent index additions with 60-199 bars are covered by the live golden).
MIN_HISTORY_BARS = 60


def live_ensemble_frames(bars: Bars) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry and exit frames (``dates x symbols``) of the deployed ensemble over ``bars``."""
    out = Ensemble().build(bars, Ensemble.live_grid())
    symbols = bars.symbols
    return out.entries.set_axis(symbols, axis=1), out.exits.set_axis(symbols, axis=1)


def history_eligibility(close: pd.DataFrame, min_bars: int = MIN_HISTORY_BARS) -> pd.DataFrame:
    """Per bar and symbol: has the symbol shown at least ``min_bars`` valid closes up to here?"""
    return close.notna().cumsum() >= min_bars


def subset_bars(bars: Bars, symbols: list[str]) -> Bars:
    """The same bars restricted to ``symbols`` (columns only; rows and provenance unchanged)."""
    return replace(bars, **{name: bars.frame(name)[symbols] for name in BAR_FIELDS})
