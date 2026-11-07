"""Bollinger reversion with conviction sizing on the engine contract.

Entries and exits are the Bollinger voter's. The entry size grows with how far the close is
below the lower band: ``min_size + depth * (max_size - min_size)`` where ``depth`` is the
distance below the band in band-half-widths, clipped to [0, 1]. The band half-width is
``middle - lower`` of the engine's Bollinger preset (``std * rolling std``), so the Bollinger
maths stays in vectorbt and the strategy only adds the clip and the interpolation.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.lab.strategies.contract import (
    ParamGrid,
    SignalOut,
    group_columns,
    resolve_combos,
)
from ggTrader.lab.strategies.voters import bb_bands, bb_pairs


def conviction_sizes(
    close: np.ndarray,
    lower: np.ndarray,
    middle: np.ndarray,
    entries: np.ndarray,
    min_size: float,
    max_size: float,
) -> np.ndarray:
    """Per-bar entry size, NaN where there is no entry (the base size then applies)."""
    width = middle - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.clip((lower - close) / np.where(width == 0.0, np.nan, width), 0.0, 1.0)
    sizes = min_size + depth * (max_size - min_size)
    return np.where(entries, sizes, np.nan)


class ConvictionBb:
    """Bollinger reversion; deeper oversold means a larger position (up to ``max_size``)."""

    name: ClassVar[str] = "conviction_bb"
    kind: ClassVar[str] = "signals"
    grid: ClassVar[ParamGrid] = ParamGrid.from_sweep(
        {
            "bb_period": [10, 15, 20, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "min_size": [0.01],
            "max_size": [0.03, 0.05, 0.08],
        }
    )
    live: ClassVar[dict[str, Any] | None] = None
    defaults: ClassVar[dict[str, Any]] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "min_size": 0.01,
        "max_size": 0.05,
    }

    def build(self, bars: Bars, grid: ParamGrid | None = None) -> SignalOut:
        """Entries, exits and per-bar sizes for every combination of ``grid``."""
        grid = grid if grid is not None else self.grid
        combos = resolve_combos(grid, self.defaults)
        if not combos:
            raise ValueError("grid has no valid combination")
        pairs = list(dict.fromkeys((c["bb_period"], c["bb_std"]) for c in combos))
        signals = bb_pairs(bars.close, pairs)
        bands = bb_bands(bars.close, pairs)
        close = bars.close.to_numpy()
        n_bars, n_sym = close.shape
        levels = grid.swept()
        columns = group_columns(combos, levels, bars.symbols)
        ent = np.empty((n_bars, len(combos) * n_sym), dtype=bool)
        ext = np.empty_like(ent)
        size = np.empty((n_bars, len(combos) * n_sym), dtype="float64")
        for i, combo in enumerate(combos):
            key = (combo["bb_period"], combo["bb_std"])
            entry, exit_ = signals[key]
            lower, middle = bands[key]
            sl = slice(i * n_sym, (i + 1) * n_sym)
            ent[:, sl], ext[:, sl] = entry, exit_
            size[:, sl] = conviction_sizes(
                close, lower, middle, entry, float(combo["min_size"]), float(combo["max_size"])
            )
        index = bars.close.index
        return SignalOut(
            entries=pd.DataFrame(ent, index=index, columns=columns),
            exits=pd.DataFrame(ext, index=index, columns=columns),
            combos=tuple(combos),
            size=pd.DataFrame(size, index=index, columns=columns),
            meta={"levels": levels},
        )
