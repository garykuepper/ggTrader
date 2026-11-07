"""The five voter strategies on the engine contract: each is one voter, no voting.

``ema_cross``, ``bb_reversion``, ``rsi_reversion``, ``macd_divergence`` and
``volume_bb_reversion`` share the voter arrays of the ensemble (``voters.py``), so a voter is
defined once. Grids are the legacy ``sweep_params`` with their validity rules as data.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.lab.strategies.contract import (
    Lt,
    ParamGrid,
    SignalOut,
    group_columns,
    resolve_combos,
)
from ggTrader.lab.strategies.voters import VOTER_DEFAULTS, VOTER_SPECS, voter_arrays


class _SingleVoter:
    """Shared ``build`` for a strategy that trades one voter's entries and exits."""

    name: ClassVar[str]
    kind: ClassVar[str] = "signals"
    voter: ClassVar[str]
    grid: ClassVar[ParamGrid]
    live: ClassVar[dict[str, Any] | None] = None

    defaults: ClassVar[dict[str, Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Legacy constructor defaults for the voter's own parameters.
        cls.defaults = {k: VOTER_DEFAULTS[k] for k in VOTER_SPECS[cls.voter].keys}

    def build(self, bars: Bars, grid: ParamGrid | None = None) -> SignalOut:
        """Entry and exit signals for every combination of ``grid`` over ``bars``."""
        grid = grid if grid is not None else self.grid
        combos = resolve_combos(grid, self.defaults)
        if not combos:
            raise ValueError("grid has no valid combination")
        arrays = voter_arrays(self.voter, bars, combos)
        levels = grid.swept()
        columns = group_columns(combos, levels, bars.symbols)
        n_bars, n_sym = bars.close.shape
        ent = np.empty((n_bars, len(combos) * n_sym), dtype=bool)
        ext = np.empty_like(ent)
        for i, combo in enumerate(combos):
            entry, exit_ = arrays.pick(combo)
            ent[:, i * n_sym : (i + 1) * n_sym] = entry
            ext[:, i * n_sym : (i + 1) * n_sym] = exit_
        index = bars.close.index
        return SignalOut(
            entries=pd.DataFrame(ent, index=index, columns=columns),
            exits=pd.DataFrame(ext, index=index, columns=columns),
            combos=tuple(combos),
            meta={"voter": self.voter, "levels": levels},
        )


class EmaCross(_SingleVoter):
    """EMA crossover: long when the fast EMA crosses above the slow one, flat on the reverse."""

    name = "ema_cross"
    voter = "ema"
    grid = ParamGrid.from_sweep(
        {"ema_fast": [5, 10, 20, 50], "ema_slow": [20, 30, 50, 100, 200]},
        rules=(Lt("ema_fast", "ema_slow"),),
    )


class BbReversion(_SingleVoter):
    """Bollinger reversion: buy below the lower band, sell at the middle band."""

    name = "bb_reversion"
    voter = "bb"
    grid = ParamGrid.from_sweep({"bb_period": [10, 15, 20, 30], "bb_std": [1.5, 2.0, 2.5, 3.0]})


class RsiReversion(_SingleVoter):
    """RSI reversion: buy when RSI crosses below the oversold level, sell above the exit level."""

    name = "rsi_reversion"
    voter = "rsi"
    grid = ParamGrid.from_sweep(
        {"rsi_period": [7, 14, 21], "rsi_oversold": [20, 25, 30], "rsi_exit": [50, 55, 60]}
    )


class MacdDivergence(_SingleVoter):
    """MACD bullish divergence entry; exit when the histogram crosses below zero."""

    name = "macd_divergence"
    voter = "macd"
    grid = ParamGrid.from_sweep(
        {
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [9],
            "divergence_window": [10, 20],
        },
        rules=(Lt("macd_fast", "macd_slow"),),
    )


class VolumeBbReversion(_SingleVoter):
    """Bollinger reversion gated by a volume spike (capitulation filter)."""

    name = "volume_bb_reversion"
    voter = "vbb"
    grid = ParamGrid.from_sweep(
        {
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "vol_period": [20],
            "vol_mult": [1.5, 2.0, 2.5],
        }
    )
