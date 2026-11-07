"""Ensemble (N-of-M vote) on the engine contract: ``ensemble`` plus the five voter strategies.

``Ensemble.grid`` is the 48-combination WFO grid (``divergence_window=10, vol_mult=1.5``);
``Ensemble.live`` holds the deployed defaults (20 and 2.0), which are *not* in that grid
(spec OD14). Both are reference configurations: run the grid for WFO references and
``Ensemble.live_grid()`` for the pinned control row, the live golden and the trader.

Definitions follow epoch E1 (OD1(a)): configured vectorbt presets; the volume-BB voter keeps the
double vote (OD3(a)). The ``mtf`` voter and the ensemble exit-rule options (``exits_enabled``,
a swept ``td_stop``) are legacy-only; a fixed ``td_stop`` is passed through for the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from ggTrader.data.bars import Bars
from ggTrader.lab.strategies.contract import (
    Lt,
    ParamGrid,
    SignalOut,
    group_columns,
    resolve_combos,
)
from ggTrader.lab.strategies.vote import vote
from ggTrader.lab.strategies.voters import VOTER_DEFAULTS, VOTER_SPECS, voter_arrays

FIVE_VOTERS: tuple[str, ...] = ("bb", "rsi", "ema", "macd", "vbb")
#: Constructor defaults of the legacy ``EnsembleSignal`` (the deployed values).
ENSEMBLE_DEFAULTS: dict[str, Any] = {
    "min_agree": 2,
    "min_agree_exit": 2,
    **VOTER_DEFAULTS,
}


class Ensemble:
    """Majority vote of five voters; RSI exits on its own, the rest need ``min_agree_exit``."""

    name: ClassVar[str] = "ensemble"
    kind: ClassVar[str] = "signals"
    #: Legacy ``EnsembleSignal.sweep_params`` (48 valid combinations).
    grid: ClassVar[ParamGrid] = ParamGrid.from_sweep(
        {
            "min_agree": [2, 3, 4],
            "min_agree_exit": [1, 2],
            "bb_period": [20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50],
            "macd_fast": [12],
            "macd_slow": [26],
            "macd_signal": [9],
            "divergence_window": [10],
            "vol_period": [20],
            "vol_mult": [1.5],
            "weekly_rsi_period": [14],
            "weekly_rsi_oversold": [30],
            "weekly_rsi_exit": [50],
        },
        rules=(Lt("ema_fast", "ema_slow"), Lt("macd_fast", "macd_slow")),
    )
    #: Deployed defaults (OD14); outside the grid on ``divergence_window`` and ``vol_mult``.
    live: ClassVar[dict[str, Any]] = dict(ENSEMBLE_DEFAULTS)
    defaults: ClassVar[dict[str, Any]] = ENSEMBLE_DEFAULTS

    def __init__(self, voters: Sequence[str] = FIVE_VOTERS, td_stop: int | None = None) -> None:
        unknown = [v for v in voters if v not in VOTER_SPECS]
        if unknown or not voters:
            raise ValueError(
                f"voters must be a non-empty subset of {FIVE_VOTERS} (mtf is legacy-only); "
                f"got {tuple(voters)}"
            )
        self.voters = tuple(voters)
        self.td_stop = td_stop

    @classmethod
    def live_grid(cls) -> ParamGrid:
        """One-combination grid of the deployed defaults (OD14 control configuration)."""
        return ParamGrid.single(cls.live)

    def build(self, bars: Bars, grid: ParamGrid | None = None) -> SignalOut:
        """Votes for every combination of ``grid`` (default: the WFO grid) over ``bars``."""
        grid = grid if grid is not None else self.grid
        combos = resolve_combos(grid, self.defaults)
        if not combos:
            raise ValueError("grid has no valid combination")
        arrays = [voter_arrays(name, bars, combos) for name in self.voters]
        levels = grid.swept()
        columns = group_columns(combos, levels, bars.symbols)
        entries, exits = vote(arrays, combos, bars.close.index, columns, independent_exit="rsi")
        return SignalOut(
            entries=entries,
            exits=exits,
            combos=tuple(combos),
            td_stop=self.td_stop,
            meta={"voters": list(self.voters), "levels": levels},
        )
