"""Worked example for docs/adding_a_strategy.md: an SMA crossover on the engine contract.

Copy this file into ``src/ggTrader/lab/strategies/`` and ``ggt lab --strategy sma_cross --wfo``
finds it; nothing else needs editing. ``tests/lab/strategies/test_docs_example.py`` runs it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.engine import indicators as ind
from ggTrader.engine import signals as sig
from ggTrader.lab.strategies.contract import Lt, ParamGrid, SignalOut, group_columns, resolve_combos


class SmaCross:
    """Long when the fast SMA crosses above the slow one; exit when it crosses back below."""

    name: ClassVar[str] = "sma_cross"
    kind: ClassVar[str] = "signals"
    grid: ClassVar[ParamGrid] = ParamGrid.from_sweep(
        {"fast": [10, 20], "slow": [50, 100]}, rules=(Lt("fast", "slow"),)
    )
    live: ClassVar[dict[str, Any] | None] = None

    def build(self, bars: Bars, grid: ParamGrid | None = None) -> SignalOut:
        """Entries and exits for every valid (fast, slow) pair, one column block per pair."""
        grid = grid or self.grid
        combos = resolve_combos(grid, {})
        close = bars.close
        entries, exits = [], []
        for combo in combos:
            fast = ind.values(ind.sma(close, combo["fast"]).ma, like=close)
            slow = ind.values(ind.sma(close, combo["slow"]).ma, like=close)
            entries.append(sig.cross_above(fast, slow))
            exits.append(sig.cross_below(fast, slow))
        columns = group_columns(combos, grid.swept(), bars.symbols)
        return SignalOut(
            entries=pd.concat(entries, axis=1).set_axis(columns, axis=1),
            exits=pd.concat(exits, axis=1).set_axis(columns, axis=1),
            combos=tuple(combos),
        )
