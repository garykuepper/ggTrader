"""SPY + cross-asset sleeve on the weight contract (static shadow GO, OD11).

Hold ``1 - sleeve_weight`` in SPY and ``sleeve_weight`` split equally across TLT, GLD and PDBC,
rebalanced on the decision dates. The slow-trend arm (``trend=True``, the legacy constructor
default) moves a leg to BIL when its trailing ``lookback``-bar return is not positive; the
static arm is ``ParamGrid.single({"trend": False})``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.lab.strategies.contract import (
    ParamGrid,
    resolve_combos,
    stack_weight_plans,
)

CORE = "SPY"
LEGS: tuple[str, ...] = ("TLT", "GLD", "PDBC")
CASH = "BIL"
UNIVERSE: list[str] = [CORE, *LEGS, CASH]


def leg_on_flags(close: pd.DataFrame, legs: Sequence[str], lookback: int) -> pd.DataFrame:
    """True where a leg's trailing ``lookback``-bar return is strictly positive.

    Counted on each leg's valid closes and carried forward to bars after its last print, as the
    legacy ``leg_is_on`` evaluated on ``data.loc[:asof]``. Too little history counts as off.
    """
    flags: dict[str, pd.Series] = {}
    for leg in legs:
        valid = close[leg].dropna()
        past = valid.shift(lookback)
        on = np.isfinite(past) & (past > 0.0) & (valid / past - 1.0 > 0.0)
        flags[leg] = on.astype(float).reindex(close.index, method="ffill").fillna(0.0) > 0.5
    return pd.DataFrame(flags, index=close.index)


class CrossAssetSleeve:
    """Monthly SPY core plus an equal-weight TLT/GLD/PDBC sleeve, optionally trend-timed."""

    name: ClassVar[str] = "cross_asset_sleeve"
    kind: ClassVar[str] = "weights"
    grid: ClassVar[ParamGrid] = ParamGrid.from_sweep({"lookback": [126, 189, 252]})
    live: ClassVar[dict[str, Any] | None] = None
    #: Legacy constructor and ``LabConfig`` defaults.
    defaults: ClassVar[dict[str, Any]] = {"sleeve_weight": 0.20, "trend": True, "lookback": 252}

    def __init__(self, legs: Sequence[str] = LEGS) -> None:
        self.legs = tuple(legs)

    def build(
        self,
        bars: Bars,
        grid: ParamGrid | None = None,
        rebal: pd.DatetimeIndex | None = None,
        eligible: pd.DataFrame | None = None,  # noqa: ARG002  (fixed ETF universe)
    ) -> pd.DataFrame:
        """Target weights on the bar after each decision date, for every combination."""
        if rebal is None:
            raise ValueError("cross_asset_sleeve needs the rebalance dates")
        grid = grid if grid is not None else self.grid
        combos = resolve_combos(grid, self.defaults)
        close = bars.close
        missing = [s for s in (CORE, CASH, *self.legs) if s not in close.columns]
        if missing:
            raise ValueError(f"cross_asset_sleeve needs bars for {missing}")
        rebal = pd.DatetimeIndex(rebal)
        flag_cache: dict[int, pd.DataFrame] = {}
        plans: list[dict[pd.Timestamp, dict[str, float]]] = []
        for combo in combos:
            sleeve = float(combo["sleeve_weight"])
            leg_w = sleeve / len(self.legs)
            trend = bool(combo["trend"])
            lookback = int(combo["lookback"])
            if trend and lookback not in flag_cache:
                flag_cache[lookback] = leg_on_flags(close, self.legs, lookback)
            plan: dict[pd.Timestamp, dict[str, float]] = {}
            for asof in rebal:
                weights = {CORE: 1.0 - sleeve, CASH: 0.0}
                for leg in self.legs:
                    on = (not trend) or bool(flag_cache[lookback].at[asof, leg])
                    weights[leg] = leg_w if on else 0.0
                    if not on:
                        weights[CASH] += leg_w
                plan[asof] = weights
            plans.append(plan)
        return stack_weight_plans(close.index, list(rebal), plans, combos, grid.swept())
