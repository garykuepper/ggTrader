"""12-1 cross-sectional momentum on the weight contract (the weight-path reference, OD11).

Top-N by trailing return, equal weight, decided at each rebalance date and traded on the next
bar. Positional look-backs follow the legacy ``select``: they count *valid* closes of each symbol
(``dropna()`` first), so a symbol with holes is scored on its own valid history. Eligibility
(the PIT universe at each date) is an input, because the ranking must only see eligible names.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.lab.strategies.contract import (
    ParamGrid,
    resolve_combos,
    stack_weight_plans,
)


def momentum_scores(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """``close[-(skip+1)] / close[-(lookback+1)] - 1`` on each symbol's valid closes.

    The value on a date is the score a decision on that date would use: the last valid close at
    or before it, stale when the symbol has stopped printing (the legacy behaviour).
    """
    scored: dict[str, pd.Series] = {}
    for symbol in close.columns:
        valid = close[symbol].dropna()
        past = valid.shift(lookback)
        recent = valid.shift(skip)
        score = (recent / past - 1.0).where((past > 0.0) & np.isfinite(past) & np.isfinite(recent))
        scored[symbol] = score.reindex(close.index, method="ffill")
    return pd.DataFrame(scored, index=close.index, columns=close.columns)


class XsMomentum:
    """Cross-sectional 12-1 momentum, long top-N equal weight, monthly decisions."""

    name: ClassVar[str] = "xs_momentum"
    kind: ClassVar[str] = "weights"
    grid: ClassVar[ParamGrid] = ParamGrid.from_sweep(
        {"top_n": [10, 20, 50], "lookback": [126, 252], "skip": [0, 21]}
    )
    live: ClassVar[dict[str, Any] | None] = None
    #: ``LabConfig`` defaults (top_n 50, lookback 252, skip 21).
    defaults: ClassVar[dict[str, Any]] = {"top_n": 50, "lookback": 252, "skip": 21}

    def build(
        self,
        bars: Bars,
        grid: ParamGrid | None = None,
        rebal: pd.DatetimeIndex | None = None,
        eligible: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Target weights for every combination.

        Args:
            bars: daily bars.
            grid: parameter grid (default ``XsMomentum.grid``).
            rebal: decision dates (from ``lab.data.rebalance_dates``); required.
            eligible: bool ``dates x symbols`` eligibility (``lab.universe.pit_mask``); ``None``
                means every symbol with a score.
        """
        if rebal is None:
            raise ValueError("xs_momentum needs the rebalance dates")
        grid = grid if grid is not None else self.grid
        combos = resolve_combos(grid, self.defaults)
        close = bars.close
        rebal = pd.DatetimeIndex(rebal)
        elig_rows = (
            np.ones((len(rebal), close.shape[1]), dtype=bool)
            if eligible is None
            else eligible.reindex(index=rebal, columns=close.columns, fill_value=False).to_numpy()
        )
        symbols = np.array(close.columns)
        score_cache: dict[tuple[int, int], np.ndarray] = {}
        plans: list[dict[pd.Timestamp, dict[str, float]]] = []
        for combo in combos:
            key = (int(combo["lookback"]), int(combo["skip"]))
            if key not in score_cache:
                scores = momentum_scores(close, *key)
                score_cache[key] = scores.reindex(rebal).to_numpy()
            rows = score_cache[key]
            plan: dict[pd.Timestamp, dict[str, float]] = {}
            for i, asof in enumerate(rebal):
                usable = np.flatnonzero(elig_rows[i] & np.isfinite(rows[i]))
                order = np.argsort(-rows[i][usable], kind="stable")
                top = usable[order[: int(combo["top_n"])]]
                plan[asof] = {str(s): 1.0 / len(top) for s in symbols[top]} if len(top) else {}
            plans.append(plan)
        return stack_weight_plans(close.index, list(rebal), plans, combos, grid.swept())
