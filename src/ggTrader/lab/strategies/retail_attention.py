"""Retail-attention (Google Trends search-spike) strategy on the weight contract.

Faithful port of the legacy logic: at each decision date, for the eligible symbols, take the
search-interest readings available as of the date (publication lag applied), score each symbol
by ``latest / mean(previous lookback_months readings) - 1``, and hold the top
``1/quintile`` bucket equal weight until the next decision.

The ``google_trends_interest`` table has 0 rows, so with the real loader this book is flat cash
(no verdict, paused). The interest loader is injectable so the logic is tested on synthetic
input; real data are absent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.lab.google_trends_data import (
    PUBLISH_LAG_DAYS,
    available_as_of,
    load_search_interest,
)
from ggTrader.lab.strategies.contract import ParamGrid, resolve_combos, stack_weight_plans

#: How far back to pull search-interest history per decision date (days).
LOOKBACK_DAYS = 400
InterestLoader = Callable[[list[str], str, str], pd.DataFrame]


def spike_scores(interest: pd.DataFrame, lookback_months: int) -> dict[str, float]:
    """Per symbol: ``latest / mean(previous lookback_months readings) - 1`` (enough readings)."""
    out: dict[str, float] = {}
    for sym, group in interest.sort_values("date").groupby("symbol"):
        if len(group) <= lookback_months:
            continue
        values = group["search_interest"].to_numpy()
        latest = values[-1]
        baseline = values[-1 - lookback_months : -1].mean()
        if pd.isna(latest) or pd.isna(baseline) or baseline == 0:
            continue
        out[sym] = (latest / baseline) - 1.0
    return out


class RetailAttention:
    """Long-only sleeve: the top bucket by search-interest spike, equal weight, monthly."""

    name: ClassVar[str] = "retail_attention"
    kind: ClassVar[str] = "weights"
    grid: ClassVar[ParamGrid] = ParamGrid.from_sweep(
        {"lookback_months": [2, 3, 6], "quintile": [4, 5]}
    )
    live: ClassVar[dict[str, Any] | None] = None
    #: Legacy constructor defaults.
    defaults: ClassVar[dict[str, Any]] = {"lookback_months": 3, "quintile": 5}

    def __init__(
        self,
        interest_loader: InterestLoader | None = None,
        publish_lag_days: int = PUBLISH_LAG_DAYS,
        min_history_bars: int = 400,
    ) -> None:
        self._loader: InterestLoader = interest_loader or load_search_interest
        self.publish_lag_days = publish_lag_days
        self.min_history_bars = min_history_bars
        self._cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    def _interest(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        key = (tuple(sorted(symbols)), start, end)
        if key not in self._cache:
            self._cache[key] = self._loader(symbols, start, end)
        return self._cache[key]

    def build(
        self,
        bars: Bars,
        grid: ParamGrid | None = None,
        rebal: pd.DatetimeIndex | None = None,
        eligible: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Target weights for every combination (flat cash when there is no interest data).

        Args:
            eligible: bool ``dates x symbols`` PIT eligibility; ``None`` means every symbol with
                at least ``min_history_bars`` closes up to the date.
        """
        if rebal is None:
            raise ValueError("retail_attention needs the rebalance dates")
        grid = grid if grid is not None else self.grid
        combos = resolve_combos(grid, self.defaults)
        close = bars.close
        rebal = pd.DatetimeIndex(rebal)
        history = close.notna().cumsum()
        plans: list[dict[pd.Timestamp, dict[str, float]]] = [{} for _ in combos]
        for asof in rebal:
            allowed = history.loc[asof] >= self.min_history_bars
            if eligible is not None:
                allowed &= eligible.reindex(columns=close.columns, fill_value=False).loc[asof]
            elig = sorted(allowed[allowed].index.tolist())
            per_combo = self._select(asof, elig, combos)
            for plan, picks in zip(plans, per_combo, strict=True):
                plan[asof] = picks
        return stack_weight_plans(close.index, list(rebal), plans, combos, grid.swept())

    def _select(
        self, asof: pd.Timestamp, elig: list[str], combos: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        """Weights picked at one decision date for every combination."""
        empty: list[dict[str, float]] = [{} for _ in combos]
        if not elig:
            return empty
        start = (asof - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        interest = self._interest(elig, start, asof.strftime("%Y-%m-%d"))
        if interest.empty:
            return empty
        interest = available_as_of(interest, asof, self.publish_lag_days)
        if interest.empty:
            return empty
        picks: list[dict[str, float]] = []
        for combo in combos:
            quintile = int(combo["quintile"])
            if len(elig) < quintile:
                picks.append({})
                continue
            spikes = spike_scores(interest, int(combo["lookback_months"]))
            if len(spikes) < quintile:
                picks.append({})
                continue
            ranked = pd.Series(spikes).sort_values(ascending=False)
            top = ranked.index[: max(1, len(ranked) // quintile)].tolist()
            picks.append({s: 1.0 / len(top) for s in top} if top else {})
        return picks
