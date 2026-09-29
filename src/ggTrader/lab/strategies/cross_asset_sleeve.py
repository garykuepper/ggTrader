"""SPY + cross-asset sleeve -- RESEARCH_SNAPSHOT.md §6 Tier 1 #1, brief
docs/research/briefs/2026-09-28-cross-asset-sleeve.md.

Hold (1 - sleeve_weight) in SPY and sleeve_weight split equally across
long-duration Treasuries (TLT), gold (GLD) and a broad commodity fund
(PDBC, the 1099 version of DBC), rebalanced at every month-end close.

Two arms, pre-registered:
  * static (``trend=False``): the sleeve is always held;
  * slow trend (``trend=True``): a leg whose trailing ``cfg.lookback``-bar
    total return is <= 0 at the month-end close has its weight moved to BIL
    (1-3 month T-bills) rather than lab cash, which earns 0%.

Sources: Hurst, Ooi & Pedersen (2017), SSRN 2993026; Kurth et al., arXiv
2607.01550 (slow trend survives in bonds/commodities). Close adaptation:
the papers are long/short futures, this is long/flat ETFs.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategy import LabConfig, Plan

CORE = "SPY"
LEGS = ("TLT", "GLD", "PDBC")
CASH = "BIL"
CROSS_ASSET_SLEEVE_UNIVERSE = [CORE, *LEGS, CASH]


def leg_is_on(close: pd.Series, lookback: int) -> bool:
    """True if the trailing ``lookback``-bar total return is strictly positive.

    Too little history counts as off: the rule can't be evaluated yet.
    """
    closes = close.dropna()
    if len(closes) < lookback + 1:
        return False
    past, last = float(closes.iloc[-(lookback + 1)]), float(closes.iloc[-1])
    return bool(np.isfinite(past) and past > 0 and last / past - 1.0 > 0.0)


class CrossAssetSleeveStrategy:
    """Monthly-rebalanced SPY + equal-weight TLT/GLD/PDBC sleeve, optionally trend-timed."""

    name = "cross_asset_sleeve"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        sleeve_weight: float = 0.20,
        trend: bool = True,
        legs: Sequence[str] = LEGS,
    ) -> None:
        self.cfg = cfg
        self.sleeve_weight = sleeve_weight
        self.trend = trend
        self.legs = tuple(legs)

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        # Pre-registered robustness grid for the trend arm only (brief step 3).
        return {"lookback": [126, 189, 252]}

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        leg_w = self.sleeve_weight / len(self.legs)
        weights = {CORE: 1.0 - self.sleeve_weight, CASH: 0.0}
        close = extract_close(data, list(self.legs)) if self.trend else None
        for leg in self.legs:
            on = not self.trend or leg_is_on(close[leg], self.cfg.lookback)
            weights[leg] = leg_w if on else 0.0
            if not on:
                weights[CASH] += leg_w
        return [{"symbol": s, "weight": w} for s, w in weights.items()]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        """Each month-end plan takes effect on the next bar (no same-bar lookahead)."""
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
