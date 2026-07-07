"""Cross-sectional and dual momentum (weight-based lab strategies)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategy import LabConfig, Plan


class CrossSectionalMomentum:
    """12-1 cross-sectional momentum: top-N by trailing return, equal weight."""

    name = "xs_momentum"
    target_kind = "weights"

    def __init__(self, cfg: LabConfig) -> None:
        self.cfg = cfg

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "top_n": [10, 20, 50],
            "lookback": [126, 252],
            "skip": [0, 21],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]  # defense in depth: invariant to post-asof rows
        lookback, skip = self.cfg.lookback, self.cfg.skip
        scores: Dict[str, float] = {}
        for sym in eligible:
            closes = data[sym]["close"].dropna()
            if len(closes) < lookback + 1:
                continue
            past = float(closes.iloc[-(lookback + 1)])
            recent = float(closes.iloc[-(skip + 1)])
            if past <= 0.0 or not np.isfinite(past) or not np.isfinite(recent):
                continue
            scores[sym] = recent / past - 1.0
        ranked_all = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked_all:
            return []

        # Enforce GICS sector constraints if configured
        max_sec = self.cfg.max_sector_count
        if max_sec is not None:
            from ggTrader.lab.strategies.registry import apply_sector_constraints

            selected_symbols = apply_sector_constraints([sym for sym, _ in ranked_all], max_sec)
            ranked = [(sym, scores[sym]) for sym in selected_symbols][: self.cfg.top_n]
        else:
            ranked = ranked_all[: self.cfg.top_n]

        if not ranked:
            return []
        weight = 1.0 / len(ranked)
        return [{"symbol": s, "weight": weight, "momentum": m} for s, m in ranked]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # default: exit anything not re-selected
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets


class DualMomentum(CrossSectionalMomentum):
    """Cross-sectional momentum + absolute filter: negative-momentum picks go to cash.

    Weights are NOT renormalized — a dropped pick's slot stays in cash.
    """

    name = "dual_momentum"

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        return [p for p in super().select(asof, data, eligible) if p["momentum"] >= 0.0]


def __getattr__(name: str):  # PEP 562 — derive public names from the single registry
    from ggTrader.lab.strategies import registry

    if name == "STRATEGY_NAMES":
        return registry.weight_strategy_names()
    if name == "build_strategy":
        return registry.build_strategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
