"""The defensive anchor parameter set, fit only on data up to each fold's ``train_end``.

Causality (spec 5.4 item 7): the anchor of fold ``k`` is scored on the equity from the first
loaded bar to ``train_end_k``, read from the same single simulation as the train sweep, never
from a window that reaches past ``train_end_k``. The expanding window means the input differs
for every fold, so the anchor is recomputed per fold; only the simulation is shared.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple

#: Annual CAGR (percent) a combo must beat to be an anchor candidate.
RISK_FREE_RATE_PCT = 4.0


class AnchorSet(NamedTuple):
    """Global min-drawdown parameter set for defensive fallback."""

    combo: str
    params: Dict[str, Any]
    max_drawdown_pct: float
    cagr_pct: float
    sharpe: float


NO_ANCHOR = AnchorSet(
    combo="none", params={}, max_drawdown_pct=0.0, cagr_pct=0.0, sharpe=float("nan")
)


def pick_anchor(
    metrics_list: List[Dict[str, Any]], risk_free_rate: float = RISK_FREE_RATE_PCT
) -> AnchorSet:
    """Minimise max drawdown subject to CAGR > risk-free; least-drawdown combo if none clears it.

    ``metrics_list`` holds one row per combo, scored from the first loaded bar to ``train_end``.
    A combination whose Sharpe is NaN never traded (or is flat): its drawdown of 0 would make it
    the "safest" fallback and the book would sit in cash by accident, so it is not a candidate
    (review MINOR-3; legacy would pick it). With no candidate left there is no anchor.
    """
    metrics_list = [m for m in metrics_list if math.isfinite(m.get("sharpe", float("nan")))]
    if not metrics_list:
        return NO_ANCHOR
    viable = [m for m in metrics_list if m.get("cagr_pct", 0.0) > risk_free_rate]
    candidates = viable if viable else metrics_list
    # max_drawdown_pct is negative, so the maximum is the least drawdown.
    best = max(candidates, key=lambda m: m.get("max_drawdown_pct", float("-inf")))
    return AnchorSet(
        combo=best["combo"],
        params=best["params"],
        max_drawdown_pct=best.get("max_drawdown_pct", 0.0),
        cagr_pct=best.get("cagr_pct", 0.0),
        sharpe=best.get("sharpe", float("nan")),
    )
