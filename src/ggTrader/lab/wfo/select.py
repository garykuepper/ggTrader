"""Parameter selection: composite score, winner pick and the durable live-parameter pick.

``composite_score`` is the legacy math unchanged (0.5 x Sharpe + 0.3 x Sortino - 0.2 x |MaxDD|,
each min-max normalised over the grid, NaN floored to the grid's worst value). The inputs now
come from ``engine.stats`` (finite-guarded), so a flat or dead combination is NaN and floored
instead of winning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

#: Minimum number of walk-forward folds a combo must have won to be eligible as a live-param
#: recommendation. Guards against deploying a combo that scores best only on the most recent
#: window but never proved out-of-sample. Set to 0 to disable the filter.
MIN_LIVE_STABILITY = 1
#: Scores are rounded to this many decimals before a maximum is taken, so two combinations whose
#: composite differs only by float noise (about 1e-13: the normalisation sums in another order)
#: tie, and the first one in grid order wins as the legacy ``max`` did on exact ties (review
#: MINOR-1). A real difference is many orders of magnitude larger.
TIE_DECIMALS = 9


def _tie_key(score: float) -> float:
    return round(score, TIE_DECIMALS)


def _min_max_normalize(values: List[float]) -> List[float]:
    """Min-max scale to [0, 1]. Returns all 0.0 if min == max."""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def composite_score(metrics_list: List[Dict[str, float]]) -> List[float]:
    """Composite rank: 0.5*norm(sharpe) + 0.3*norm(sortino) - 0.2*norm(|maxdd|).

    NaN values are replaced with the worst value in each metric's range.
    """
    sharpes: List[float] = []
    sortinos: List[float] = []
    drawdowns: List[float] = []
    for m in metrics_list:
        sharpes.append(m.get("sharpe", float("nan")))
        sortinos.append(m.get("sortino", float("nan")))
        drawdowns.append(abs(m.get("max_drawdown_pct", 0.0)))

    def _floor_nan(vals: List[float]) -> List[float]:
        finite = [v for v in vals if not math.isnan(v)]
        floor = min(finite) if finite else 0.0
        return [floor if math.isnan(v) else v for v in vals]

    sharpes = _floor_nan(sharpes)
    sortinos = _floor_nan(sortinos)
    drawdowns = _floor_nan(drawdowns)

    ns = _min_max_normalize(sharpes)
    no = _min_max_normalize(sortinos)
    nd = _min_max_normalize(drawdowns)

    return [0.5 * ns[i] + 0.3 * no[i] - 0.2 * nd[i] for i in range(len(metrics_list))]


def best_index(scores: List[float]) -> int:
    """Index of the highest score; the first one wins a tie (float noise counts as a tie)."""
    return max(range(len(scores)), key=lambda j: _tie_key(scores[j]))


def pick_live_winner(
    train_metrics: List[Dict[str, Any]],
    scores: List[float],
    fold_win_counts: Dict[str, int],
    min_stability: int = MIN_LIVE_STABILITY,
) -> tuple[int, int]:
    """Pick the live-param combo, preferring out-of-sample durability.

    Among combos that won at least ``min_stability`` walk-forward folds, return the one with the
    best composite score on the recent training window. If no combo cleared that bar (e.g. every
    fold failed its gates), fall back to the global best composite score so a recommendation is
    still produced.

    Returns ``(winner_index, stability)`` where ``stability`` is the winner's fold-win count.
    """
    durable = [
        j
        for j in range(len(train_metrics))
        if fold_win_counts.get(train_metrics[j]["combo"], 0) >= min_stability
    ]
    pool = durable or list(range(len(train_metrics)))
    best_idx = max(pool, key=lambda j: _tie_key(scores[j]))
    stability = fold_win_counts.get(train_metrics[best_idx]["combo"], 0)
    return best_idx, stability
