"""Study logic for dead-stock-random-baseline: spreads, the rule's place among the draws.

Plain functions over numbers; no file, database or network access. A draw record holds, per
dead-stock loss reading, the small set of fields below (never the month-end curves).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

LOSSES = (1.0, 0.5, 0.0)
COST = 0.00183  # the headline cost: 18.3 basis points per side
TAX = "low"
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)
DRAW_FIELDS = (
    "after_tax_final",
    "cagr_after_tax",
    "sharpe",
    "max_drawdown_pct",
    "mean_yearly_turnover",
    "cost_paid",
    "n_dead_exits",
    "skipped_buys",
)
HALF_CENT = 0.005


def loss_key(loss: float) -> str:
    """The key of one loss reading inside a draw record, for example ``loss1`` or ``loss0.5``."""
    return f"loss{loss:g}"


def draw_record(seed: int, cells: dict[float, dict[str, Any]]) -> dict[str, Any]:
    """One draw's record from the finished cells of its three loss readings."""
    if sorted(cells) != sorted(LOSSES):
        raise ValueError(f"a draw needs exactly the loss readings {LOSSES}, got {sorted(cells)}")
    record: dict[str, Any] = {"seed": int(seed)}
    for loss in LOSSES:
        record[loss_key(loss)] = {name: cells[loss][name] for name in DRAW_FIELDS}
    return record


def spread(values: list[float]) -> dict[str, float]:
    """Percentiles of finite ``values`` under the keys ``p0`` ... ``p100`` plus ``n``."""
    finite = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if len(finite) == 0:
        return {"n": 0}
    out: dict[str, float] = {"n": int(len(finite))}
    for q in PERCENTILES:
        out[f"p{q}"] = float(np.percentile(finite, q))
    out["mean"] = float(finite.mean())
    return out


def gaps_pp(draws: list[dict[str, Any]], loss: float, spy_cagr: float) -> list[float]:
    """Each draw's yearly after-tax growth minus SPY's, in percentage points."""
    return [(d[loss_key(loss)]["cagr_after_tax"] - spy_cagr) * 100.0 for d in draws]


def rule_position(rule_gap_pp: float, gaps: list[float]) -> dict[str, float]:
    """Where the rule's gap lies among the draws' gaps.

    ``share_draws_behind_rule`` is the share of draws that ended with a lower gap than the
    rule (the rule did better than that share of chance); ``share_draws_ahead_of_rule`` the
    share with a higher gap; ties count in neither.
    """
    finite = [g for g in gaps if math.isfinite(g)]
    if not finite:
        return {"n": 0}
    n = len(finite)
    behind = sum(1 for g in finite if g < rule_gap_pp)
    ahead = sum(1 for g in finite if g > rule_gap_pp)
    return {
        "n": n,
        "rule_gap_pp": rule_gap_pp,
        "share_draws_behind_rule": behind / n,
        "share_draws_ahead_of_rule": ahead / n,
    }


def summarise_arm(
    draws: list[dict[str, Any]], spy_cagr: float, rule_cells: dict[float, dict[str, Any]]
) -> dict[str, Any]:
    """Per loss reading: spreads of gap, final value, fall, turnover and cost, and the rule's place.

    Args:
        draws: draw records (``draw_record``).
        spy_cagr: SPY's yearly after-tax growth at the same cost and tax set (a fraction).
        rule_cells: the rule's saved cell per loss reading (needs ``gap_cagr_after_tax_pp``,
            ``after_tax_final``, ``max_drawdown_pct``, ``mean_yearly_turnover``).
    """
    out: dict[str, Any] = {"n_draws": len(draws)}
    for loss in LOSSES:
        key = loss_key(loss)
        gaps = gaps_pp(draws, loss, spy_cagr)
        rule = rule_cells[loss]
        out[key] = {
            "gap_pp": spread(gaps),
            "after_tax_final": spread([d[key]["after_tax_final"] for d in draws]),
            "max_drawdown_pct": spread([d[key]["max_drawdown_pct"] for d in draws]),
            "mean_yearly_turnover": spread([d[key]["mean_yearly_turnover"] for d in draws]),
            "cost_paid": spread([d[key]["cost_paid"] for d in draws]),
            "n_dead_exits": spread([float(d[key]["n_dead_exits"]) for d in draws]),
            "rule": {
                "gap_pp": rule["gap_cagr_after_tax_pp"],
                "after_tax_final": rule["after_tax_final"],
                "max_drawdown_pct": rule["max_drawdown_pct"],
                "mean_yearly_turnover": rule["mean_yearly_turnover"],
                "cost_paid": rule["cost_paid"],
                "n_dead_exits": rule["n_dead_exits"],
            },
            "position": rule_position(rule["gap_cagr_after_tax_pp"], gaps),
        }
    return out


def check_against_saved(
    computed: dict[str, dict[str, Any]], saved: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Differences larger than half a cent in ``after_tax_final`` between two sets of cells.

    Both dictionaries are keyed by cell name; every name in ``computed`` must be in ``saved``.
    A missing name is reported as a difference, never skipped.
    """
    diffs = []
    for name, cell in computed.items():
        if name not in saved:
            diffs.append({"cell": name, "problem": "not in the saved file"})
            continue
        a, b = float(cell["after_tax_final"]), float(saved[name]["after_tax_final"])
        if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > HALF_CENT:
            diffs.append({"cell": name, "computed": a, "saved": b})
    return diffs


def sanity_block(
    all_eligible_cagr: float, dice_cagrs: list[float], floor_pp: float = 0.5
) -> dict[str, Any]:
    """Compare the dice with an equal hold of the whole eligible list, loss 0.0, no cost, no tax.

    Chance picks should on average land close to the equal-weighted average of the same list.
    The block reports the difference in percentage points a year and calls it fine when it is
    within two standard errors of the dice mean or within ``floor_pp``, whichever is larger
    (the rule is fixed in the brief's amendment of 2026-11-25, before any draw).
    """
    finite = [c for c in dice_cagrs if math.isfinite(c)]
    if len(finite) < 2 or not math.isfinite(all_eligible_cagr):
        return {"ok": False, "reason": "fewer than two finite dice results or no finite hold"}
    mean = float(np.mean(finite))
    diff = (mean - all_eligible_cagr) * 100.0
    se = float(np.std(finite, ddof=1) / math.sqrt(len(finite))) * 100.0
    allowed = max(2.0 * se, floor_pp)
    return {
        "ok": bool(abs(diff) <= allowed),
        "all_eligible_cagr_pct": all_eligible_cagr * 100.0,
        "dice_mean_cagr_pct": mean * 100.0,
        "dice_min_cagr_pct": min(finite) * 100.0,
        "dice_max_cagr_pct": max(finite) * 100.0,
        "n_dice": len(finite),
        "difference_pp": diff,
        "standard_error_pp": se,
        "allowed_pp": allowed,
    }
