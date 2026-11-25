"""Random-pick monthly plans for the baseline: the rule's eligible set, a random order.

``eligible_sets`` repeats the first test's planner exactly up to the sorting step, so the
stocks a random portfolio may hold on a decision day are the ones the momentum rule could hold.
``rule_ranking_plan`` sorts that set by score and must reproduce ``rb_replay.plan_trades``
(a test pins this). ``plan_fresh_dice`` (arm R) draws the portfolio anew every month;
``plan_matched`` (arm M) replaces, each month, about as many names as the rule replaced.

Matching rule of arm M (fixed before any draw): with ``need`` = the number of names to hold
(50, or fewer when fewer are eligible) and ``k`` = names the rule actually bought for the
first time that month (its fills), keep
``need - k`` of the names already held that are still eligible, chosen at random, then fill
with random eligible names not held last month; if too few such names exist, all of them
are taken and the rest come from names dropped this month. If fewer names remain
eligible than ``need - k`` the portfolio replaces more than ``k`` names (deaths and lost
eligibility force it). The first month holds ``need`` fresh random names, as the rule buys
``need`` names then.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rb_replay import (
    TargetStep,
    _aligned_eligibility,
    _decision_days,
    _localize,
    span_end_dates,
)

from ggTrader.lab.strategies.xs_momentum import momentum_scores

SEED_BASE = 20261125
DUST = 1e-9  # a position below this share of its own peak is float dust, not a holding
WINDOW_INDEX = {"early": 0, "dev": 1, "holdout": 2}
ARM_INDEX = {"R": 0, "M": 1}


@dataclass(frozen=True)
class DecisionSet:
    """One decision day: when it acts and which stocks may be held, with their scores."""

    decision_date: pd.Timestamp
    date: pd.Timestamp
    usable: np.ndarray  # column positions of the stocks the rule could hold
    scores: np.ndarray  # momentum score of every column on this decision day


def draw_seed(window: str, arm: str, draw: int) -> int:
    """The registered seed of one draw: base + 1,000,000 x window + 10,000 x arm + draw."""
    if window not in WINDOW_INDEX or arm not in ARM_INDEX or not 0 <= draw < 10_000:
        raise ValueError(f"bad draw identity: {window!r}, {arm!r}, {draw!r}")
    return SEED_BASE + 1_000_000 * WINDOW_INDEX[window] + 10_000 * ARM_INDEX[arm] + draw


def eligible_sets(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    lookback: int,
    skip: int,
    start: str,
    end: str,
) -> list[DecisionSet]:
    """Every decision day with the stocks the momentum rule could choose from.

    Mirrors ``rb_replay.plan_trades`` line for line up to the sort: the same decision days,
    the same execution day, the same eligibility, a finite score, and a series that has not
    already ended before the decision day.
    """
    index = close.index
    days = _decision_days(index, start, end)
    if not days:
        return []
    scores = momentum_scores(close, lookback, skip).reindex(pd.DatetimeIndex(days)).to_numpy()
    allowed = _aligned_eligibility(eligible, close, days)
    ended = span_end_dates(close, spans).reindex(close.columns)
    last_exec = _localize(pd.Timestamp(end), index)
    sets: list[DecisionSet] = []
    for i, decision in enumerate(days):
        pos = index.get_loc(decision) + 1
        if pos >= len(index) or index[pos] > last_exec:
            continue
        already_dead = (ended < decision).to_numpy()
        usable = np.flatnonzero(allowed[i] & np.isfinite(scores[i]) & ~already_dead)
        sets.append(DecisionSet(decision, index[pos], usable, scores[i]))
    return sets


def _step(sets_item: DecisionSet, symbols: np.ndarray, chosen: np.ndarray) -> TargetStep:
    targets = {str(s): 1.0 / len(chosen) for s in symbols[chosen]} if len(chosen) else {}
    return TargetStep(sets_item.decision_date, sets_item.date, targets)


def rule_ranking_plan(
    sets: list[DecisionSet], symbols: np.ndarray, *, top_n: int
) -> list[TargetStep]:
    """The momentum rule's plan rebuilt from ``sets`` (stable sort, highest score first)."""
    steps = []
    for item in sets:
        order = np.argsort(-item.scores[item.usable], kind="stable")
        steps.append(_step(item, symbols, item.usable[order[:top_n]]))
    return steps


def plan_fresh_dice(
    sets: list[DecisionSet], symbols: np.ndarray, *, top_n: int, seed: int
) -> list[TargetStep]:
    """Arm R: each month hold ``top_n`` stocks drawn at random from the eligible set."""
    rng = np.random.default_rng(seed)
    steps = []
    for item in sets:
        size = min(top_n, len(item.usable))
        chosen = np.sort(rng.choice(item.usable, size=size, replace=False)) if size else item.usable
        steps.append(_step(item, symbols, chosen))
    return steps


def _naive(stamp: pd.Timestamp) -> pd.Timestamp:
    """The same wall-clock day without a time zone (the ledger's own convention)."""
    stamp = pd.Timestamp(stamp)
    return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp


def bought_counts(rule_steps: list[TargetStep]) -> list[int]:
    """Names the rule bought at each step: its targets not held after the previous step."""
    counts, previous = [], set()
    for step in rule_steps:
        now = set(step.targets)
        counts.append(len(now - previous))
        previous = now
    return counts


def executed_new_names(trades: list, steps: list[TargetStep]) -> list[int]:
    """Names the rule actually bought for the first time at each step, from its fills.

    A buy counts as a new name when the stock was not held just before it; a buy that only
    resizes a held position does not. A target the ledger could not fill is not a purchase,
    and counts the month it is finally bought. Counted from the ledger's own trade list of
    the rule's reference cell (the one the check-first run reproduced from the saved file).
    """
    held: dict[str, float] = {}
    peak: dict[str, float] = {}
    new_by_date: dict[pd.Timestamp, int] = {}
    total_new = 0
    for trade in sorted(trades, key=lambda t: _naive(t.date)):
        before = held.get(trade.symbol, 0.0)
        # "Not held" means the ledger sold the whole position (what is left is rounding of the
        # same size order as the position itself), never an absolute share count.
        is_held = before > DUST * peak.get(trade.symbol, 0.0) and before > 0.0
        if trade.qty > 0 and trade.reason == "rebalance" and not is_held:
            day = _naive(trade.date)
            new_by_date[day] = new_by_date.get(day, 0) + 1
            total_new += 1
        held[trade.symbol] = before + trade.qty
        peak[trade.symbol] = max(peak.get(trade.symbol, 0.0), held[trade.symbol])
    counts = [new_by_date.get(_naive(step.date), 0) for step in steps]
    if sum(counts) != total_new:
        # The ledger stamps trades without a time zone and the plan's days may carry one; a
        # purchase on a day that is not a plan day would silently vanish from the counts.
        raise ValueError(
            f"{total_new - sum(counts)} of {total_new} first purchases fall on days that are "
            "not execution days of the plan"
        )
    return counts


def plan_matched(
    sets: list[DecisionSet],
    symbols: np.ndarray,
    bought: list[int],
    *,
    top_n: int,
    seed: int,
) -> list[TargetStep]:
    """Arm M: replace about as many names each month as the rule bought, the picks at random.

    Args:
        sets: the eligible sets (``eligible_sets``).
        symbols: column names of the close frame.
        bought: names the rule bought at each step (``executed_new_names``), same length as sets.
        top_n: portfolio size.
        seed: the draw's registered seed.
    """
    if len(sets) != len(bought):
        raise ValueError(
            "the rule's purchase counts and the eligible sets are not for the same days"
        )
    rng = np.random.default_rng(seed)
    held = np.array([], dtype=int)
    steps = []
    for item, k in zip(sets, bought, strict=True):
        need = min(top_n, len(item.usable))
        still = np.intersect1d(held, item.usable)
        keep_n = min(len(still), max(0, need - k))
        keep = rng.choice(still, size=keep_n, replace=False) if keep_n else still[:0]
        add_n = need - len(keep)
        # New names come from stocks not held last month, so a replaced name is a real
        # replacement; only when too few such stocks exist may a dropped name come back.
        pool = np.setdiff1d(item.usable, held)
        if len(pool) >= add_n:
            add = rng.choice(pool, size=add_n, replace=False) if add_n else pool[:0]
        else:
            # Every genuinely new name is taken; the rest come from names dropped this month.
            dropped = np.setdiff1d(np.setdiff1d(item.usable, keep), pool)
            extra = rng.choice(dropped, size=add_n - len(pool), replace=False)
            add = np.concatenate([pool, extra])
        chosen = np.sort(np.concatenate([keep, add]).astype(int))
        steps.append(_step(item, symbols, chosen))
        held = chosen
    return steps


def plan_all_eligible(sets: list[DecisionSet], symbols: np.ndarray) -> list[TargetStep]:
    """The yardstick for the sanity check: hold every eligible stock equally each month."""
    return [_step(item, symbols, item.usable) for item in sets]
