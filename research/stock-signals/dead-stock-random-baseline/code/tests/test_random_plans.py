"""Tests for the random-pick planners: same eligible set as the rule, only the order differs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rb_replay
from random_plans import (
    SEED_BASE,
    DecisionSet,
    bought_counts,
    draw_seed,
    eligible_sets,
    executed_new_names,
    plan_all_eligible,
    plan_fresh_dice,
    plan_matched,
    rule_ranking_plan,
)
from rb_replay import TargetStep

pytestmark = pytest.mark.study

DAYS = pd.bdate_range("2018-01-01", "2019-12-31")
START, END = "2019-01-02", "2019-12-31"
LOOKBACK, SKIP, TOP_N = 40, 5, 3


def make_close(n: int = 14, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, size=(len(DAYS), n))
    prices = 50.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=DAYS, columns=[f"S{i:02d}" for i in range(n)])


def empty_spans() -> pd.DataFrame:
    return pd.DataFrame({"symbol": [], "last_date": pd.to_datetime([])})


def eligibility(close: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(True, index=close.index, columns=close.columns)
    frame.loc["2019-05-01":"2019-08-31", ["S00", "S01"]] = False  # two names drop out for a while
    return frame


def sets_for(close, eligible=None, spans=None):
    return eligible_sets(
        close,
        eligibility(close) if eligible is None else eligible,
        empty_spans() if spans is None else spans,
        lookback=LOOKBACK,
        skip=SKIP,
        start=START,
        end=END,
    )


def test_rule_ranking_plan_reproduces_the_copied_planner_exactly() -> None:
    close = make_close()
    eligible = eligibility(close)
    spans = pd.DataFrame({"symbol": ["S03"], "last_date": pd.to_datetime(["2019-07-15"])})
    expected = rb_replay.plan_trades(
        close,
        eligible,
        spans,
        top_n=TOP_N,
        lookback=LOOKBACK,
        skip=SKIP,
        dead_loss=1.0,
        start=START,
        end=END,
    )
    got = rule_ranking_plan(sets_for(close, eligible, spans), np.array(close.columns), top_n=TOP_N)
    assert got == expected


def test_random_plans_only_hold_stocks_the_rule_could_hold() -> None:
    close = make_close()
    sets = sets_for(close)
    symbols = np.array(close.columns)
    for plan in (
        plan_fresh_dice(sets, symbols, top_n=TOP_N, seed=1),
        plan_matched(
            sets,
            symbols,
            bought_counts(rule_ranking_plan(sets, symbols, top_n=TOP_N)),
            top_n=TOP_N,
            seed=1,
        ),
    ):
        assert len(plan) == len(sets)
        for step, item in zip(plan, sets, strict=True):
            allowed = {symbols[j] for j in item.usable}
            assert set(step.targets) <= allowed
            assert len(step.targets) == min(TOP_N, len(item.usable))
            assert sum(step.targets.values()) == pytest.approx(1.0)


def test_same_seed_same_plan_and_other_seed_other_plan() -> None:
    close = make_close()
    sets = sets_for(close)
    symbols = np.array(close.columns)
    a = plan_fresh_dice(sets, symbols, top_n=TOP_N, seed=5)
    assert a == plan_fresh_dice(sets, symbols, top_n=TOP_N, seed=5)
    assert a != plan_fresh_dice(sets, symbols, top_n=TOP_N, seed=6)


def test_fewer_eligible_than_top_n_holds_all_of_them() -> None:
    close = make_close(n=14)
    eligible = pd.DataFrame(False, index=close.index, columns=close.columns)
    eligible["S00"] = True
    eligible["S01"] = True
    sets = sets_for(close, eligible)
    plan = plan_fresh_dice(sets, np.array(close.columns), top_n=TOP_N, seed=2)
    assert all(set(s.targets) == {"S00", "S01"} for s in plan)


def test_nothing_eligible_gives_empty_targets() -> None:
    close = make_close()
    eligible = pd.DataFrame(False, index=close.index, columns=close.columns)
    sets = sets_for(close, eligible)
    symbols = np.array(close.columns)
    assert all(s.targets == {} for s in plan_fresh_dice(sets, symbols, top_n=TOP_N, seed=2))
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    assert all(
        s.targets == {}
        for s in plan_matched(sets, symbols, bought_counts(rule), top_n=TOP_N, seed=2)
    )


def test_matched_arm_replaces_no_fewer_names_than_the_rule_and_exactly_when_unforced() -> None:
    close = make_close(n=30)
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)
    sets = sets_for(close, eligible)
    symbols = np.array(close.columns)
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    mine = plan_matched(sets, symbols, bought_counts(rule), top_n=TOP_N, seed=3)
    # Nothing is ever forced out here (everything stays eligible and alive), so the matched
    # arm buys exactly as many names each month as the rule did.
    assert bought_counts(mine) == bought_counts(rule)


def test_matched_arm_can_replace_more_when_held_names_lose_eligibility() -> None:
    close = make_close(n=14)
    sets = sets_for(close)
    symbols = np.array(close.columns)
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    mine = plan_matched(sets, symbols, bought_counts(rule), top_n=TOP_N, seed=4)
    assert all(m >= r for m, r in zip(bought_counts(mine), bought_counts(rule), strict=True))


def test_first_month_matched_buys_a_full_portfolio() -> None:
    close = make_close()
    sets = sets_for(close)
    symbols = np.array(close.columns)
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    mine = plan_matched(sets, symbols, bought_counts(rule), top_n=TOP_N, seed=1)
    assert bought_counts(mine)[0] == TOP_N


def test_matched_refuses_counts_for_other_days() -> None:
    close = make_close()
    sets = sets_for(close)
    symbols = np.array(close.columns)
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    with pytest.raises(ValueError, match="same days"):
        plan_matched(sets, symbols, bought_counts(rule)[1:], top_n=TOP_N, seed=1)


def test_all_eligible_plan_is_equal_weight_over_the_eligible_set() -> None:
    close = make_close()
    sets = sets_for(close)
    symbols = np.array(close.columns)
    for step, item in zip(plan_all_eligible(sets, symbols), sets, strict=True):
        assert set(step.targets) == {symbols[j] for j in item.usable}


def test_draw_seed_formula_and_refusals() -> None:
    assert SEED_BASE == 20261125
    assert draw_seed("early", "R", 0) == 20261125
    assert draw_seed("dev", "R", 7) == 20261125 + 1_000_000 + 7
    assert draw_seed("holdout", "M", 99) == 20261125 + 2_000_000 + 10_000 + 99
    for bad in (("other", "R", 0), ("early", "X", 0), ("early", "R", -1), ("early", "R", 10_000)):
        with pytest.raises(ValueError):
            draw_seed(*bad)


def test_a_dead_stock_is_never_drawn_after_its_series_ended() -> None:
    close = make_close()
    spans = pd.DataFrame({"symbol": ["S05"], "last_date": pd.to_datetime(["2019-03-20"])})
    close.loc["2019-03-21":, "S05"] = np.nan
    sets = sets_for(close, None, spans)
    symbols = np.array(close.columns)
    for seed in range(20):
        for step in plan_fresh_dice(sets, symbols, top_n=TOP_N, seed=seed):
            if step.decision_date > pd.Timestamp("2019-03-20"):
                assert "S05" not in step.targets


# --- arm M on tiny explicit fixtures: the exact forced-exit, fill and fallback behaviour ---

SYMS = np.array([f"X{i}" for i in range(8)])
DAY1, DAY2 = pd.Timestamp("2019-01-31"), pd.Timestamp("2019-02-28")


def tiny_set(day: pd.Timestamp, usable: list[int]) -> DecisionSet:
    return DecisionSet(day, day + pd.Timedelta(days=1), np.array(usable), np.zeros(len(SYMS)))


def test_matched_keeps_the_unreplaced_names_and_buys_exactly_k_new_ones() -> None:
    sets = [tiny_set(DAY1, list(range(8))), tiny_set(DAY2, list(range(8)))]
    for seed in range(30):
        first, second = plan_matched(sets, SYMS, [3, 1], top_n=3, seed=seed)
        assert len(first.targets) == 3
        assert len(set(second.targets) - set(first.targets)) == 1
        assert len(set(second.targets) & set(first.targets)) == 2


def test_matched_replaces_a_held_stock_that_stopped_being_eligible() -> None:
    for seed in range(30):
        sets = [tiny_set(DAY1, list(range(8))), tiny_set(DAY2, list(range(8)))]
        first = plan_matched(sets, SYMS, [3, 0], top_n=3, seed=seed)[0]
        gone = sorted(first.targets)[0]
        gone_index = int(np.flatnonzero(SYMS == gone)[0])
        sets[1] = tiny_set(DAY2, [i for i in range(8) if i != gone_index])
        a, b = plan_matched(sets, SYMS, [3, 0], top_n=3, seed=seed)
        assert a == first  # month one does not depend on month two
        assert gone not in b.targets and len(b.targets) == 3
        assert len(set(b.targets) - set(a.targets)) == 1  # one forced replacement, k was 0


def test_matched_takes_every_new_name_first_then_names_dropped_this_month() -> None:
    # Four eligible names, three held, the rule bought two: only one name was not held, so it
    # is always bought, and the second replacement must come from the names dropped this month.
    sets = [tiny_set(DAY1, [0, 1, 2, 3]), tiny_set(DAY2, [0, 1, 2, 3])]
    seen_dropped_back = set()
    for seed in range(60):
        a, b = plan_matched(sets, SYMS, [3, 2], top_n=3, seed=seed)
        outside = {"X0", "X1", "X2", "X3"} - set(a.targets)
        assert len(outside) == 1 and len(b.targets) == 3
        assert next(iter(outside)) in b.targets  # the only genuinely new name is always taken
        assert len(set(b.targets) & set(a.targets)) == 2
        seen_dropped_back.add(tuple(sorted(set(b.targets) & set(a.targets))))
    assert len(seen_dropped_back) > 1  # which held names stay is random


def test_matched_and_dice_hold_everything_when_fewer_than_top_n_are_eligible() -> None:
    sets = [tiny_set(DAY1, [0, 1]), tiny_set(DAY2, [0, 1])]
    for step in plan_matched(sets, SYMS, [2, 0], top_n=3, seed=1):
        assert set(step.targets) == {"X0", "X1"}
    for step in plan_fresh_dice(sets, SYMS, top_n=3, seed=1):
        assert set(step.targets) == {"X0", "X1"}


def test_matched_never_draws_a_stock_that_is_not_eligible() -> None:
    sets = [tiny_set(DAY1, [0, 2, 4, 6]), tiny_set(DAY2, [1, 3, 5, 7])]
    for seed in range(30):
        a, b = plan_matched(sets, SYMS, [3, 3], top_n=3, seed=seed)
        assert set(a.targets) <= {"X0", "X2", "X4", "X6"}
        assert set(b.targets) <= {"X1", "X3", "X5", "X7"}


# --- executed new names, from the ledger's fills ---


class Fill:
    def __init__(self, date: str, symbol: str, qty: float, reason: str = "rebalance") -> None:
        self.date, self.symbol, self.qty, self.reason = pd.Timestamp(date), symbol, qty, reason


def step_on(date: str) -> TargetStep:
    stamp = pd.Timestamp(date)
    return TargetStep(stamp - pd.Timedelta(days=1), stamp, {})


def test_executed_new_names_counts_first_purchases_not_resizes() -> None:
    steps = [step_on("2019-02-01"), step_on("2019-03-01"), step_on("2019-04-01")]
    trades = [
        Fill("2019-02-01", "A", 10),
        Fill("2019-02-01", "B", 10),
        Fill("2019-03-01", "A", 3),  # resize of a held name: not a new name
        Fill("2019-03-01", "B", -10, "rebalance"),  # sold out
        Fill("2019-03-01", "C", 10),
        Fill("2019-04-01", "B", 5),  # B comes back after being sold out: a new purchase
    ]
    assert executed_new_names(trades, steps) == [2, 1, 1]


def test_a_target_the_ledger_could_not_fill_counts_when_it_is_finally_bought() -> None:
    steps = [step_on("2019-02-01"), step_on("2019-03-01")]
    # A was a target in February but had no price (no fill); it is bought in March.
    trades = [Fill("2019-02-01", "B", 10), Fill("2019-03-01", "A", 10)]
    assert executed_new_names(trades, steps) == [1, 1]


def test_dead_exits_and_liquidation_are_not_purchases() -> None:
    steps = [step_on("2019-02-01"), step_on("2019-03-01")]
    trades = [
        Fill("2019-02-01", "A", 10),
        Fill("2019-02-15", "A", -10, "dead_exit"),
        Fill("2019-03-01", "A", 4, "tax_sale"),
    ]
    assert executed_new_names(trades, steps) == [1, 0]


def test_small_positive_holdings_are_still_holdings() -> None:
    steps = [step_on("2019-02-01"), step_on("2019-03-01")]
    trades = [Fill("2019-02-01", "A", 5e-10), Fill("2019-03-01", "A", 5e-10)]
    assert executed_new_names(trades, steps) == [1, 0]


def test_a_held_stock_that_dies_is_replaced_in_arm_m_through_the_real_eligible_sets() -> None:
    close = make_close(n=14)
    spans = pd.DataFrame({"symbol": ["S05"], "last_date": pd.to_datetime(["2019-06-20"])})
    close.loc["2019-06-21":, "S05"] = np.nan
    sets = sets_for(close, pd.DataFrame(True, index=close.index, columns=close.columns), spans)
    symbols = np.array(close.columns)
    held_before_death = 0
    for seed in range(200):
        plan = plan_matched(sets, symbols, [3] + [0] * (len(sets) - 1), top_n=TOP_N, seed=seed)
        for before, after in zip(plan, plan[1:], strict=False):
            if after.decision_date > pd.Timestamp("2019-06-20"):
                assert "S05" not in after.targets and len(after.targets) == TOP_N
                if "S05" in before.targets:
                    held_before_death += 1
                    assert len(set(after.targets) - set(before.targets)) == 1  # forced swap
    assert held_before_death > 0


def test_new_names_match_when_the_plan_days_carry_a_time_zone_and_the_fills_do_not() -> None:
    stamp = pd.Timestamp("2019-02-01", tz="UTC")
    steps = [TargetStep(stamp - pd.Timedelta(days=1), stamp, {})]
    assert executed_new_names([Fill("2019-02-01", "A", 5), Fill("2019-02-01", "B", 5)], steps) == [
        2
    ]


def test_a_purchase_on_a_day_that_is_not_a_plan_day_is_refused_not_dropped() -> None:
    steps = [step_on("2019-02-01")]
    with pytest.raises(ValueError, match="not execution days"):
        executed_new_names([Fill("2019-02-01", "A", 5), Fill("2019-02-04", "B", 5)], steps)
