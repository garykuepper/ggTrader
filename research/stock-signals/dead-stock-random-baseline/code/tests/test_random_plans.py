"""Tests for the random-pick planners: same eligible set as the rule, only the order differs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rb_replay
from random_plans import (
    SEED_BASE,
    bought_counts,
    draw_seed,
    eligible_sets,
    plan_all_eligible,
    plan_fresh_dice,
    plan_matched,
    rule_ranking_plan,
)

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
            sets, symbols, rule_ranking_plan(sets, symbols, top_n=TOP_N), top_n=TOP_N, seed=1
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
    assert all(s.targets == {} for s in plan_matched(sets, symbols, rule, top_n=TOP_N, seed=2))


def test_matched_arm_replaces_no_fewer_names_than_the_rule_and_exactly_when_unforced() -> None:
    close = make_close(n=30)
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)
    sets = sets_for(close, eligible)
    symbols = np.array(close.columns)
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    mine = plan_matched(sets, symbols, rule, top_n=TOP_N, seed=3)
    # Nothing is ever forced out here (everything stays eligible and alive), so the matched
    # arm buys exactly as many names each month as the rule did.
    assert bought_counts(mine) == bought_counts(rule)


def test_matched_arm_can_replace_more_when_held_names_lose_eligibility() -> None:
    close = make_close(n=14)
    sets = sets_for(close)
    symbols = np.array(close.columns)
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    mine = plan_matched(sets, symbols, rule, top_n=TOP_N, seed=4)
    assert all(m >= r for m, r in zip(bought_counts(mine), bought_counts(rule), strict=True))


def test_first_month_matched_buys_a_full_portfolio() -> None:
    close = make_close()
    sets = sets_for(close)
    symbols = np.array(close.columns)
    mine = plan_matched(
        sets, symbols, rule_ranking_plan(sets, symbols, top_n=TOP_N), top_n=TOP_N, seed=1
    )
    assert bought_counts(mine)[0] == TOP_N


def test_matched_refuses_a_rule_plan_for_other_days() -> None:
    close = make_close()
    sets = sets_for(close)
    symbols = np.array(close.columns)
    rule = rule_ranking_plan(sets, symbols, top_n=TOP_N)
    with pytest.raises(ValueError, match="same days"):
        plan_matched(sets, symbols, rule[1:], top_n=TOP_N, seed=1)


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
