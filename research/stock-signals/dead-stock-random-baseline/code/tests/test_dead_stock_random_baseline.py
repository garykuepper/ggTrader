"""Tests for the summary logic: spreads, the rule's place, the saved-file check, sanity."""

from __future__ import annotations

import math

import dead_stock_random_baseline as m
import pytest

pytestmark = pytest.mark.study


def cell(cagr: float, final: float = 50_000.0) -> dict:
    return {
        "after_tax_final": final,
        "cagr_after_tax": cagr,
        "sharpe": 0.1,
        "max_drawdown_pct": -50.0,
        "mean_yearly_turnover": 9.0,
        "cost_paid": 1000.0,
        "n_dead_exits": 3,
        "skipped_buys": 0,
    }


def record(seed: int, cagr: float) -> dict:
    return m.draw_record(seed, {1.0: cell(cagr), 0.5: cell(cagr), 0.0: cell(cagr)})


def rule_cell(gap: float) -> dict:
    return {**cell(0.0), "gap_cagr_after_tax_pp": gap}


def test_draw_record_needs_all_three_loss_readings() -> None:
    with pytest.raises(ValueError, match="loss readings"):
        m.draw_record(1, {1.0: cell(0.0), 0.5: cell(0.0)})


def test_draw_record_keeps_only_the_small_fields() -> None:
    rec = record(7, 0.01)
    assert rec["seed"] == 7
    assert set(rec["loss1"]) == set(m.DRAW_FIELDS)
    assert set(rec) == {"seed", "loss1", "loss0.5", "loss0"}


def test_spread_percentiles_and_nonfinite_values_are_ignored() -> None:
    s = m.spread([1.0, 2.0, 3.0, 4.0, 5.0, math.nan, None])
    assert s["n"] == 5 and s["p0"] == 1.0 and s["p50"] == 3.0 and s["p100"] == 5.0
    assert m.spread([math.nan]) == {"n": 0}


def test_rule_position_shares() -> None:
    gaps = [-10.0, -8.0, -6.0, -4.0]
    pos = m.rule_position(-7.0, gaps)
    assert pos["share_draws_behind_rule"] == 0.5
    assert pos["share_draws_ahead_of_rule"] == 0.5
    tie = m.rule_position(-6.0, gaps)
    assert tie["share_draws_behind_rule"] == 0.5 and tie["share_draws_ahead_of_rule"] == 0.25


def test_gaps_are_in_points_a_year_against_spy() -> None:
    draws = [record(1, 0.05), record(2, -0.10)]
    assert m.gaps_pp(draws, 1.0, 0.02) == pytest.approx([3.0, -12.0])


def test_summarise_arm_reports_every_loss_reading_and_the_rule_place() -> None:
    draws = [record(i, 0.0 - i / 100.0) for i in range(5)]
    out = m.summarise_arm(
        draws, 0.0, {1.0: rule_cell(-2.5), 0.5: rule_cell(-2.5), 0.0: rule_cell(-9)}
    )
    assert out["n_draws"] == 5
    assert out["loss1"]["position"]["share_draws_behind_rule"] == pytest.approx(0.4)
    assert out["loss0"]["position"]["share_draws_behind_rule"] == 0.0
    assert out["loss1"]["rule"]["gap_pp"] == -2.5


def test_check_against_saved_flags_a_difference_and_a_missing_cell() -> None:
    saved = {"a": {"after_tax_final": 100.0}, "b": {"after_tax_final": 200.0}}
    ok = m.check_against_saved({"a": {"after_tax_final": 100.004}}, saved)
    assert ok == []
    bad = m.check_against_saved(
        {"a": {"after_tax_final": 100.01}, "c": {"after_tax_final": 1.0}}, saved
    )
    assert {d["cell"] for d in bad} == {"a", "c"}
    assert m.check_against_saved({"a": {"after_tax_final": math.nan}}, saved)


def test_sanity_block_accepts_close_and_flags_far() -> None:
    close = m.sanity_block(0.05, [0.049, 0.051, 0.05, 0.052])
    assert close["ok"] and close["n_dice"] == 4
    far = m.sanity_block(0.05, [0.01, 0.011, 0.012, 0.0105])
    assert not far["ok"]
    assert not m.sanity_block(math.nan, [0.01, 0.02])["ok"]
    assert not m.sanity_block(0.05, [0.01])["ok"]
