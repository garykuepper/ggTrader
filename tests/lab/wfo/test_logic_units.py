"""Unit tests of the pure walk-forward logic: folds, selection, anchor, breaker, gates.

The randomised comparisons that used to call the legacy ``lab.wfo`` now compare with the outputs
frozen in ``golden/legacy_logic_units.json`` (captured from the legacy module before it was deleted
in P5c by ``research/programme/vectorbt-engine/p5c-evidence/capture_legacy_logic_units.py``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.wfo import breaker, folds, gates, select
from ggTrader.lab.wfo.anchor import NO_ANCHOR, pick_anchor
from tests.oracles.legacy_dsr import dsr_check

GOLDEN = json.loads((Path(__file__).parent / "golden" / "legacy_logic_units.json").read_text())


def _num(x):
    """Golden JSON stores NaN as the string 'nan'."""
    return math.nan if x == "nan" else x


def _same_num(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return (math.isnan(a) and math.isnan(b)) or a == b


# --- folds ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2021-01-31", "2026-04-30"),
        ("2021-01-31", "2022-07-31"),
        ("2020-02-29", "2023-05-31"),
        ("2021-01-31", "2021-12-31"),
    ],
)
def test_generate_folds_equals_the_frozen_legacy_output(start: str, end: str) -> None:
    es, ee = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    new = folds.generate_folds(es, ee)
    assert [[str(x) for x in f] for f in new] == GOLDEN["folds"][f"{start}|{end}"]


def test_17_folds_for_the_pinned_window() -> None:
    got = folds.generate_folds(
        pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC")
    )
    assert len(got) == 17
    assert got[0].train_end == got[0].test_start  # the shared bar of legacy folds
    assert got[-1].test_end == pd.Timestamp("2026-04-30", tz="UTC")


def test_fold_windows_equal_label_slicing_including_a_sunday_start() -> None:
    index = pd.bdate_range("2020-01-01", "2023-12-29", tz="UTC")
    fold = folds.generate_folds(
        pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC")
    )[1]
    win = folds.fold_windows(index, fold)
    pd.testing.assert_index_equal(
        index[win.train[0] : win.train[1]],
        index[(index >= fold.train_start) & (index <= fold.train_end)],
    )
    pd.testing.assert_index_equal(
        index[win.test[0] : win.test[1]],
        index[(index >= fold.test_start) & (index <= fold.test_end)],
    )
    assert win.anchor == (0, win.train[1])  # anchor never reaches past train_end


# --- selection --------------------------------------------------------------------------------


def _random_metrics(rng: np.random.Generator, n: int) -> list[dict[str, float]]:
    rows = []
    for _ in range(n):
        rows.append(
            {
                "sharpe": float(rng.normal(0.5, 1.0)) if rng.random() > 0.2 else math.nan,
                "sortino": float(rng.normal(0.7, 1.2)) if rng.random() > 0.2 else math.nan,
                "max_drawdown_pct": -float(rng.uniform(0, 30)),
            }
        )
    return rows


def test_composite_score_equals_the_frozen_legacy_output_on_random_grids() -> None:
    rng = np.random.default_rng(3)
    for n in (1, 2, 5, 48):
        rows = _random_metrics(rng, n)
        expected = [_num(x) for x in GOLDEN["composite"][str(n)]]
        got = select.composite_score(rows)
        assert len(got) == len(expected)
        assert all(_same_num(g, e) for g, e in zip(got, expected, strict=True))


def test_best_index_takes_the_first_of_equal_scores() -> None:
    assert select.best_index([0.1, 0.5, 0.5, 0.2]) == 1


def test_a_flat_combination_cannot_win_a_grid_with_real_edges() -> None:
    rows = [
        {"sharpe": 1.2, "sortino": 1.5, "max_drawdown_pct": -4.0},
        {"sharpe": 0.9, "sortino": 1.1, "max_drawdown_pct": -6.0},
        {"sharpe": 0.4, "sortino": 0.5, "max_drawdown_pct": -9.0},
        {"sharpe": math.nan, "sortino": math.nan, "max_drawdown_pct": 0.0},  # flat, guarded
    ]
    assert select.best_index(select.composite_score(rows)) == 0


def test_pick_live_winner_equals_the_frozen_legacy_output() -> None:
    rows = [{"combo": c} for c in "abcd"]
    scores = [0.2, 0.9, 0.4, 0.8]
    for counts, expected in zip(
        ({}, {"a": 2}, {"b": 0, "c": 3}, {"d": 1, "c": 1}), GOLDEN["pick_live_winner"], strict=True
    ):
        assert list(select.pick_live_winner(rows, scores, counts)) == expected


# --- anchor -----------------------------------------------------------------------------------


def _row(combo: str, cagr: float, mdd: float) -> dict:
    return {
        "combo": combo,
        "params": {"k": combo},
        "cagr_pct": cagr,
        "max_drawdown_pct": mdd,
        "sharpe": 1.0,
    }


def test_anchor_is_the_least_drawdown_among_combos_beating_the_risk_free_rate() -> None:
    rows = [_row("a", 10.0, -12.0), _row("b", 6.0, -5.0), _row("c", 2.0, -1.0)]
    assert pick_anchor(rows).combo == "b"  # c has the least drawdown but CAGR <= 4%


def test_anchor_falls_back_to_the_least_drawdown_when_none_clears_the_rate() -> None:
    rows = [_row("a", 1.0, -12.0), _row("b", 2.0, -5.0)]
    assert pick_anchor(rows).combo == "b"


def test_anchor_without_rows_is_the_none_set() -> None:
    assert pick_anchor([]) == NO_ANCHOR
    assert NO_ANCHOR.combo == "none" and NO_ANCHOR.params == {}


# --- breaker ----------------------------------------------------------------------------------


def _state(wfe: list, oos: list, halted: bool = False, shadow: list | None = None):
    return breaker.WfoState(list(wfe), list(oos), halted, None, list(shadow or []))


def _tuple(state) -> list:
    """The state as the golden stores it (NaN as 'nan')."""
    return [
        [_num_out(x) for x in state.wfe_history],
        [_num_out(x) for x in state.oos_sharpes],
        bool(state.halted),
        state.halt_reason,
        list(state.shadow_window),
    ]


def _num_out(x):
    return None if x is None else ("nan" if math.isnan(x) else float(x))


def test_circuit_breaker_equals_the_frozen_legacy_output_on_random_histories() -> None:
    rng = np.random.default_rng(11)
    for case in GOLDEN["breaker"]:
        n = int(rng.integers(0, 9))
        wfe = [None if rng.random() < 0.25 else float(rng.normal(0.4, 0.8)) for _ in range(n)]
        if rng.random() < 0.2 and wfe:
            wfe[int(rng.integers(0, len(wfe)))] = math.nan
        oos = [float(rng.normal(0, 1)) for _ in range(int(rng.integers(0, 6)))]
        for policy in ("legacy", "exclude"):
            got = _tuple(breaker.check_circuit_breaker(_state(wfe, oos), nan_policy=policy))
            assert got == case[policy]


def test_acute_failure_two_negative_oos_sharpes_halt() -> None:
    new = _state([], [0.5, -0.1, -0.2])
    out = breaker.check_circuit_breaker(new)
    assert out.halted and "Acute" in (out.halt_reason or "")


def test_chronic_decay_halts_on_a_low_trailing_wfe_average() -> None:
    new = _state([0.1, 0.2, 0.1, 0.2], [])
    assert breaker.check_circuit_breaker(new).halted


def test_shadow_reentry_two_of_three_clean_windows_equals_the_frozen_legacy_output() -> None:
    rng = np.random.default_rng(5)
    for expected in GOLDEN["shadow"]:
        state = _state([], [], True, [bool(x) for x in rng.integers(0, 2, int(rng.integers(0, 3)))])
        args = (
            bool(rng.integers(0, 2)),
            bool(rng.integers(0, 2)),
            [None, 0.2, 0.9][int(rng.integers(0, 3))],
        )
        assert _tuple(breaker.check_shadow_reentry(state, *args)) == expected


def test_wfe_is_none_below_the_floor_and_a_ratio_above() -> None:
    assert breaker.compute_wfe(0.3, 1.0) is None
    assert breaker.compute_wfe(1.0, 0.5) == pytest.approx(0.5)
    assert breaker.compute_wfe(0.5, 0.2) == GOLDEN["wfe_0.5_0.2"]


# --- gates ------------------------------------------------------------------------------------


def _grid(k_values=(1, 2, 3)) -> list[dict]:
    return [{"a": a, "b": b} for a in k_values for b in (10, 20)]


def test_extract_grid_arrays_equals_the_frozen_legacy_output() -> None:
    grid = _grid()
    rng = np.random.default_rng(2)
    rows = [
        {
            "params": g,
            "sharpe": float(rng.normal()),
            "total_return_pct": float(rng.normal(5, 3)),
            "n_trades": int(rng.integers(0, 9)),
        }
        for g in grid
    ]
    new = gates.extract_grid_arrays(rows, grid)
    for x, y in zip(new[:3], GOLDEN["grid_arrays"]["arrays"], strict=True):
        np.testing.assert_array_equal(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    assert repr(new[3]) == GOLDEN["grid_arrays"]["fourth"]


def test_extract_grid_arrays_handles_none_axis_values_and_ragged_grids() -> None:
    grid = [{"stop": None, "k": 1}, {"stop": 3, "k": 1}]
    rows = [{"params": g, "sharpe": 1.0, "total_return_pct": 1.0, "n_trades": 1} for g in grid]
    assert gates.extract_grid_arrays(rows, grid)[2] == (1, 2)
    assert gates.extract_grid_arrays(rows, [{"a": 1}, {"b": 2}])[2] == ()


def test_ndh_passes_on_a_plateau_and_fails_on_an_isolated_spike() -> None:
    grid = _grid()
    plateau = [{"params": g, "sharpe": 1.0, "total_return_pct": 4.0, "n_trades": 2} for g in grid]
    assert gates.run_ndh(plateau, grid, 2, 0.85).passed
    spike = [dict(r, sharpe=-0.5) for r in plateau]
    spike[2] = dict(plateau[2], sharpe=2.0)
    assert not gates.run_ndh(spike, grid, 2, 0.85).passed


def test_ndh_excludes_regime_axes_from_the_neighbourhood() -> None:
    grid = [{"min_agree": m, "k": k} for m in (2, 3) for k in (1, 2)]
    rows = [
        {
            "params": g,
            "sharpe": 1.0 if g["min_agree"] == 2 else -1.0,
            "total_return_pct": 1.0,
            "n_trades": 1,
        }
        for g in grid
    ]
    # winner min_agree=2,k=1; its only smooth neighbour (k=2) is profitable, the regime-axis
    # neighbour (min_agree=3) is not and must not count.
    assert gates.run_ndh(rows, grid, 0, 0.85).passed


def test_dsr_gate_matches_the_legacy_statistic_on_seeded_returns() -> None:
    rng = np.random.default_rng(9)
    for seed_shift in range(6):
        rets = pd.Series(rng.normal(0.0008, 0.011, 250 + 20 * seed_shift))
        sharpe = rets.mean() / rets.std() * math.sqrt(252)
        passed, value = gates.run_dsr(rets, 48, 0.80)
        old = dsr_check(
            float(sharpe), len(rets), 48, float(rets.skew()), float(rets.kurtosis()), 0.80
        )
        assert value == pytest.approx(old.dsr_value, abs=1e-10)
        assert passed == old.passed


def test_dsr_gate_short_window_flat_window_and_single_trial() -> None:
    assert gates.run_dsr(pd.Series(np.ones(9) * 0.001), 48, 0.8) == (False, 0.0)
    flat = pd.Series(np.zeros(100))
    assert gates.run_dsr(flat, 48, 0.8) == (False, 0.0)  # NaN reported as 0.0
    rets = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
    passed, value = gates.run_dsr(rets, 1, 0.5)  # no deflation with one trial
    assert 0.0 <= value <= 1.0 and passed == (value >= 0.5)
    legacy = dsr_check(sharpe, len(rets), 1, float(rets.skew()), float(rets.kurtosis()), 0.5)
    assert value == pytest.approx(legacy.dsr_value, abs=1e-12) and passed == legacy.passed
