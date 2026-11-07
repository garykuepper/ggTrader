"""Hand-written known-answer tests for the walk-forward decision logic (folds, composite score,
live-winner pick, WFE, circuit breaker, shadow re-entry, anchor, table).

Moved from the deleted ``tests/lab/test_wfo.py`` in P5c and pointed at ``ggTrader.lab.wfo``;
the expected values were written by hand against the legacy module and are independent of the
code under test.
"""

import pandas as pd
import pytest

from ggTrader.lab.wfo.breaker import (
    WfoState,
    check_circuit_breaker,
    check_shadow_reentry,
    compute_wfe,
)
from ggTrader.lab.wfo.folds import generate_folds
from ggTrader.lab.wfo.report import format_wfo_table
from ggTrader.lab.wfo.select import composite_score
from ggTrader.lab.wfo.select import pick_live_winner as _pick_live_winner


def test_generate_folds_count_and_boundaries():
    """5-year span with 12mo train / 3mo test -> 16 folds, no overlap."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 16
    for f in folds:
        assert f.test_start == f.train_end  # no gap
        train_months = (f.train_end - f.train_start).days / 30.44
        test_months = (f.test_end - f.test_start).days / 30.44
        assert abs(train_months - 12.0) < 1.0
        assert abs(test_months - 3.0) < 1.0
    assert all(f.test_end <= end for f in folds)
    # Folds slide by 3 months
    for i in range(1, len(folds)):
        delta_months = (folds[i].train_start - folds[i - 1].train_start).days / 30.44
        assert abs(delta_months - 3.0) < 1.0


def test_generate_folds_short_data_returns_fewer():
    """Only 12 months of data -> not enough for 12mo train + 3mo test."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 0


def test_generate_folds_exact_15_months():
    """Exactly 15 months (12 + 3) -> 1 fold."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2025-04-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 1
    assert folds[0].train_start == start
    assert folds[0].test_end == end


def test_composite_score_ranking():
    """Combo with best sharpe+sortino and least drawdown scores highest."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.2, "max_drawdown_pct": -10.0},
        {"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -20.0},
        {"sharpe": 0.2, "sortino": 0.3, "max_drawdown_pct": -30.0},
    ]
    scores = composite_score(metrics)
    assert len(scores) == 3
    assert scores[0] > scores[1] > scores[2]
    # Best combo should get close to 1.0 (all normalized to 1.0)
    # 0.5 * 1.0 + 0.3 * 1.0 - 0.2 * 0.0 = 0.8
    assert abs(scores[0] - 0.8) < 1e-9


def test_composite_score_single_combo():
    """Single combo: min==max for all metrics, all normalized to 0.0, score=0.0."""
    metrics = [{"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -15.0}]
    scores = composite_score(metrics)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_composite_score_nan_handling():
    """NaN sharpe/sortino treated as worst (floor of range)."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.0, "max_drawdown_pct": -10.0},
        {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown_pct": -20.0},
    ]
    scores = composite_score(metrics)
    assert scores[0] > scores[1]


def test_pick_live_winner_prefers_fold_proven_combo():
    """A combo that wins folds is chosen over a higher-scoring combo that never
    won out-of-sample (the overfit-to-recent-window trap)."""
    train_metrics = [
        {"combo": "durable"},  # won folds, lower recent score
        {"combo": "overfit"},  # best recent score, never won a fold
    ]
    scores = [0.40, 0.99]
    fold_win_counts = {"durable": 11}
    idx, stability = _pick_live_winner(train_metrics, scores, fold_win_counts)
    assert train_metrics[idx]["combo"] == "durable"
    assert stability == 11


def test_pick_live_winner_picks_best_score_among_durable():
    """Among combos that cleared the stability bar, the best recent score wins."""
    train_metrics = [
        {"combo": "a"},
        {"combo": "b"},
        {"combo": "overfit"},
    ]
    scores = [0.50, 0.70, 0.99]
    fold_win_counts = {"a": 3, "b": 6}
    idx, stability = _pick_live_winner(train_metrics, scores, fold_win_counts)
    assert train_metrics[idx]["combo"] == "b"
    assert stability == 6


def test_pick_live_winner_falls_back_when_no_fold_winners():
    """With no fold winners (e.g. every fold failed gates), fall back to the
    global best composite score rather than recommending nothing."""
    train_metrics = [{"combo": "a"}, {"combo": "b"}]
    scores = [0.30, 0.90]
    idx, stability = _pick_live_winner(train_metrics, scores, {})
    assert train_metrics[idx]["combo"] == "b"
    assert stability == 0


def test_compute_wfe_normal():
    """WFE = OOS sharpe / IS sharpe when IS >= floor."""

    assert compute_wfe(is_sharpe=1.0, oos_sharpe=0.6) == pytest.approx(0.6)
    assert compute_wfe(is_sharpe=2.0, oos_sharpe=1.0) == pytest.approx(0.5)


def test_compute_wfe_neutral_window():
    """IS Sharpe below floor returns None (neutral, excluded from average)."""

    assert compute_wfe(is_sharpe=0.3, oos_sharpe=0.1) is None
    assert compute_wfe(is_sharpe=0.0, oos_sharpe=-0.5) is None


def test_compute_wfe_negative_oos():
    """Negative OOS sharpe -> negative WFE (valid, not neutral)."""

    result = compute_wfe(is_sharpe=1.0, oos_sharpe=-0.5)
    assert result == pytest.approx(-0.5)


def test_compute_wfe_custom_floor():
    """Custom IS floor is respected."""

    assert compute_wfe(is_sharpe=0.35, oos_sharpe=0.1, is_floor=0.3) == pytest.approx(0.1 / 0.35)
    assert compute_wfe(is_sharpe=0.35, oos_sharpe=0.1, is_floor=0.4) is None


def test_circuit_breaker_chronic_decay():
    """Trailing 4-window WFE avg < 0.25 triggers halt."""

    state = WfoState(wfe_history=[0.2, 0.1, 0.3, 0.15])
    result = check_circuit_breaker(state)
    assert result.halted is True
    assert "chronic" in result.halt_reason.lower()


def test_circuit_breaker_acute_failure():
    """Two consecutive negative OOS Sharpe windows trigger halt."""

    state = WfoState(
        wfe_history=[0.8, 0.6],
        oos_sharpes=[-0.1, -0.2],
    )
    result = check_circuit_breaker(state)
    assert result.halted is True
    assert "acute" in result.halt_reason.lower()


def test_circuit_breaker_healthy():
    """Healthy WFE history does not trigger halt."""

    state = WfoState(wfe_history=[0.6, 0.7, 0.5, 0.8])
    result = check_circuit_breaker(state)
    assert result.halted is False
    assert result.halt_reason is None


def test_circuit_breaker_neutral_windows_excluded():
    """None values in WFE history are excluded from average, not counted as 0."""

    state = WfoState(wfe_history=[0.6, None, None, 0.8])
    result = check_circuit_breaker(state)
    # avg of [0.6, 0.8] = 0.7 > 0.25 -> healthy
    assert result.halted is False


def test_circuit_breaker_single_negative_oos_not_halt():
    """One negative OOS Sharpe is not enough (need 2 consecutive)."""

    state = WfoState(
        wfe_history=[0.8],
        oos_sharpes=[0.5, -0.1, 0.3],
    )
    result = check_circuit_breaker(state)
    assert result.halted is False


def test_shadow_reentry_needs_two_clean():
    """First clean shadow window doesn't restore -- need 2 of last 3."""

    state = WfoState(halted=True, halt_reason="test")
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    assert result.halted is True
    assert result.shadow_window == [True]


def test_shadow_reentry_two_consecutive_restores():
    """Two consecutive clean windows end the halt (2 of last 2)."""

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    assert result.halted is False
    assert result.shadow_window == []
    assert result.halt_reason is None


def test_shadow_reentry_two_of_three_restores():
    """A dirty window BETWEEN two clean windows no longer blocks re-entry.

    This is the mid-cap fix: noisy universes alternate clean/dirty, so the old
    '2 consecutive' rule left the halt permanently stuck. 'clean, dirty, clean'
    is 2 of the last 3 and must restore live trading.
    """

    # window so far: clean, then dirty
    state = WfoState(halted=True, halt_reason="test", shadow_window=[True, False])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    assert result.halted is False
    assert result.shadow_window == []
    assert result.halt_reason is None


def test_shadow_reentry_dirty_window_does_not_wipe_history():
    """A single dirty window appends False but keeps the rolling history."""

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=False,
        wfe=0.6,
    )
    assert result.halted is True
    assert result.shadow_window == [True, False]


def test_shadow_reentry_only_last_three_count():
    """Rolling window caps at 3: an old clean window ages out and can't restore."""

    # last 3 would be [False, False, <new>]; one stale clean is dropped
    state = WfoState(halted=True, halt_reason="test", shadow_window=[True, False, False])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    # window becomes [False, False, True] -> only 1 of last 3 clean -> stay halted
    assert result.halted is True
    assert result.shadow_window == [False, False, True]


def test_shadow_reentry_requires_wfe_healthy():
    """WFE < 0.5 (healthy target) is not a clean window."""

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.3,
    )
    assert result.halted is True
    assert result.shadow_window == [True, False]


def test_shadow_reentry_neutral_wfe_fails():
    """Neutral WFE (None) is not a clean window -- can't confirm recovery."""

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=None,
    )
    assert result.halted is True
    assert result.shadow_window == [True, False]


class TestWfoTableOosColumn:
    """The per-fold OOS column must report the fold's actual out-of-sample
    Sharpe.

    Regression test for audit 2026-07-25 §2.1A: the column was rendered
    from `oos_score`, which is `composite_score(test_metrics)[0]` where
    `test_metrics` always holds exactly one combo (`winner_grid =
    [deploy_params]`). `composite_score` min-max normalizes within the
    list it is given, and `_min_max_normalize` returns [0.0] when
    min == max -- so the column was a hardcoded 0.00 in every WFO table
    ever printed, regardless of performance.
    """

    @staticmethod
    def _fold(oos_sharpe, oos_score=0.0):
        return {
            "fold_num": 1,
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2025-01-01"),
            "test_start": pd.Timestamp("2025-01-01"),
            "test_end": pd.Timestamp("2025-04-01"),
            "winner_combo": "demo__x1",
            "winner_params": {"x": 1},
            "train_score": 0.80,
            "oos_score": oos_score,
            "gates_passed": True,
            "wfe": 0.5,
            "is_sharpe": 1.0,
            "oos_sharpe": oos_sharpe,
            "halted": False,
            "used_anchor": False,
        }

    def _render(self, oos_sharpe):

        return format_wfo_table(
            [self._fold(oos_sharpe)],
            {"sharpe": 0.5, "cagr_pct": 5.0, "max_drawdown_pct": -10.0},
            {"sharpe": 1.0, "cagr_pct": 10.0, "max_drawdown_pct": -8.0},
            {"combo": "demo__x1", "params": {"x": 1}, "train_metrics": {}, "stability": 1.0},
            "demo",
            1,
            1,
        )

    def test_positive_oos_sharpe_is_shown_not_zero(self):
        out = self._render(1.44)
        assert "1.44" in out, f"fold OOS Sharpe missing from table:\n{out}"

    def test_negative_oos_sharpe_is_shown(self):
        out = self._render(-0.38)
        assert "-0.38" in out, f"negative fold OOS Sharpe missing:\n{out}"

    def test_distinct_oos_values_render_differently(self):
        """Two folds with very different OOS results must not print the
        same number -- the exact symptom of the constant-0.00 bug."""
        assert self._render(1.44) != self._render(-0.38)

    def test_train_and_oos_columns_are_both_sharpe_so_wfe_is_verifiable(self):
        """WFE is OOS Sharpe / IS Sharpe, so the two columns either side of
        it must be those same Sharpes -- otherwise the row cannot be
        checked by eye and mixes units (composite score vs Sharpe)."""

        fold = self._fold(0.50)
        fold["is_sharpe"] = 2.00
        fold["train_score"] = 0.80  # composite score -- must NOT be shown
        fold["wfe"] = 0.25
        out = format_wfo_table(
            [fold],
            {"sharpe": 0.5, "cagr_pct": 5.0, "max_drawdown_pct": -10.0},
            {"sharpe": 1.0, "cagr_pct": 10.0, "max_drawdown_pct": -8.0},
            {"combo": "demo__x1", "params": {"x": 1}, "train_metrics": {}, "stability": 1.0},
            "demo",
            1,
            1,
        )
        row = [ln for ln in out.splitlines() if ln.startswith("1  ")][0]
        assert "2.00" in row, f"train Sharpe missing from row: {row!r}"
        assert "0.80" not in row, f"composite score leaked into row: {row!r}"


def test_compute_wfe_returns_nan_when_oos_nan():
    """NaN OOS Sharpe with a healthy IS Sharpe gives NaN WFE, not None."""
    import math

    result = compute_wfe(is_sharpe=1.0, oos_sharpe=float("nan"))
    assert result is not None
    assert math.isnan(result)


def test_circuit_breaker_none_wfe_excluded_from_window():
    """Control: a trailing None is dropped, so the last four real values are averaged."""

    state = WfoState(wfe_history=[0.1, 0.1, 0.1, 0.1, None])
    assert check_circuit_breaker(state).halted is True


def test_circuit_breaker_nan_wfe_suppresses_chronic_decay():
    """OPEN DECISION: a NaN is not filtered, so it poisons the window average.

    Excluding NaN like None would average [0.1] * 4 = 0.1 < 0.25 and halt.
    """

    state = WfoState(wfe_history=[0.1, 0.1, 0.1, 0.1, float("nan")])
    assert check_circuit_breaker(state).halted is False


def test_circuit_breaker_nan_policy_exclude_halts():
    """Excluding NaN exposes chronic decay."""

    assert check_circuit_breaker(
        WfoState(wfe_history=[0.1] * 4 + [float("nan")]), nan_policy="exclude"
    ).halted


def test_circuit_breaker_nan_policy_legacy_unchanged():
    """The default preserves NaN behavior."""

    assert not check_circuit_breaker(WfoState(wfe_history=[0.1] * 4 + [float("nan")])).halted
