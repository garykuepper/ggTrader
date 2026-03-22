"""Unit tests for WFO robustness winner selection and empty robustness handling."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.orchestrator import (
    _calculate_robustness,
    _default_params_from_grid,
    _is_better_robustness,
    _wfo_per_coin_fallback_triple,
)


class TestIsBetterRobustness:
    def test_nan_does_not_beat_negative_inf(self):
        assert not _is_better_robustness(float("nan"), float("-inf"))

    def test_finite_beats_negative_inf(self):
        assert _is_better_robustness(0.1, float("-inf"))

    def test_finite_beats_nan_best(self):
        assert _is_better_robustness(0.1, float("nan"))

    def test_higher_finite_wins(self):
        assert _is_better_robustness(2.0, 1.0)
        assert not _is_better_robustness(1.0, 2.0)

    def test_negative_inf_candidate_never_wins(self):
        assert not _is_better_robustness(float("-inf"), float("-inf"))
        assert not _is_better_robustness(float("-inf"), 1.0)


class TestCalculateRobustnessEmpty:
    def test_empty_train_metrics_returns_empty(self):
        s_empty = pd.Series(dtype=float)
        top, params = _calculate_robustness(
            {1: s_empty, 2: s_empty},
            ["adx_length"],
            {"adx_length": [14]},
            {1: 0.5, 2: 0.5},
        )
        assert top == []
        assert params == {}


class TestFallbackTriple:
    def test_first_strategy_and_exit(self):
        grids = {
            "ema_cross": {"fast": [5, 10], "slow": [20]},
            "rsi_reversal": {"rsi_len": [14]},
        }
        s, e, p = _wfo_per_coin_fallback_triple(grids, ["atr_trailing", "fixed_sl_tp"])
        assert s == "ema_cross"
        assert e == "atr_trailing"
        assert p["fast"] == 5
        assert p["slow"] == 20


class TestDefaultParamsFromGrid:
    def test_scalars_and_lists(self):
        g = {"a": [1, 2], "b": 3.0}
        d = _default_params_from_grid(g)
        assert d["a"] == 1
        assert d["b"] == 3.0
