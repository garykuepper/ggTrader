"""Tests for the per-cell IS-OOS gap penalty in _calculate_robustness.

Step 4b of the WFO overfitting work. The change adds an optional
``test_metrics_by_fold`` arg to _calculate_robustness; when
``PARAM_OOS_GAP_PENALTY > 0`` the function subtracts ``gamma * |IS_mean - OOS_mean|``
from each cell's score so the picked cell must agree across train and test.

Synthetic 3-cell × 2-fold case (fold weights equal):
    Cell A: IS = [3.0, 1.0]  → IS_mean 2.0, OOS = [0.0, 0.0]  → OOS_mean 0.0, gap 2.0
    Cell B: IS = [1.0, 1.0]  → IS_mean 1.0, OOS = [1.0, 1.0]  → OOS_mean 1.0, gap 0.0
    Cell C: IS = [0.5, 0.5]  → IS_mean 0.5, OOS = [0.5, 0.5]  → OOS_mean 0.5, gap 0.0

Under raw IS-only (gamma=0): A wins (IS_mean 2.0 > B 1.0 > C 0.5).
Under gap penalty (gamma=1.0):
    A score = 2.0 - 1.0*2.0 = 0.0
    B score = 1.0 - 1.0*0.0 = 1.0
    C score = 0.5 - 1.0*0.0 = 0.5
B wins.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.wfo import _calculate_robustness


def _make_fixture():
    """3 cells × 2 folds with hand-designed IS / OOS arrays."""
    cells = [("A",), ("B",), ("C",)]
    is_fold1 = pd.Series([3.0, 1.0, 0.5], index=pd.Index(cells, dtype=object))
    is_fold2 = pd.Series([1.0, 1.0, 0.5], index=pd.Index(cells, dtype=object))
    oos_fold1 = pd.Series([0.0, 1.0, 0.5], index=pd.Index(cells, dtype=object))
    oos_fold2 = pd.Series([0.0, 1.0, 0.5], index=pd.Index(cells, dtype=object))
    is_metrics = {1: is_fold1, 2: is_fold2}
    test_metrics = {1: oos_fold1, 2: oos_fold2}
    return is_metrics, test_metrics


def test_gamma_zero_reproduces_legacy_is_only_selection():
    """With PARAM_OOS_GAP_PENALTY=0, picks the IS-best cell (A)."""
    is_metrics, test_metrics = _make_fixture()
    config = {"PARAM_OOS_GAP_PENALTY": 0.0, "PARAM_STABILITY_WEIGHT": 0.0}
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics,
        param_names=["x"],
        param_grid={"x": ["A", "B", "C"]},
        oos_metrics_by_fold=None,
        config=config,
        test_metrics_by_fold=test_metrics,
    )
    assert top, "expected non-empty top list"
    assert top[0]["params"]["x"] == "A", (
        f"expected A to win at gamma=0 (IS-only), got {top[0]['params']}"
    )


def test_gamma_one_picks_consistent_cell():
    """With PARAM_OOS_GAP_PENALTY=1.0, B (consistent) outranks A (spike)."""
    is_metrics, test_metrics = _make_fixture()
    config = {"PARAM_OOS_GAP_PENALTY": 1.0, "PARAM_STABILITY_WEIGHT": 0.0}
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics,
        param_names=["x"],
        param_grid={"x": ["A", "B", "C"]},
        oos_metrics_by_fold=None,
        config=config,
        test_metrics_by_fold=test_metrics,
    )
    assert top, "expected non-empty top list"
    assert top[0]["params"]["x"] == "B", (
        f"expected B to win at gamma=1 (consistent), got {top[0]['params']}"
    )


def test_no_test_metrics_is_legacy_behavior():
    """When test_metrics_by_fold is None, behavior is bit-identical to legacy."""
    is_metrics, _ = _make_fixture()
    config = {"PARAM_OOS_GAP_PENALTY": 1.0, "PARAM_STABILITY_WEIGHT": 0.0}
    top, _ = _calculate_robustness(
        is_metrics_by_fold=is_metrics,
        param_names=["x"],
        param_grid={"x": ["A", "B", "C"]},
        oos_metrics_by_fold=None,
        config=config,
        test_metrics_by_fold=None,
    )
    assert top[0]["params"]["x"] == "A", "no test data → no penalty → A still wins"


def test_partial_oos_falls_back_to_is(monkeypatch=None):
    """A cell with NaN OOS_mean (no test data for that cell) gets no penalty."""
    cells = [("A",), ("B",)]
    is_metrics = {
        1: pd.Series([3.0, 1.0], index=pd.Index(cells, dtype=object)),
        2: pd.Series([3.0, 1.0], index=pd.Index(cells, dtype=object)),
    }
    # Test fold 1 has no entry for cell A — only B has OOS data.
    test_metrics = {
        1: pd.Series([np.nan, 1.0], index=pd.Index(cells, dtype=object)),
        2: pd.Series([np.nan, 1.0], index=pd.Index(cells, dtype=object)),
    }
    config = {"PARAM_OOS_GAP_PENALTY": 1.0, "PARAM_STABILITY_WEIGHT": 0.0}
    top, _ = _calculate_robustness(
        is_metrics_by_fold=is_metrics,
        param_names=["x"],
        param_grid={"x": ["A", "B"]},
        oos_metrics_by_fold=None,
        config=config,
        test_metrics_by_fold=test_metrics,
    )
    # Cell A: IS_mean=3, OOS_mean=NaN → gap=NaN→0 → score=3
    # Cell B: IS_mean=1, OOS_mean=1 → gap=0 → score=1
    # A still wins because its missing OOS gets no penalty.
    assert top[0]["params"]["x"] == "A", (
        f"expected A to win when its OOS is NaN (no penalty applied), got {top[0]['params']}"
    )
