"""Tests for the per-fold z-rank blend in _weighted_robustness_series.

Step 1.5 of the WFO overfitting work. The change adds an optional ``config``
kwarg to the function; when ``PARAM_ZRANK_WEIGHT > 0`` the function blends
the existing raw weighted-mean-of-IS with a per-fold-z-score weighted mean.
At alpha=0 the function must reproduce its prior behavior bit-for-bit.

Synthetic 3-cell × 2-fold case:
    Cell A: IS = [4.0, 0.0]  → raw mean 2.0, single-fold spike
    Cell B: IS = [1.8, 1.8]  → raw mean 1.8, consistent
    Cell C: IS = [0.5, 0.5]  → raw mean 0.5, consistent low

Under raw mean (alpha=0): A wins.
Under per-fold z-rank (alpha=1): B wins (highest mean rank position).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.wfo import _weighted_robustness_series


def _make_fixture():
    """3 cells × 2 folds. Returns (is_metrics_by_fold, weights)."""
    cells = [("A",), ("B",), ("C",)]
    fold1 = pd.Series([4.0, 1.8, 0.5], index=pd.Index(cells, dtype=object))
    fold2 = pd.Series([0.0, 1.8, 0.5], index=pd.Index(cells, dtype=object))
    return {1: fold1, 2: fold2}, {1: 1.0, 2: 1.0}


def test_alpha_zero_reproduces_raw_weighted_mean():
    """With PARAM_ZRANK_WEIGHT=0 (or no config), output equals raw weighted mean."""
    is_metrics, weights = _make_fixture()
    out_no_config = _weighted_robustness_series(is_metrics, weights)
    out_alpha0 = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 0.0}
    )
    pd.testing.assert_series_equal(out_no_config, out_alpha0)
    # And matches hand-computed weighted mean.
    expected = pd.Series([2.0, 1.8, 0.5], index=pd.Index([("A",), ("B",), ("C",)], dtype=object))
    pd.testing.assert_series_equal(out_no_config.sort_index(), expected.sort_index())


def test_alpha_one_picks_consistent_cell():
    """With PARAM_ZRANK_WEIGHT=1.0, B (consistent rank) outranks A (single-fold spike)."""
    is_metrics, weights = _make_fixture()
    out = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 1.0}
    )
    out_sorted = out.sort_values(ascending=False)
    order = list(out_sorted.index)
    assert out_sorted.index[0] == ("B",), f"expected B to win at alpha=1, got {order}"
    assert out_sorted.index[-1] == ("C",), "expected C last (lowest in both folds)"


def test_alpha_half_blends_smoothly():
    """At alpha=0.5 the result is between raw and z-rank, not equal to either."""
    is_metrics, weights = _make_fixture()
    raw = _weighted_robustness_series(is_metrics, weights)
    zrank = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 1.0}
    )
    blend = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 0.5}
    )
    # Per-cell: blend ≈ 0.5*raw + 0.5*zrank
    expected = 0.5 * raw + 0.5 * zrank
    pd.testing.assert_series_equal(blend.sort_index(), expected.sort_index())


def test_degenerate_fold_skipped_in_zrank():
    """A fold where every cell has the same IS contributes nothing to z-rank."""
    cells = [("A",), ("B",), ("C",)]
    fold1 = pd.Series([4.0, 1.8, 0.5], index=pd.Index(cells, dtype=object))
    # Every cell has identical IS in fold 2: std==0, contributes 0 to z-rank.
    fold2 = pd.Series([1.0, 1.0, 1.0], index=pd.Index(cells, dtype=object))
    is_metrics = {1: fold1, 2: fold2}
    weights = {1: 1.0, 2: 1.0}

    zrank_only = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 1.0}
    )
    # With fold 2 contributing 0, the z-rank is essentially fold-1's z divided by
    # weight sum that included fold 2 — but since fold 2's z is NaN (std==0),
    # it's excluded from the denominator too. Expected: pure fold-1 z-scores.
    fold1_vals = np.array([4.0, 1.8, 0.5])
    fold1_z = (fold1_vals - fold1_vals.mean()) / fold1_vals.std()
    expected = pd.Series(fold1_z, index=pd.Index(cells, dtype=object))
    pd.testing.assert_series_equal(zrank_only.sort_index(), expected.sort_index())


def test_signature_backward_compat():
    """Existing callers without config kwarg must still work."""
    is_metrics, weights = _make_fixture()
    # Positional-only call (current convention):
    out = _weighted_robustness_series(is_metrics, weights)
    assert len(out) == 3
    assert all(np.isfinite(out.to_numpy()))
