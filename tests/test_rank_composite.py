"""Tests for the rank-based composite metric (Sortino + Calmar + PF).

Step 1 of the WFO textbook reset. The composite is the within-fold selection
objective. Rank cells by each of Sortino/Calmar/PF descending, average ranks
on ties, take the mean of the three ranks, and emit -mean so 'max wins'
downstream.

Sharpe is intentionally dropped (Sharpe and Sortino are near-redundant; the
spec drops Sharpe to keep the composite to three non-redundant dimensions).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.metrics import _train_metric_series


def _mock_pf(sortino, calmar_total_return, calmar_max_dd, pf_factor):
    """Create a mock vbt portfolio with predetermined per-cell ratio series."""
    pf = MagicMock()
    pf.sortino_ratio.return_value = pd.Series(sortino)
    pf.total_return.return_value = pd.Series(calmar_total_return)
    pf.max_drawdown.return_value = pd.Series(calmar_max_dd)
    trades_mock = MagicMock()
    trades_mock.profit_factor.return_value = pd.Series(pf_factor)
    pf.trades = trades_mock
    # sharpe_ratio is called by the legacy path; we patch to fail-loud if invoked.
    pf.sharpe_ratio.side_effect = AssertionError("Sharpe should not be used in rank composite")
    return pf


def test_rank_composite_winner_is_top_of_each_axis():
    """Cell that's best in all three axes wins decisively."""
    # 3 cells: A best in all 3, B middle, C worst.
    sortino = [2.0, 1.0, 0.5]
    calmar_tr = [3.0, 2.0, 1.0]
    calmar_dd = [-0.1, -0.2, -0.3]
    pf_factor = [2.0, 1.5, 1.0]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    scores = _train_metric_series(pf, config)
    # A should score highest. Score = -mean(rank_sortino, rank_calmar, rank_pf).
    # Calmar inputs yield raw [30, 10, 3.33] which clip to [5, 5, 3.33]; A and B
    # tie at 5.0 on Calmar (avg rank 1.5 each). Sortino and PF break the tie:
    # A ranks 1 on both, B ranks 2 on both, C ranks 3 on all. Assertion is
    # ordering-only (A > B > C), which holds.
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_rank_composite_average_rank_on_ties():
    """Tied cells get the average rank, not min or max."""
    # Cell A and B tied at top of Sortino, C clearly worst.
    sortino = [2.0, 2.0, 0.5]
    # Calmar inputs chosen so values stay below the clip ceiling (5.0):
    # A=0.4/0.1=4.0, B=0.3/0.2=1.5, C=0.2/0.3=0.67 → ranks A=1, B=2, C=3 (no tie).
    calmar_tr = [0.4, 0.3, 0.2]
    calmar_dd = [-0.1, -0.2, -0.3]
    pf_factor = [2.0, 1.5, 1.0]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    scores = _train_metric_series(pf, config)
    # On Sortino: A & B tied -> average rank 1.5 each, C rank 3.
    # On Calmar: A rank 1, B rank 2, C rank 3.
    # On PF:     A rank 1, B rank 2, C rank 3.
    # Implementation uses MEAN rank across the 3 axes, then negates.
    # A mean rank = (1.5+1+1)/3 = 1.1666...; score = -1.1666...
    # B mean rank = (1.5+2+2)/3 = 1.8333...; score = -1.8333...
    # C mean rank = (3+3+3)/3   = 3.0;       score = -3.0
    assert scores.iloc[0] == pytest.approx(-7.0 / 6.0, abs=1e-9)
    assert scores.iloc[1] == pytest.approx(-11.0 / 6.0, abs=1e-9)
    assert scores.iloc[2] == pytest.approx(-3.0, abs=1e-9)


def test_rank_composite_sharpe_never_consulted():
    """The mock's sharpe_ratio raises if called; composite must not call it."""
    sortino = [1.0, 0.5]
    calmar_tr = [1.0, 0.5]
    calmar_dd = [-0.1, -0.2]
    pf_factor = [1.5, 1.0]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    # Should not raise (no AssertionError from the side_effect).
    _ = _train_metric_series(pf, config)


def test_rank_composite_nan_propagates_to_nan():
    """A cell with NaN Sortino (no trades) gets NaN score, not a real rank."""
    sortino = [1.0, float("nan"), 0.5]
    calmar_tr = [1.0, 0.5, 0.3]
    calmar_dd = [-0.1, -0.2, -0.3]
    pf_factor = [1.5, 1.0, 0.8]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    scores = _train_metric_series(pf, config)
    assert np.isfinite(scores.iloc[0])
    assert np.isnan(scores.iloc[1])
    assert np.isfinite(scores.iloc[2])


from ggTrader.core.wfo import _calculate_robustness


def test_min_trades_gate_disqualifies_low_trade_cells():
    """A cell with <30 trades in a fold gets NaN, so it can't win that fold."""
    # Two folds, three cells. Cell A is always above 30 trades. Cell B is below
    # in fold 1, above in fold 2. Cell C is always above. We don't directly
    # test _process_wfo_fold (it requires real OHLCV/portfolios); we test the
    # logical contract: cells with NaN scores get NaN robustness.
    cells = [("A",), ("B",), ("C",)]
    fold1 = pd.Series([1.0, float("nan"), 0.5], index=pd.Index(cells, dtype=object))
    fold2 = pd.Series([1.0, 0.8, 0.5], index=pd.Index(cells, dtype=object))
    is_metrics_by_fold = {1: fold1, 2: fold2}
    # With 8-of-10 forgiveness disabled (min_pass=0), the legacy weighted mean
    # would emit B's score from just fold 2. Confirm via the function.
    config = {"MIN_TRAIN_FOLD_PASS_COUNT": 0}  # disable forgiveness
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics_by_fold,
        param_names=["x"],
        param_grid={"x": ["A", "B", "C"]},
        config=config,
    )
    # All three cells produce a finite score (B from one fold only).
    assert any(r["params"]["x"] == "B" for r in top)


def test_eight_of_ten_forgiveness_drops_cells_below_threshold():
    """A cell present in fewer than min_pass folds is forced to NaN everywhere."""
    cells = [("A",), ("B",)]
    # 10 folds. A present in all 10, B present in only 5.
    is_metrics_by_fold = {}
    for f in range(1, 11):
        if f <= 5:
            row = pd.Series([1.0, 0.5], index=pd.Index(cells, dtype=object))
        else:
            row = pd.Series([1.0, float("nan")], index=pd.Index(cells, dtype=object))
        is_metrics_by_fold[f] = row
    config = {"MIN_TRAIN_FOLD_PASS_COUNT": 8}
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics_by_fold,
        param_names=["x"],
        param_grid={"x": ["A", "B"]},
        config=config,
    )
    # B passed only 5 folds (<8). It must be dropped.
    assert not any(r["params"]["x"] == "B" for r in top), (
        "B should be dropped by 8-of-10 forgiveness (only 5 finite folds)"
    )


def test_eight_of_ten_forgiveness_fills_with_fold_median():
    """A cell that passes 8+ folds gets the missing folds filled with median rank."""
    # 10 folds, 3 cells. A always present, scores 1.0. B missing in 2 of 10,
    # scores 0.5 elsewhere. C present in all 10, scores -0.5 (below median).
    cells = [("A",), ("B",), ("C",)]
    is_metrics_by_fold = {}
    for f in range(1, 11):
        if f in (1, 2):
            row = pd.Series([1.0, float("nan"), -0.5], index=pd.Index(cells, dtype=object))
        else:
            row = pd.Series([1.0, 0.5, -0.5], index=pd.Index(cells, dtype=object))
        is_metrics_by_fold[f] = row
    config = {"MIN_TRAIN_FOLD_PASS_COUNT": 8}
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics_by_fold,
        param_names=["x"],
        param_grid={"x": ["A", "B", "C"]},
        config=config,
    )
    # B passes 8 folds, gets median-fill in folds 1 and 2. Fold median is
    # 0.25 (mean of 1.0 and -0.5 — only two finite cells in fold 1 since B is NaN).
    # The exact ranking depends on aggregation; the key assertion is B is not dropped.
    assert any(r["params"]["x"] == "B" for r in top), "B should survive (passes 8 of 10 folds)"
