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
