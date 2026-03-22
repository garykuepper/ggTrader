"""Tests for WFO train metric composite blend."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.orchestrator import _train_metric_series


class _MockPF:
    def __init__(self) -> None:
        idx = pd.MultiIndex.from_tuples([(0, "S"), (1, "S")], names=["combo", "sym"])
        self._idx = idx

    def sharpe_ratio(self) -> pd.Series:
        return pd.Series([1.0, 0.0], index=self._idx)

    def sortino_ratio(self) -> pd.Series:
        return pd.Series([0.5, 1.0], index=self._idx)

    def total_return(self) -> pd.Series:
        return pd.Series([0.2, 0.1], index=self._idx)

    def max_drawdown(self) -> pd.Series:
        return pd.Series([-0.1, -0.2], index=self._idx)


def test_composite_equal_weights() -> None:
    cfg = {
        "TRAIN_METRIC": "composite",
        "TRAIN_METRIC_COMPOSITE_WEIGHTS": {
            "sharpe": 1.0,
            "sortino": 1.0,
            "calmar": 1.0,
        },
    }
    pf = _MockPF()
    out = _train_metric_series(pf, cfg)
    # calmar row0: 0.2/0.1=2, row1: 0.1/0.2=0.5
    exp0 = (1.0 + 0.5 + 2.0) / 3.0
    exp1 = (0.0 + 1.0 + 0.5) / 3.0
    np.testing.assert_allclose(float(out.iloc[0]), exp0, rtol=1e-6)
    np.testing.assert_allclose(float(out.iloc[1]), exp1, rtol=1e-6)


def test_composite_custom_weights_normalized() -> None:
    cfg = {
        "TRAIN_METRIC": "composite",
        "TRAIN_METRIC_COMPOSITE_WEIGHTS": {"sharpe": 2.0, "sortino": 0.0, "calmar": 0.0},
    }
    pf = _MockPF()
    out = _train_metric_series(pf, cfg)
    np.testing.assert_allclose(float(out.iloc[0]), 1.0)
    np.testing.assert_allclose(float(out.iloc[1]), 0.0)


def test_calmar_path_unchanged() -> None:
    cfg = {"TRAIN_METRIC": "calmar"}
    pf = _MockPF()
    out = _train_metric_series(pf, cfg)
    np.testing.assert_allclose(float(out.iloc[0]), 2.0)
    np.testing.assert_allclose(float(out.iloc[1]), 0.5)
