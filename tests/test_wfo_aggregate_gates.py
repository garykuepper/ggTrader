"""Tests for WFO aggregate metrics and the 4 PASS/FAIL gates.

Per (coin, entry, exit) combo, after the 10-fold WFO completes, four metrics
are computed from the per-fold results and judged against fixed thresholds.
Gates are filters (pass/fail), never selection.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.wfo_aggregate import (  # noqa: E402
    apply_gates,
    compute_wfe,
    dd_ratio,
    fraction_profitable_folds,
    infer_bars_per_year,
    parameter_cv,
)


def test_infer_bars_per_year_4h():
    """4h bars give 2191.5 bars/year."""
    idx = pd.date_range("2023-01-01", periods=100, freq="4h")
    assert infer_bars_per_year(idx) == pytest.approx(2191.5, rel=1e-4)


def test_infer_bars_per_year_1d():
    """Daily bars give 365.25 bars/year."""
    idx = pd.date_range("2023-01-01", periods=100, freq="D")
    assert infer_bars_per_year(idx) == pytest.approx(365.25, rel=1e-4)


def test_infer_bars_per_year_1h():
    """Hourly bars give 8766 bars/year."""
    idx = pd.date_range("2023-01-01", periods=100, freq="h")
    assert infer_bars_per_year(idx) == pytest.approx(8766.0, rel=1e-4)


def test_infer_bars_per_year_config_override():
    """Explicit WFO_BARS_PER_YEAR overrides inference."""
    idx = pd.date_range("2023-01-01", periods=100, freq="4h")
    # Index says 4h (2191.5), but config forces 1000.
    assert infer_bars_per_year(idx, config={"WFO_BARS_PER_YEAR": 1000}) == 1000.0


def test_infer_bars_per_year_fallback_on_empty():
    """Empty/too-short index falls back to 4h default."""
    idx = pd.DatetimeIndex([])
    assert infer_bars_per_year(idx) == pytest.approx(2191.5, rel=1e-4)


def test_wfe_basic_ratio():
    """WFE = mean(test) / mean(train)."""
    train_returns = [0.20, 0.15, 0.10]  # mean 0.15
    test_returns = [0.10, 0.075, 0.05]  # mean 0.075
    assert compute_wfe(train_returns, test_returns) == pytest.approx(0.5, abs=1e-9)


def test_wfe_handles_zero_train_mean():
    """If train mean is ~0, WFE is NaN (avoids division by zero)."""
    train_returns = [0.01, -0.01, 0.0]  # mean 0
    test_returns = [0.05, 0.05, 0.05]
    assert np.isnan(compute_wfe(train_returns, test_returns))


def test_fraction_profitable_folds():
    """Counts folds where test return > 0."""
    test_returns = [0.10, -0.05, 0.20, 0.0, 0.15]
    # 3 of 5 strictly positive (zero is not strictly positive).
    assert fraction_profitable_folds(test_returns) == pytest.approx(0.6, abs=1e-9)


def test_parameter_cv_takes_worst_axis():
    """CV is computed per axis; the WORST (max) axis CV is reported.

    Reasoning: a coin's stability is bottlenecked by its least stable axis.
    Averaging across axes lets one unstable axis hide behind several stable
    ones. Max forces every axis to satisfy the CV gate.
    """
    # Two params, each with 4 fold-winners.
    # adx_length: [14, 14, 14, 14] -> std 0, mean 14, CV 0.
    # adx_threshold: [20, 25, 30, 35] -> std ~5.59 (ddof=0), mean 27.5, CV ~0.203.
    # max(0.0, 0.203) = 0.203.
    fold_params = [
        {"adx_length": 14, "adx_threshold": 20},
        {"adx_length": 14, "adx_threshold": 25},
        {"adx_length": 14, "adx_threshold": 30},
        {"adx_length": 14, "adx_threshold": 35},
    ]
    cv = parameter_cv(fold_params)
    assert cv == pytest.approx(0.2032, abs=0.005)


def test_parameter_cv_max_dominated_by_worst_axis():
    """A coin with one stable axis and one wild axis fails on the wild axis."""
    # adx_length: stable [14, 14, 14, 14] -> CV 0.
    # adx_threshold: wild [10, 20, 30, 40] -> std ~11.18, mean 25, CV ~0.447.
    # max = 0.447. If we averaged we'd get ~0.224 — that would let the
    # combo squeak by the 0.3 gate despite the wild axis. With max, it fails.
    fold_params = [
        {"adx_length": 14, "adx_threshold": 10},
        {"adx_length": 14, "adx_threshold": 20},
        {"adx_length": 14, "adx_threshold": 30},
        {"adx_length": 14, "adx_threshold": 40},
    ]
    cv = parameter_cv(fold_params)
    assert cv == pytest.approx(0.4472, abs=0.005)
    assert cv > 0.3  # would fail the spec gate


def test_parameter_cv_constant_params():
    """All folds choose identical params -> CV = 0."""
    fold_params = [{"a": 1, "b": 2}] * 4
    assert parameter_cv(fold_params) == 0.0


def test_dd_ratio():
    """DD ratio = mean(|test_dd|) / mean(|train_dd|)."""
    train_dds = [-0.10, -0.12, -0.08]  # |mean| = 0.10
    test_dds = [-0.20, -0.18, -0.22]  # |mean| = 0.20
    assert dd_ratio(train_dds, test_dds) == pytest.approx(2.0, abs=1e-9)


def test_apply_gates_all_pass():
    """All 4 gates pass -> True."""
    result = apply_gates(
        wfe=0.6,
        profitable_fraction=0.7,
        param_cv=0.25,
        dd_ratio_val=1.8,
        thresholds={"wfe_min": 0.5, "profitable_min": 0.6, "cv_max": 0.3, "dd_max": 2.0},
    )
    assert result["passed"] is True
    assert result["failures"] == []


def test_apply_gates_one_fails():
    """If one gate fails, passed=False and failures lists the failing metric."""
    result = apply_gates(
        wfe=0.4,  # below 0.5 threshold
        profitable_fraction=0.7,
        param_cv=0.25,
        dd_ratio_val=1.8,
        thresholds={"wfe_min": 0.5, "profitable_min": 0.6, "cv_max": 0.3, "dd_max": 2.0},
    )
    assert result["passed"] is False
    assert "wfe" in result["failures"]
