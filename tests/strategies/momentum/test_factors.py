"""Unit tests for Numba factor functions in ggTrader.strategies.momentum.factors."""

from __future__ import annotations

import numpy as np
import pytest

from ggTrader.strategies.momentum.factors import (
    compute_momentum_nb,
    compute_liquidity_shock_nb,
    strip_btc_beta_nb,
)


def test_momentum_no_lookahead() -> None:
    """Verify that momentum at bar i never uses data from bar i+1."""
    np.random.seed(42)
    close_arr = np.random.uniform(10.0, 100.0, (100, 5))
    window = 10
    gap = 2

    # First run
    res_1 = compute_momentum_nb(close_arr, window, gap)

    # Mutate the last row (bar i+1)
    close_arr_mutated = close_arr.copy()
    close_arr_mutated[-1, :] += 50.0

    # Second run
    res_2 = compute_momentum_nb(close_arr_mutated, window, gap)

    # Assert that all outputs up to index -2 (which corresponds to bar i) are identical
    np.testing.assert_array_equal(res_1[:-1, :], res_2[:-1, :])


def test_momentum_output_shape() -> None:
    """Assert output shape == input shape."""
    np.random.seed(42)
    close_arr = np.random.uniform(10.0, 100.0, (100, 5))
    window = 10
    gap = 2
    res = compute_momentum_nb(close_arr, window, gap)
    assert res.shape == close_arr.shape


def test_momentum_nan_warmup() -> None:
    """Assert first (window) rows are NaN."""
    np.random.seed(42)
    close_arr = np.random.uniform(10.0, 100.0, (100, 5))
    window = 15
    gap = 3
    res = compute_momentum_nb(close_arr, window, gap)

    # The first 'window' rows (indices 0 to window-1) must be NaN
    assert np.all(np.isnan(res[:window, :]))
    # After that, there should be some non-NaN values
    assert np.any(~np.isnan(res[window:, :]))


def test_liquidity_shock_positive() -> None:
    """Assert all non-NaN values of liquidity shock >= 0."""
    np.random.seed(42)
    close_arr = np.random.uniform(10.0, 100.0, (100, 5))
    vol_arr = np.random.uniform(1000.0, 10000.0, (100, 5))

    res = compute_liquidity_shock_nb(close_arr, vol_arr)

    # First row is NaN
    assert np.all(np.isnan(res[0, :]))
    # Non-NaN values must be >= 0
    non_nan_vals = res[1:, :][~np.isnan(res[1:, :])]
    assert np.all(non_nan_vals >= 0.0)


def test_btc_beta_residual_uncorrelated() -> None:
    """Verify corr(residual, btc_ret) < 0.05 on synthetic data."""
    np.random.seed(42)
    n_t = 1000
    window = 60

    # Generate synthetic BTC returns and altcoin returns with high correlation
    btc_ret = np.random.normal(0, 0.01, n_t)
    alt_ret = 1.5 * btc_ret + np.random.normal(0, 0.005, n_t)

    # Reshape for 2D inputs
    alt_ret_2d = alt_ret.reshape(-1, 1)

    residual = strip_btc_beta_nb(alt_ret_2d, btc_ret, window)

    # Warmup check
    assert np.all(np.isnan(residual[:window, 0]))

    # Valid values check
    valid_res = residual[window:, 0]
    valid_btc = btc_ret[window:]

    # Calculate correlation
    corr = np.corrcoef(valid_res, valid_btc)[0, 1]
    assert abs(corr) < 0.05
