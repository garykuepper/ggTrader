"""Unit tests for WFO robustness gates: NDH plateau filter and DSR gate."""

import numpy as np

from ggTrader.lab.gates import (
    DsrResult,
    NdhResult,
    _neighbor_flat_indices,
    dsr_check,
    expected_max_sr,
    ndh_check,
    psr,
)


def _make_plateau_grid() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """A 5×5×5 grid where the center region is a stable plateau.

    Returns (sharpe_grid, expectancy_grid, shape).
    sharpe_grid and expectancy_grid are 1D arrays of length 125, indexed
    the same way as np.ravel_multi_index over shape (5, 5, 5).
    """
    shape = (5, 5, 5)
    n = 125
    sharpe = np.full(n, 0.8)
    expectancy = np.full(n, 0.01)
    # Peak at center (2, 2, 2) -> flat index 62
    sharpe[62] = 1.2
    return sharpe, expectancy, shape


def test_ndh_passes_stable_plateau():
    """A plateau where all 6 axis-aligned neighbors are positive passes."""
    sharpe, expectancy, shape = _make_plateau_grid()
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    assert isinstance(result, NdhResult)
    assert result.passed is True
    # Axis-aligned (von-Neumann) neighborhood: 2 per axis × 3 axes = 6.
    assert result.n_neighbors == 6
    assert result.n_positive == 6
    assert result.density == 1.0


def test_ndh_fails_isolated_spike():
    """A spike surrounded by negative-sharpe neighbors fails density."""
    shape = (5, 5, 5)
    n = 125
    sharpe = np.full(n, -0.2)  # all neighbors negative
    expectancy = np.full(n, -0.005)
    sharpe[62] = 1.5  # isolated peak
    expectancy[62] = 0.02
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    assert result.passed is False
    assert result.n_positive == 0
    assert result.density == 0.0


def test_ndh_fails_high_variance():
    """A neighborhood with wildly varying sharpes fails variance cap."""
    sharpe, expectancy, shape = _make_plateau_grid()
    # Make some neighbors very high, others near zero -> high std
    idxs = _neighbor_indices(62, shape)
    for i, idx in enumerate(idxs):
        sharpe[idx] = 2.0 if i % 2 == 0 else 0.01
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    # Density passes (all positive), but variance should fail
    assert result.density > 0.85
    assert result.passed is False
    assert result.variance_ratio > 0.20


def test_ndh_singleton_dims_do_not_explode():
    """Pinned (size-1) params must not blow up the neighbor offset grid.

    The ensemble grid has 17 param keys but only ~5 vary; the rest are size-1.
    Trailing size-1 dims don't change C-order flat indices, so the neighbor set
    for a 17-dim shape must equal the 5-dim base case. Before the fix, a 17-dim
    shape allocated 3**17 (~129M) offset vectors and OOM-killed the process.
    """
    base_shape = (3, 2, 2, 2, 2)
    full_shape = base_shape + (1,) * 12  # mirrors the 17-key ensemble grid
    peak = int(np.ravel_multi_index((1, 0, 0, 0, 0), base_shape))
    expected = sorted(_neighbor_flat_indices(peak, base_shape))
    got = sorted(_neighbor_flat_indices(peak, full_shape))
    assert got == expected


def test_ndh_edge_peak_has_fewer_neighbors():
    """Peak at grid corner (0,0,0) has only 3 axis-aligned neighbors."""
    shape = (5, 5, 5)
    n = 125
    sharpe = np.full(n, 0.5)
    expectancy = np.full(n, 0.01)
    sharpe[0] = 1.0  # corner
    result = ndh_check(
        peak_idx=0,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    # Corner: only +1 reachable in each of 3 axes = 3 neighbors.
    assert result.n_neighbors == 3
    assert result.passed is True


def test_ndh_needs_both_sharpe_and_expectancy_positive():
    """Neighbor with positive sharpe but negative expectancy doesn't count."""
    sharpe, expectancy, shape = _make_plateau_grid()
    idxs = _neighbor_indices(62, shape)
    # One of 6 neighbors gets negative expectancy -> 5/6 = 0.833 < 0.85.
    expectancy[idxs[0]] = -0.001
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    assert result.n_neighbors == 6
    assert result.n_positive == 5
    assert result.passed is False


def test_ndh_excludes_regime_axes():
    """neighbor_axes restricts the neighborhood to the given (tuning) axes."""
    shape = (3, 2, 2, 2, 2)  # axis 0 = a regime param like min_agree
    n = int(np.prod(shape))
    sharpe = np.full(n, 0.8)
    expectancy = np.full(n, 0.01)
    peak = int(np.ravel_multi_index((1, 0, 0, 0, 0), shape))
    sharpe[peak] = 1.0
    # Make the axis-0 (regime) neighbors terrible — they must be ignored.
    for nb in _neighbor_flat_indices(peak, shape, axes=(0,)):
        sharpe[nb] = -5.0
        expectancy[nb] = -1.0
    result = ndh_check(
        peak_idx=peak,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
        neighbor_axes=(1, 2, 3, 4),  # exclude regime axis 0
    )
    assert result.density == 1.0
    assert result.passed is True


def _neighbor_indices(peak_idx: int, shape: tuple[int, ...]) -> list[int]:
    """Helper delegating to the real axis-aligned neighbor implementation."""
    return _neighbor_flat_indices(peak_idx, shape)


def test_psr_known_values():
    """PSR with zero skew/kurtosis and large T should match basic z-test."""
    # SR=1.0 vs benchmark=0.0, T=252, normal returns
    result = psr(
        observed_sr=1.0,
        benchmark_sr=0.0,
        n_obs=252,
        skew=0.0,
        kurtosis_excess=0.0,
    )
    # SE(SR) = sqrt(1/251) ≈ 0.0631, z = 1.0/0.0631 ≈ 15.85 -> PSR ≈ 1.0
    assert result > 0.99


def test_psr_low_sr_low_probability():
    """PSR with SR barely above benchmark should give ~0.5."""
    result = psr(
        observed_sr=0.5,
        benchmark_sr=0.5,
        n_obs=252,
        skew=0.0,
        kurtosis_excess=0.0,
    )
    assert abs(result - 0.5) < 0.01


def test_psr_negative_skew_reduces_confidence():
    """Negative skew inflates SR variance -> lower PSR."""
    psr_normal = psr(
        observed_sr=1.0,
        benchmark_sr=0.0,
        n_obs=252,
        skew=0.0,
        kurtosis_excess=0.0,
    )
    psr_skewed = psr(
        observed_sr=1.0,
        benchmark_sr=0.0,
        n_obs=252,
        skew=-2.0,
        kurtosis_excess=5.0,
    )
    assert psr_skewed < psr_normal


def test_expected_max_sr_increases_with_trials():
    """More trials -> higher expected max Sharpe."""
    sr10 = expected_max_sr(n_trials=10, n_obs=252, skew=0.0, kurtosis_excess=0.0)
    sr1000 = expected_max_sr(n_trials=1000, n_obs=252, skew=0.0, kurtosis_excess=0.0)
    sr10000 = expected_max_sr(n_trials=10000, n_obs=252, skew=0.0, kurtosis_excess=0.0)
    assert sr10 < sr1000 < sr10000
    # With 10 trials and T=252, expected max SR should be modest
    assert 0.05 < sr10 < 0.5


def test_expected_max_sr_longer_track_record_narrows():
    """Longer T -> smaller SE -> smaller expected max SR."""
    sr_short = expected_max_sr(n_trials=100, n_obs=60, skew=0.0, kurtosis_excess=0.0)
    sr_long = expected_max_sr(n_trials=100, n_obs=1000, skew=0.0, kurtosis_excess=0.0)
    assert sr_long < sr_short


def test_dsr_check_strong_signal_passes():
    """High SR with few trials should pass easily."""
    result = dsr_check(
        observed_sr=1.5,
        n_obs=504,  # 2 years daily
        n_trials=20,
        skew=0.0,
        kurtosis_excess=0.0,
    )
    assert isinstance(result, DsrResult)
    assert result.passed is True
    assert result.dsr_value > 0.80


def test_dsr_check_data_mined_fails():
    """Mediocre SR found after thousands of trials should fail."""
    result = dsr_check(
        observed_sr=0.15,
        n_obs=252,
        n_trials=10000,
        skew=-1.0,
        kurtosis_excess=3.0,
    )
    assert result.passed is False
    assert result.dsr_value < 0.80


def test_dsr_check_threshold_respected():
    """Custom threshold is used, not hardcoded 0.80."""
    kwargs = dict(observed_sr=0.3, n_obs=252, n_trials=500, skew=0.0, kurtosis_excess=0.0)
    result_strict = dsr_check(**kwargs, threshold=0.999)
    result_relaxed = dsr_check(**kwargs, threshold=0.01)
    assert result_relaxed.passed is True
    assert result_strict.passed is False
    assert result_relaxed.dsr_value == result_strict.dsr_value
