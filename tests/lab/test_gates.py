"""Unit tests for the WFO robustness gate NDH (the DSR gate is ``engine.stats.dsr``)."""

import numpy as np

from ggTrader.lab.gates import NdhResult, _neighbor_flat_indices, ndh_check


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


def test_ndh_perfect_plateau_exempt_from_variance_cap():
    """A perfect-density plateau (every neighbor profitable) passes even with
    high Sharpe-magnitude dispersion. Dispersion among *positive* neighbors is
    not an overfit signal — density already proves robustness. (2026-06-25:
    the variance cap was over-rejecting these on midcaps.)"""
    sharpe, expectancy, shape = _make_plateau_grid()
    idxs = _neighbor_flat_indices(62, shape)
    for i, idx in enumerate(idxs):
        sharpe[idx] = 2.0 if i % 2 == 0 else 0.01  # all positive, high std
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    assert result.density == 1.0
    assert result.variance_ratio > 0.20  # would have failed the old cap
    assert result.passed is True


def test_ndh_variance_cap_still_applies_below_perfect_density():
    """When density is below 1.0 (a neighbor is unprofitable) but still clears
    the density threshold, the variance cap still bites — it is only exempted
    for perfect plateaus."""
    sharpe, expectancy, shape = _make_plateau_grid()
    idxs = _neighbor_flat_indices(62, shape)
    for i, idx in enumerate(idxs):
        sharpe[idx] = 2.0 if i % 2 == 0 else 0.01
    # Make ONE neighbor unprofitable -> density 5/6 ≈ 0.83 (< 1.0)
    sharpe[idxs[1]] = -0.3
    expectancy[idxs[1]] = -0.005
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
        density_threshold=0.8,  # 0.83 clears density, so only variance can fail
    )
    assert result.density < 1.0
    assert result.density >= 0.8
    assert result.variance_ratio > 0.20
    assert result.passed is False


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
