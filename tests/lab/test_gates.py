"""Unit tests for WFO robustness gates: NDH plateau filter and DSR gate."""

import numpy as np

from ggTrader.lab.gates import NdhResult, ndh_check


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
    """A plateau where all 26 neighbors have positive sharpe+expectancy passes."""
    sharpe, expectancy, shape = _make_plateau_grid()
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    assert isinstance(result, NdhResult)
    assert result.passed is True
    assert result.n_neighbors == 26
    assert result.n_positive == 26
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


def test_ndh_edge_peak_has_fewer_neighbors():
    """Peak at grid corner (0,0,0) has only 7 neighbors, not 26."""
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
    assert result.n_neighbors == 7
    assert result.passed is True


def test_ndh_needs_both_sharpe_and_expectancy_positive():
    """Neighbor with positive sharpe but negative expectancy doesn't count."""
    sharpe, expectancy, shape = _make_plateau_grid()
    idxs = _neighbor_indices(62, shape)
    # Make 5 neighbors have negative expectancy
    for idx in idxs[:5]:
        expectancy[idx] = -0.001
    result = ndh_check(
        peak_idx=62,
        sharpe_grid=sharpe,
        expectancy_grid=expectancy,
        grid_shape=shape,
    )
    # 21/26 positive = 0.808 < 0.85 threshold
    assert result.n_positive == 21
    assert result.passed is False


def _neighbor_indices(peak_idx: int, shape: tuple[int, ...]) -> list[int]:
    """Helper to get neighbor flat indices (same logic the implementation should use)."""
    coords = np.array(np.unravel_index(peak_idx, shape))
    ndim = len(shape)
    offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * ndim)).T.reshape(-1, ndim)
    neighbors = []
    for offset in offsets:
        if np.all(offset == 0):
            continue
        nc = coords + offset
        if np.all(nc >= 0) and np.all(nc < np.array(shape)):
            neighbors.append(int(np.ravel_multi_index(nc, shape)))
    return neighbors
