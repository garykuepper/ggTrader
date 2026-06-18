"""Statistical robustness gates for WFO parameter validation."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class NdhResult(NamedTuple):
    """Result of the Neighborhood Density Hurdle check."""

    passed: bool
    density: float
    variance_ratio: float
    n_positive: int
    n_neighbors: int


def _neighbor_flat_indices(peak_idx: int, grid_shape: tuple[int, ...]) -> list[int]:
    """Return flat indices of all ±1-step neighbors in an N-dim grid."""
    coords = np.array(np.unravel_index(peak_idx, grid_shape))
    ndim = len(grid_shape)
    offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * ndim)).T.reshape(-1, ndim)
    neighbors: list[int] = []
    for offset in offsets:
        if np.all(offset == 0):
            continue
        nc = coords + offset
        if np.all(nc >= 0) and np.all(nc < np.array(grid_shape)):
            neighbors.append(int(np.ravel_multi_index(nc, grid_shape)))
    return neighbors


def ndh_check(
    peak_idx: int,
    sharpe_grid: np.ndarray,
    expectancy_grid: np.ndarray,
    grid_shape: tuple[int, ...],
    density_threshold: float = 0.85,
    variance_cap: float = 0.20,
) -> NdhResult:
    """Neighborhood Density Hurdle: reject isolated parameter spikes.

    Checks a ±1-step neighborhood around peak_idx in an N-dimensional grid:
    1. Density: fraction of neighbors with BOTH positive Sharpe AND positive
       trade expectancy must be >= density_threshold (default 85%).
    2. Variance cap: std(neighbor Sharpes) / peak Sharpe <= variance_cap
       (default 20%).
    """
    neighbor_idxs = _neighbor_flat_indices(peak_idx, grid_shape)
    n_neighbors = len(neighbor_idxs)
    if n_neighbors == 0:
        return NdhResult(
            passed=False,
            density=0.0,
            variance_ratio=float("inf"),
            n_positive=0,
            n_neighbors=0,
        )

    neighbor_sharpes = sharpe_grid[neighbor_idxs]
    neighbor_expectancy = expectancy_grid[neighbor_idxs]

    n_positive = int(np.sum((neighbor_sharpes > 0) & (neighbor_expectancy > 0)))
    density = n_positive / n_neighbors

    peak_sharpe = sharpe_grid[peak_idx]
    if peak_sharpe <= 0:
        return NdhResult(
            passed=False,
            density=density,
            variance_ratio=float("inf"),
            n_positive=n_positive,
            n_neighbors=n_neighbors,
        )

    variance_ratio = float(np.std(neighbor_sharpes) / peak_sharpe)

    passed = density >= density_threshold and variance_ratio <= variance_cap
    return NdhResult(
        passed=passed,
        density=density,
        variance_ratio=variance_ratio,
        n_positive=n_positive,
        n_neighbors=n_neighbors,
    )
