"""Neighborhood-density robustness gate for WFO parameter validation (NDH).

The deflated Sharpe gate is ``ggTrader.engine.stats.dsr``; the hand-written version that lived
here was removed in P10 (its reference copy is ``tests/oracles/legacy_dsr.py``).
"""

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


def _neighbor_flat_indices(
    peak_idx: int,
    grid_shape: tuple[int, ...],
    axes: tuple[int, ...] | list[int] | None = None,
) -> list[int]:
    """Return flat indices of axis-aligned ±1-step neighbors in an N-dim grid.

    Uses a von-Neumann neighborhood (one axis changed at a time, no diagonals).
    A Moore neighborhood (diagonals) over a coarse grid connects nearly every
    cell to every other — on a (3,2,2,2,2) grid a single cell had 47 of 48
    "neighbors", so NDH degenerated into a grid-wide quality check instead of a
    local-plateau check.

    ``axes`` restricts which dimensions are treated as neighbors. Pass the
    smooth tuning axes only; regime axes (e.g. min_agree, where ±1 is a
    different strategy rather than a small perturbation) should be excluded so
    the neighborhood measures genuine parameter robustness. Defaults to every
    dimension with size > 1.
    """
    shape_arr = np.array(grid_shape)
    coords = np.array(np.unravel_index(peak_idx, grid_shape))
    if axes is None:
        axes = tuple(d for d, size in enumerate(grid_shape) if size > 1)
    neighbors: list[int] = []
    for d in axes:
        if grid_shape[d] <= 1:
            continue
        for delta in (-1, 1):
            nc = coords.copy()
            nc[d] += delta
            if 0 <= nc[d] < shape_arr[d]:
                neighbors.append(int(np.ravel_multi_index(nc, grid_shape)))
    return neighbors


def ndh_check(
    peak_idx: int,
    sharpe_grid: np.ndarray,
    expectancy_grid: np.ndarray,
    grid_shape: tuple[int, ...],
    density_threshold: float = 0.85,
    variance_cap: float = 0.20,
    neighbor_axes: tuple[int, ...] | list[int] | None = None,
) -> NdhResult:
    """Neighborhood Density Hurdle: reject isolated parameter spikes.

    Checks an axis-aligned ±1-step neighborhood around peak_idx:
    1. Density: fraction of neighbors with BOTH positive Sharpe AND positive
       trade expectancy must be >= density_threshold (default 85%).
    2. Variance cap: std(neighbor Sharpes) / peak Sharpe <= variance_cap
       (default 20%).

    ``neighbor_axes`` restricts the neighborhood to the smooth tuning axes;
    regime axes (min_agree, etc.) should be excluded so a ±1 step is a genuine
    small perturbation rather than a different strategy.
    """
    neighbor_idxs = _neighbor_flat_indices(peak_idx, grid_shape, axes=neighbor_axes)
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

    # Variance cap is exempted on a perfect plateau: when every neighbor is
    # profitable (density == 1.0), Sharpe-magnitude dispersion among positive
    # neighbors is not an overfit signal — density already proves robustness.
    # The cap still applies when a neighbor is unprofitable (density < 1.0).
    variance_ok = density >= 1.0 or variance_ratio <= variance_cap
    passed = density >= density_threshold and variance_ok
    return NdhResult(
        passed=passed,
        density=density,
        variance_ratio=variance_ratio,
        n_positive=n_positive,
        n_neighbors=n_neighbors,
    )
