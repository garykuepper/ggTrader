"""Statistical robustness gates for WFO parameter validation."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import norm


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

    passed = density >= density_threshold and variance_ratio <= variance_cap
    return NdhResult(
        passed=passed,
        density=density,
        variance_ratio=variance_ratio,
        n_positive=n_positive,
        n_neighbors=n_neighbors,
    )


class DsrResult(NamedTuple):
    """Result of the Deflated Sharpe Ratio check."""

    passed: bool
    dsr_value: float
    expected_max_sr: float
    threshold: float


_EULER_MASCHERONI = 0.5772156649015329


def _sr_standard_error(sr: float, n_obs: int, skew: float, kurtosis_excess: float) -> float:
    """Standard error of the Sharpe Ratio estimate (Bailey & López de Prado 2014).

    σ̂(SR) = sqrt( (1 - γ₃·SR + (γ₄-1)/4 · SR²) / (T-1) )
    """
    if n_obs <= 1:
        return float("inf")
    numerator = 1.0 - skew * sr + (kurtosis_excess - 1.0) / 4.0 * sr**2
    numerator = max(numerator, 1e-12)
    return float(np.sqrt(numerator / (n_obs - 1)))


def psr(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis_excess: float,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark_sr).

    PSR(SR*) = Φ( (SR_obs - SR*) / σ̂(SR_obs) )
    """
    se = _sr_standard_error(observed_sr, n_obs, skew, kurtosis_excess)
    if se <= 0 or not np.isfinite(se):
        return 0.0
    z = (observed_sr - benchmark_sr) / se
    return float(norm.cdf(z))


def expected_max_sr(
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis_excess: float,
) -> float:
    """Expected maximum Sharpe across N independent trials.

    E[max(SR)] ≈ σ̂(SR) · ( (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)) )

    where γ is the Euler-Mascheroni constant and σ̂(SR) is evaluated at SR=0
    (the null hypothesis that the true SR is zero).
    """
    if n_trials < 2:
        return 0.0
    se = _sr_standard_error(0.0, n_obs, skew, kurtosis_excess)
    gamma = _EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(se * ((1.0 - gamma) * z1 + gamma * z2))


def dsr_check(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    skew: float,
    kurtosis_excess: float,
    threshold: float = 0.80,
) -> DsrResult:
    """Deflated Sharpe Ratio: PSR evaluated against E[max(SR)] as the benchmark.

    Passes if DSR >= threshold (default 0.80, per spec Section 6).
    """
    e_max = expected_max_sr(n_trials, n_obs, skew, kurtosis_excess)
    dsr_value = psr(observed_sr, e_max, n_obs, skew, kurtosis_excess)
    return DsrResult(
        passed=dsr_value >= threshold,
        dsr_value=dsr_value,
        expected_max_sr=e_max,
        threshold=threshold,
    )
