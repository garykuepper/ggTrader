"""WFO gates: the NDH plateau lattice and the deflated Sharpe ratio (DSR) check.

The NDH logic itself is ``lab.gates.ndh_check``, reused unchanged. This module owns the glue
between a fold's per-combo metrics and that function (the grid lattice, the regime axes), and
the DSR call through ``engine.stats.dsr`` (vectorbt's deflated Sharpe, configured to the legacy
null; spec 5.4 item 6, OD2). It is the only DSR implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple

import numpy as np
import pandas as pd

from ggTrader.engine import stats
from ggTrader.lab.gates import ndh_check

#: Discrete "regime" params where a +-1 grid step is a different strategy rather than a small
#: perturbation. Excluded from the NDH plateau neighborhood.
REGIME_PARAMS: frozenset = frozenset({"min_agree", "min_agree_exit"})
#: Windows of at most this many bars are not scored for DSR (the legacy ``len(eq) > 10``).
MIN_DSR_BARS = 10


class NdhOutcome(NamedTuple):
    """NDH gate result plus the fields the fold table reports."""

    passed: bool
    density: float
    variance_ratio: float


def extract_grid_arrays(
    train_metrics: List[Dict[str, Any]],
    grid: List[Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], Dict[int, int]]:
    """Map sweep results to N-dim grid arrays for NDH.

    Returns:
        sharpe_grid: 1D array indexed by flat grid position.
        expectancy_grid: 1D array indexed by flat grid position.
        grid_shape: shape of the N-dim grid.
        result_to_grid: mapping from train_metrics index to grid flat index.
    """
    if not grid:
        return np.array([]), np.array([]), (), {}

    # Requires uniform keys across all grid combos (Cartesian product).
    param_keys = sorted(grid[0].keys())
    if any(sorted(c.keys()) != param_keys for c in grid):
        return np.array([]), np.array([]), (), {}

    # Axes are categorical coordinates for the NDH lattice; the sort only needs to be
    # deterministic. None ("no stop", e.g. td_stop/tp_stop) is not orderable against numbers,
    # so key it to sort last without comparing values.
    axes: Dict[str, List[Any]] = {
        k: sorted({c[k] for c in grid}, key=lambda v: (v is None, v)) for k in param_keys
    }
    shape = tuple(len(axes[k]) for k in param_keys)

    n_cells = 1
    for s in shape:
        n_cells *= s

    sharpe_arr = np.full(n_cells, float("nan"))
    expectancy_arr = np.full(n_cells, float("nan"))
    axis_idx = {k: {v: i for i, v in enumerate(axes[k])} for k in param_keys}

    def _flat_idx(params: Dict[str, Any]) -> int:
        coords = tuple(axis_idx[k][params[k]] for k in param_keys)
        return int(np.ravel_multi_index(coords, shape))

    result_to_grid: Dict[int, int] = {}
    for ri, m in enumerate(train_metrics):
        flat = _flat_idx(m["params"])
        sharpe_arr[flat] = m.get("sharpe", float("nan"))
        # Real per-trade expectancy: window return / trades. Only its sign reaches the gate.
        n_trades = max(1, int(m.get("n_trades", 0)))
        expectancy_arr[flat] = (m.get("total_return_pct", 0.0) / 100.0) / n_trades
        result_to_grid[ri] = flat

    # Replace NaN with 0 for the gate check (missing combos count as zero-edge).
    return (
        np.nan_to_num(sharpe_arr, nan=0.0),
        np.nan_to_num(expectancy_arr, nan=0.0),
        shape,
        result_to_grid,
    )


def run_ndh(
    train_metrics: List[Dict[str, Any]],
    grid: List[Dict[str, Any]],
    best_idx: int,
    threshold: float,
) -> NdhOutcome:
    """NDH gate for the winner at ``train_metrics[best_idx]`` (fails when it is off the lattice)."""
    sharpe_grid, exp_grid, grid_shape, r2g = extract_grid_arrays(train_metrics, grid)
    if best_idx not in r2g or len(grid_shape) == 0:
        return NdhOutcome(False, 0.0, float("nan"))
    # Restrict the neighborhood to smooth tuning axes: a +-1 step in a regime param (min_agree)
    # is a different strategy, not a small perturbation.
    param_keys = sorted(grid[0].keys())
    neighbor_axes = tuple(i for i, k in enumerate(param_keys) if k not in REGIME_PARAMS)
    result = ndh_check(
        peak_idx=r2g[best_idx],
        sharpe_grid=sharpe_grid,
        expectancy_grid=exp_grid,
        grid_shape=grid_shape,
        density_threshold=threshold,
        neighbor_axes=neighbor_axes,
    )
    return NdhOutcome(result.passed, result.density, result.variance_ratio)


def run_dsr(
    window_returns: pd.Series,
    n_trials: int,
    threshold: float,
) -> tuple[bool, float]:
    """DSR gate of the winner's training window: ``(passed, dsr_value)``.

    One implementation for every grid size: ``engine.stats.dsr`` (vectorbt's deflated Sharpe
    configured to the legacy statistic). With ``n_trials == 1`` nothing is deflated and the
    value is the probabilistic Sharpe ratio against zero (see ``engine.stats.dsr``).
    """
    if len(window_returns) < MIN_DSR_BARS:
        return False, 0.0
    value = float(stats.dsr(window_returns, n_trials=n_trials))
    # A flat window has no defined Sharpe; the legacy PSR reports 0.0 (failed) for it.
    value = value if np.isfinite(value) else 0.0
    return bool(value >= threshold), value
