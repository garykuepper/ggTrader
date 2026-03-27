"""Pure helper utilities for the orchestration layer."""

import gc
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None


def _eta_str(seconds: float) -> str:
    """Format seconds as HH:MM:SS for ETA display."""
    return str(timedelta(seconds=int(max(0, seconds))))


def _wall_clock_eta(seconds: float) -> str:
    """Return wall-clock estimate (e.g., '1:30 PM')."""
    from datetime import datetime, timedelta

    finish = datetime.now() + timedelta(seconds=max(0, seconds))
    return finish.strftime("%I:%M %p")


def _log_memory_usage(label: str) -> None:
    """Log current process memory for debugging."""
    if psutil is None:
        return
    try:
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / (1024**2)
        print(f"    [{label}] Memory: {mem_mb:.1f} MB")
    except Exception:
        pass


def _to_native(val: Any) -> Any:
    """Ensure basic types are JSON serializable. Converts NaNs to None."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_native(x) for x in val]
    return val


def _coerce_metric_float(x: Any) -> float:
    """Coerce WFO/OOS metrics to float; None or invalid -> nan for numpy reductions."""
    if x is None:
        return float("nan")
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return float("nan")
        return xf
    except (TypeError, ValueError):
        return float("nan")


def _safe(val: Any, default: float = 0.0) -> float:
    """Replace None, NaN, or Inf with default for JSON safety."""
    import math

    if val is None:
        return default
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(v) or math.isinf(v)) else v


def _as_optional_float(x: Any) -> Any:
    """Return a finite float or None (for JSON/report fields where 0 would mislead)."""
    import math

    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def _format_robustness_metric(x: Any) -> str:
    """Human-readable robustness for logs (handles nan, +/-inf)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf):
        return "nan"
    if np.isinf(xf):
        return "-inf" if xf < 0 else "inf"
    return f"{xf:.4f}"


def _extract_params(
    idx: Any,
    metric_series: pd.Series,
    param_names: List[str],
    full_param_grid: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper to extract native parameters from VectorBT Index/MultiIndex."""
    extracted = {}
    if isinstance(idx, tuple):
        # MultiIndex case
        names = (
            metric_series.index.names
            if hasattr(metric_series.index, "names") and metric_series.index.names[0]
            else param_names
        )
        for name, val in zip(names, idx):
            clean_name = name.replace("sf_", "") if name else "unknown"
            if clean_name in param_names:
                extracted[clean_name] = val
    else:
        # Single index case
        name = metric_series.index.name
        clean_name = name.replace("sf_", "") if name else param_names[0]
        extracted[clean_name] = idx

    # Fill defaults for any missing from extraction (constants in the grid)
    for k, v in full_param_grid.items():
        if k not in extracted:
            extracted[k] = v[0] if isinstance(v, list) else v
    return extracted


def _first_grid_value(param_grid: Dict[str, Any], key: str) -> Any:
    """Return a single default from the grid for key (first list element or scalar)."""
    v = param_grid.get(key)
    if v is None:
        return None
    return v[0] if isinstance(v, list) else v


def _default_params_from_grid(param_grid: Dict[str, Any]) -> Dict[str, Any]:
    """Scalar defaults for every key (first list element or scalar)."""
    return {k: _first_grid_value(param_grid, k) for k in param_grid}


def _wfo_per_coin_fallback_triple(
    strategy_param_grids: Dict[str, Dict[str, Any]],
    exit_tournament: List[str],
) -> Tuple[str, str, Dict[str, Any]]:
    """First entry strategy, first exit in tournament, and grid first-value params."""
    if not strategy_param_grids or not exit_tournament:
        raise ValueError("strategy_param_grids and exit_tournament must be non-empty")
    first_strategy = next(iter(strategy_param_grids))
    first_exit = exit_tournament[0]
    grid = strategy_param_grids[first_strategy]
    return first_strategy, first_exit, _default_params_from_grid(grid)


def _is_better_robustness(candidate: float, best: float) -> bool:
    """True if candidate should replace best (finite candidate beats non-finite or lower best)."""
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(best):
        return True
    return candidate > best


def _is_bad_engine_param(val: Any) -> bool:
    """True if val is None or a non-finite float (would break float() in signal code)."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        return bool(np.isnan(val) or np.isinf(val))
    return False


def _coerce_strategy_params_for_engine(
    extracted: Dict[str, Any], param_grid: Dict[str, Any]
) -> Dict[str, Any]:
    """Replace None/NaN with grid defaults so FastBacktest never gets JSON-sanitized Nones."""
    out = dict(extracted)
    for k in param_grid:
        if k not in out or _is_bad_engine_param(out.get(k)):
            out[k] = _first_grid_value(param_grid, k)
    return out
