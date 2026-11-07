"""Shared research configuration (``LabConfig``) for the lab bench."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabConfig:
    """Tunables shared by lab strategies and the walk-forward."""

    top_n: int = 50
    lookback: int = 252  # trailing bars for the momentum measurement window
    skip: int = 21  # most-recent bars excluded (12-1 momentum)
    min_history_bars: int = 400  # required non-NaN closes to be eligible
    max_stocks: int | None = None  # cap the per-rebalance universe (deterministic)
