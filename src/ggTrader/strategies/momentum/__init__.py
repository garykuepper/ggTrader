"""Cross-Sectional Momentum package."""

from __future__ import annotations

from ggTrader.strategies.momentum.config import MomentumConfig
from ggTrader.strategies.momentum.cross_sectional import CrossSectionalMomentum
from ggTrader.strategies.momentum.factors import (
    compute_momentum_nb,
    compute_liquidity_shock_nb,
    strip_btc_beta_nb,
)
from ggTrader.strategies.momentum.signals import CrossSectionalSignalGenerator

__all__ = [
    "MomentumConfig",
    "CrossSectionalMomentum",
    "CrossSectionalSignalGenerator",
    "compute_momentum_nb",
    "compute_liquidity_shock_nb",
    "strip_btc_beta_nb",
]
