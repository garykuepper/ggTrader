from __future__ import annotations

from typing import Any

from .conviction import ConvictionBBSignal
from .ensemble import EnsembleConvictionSignal, EnsembleSignal
from .ensemble_ic import EnsembleICSignal
from .ensemble_kelly import EnsembleKellySignal
from .momentum import CrossSectionalMomentum, DualMomentum
from .registry import (
    all_strategy_names,
    build_strategy,
    signal_registry,
    signal_strategy_names,
    weight_strategy_names,
)
from .signals import (
    BollingerReversionSignal,
    EmaCrossSignal,
    MACDDivergenceSignal,
    MultiTimeframeReversionSignal,
    RsiReversionSignal,
    VolumeBBReversionSignal,
    WfoTournamentSignal,
)

STRATEGY_REGISTRY: dict[str, Any] = {
    "ema_cross": EmaCrossSignal,
    "wfo_tournament": WfoTournamentSignal,
    "bb_reversion": BollingerReversionSignal,
    "rsi_reversion": RsiReversionSignal,
    "macd_divergence": MACDDivergenceSignal,
    "volume_bb_reversion": VolumeBBReversionSignal,
    "mtf_reversion": MultiTimeframeReversionSignal,
    "ensemble": EnsembleSignal,
    "ensemble_ic": EnsembleICSignal,
    "ensemble_kelly": EnsembleKellySignal,
    "conviction_bb": ConvictionBBSignal,
    "ensemble_conviction": EnsembleConvictionSignal,
    "xs_momentum": CrossSectionalMomentum,
    "dual_momentum": DualMomentum,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "all_strategy_names",
    "build_strategy",
    "signal_registry",
    "signal_strategy_names",
    "weight_strategy_names",
    "CrossSectionalMomentum",
    "DualMomentum",
    "EmaCrossSignal",
    "WfoTournamentSignal",
    "BollingerReversionSignal",
    "RsiReversionSignal",
    "MACDDivergenceSignal",
    "VolumeBBReversionSignal",
    "MultiTimeframeReversionSignal",
    "EnsembleSignal",
    "EnsembleICSignal",
    "EnsembleKellySignal",
    "ConvictionBBSignal",
    "EnsembleConvictionSignal",
]
