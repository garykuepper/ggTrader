from __future__ import annotations

from typing import Any

from .conviction import ConvictionBBSignal
from .ensemble import EnsembleConvictionSignal, EnsembleSignal
from .ensemble_ic import EnsembleICSignal
from .ensemble_kelly import EnsembleKellySignal
from .idio_vol import IdioVolStrategy
from .leveraged_rotation import (
    LeveragedRotationNasdaq100,
    LeveragedRotationRussell2000,
    LeveragedRotationSp500,
)
from .leveraged_trend import (
    LeveragedTrendNasdaq100,
    LeveragedTrendRussell2000,
    LeveragedTrendSp500,
)
from .max_effect import MaxEffectStrategy
from .momentum import CrossSectionalMomentum, DualMomentum
from .pairs_stat_arb import PairsStatArb
from .pead import PeadStrategy
from .registry import (
    all_strategy_names,
    apply_sector_constraints,
    build_strategy,
    signal_registry,
    signal_strategy_names,
    weight_strategy_names,
)
from .short_interest import ShortInterestStrategy
from .signals import (
    BollingerReversionSignal,
    EmaCrossSignal,
    MACDDivergenceSignal,
    MultiTimeframeReversionSignal,
    OvernightGapReversionSignal,
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
    "overnight_gap": OvernightGapReversionSignal,
    "ensemble": EnsembleSignal,
    "ensemble_ic": EnsembleICSignal,
    "ensemble_kelly": EnsembleKellySignal,
    "conviction_bb": ConvictionBBSignal,
    "ensemble_conviction": EnsembleConvictionSignal,
    "xs_momentum": CrossSectionalMomentum,
    "dual_momentum": DualMomentum,
    "idio_vol": IdioVolStrategy,
    "max_effect": MaxEffectStrategy,
    "pairs_stat_arb": PairsStatArb,
    "pead": PeadStrategy,
    "short_interest": ShortInterestStrategy,
    "leveraged_rotation_sp500": LeveragedRotationSp500,
    "leveraged_rotation_nasdaq100": LeveragedRotationNasdaq100,
    "leveraged_rotation_russell2000": LeveragedRotationRussell2000,
    "leveraged_trend_sp500": LeveragedTrendSp500,
    "leveraged_trend_nasdaq100": LeveragedTrendNasdaq100,
    "leveraged_trend_russell2000": LeveragedTrendRussell2000,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "all_strategy_names",
    "apply_sector_constraints",
    "build_strategy",
    "signal_registry",
    "signal_strategy_names",
    "weight_strategy_names",
    "CrossSectionalMomentum",
    "DualMomentum",
    "IdioVolStrategy",
    "MaxEffectStrategy",
    "PairsStatArb",
    "PeadStrategy",
    "ShortInterestStrategy",
    "LeveragedRotationSp500",
    "LeveragedRotationNasdaq100",
    "LeveragedRotationRussell2000",
    "LeveragedTrendSp500",
    "LeveragedTrendNasdaq100",
    "LeveragedTrendRussell2000",
    "EmaCrossSignal",
    "WfoTournamentSignal",
    "BollingerReversionSignal",
    "RsiReversionSignal",
    "OvernightGapReversionSignal",
    "MACDDivergenceSignal",
    "VolumeBBReversionSignal",
    "MultiTimeframeReversionSignal",
    "EnsembleSignal",
    "EnsembleICSignal",
    "EnsembleKellySignal",
    "ConvictionBBSignal",
    "EnsembleConvictionSignal",
]
