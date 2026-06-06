"""Configuration schema for Strategy 1 (Cross-Sectional Momentum)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class MomentumConfig(BaseModel):
    """Configuration settings for Cross-Sectional Momentum strategy."""

    asset_class: Literal["equity", "crypto"] = "crypto"

    formation_window: int = 30
    exclusion_gap: int = 3
    liq_smooth_mode: Literal["rolling", "ewm"] = "ewm"
    liq_smooth_window: int = 14
    rank_mode: Literal["sector", "residual_btc"] = "residual_btc"
    btc_beta_window: int = 60
    w_momentum: float = 0.6
    w_liquidity: float = 0.4
    entry_percentile: float = 0.90
    vol_mode: Literal["split", "continuous"] = "continuous"

    fee_rate: float = 0.001
    slippage: float = 0.002
    hold_bars: int = 3
    engaged_threshold: float = 0.60
    hmm_filter_enabled: bool = False

    @classmethod
    def for_equities(cls) -> MomentumConfig:
        """Get default MomentumConfig for equities."""
        return cls(
            asset_class="equity",
            formation_window=40,
            exclusion_gap=5,
            liq_smooth_mode="rolling",
            liq_smooth_window=20,
            rank_mode="sector",
            w_momentum=0.6,
            w_liquidity=0.4,
            entry_percentile=0.90,
            vol_mode="split",
            fee_rate=0.0005,
            slippage=0.0005,
            hold_bars=5,
            engaged_threshold=0.65,
        )

    @classmethod
    def for_crypto(cls) -> MomentumConfig:
        """Get default MomentumConfig for crypto."""
        return cls(
            asset_class="crypto",
            formation_window=30,
            exclusion_gap=3,
            liq_smooth_mode="ewm",
            liq_smooth_window=14,
            rank_mode="residual_btc",
            btc_beta_window=60,
            w_momentum=0.6,
            w_liquidity=0.4,
            entry_percentile=0.90,
            vol_mode="continuous",
            fee_rate=0.001,
            slippage=0.002,
            hold_bars=3,
            engaged_threshold=0.60,
        )

    @model_validator(mode="after")
    def validate_weights_and_percentile(self) -> MomentumConfig:
        """Validate that weights sum to 1.0 and entry percentile is in range."""
        if not np_isclose(self.w_momentum + self.w_liquidity, 1.0):
            raise ValueError(
                f"w_momentum ({self.w_momentum}) and w_liquidity ({self.w_liquidity}) "
                f"must sum to 1.0 (got {self.w_momentum + self.w_liquidity})"
            )
        if not (0.5 < self.entry_percentile < 1.0):
            raise ValueError(
                "entry_percentile must be strictly between 0.5 and 1.0 "
                f"(got {self.entry_percentile})"
            )
        return self


def np_isclose(a: float, b: float) -> bool:
    """Float equality check helper."""
    return abs(a - b) < 1e-9
