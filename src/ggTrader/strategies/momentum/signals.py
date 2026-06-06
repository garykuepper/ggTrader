"""Rank synthesis and entry signal generator for Strategy 1."""

from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from ggTrader.strategies.momentum.config import MomentumConfig


class CrossSectionalSignalGenerator:
    """Generates cross-sectional multi-factor signals based on momentum and liquidity."""

    def __init__(self, config: MomentumConfig) -> None:
        """Initialize the generator.

        Args:
            config: Configuration schema instance.
        """
        self.config = config

    def generate(
        self,
        mom_df: pd.DataFrame,
        liq_df: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Synthesize factors, rank cross-sectionally, and generate boolean signals.

        Args:
            mom_df: DataFrame of momentum scores.
            liq_df: DataFrame of liquidity scores.
            sector_map: Mapping of symbol to GICS sector (for equities).

        Returns:
            Boolean DataFrame where True indicates a long entry signal.
        """
        # Cross-sectional rank both factors pct-wise (axis=1)
        if self.config.rank_mode == "sector" and sector_map:
            # Group by sector along columns (transpose -> groupby -> rank -> transpose)
            mapped_sectors = {s: sector_map.get(s, "unknown") for s in mom_df.columns}

            mom_rank = mom_df.T.groupby(mapped_sectors).rank(pct=True).T
            liq_rank = liq_df.T.groupby(mapped_sectors).rank(pct=True).T
        else:
            mom_rank = mom_df.rank(axis=1, pct=True)
            liq_rank = liq_df.rank(axis=1, pct=True)

        # Composite score
        composite = self.config.w_momentum * mom_rank + self.config.w_liquidity * liq_rank

        # Top percentile quantile row-wise
        quantile_series = composite.quantile(self.config.entry_percentile, axis=1)

        # Boolean entries mask
        entries = composite.ge(quantile_series, axis=0)

        # Shift signals by 1 bar to prevent look-ahead bias
        entries_shifted = entries.shift(1).fillna(False).astype(bool)

        return entries_shifted
