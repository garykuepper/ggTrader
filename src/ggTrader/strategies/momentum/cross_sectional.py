"""Core strategy class for Strategy 1: Cross-Sectional Momentum."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.strategies.momentum.config import MomentumConfig
from ggTrader.strategies.momentum.factors import (
    compute_liquidity_shock_nb,
    compute_momentum_nb,
    strip_btc_beta_nb,
)
from ggTrader.strategies.momentum.signals import CrossSectionalSignalGenerator

logger = logging.getLogger(__name__)


class CrossSectionalMomentum:
    """Orchestrates cross-sectional factor generation and portfolio simulation."""

    def __init__(self, config: MomentumConfig) -> None:
        """Initialize the strategy.

        Args:
            config: Strategy configuration model.
        """
        self.config = config
        self.signal_generator = CrossSectionalSignalGenerator(config)
        self.portfolio: Optional[vbt.Portfolio] = None

    def run(
        self,
        close_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        btc_close: Optional[pd.Series] = None,
        hmm_filter_enabled: Optional[bool] = None,
        regime_gate: Optional[pd.Series] = None,
    ) -> vbt.Portfolio:
        """Execute the strategy backtest/simulation.

        Args:
            close_df: DataFrame of close prices (n_timestamps, n_assets).
            volume_df: DataFrame of volume (n_timestamps, n_assets).
            sector_map: Mapping of asset symbol to GICS sector.
            btc_close: Optional series of BTC close prices (for crypto residual).
            hmm_filter_enabled: If True, applies the HMM regime filter gate.
            regime_gate: Pre-loaded boolean regime series (True=Engaged, False=Lapse).

        Returns:
            vbt.Portfolio object containing backtest results.
        """
        if hmm_filter_enabled is None:
            hmm_filter_enabled = getattr(self.config, "hmm_filter_enabled", False)

        # VectorBT indicator wrappers
        def _mom_wrapper(
            close: pd.DataFrame | pd.Series | np.ndarray, window: int, gap: int
        ) -> np.ndarray:
            if isinstance(close, pd.DataFrame):
                arr = close.values
            elif isinstance(close, pd.Series):
                arr = close.values.reshape(-1, 1)
            else:
                arr = np.atleast_2d(close)
            return compute_momentum_nb(arr, int(window), int(gap))

        def _liq_wrapper(
            close: pd.DataFrame | pd.Series | np.ndarray,
            volume: pd.DataFrame | pd.Series | np.ndarray,
        ) -> np.ndarray:
            if isinstance(close, pd.DataFrame):
                arr_c = close.values
            elif isinstance(close, pd.Series):
                arr_c = close.values.reshape(-1, 1)
            else:
                arr_c = np.atleast_2d(close)

            if isinstance(volume, pd.DataFrame):
                arr_v = volume.values
            elif isinstance(volume, pd.Series):
                arr_v = volume.values.reshape(-1, 1)
            else:
                arr_v = np.atleast_2d(volume)
            return compute_liquidity_shock_nb(arr_c, arr_v)

        mom_ind = vbt.IndicatorFactory(
            class_name="MomentumNB",
            short_name="mom",
            input_names=["close"],
            param_names=["window", "gap"],
            output_names=["mom"],
        ).from_apply_func(_mom_wrapper, keep_pd=True)

        liq_ind = vbt.IndicatorFactory(
            class_name="LiquidityShockNB",
            short_name="liq",
            input_names=["close", "volume"],
            output_names=["liq"],
        ).from_apply_func(_liq_wrapper, keep_pd=True)

        # 1. Handle crypto BTC-beta stripping if configured
        if self.config.asset_class == "crypto":
            if btc_close is None:
                if "BTC-USD" in close_df.columns:
                    btc_close = close_df["BTC-USD"]
                else:
                    btc_cols = [c for c in close_df.columns if "BTC" in c]
                    if btc_cols:
                        btc_close = close_df[btc_cols[0]]

            if btc_close is None:
                raise ValueError("BTC close prices are required for crypto beta stripping.")

            # Calculate log returns
            alt_log_ret = np.log(close_df).diff().fillna(0.0).values
            btc_log_ret = np.log(btc_close).diff().fillna(0.0).values.flatten()

            residual_ret = strip_btc_beta_nb(alt_log_ret, btc_log_ret, self.config.btc_beta_window)

            # Reconstruct pseudo prices via cumulative summation
            start_prices = close_df.iloc[0].values
            cum_res = np.cumsum(np.nan_to_num(residual_ret), axis=0)
            pseudo_close_arr = np.exp(cum_res) * start_prices
            mom_close = pd.DataFrame(
                pseudo_close_arr, index=close_df.index, columns=close_df.columns
            )
        else:
            mom_close = close_df

        # 2. Run factor indicators
        mom_res = mom_ind.run(
            mom_close, window=self.config.formation_window, gap=self.config.exclusion_gap
        ).mom

        if self.config.liq_smooth_mode == "ewm":
            smooth_vol = volume_df.ewm(span=self.config.liq_smooth_window, adjust=False).mean()
        else:
            smooth_vol = volume_df.rolling(window=self.config.liq_smooth_window).mean().fillna(0.0)

        liq_res = liq_ind.run(close_df, smooth_vol).liq

        # Drop parameter levels from VectorBT indicator columns to yield standard asset names index
        if isinstance(mom_res.columns, pd.MultiIndex):
            mom_res.columns = mom_res.columns.get_level_values(-1)
        if isinstance(liq_res.columns, pd.MultiIndex):
            liq_res.columns = liq_res.columns.get_level_values(-1)

        # 3. Generate entries
        entries_df = self.signal_generator.generate(mom_res, liq_res, sector_map=sector_map)

        # 4. Generate exits (no longer in top decile)
        if self.config.rank_mode == "sector" and sector_map:
            mapped_sectors = {s: sector_map.get(s, "unknown") for s in mom_res.columns}
            mom_rank = mom_res.T.groupby(mapped_sectors).rank(pct=True).T
            liq_rank = liq_res.T.groupby(mapped_sectors).rank(pct=True).T
        else:
            mom_rank = mom_res.rank(axis=1, pct=True)
            liq_rank = liq_res.rank(axis=1, pct=True)

        composite = self.config.w_momentum * mom_rank + self.config.w_liquidity * liq_rank
        quantile_series = composite.quantile(self.config.entry_percentile, axis=1)
        unshifted_entries = composite.ge(quantile_series, axis=0)
        exits_df = (
            (~unshifted_entries)
            .infer_objects(copy=False)
            .fillna(False)
            .shift(1)
            .fillna(False)
            .astype(bool)
        )

        # 5. Apply HMM regime filter gate if enabled
        if hmm_filter_enabled:
            if regime_gate is None:
                from ggTrader.strategies.regime.hmm_filter import load_regime_gate

                regime_gate = load_regime_gate(
                    self.config.asset_class,
                    close_df.index.min(),
                    close_df.index.max(),
                    engaged_threshold=self.config.engaged_threshold,
                )

            raw_count = int(entries_df.sum().sum())
            regime_gate_aligned = regime_gate.reindex(entries_df.index, fill_value=False)
            entries_df = entries_df.multiply(regime_gate_aligned, axis=0)
            filtered_count = int(entries_df.sum().sum())
            suppressed = raw_count - filtered_count
            pct = (suppressed / raw_count * 100.0) if raw_count > 0 else 0.0

            logger.info(
                f"Regime gate applied — raw_signals: {raw_count}, "
                f"filtered_signals: {filtered_count}, suppressed_signals: {suppressed} "
                f"({pct:.2f}%)"
            )

        def _generate_hold_exits(entries: pd.DataFrame, hold_period: int) -> pd.DataFrame:
            exits = pd.DataFrame(False, index=entries.index, columns=entries.columns)
            for col in entries.columns:
                entry_indices = np.where(entries[col])[0]
                for idx in entry_indices:
                    exit_idx = idx + hold_period
                    if exit_idx < len(entries):
                        exits.iloc[exit_idx, exits.columns.get_loc(col)] = True
            return exits

        hold_exits = _generate_hold_exits(entries_df, self.config.hold_bars)
        exits_df = exits_df | hold_exits

        # 6. Run VectorBT Portfolio simulation
        self.entries_df = entries_df
        self.portfolio = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=exits_df,
            group_by=True,
            cash_sharing=True,
            size=0.10,
            size_type="percent",
            direction="longonly",
            fees=self.config.fee_rate,
            slippage=self.config.slippage,
        )

        return self.portfolio

    def stats(self) -> pd.Series:
        """Passthrough to portfolio stats."""
        if self.portfolio is None:
            raise ValueError("Portfolio has not been simulated yet. Run run() first.")
        return self.portfolio.stats()

    def plot(self) -> None:
        """Passthrough to portfolio plot."""
        if self.portfolio is None:
            raise ValueError("Portfolio has not been simulated yet. Run run() first.")
        self.portfolio.plot()

    def run_with_filter(
        self,
        close_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        btc_close: Optional[pd.Series] = None,
        regime_gate: Optional[pd.Series] = None,
    ) -> vbt.Portfolio:
        """Run the strategy with the HMM regime filter enabled."""
        return self.run(
            close_df=close_df,
            volume_df=volume_df,
            sector_map=sector_map,
            btc_close=btc_close,
            hmm_filter_enabled=True,
            regime_gate=regime_gate,
        )

    def run_without_filter(
        self,
        close_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        btc_close: Optional[pd.Series] = None,
    ) -> vbt.Portfolio:
        """Run the strategy with the HMM regime filter disabled."""
        return self.run(
            close_df=close_df,
            volume_df=volume_df,
            sector_map=sector_map,
            btc_close=btc_close,
            hmm_filter_enabled=False,
        )
