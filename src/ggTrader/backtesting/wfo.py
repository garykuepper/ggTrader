"""Walk-Forward Optimization (WFO) pipeline for strategy parameter tuning."""

from __future__ import annotations

import gc
import itertools
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import text

from ggTrader.core.metrics import safe_portfolio_stats
from ggTrader.utils.result_db_manager import ResultDBManager

logger = logging.getLogger(__name__)


class WFOConfig(BaseModel):
    """Configuration settings for Walk-Forward Optimization."""

    in_sample_bars: int = 252
    out_of_sample_bars: int = 63
    step_bars: int = 63
    chunk_size: int = 200
    deployment_sharpe_threshold: float = 1.0
    deployment_max_dd_threshold: float = 0.15


class WalkForwardOptimizer:
    """Institutional-grade Walk-Forward Optimization harness."""

    def __init__(
        self,
        strategy_cls: Type[Any],
        param_grid: Dict[str, List[Any]],
        wfo_config: WFOConfig,
        base_config: Optional[Any] = None,
    ) -> None:
        """Initialize the Walk-Forward Optimizer.

        Args:
            strategy_cls: The strategy class to optimize (e.g. CrossSectionalMomentum).
            param_grid: Dictionary mapping parameter names to lists of values to test.
            wfo_config: WFO pipeline configuration settings.
            base_config: Optional base configuration object for the strategy.
        """
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.wfo_config = wfo_config
        self.base_config = base_config
        self.results: List[Dict[str, Any]] = []
        self.continuous_returns: Optional[pd.Series] = None
        self.run_id: str = f"wfo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._init_db()

    def _init_db(self) -> None:
        """Create the WFO splits table if it does not exist."""
        db = ResultDBManager()
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS wfo_splits (
                run_id VARCHAR,
                window_id INTEGER,
                params JSONB,
                in_sample_sharpe DOUBLE PRECISION,
                oos_sharpe DOUBLE PRECISION,
                oos_max_dd DOUBLE PRECISION,
                status VARCHAR(16),
                PRIMARY KEY (run_id, window_id)
            );
        """
        try:
            with db.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text(create_table_sql))
        except Exception as e:
            logger.warning(f"Could not initialize wfo_splits database table: {e}")

    def run(self, close_df: pd.DataFrame, volume_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Execute the WFO sequential in-sample/out-of-sample search.

        Args:
            close_df: Price close DataFrame.
            volume_df: Volume DataFrame.
            **kwargs: Extra parameters passed to the strategy's run method
                (e.g. sector_map, btc_close).

        Returns:
            pd.DataFrame: Concatenated out-of-sample returns Series as a DataFrame.
        """
        self.results = []
        total_bars = len(close_df)

        # Generate combinations grid
        keys, values = zip(*self.param_grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        logger.info(f"WFO Run {self.run_id} started with {len(combos)} total combinations.")

        # Split history into sequential rolls
        splits: List[tuple[int, int, int, int]] = []
        i = 0
        while True:
            is_start = i * self.wfo_config.step_bars
            is_end = is_start + self.wfo_config.in_sample_bars
            oos_start = is_end
            oos_end = oos_start + self.wfo_config.out_of_sample_bars
            if oos_end > total_bars:
                break
            splits.append((is_start, is_end, oos_start, oos_end))
            i += 1

        if not splits:
            raise ValueError(
                f"Dataset size ({total_bars} bars) is too small for "
                f"in-sample ({self.wfo_config.in_sample_bars}) and "
                f"out-of-sample ({self.wfo_config.out_of_sample_bars}) settings."
            )

        logger.info(f"Generated {len(splits)} rolling window splits.")
        db = ResultDBManager()
        oos_returns_list: List[pd.Series] = []

        for window_id, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
            logger.info(
                f"Processing Window {window_id}: "
                f"IS [{is_start}:{is_end}], OOS [{oos_start}:{oos_end}]"
            )

            # Slice data for In-Sample fitting
            close_is = close_df.iloc[is_start:is_end]
            volume_is = volume_df.iloc[is_start:is_end]
            kwargs_is = {}
            for k, v in kwargs.items():
                if isinstance(v, (pd.DataFrame, pd.Series)):
                    kwargs_is[k] = v.iloc[is_start:is_end]
                else:
                    kwargs_is[k] = v

            best_sharpe = -np.inf
            best_combo = combos[0]

            # Chunk parameter grid to manage memory
            chunk_size = self.wfo_config.chunk_size
            chunks = [combos[k : k + chunk_size] for k in range(0, len(combos), chunk_size)]

            for chunk_idx, chunk in enumerate(chunks):
                for combo in chunk:
                    # Construct strategy config
                    if self.base_config is not None:
                        config_dict = self.base_config.model_dump()
                        config_dict.update(combo)
                        config = type(self.base_config)(**config_dict)
                    else:
                        config = combo

                    try:
                        strategy = self.strategy_cls(config)
                        portfolio = strategy.run(close_is, volume_is, **kwargs_is)
                        stats = safe_portfolio_stats(portfolio)
                        sharpe = stats.get("Sharpe Ratio", -np.inf)
                        if np.isnan(sharpe):
                            sharpe = -np.inf

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_combo = combo
                    except Exception as e:
                        logger.warning(
                            f"Strategy run failed on IS combo {combo} in window {window_id}: {e}"
                        )
                        continue

                gc.collect()

            logger.info(
                f"Window {window_id} IS winner: {best_combo} with Sharpe: {best_sharpe:.4f}"
            )

            # Evaluate winner on Out-of-Sample slice
            close_oos = close_df.iloc[oos_start:oos_end]
            volume_oos = volume_df.iloc[oos_start:oos_end]
            kwargs_oos = {}
            for k, v in kwargs.items():
                if isinstance(v, (pd.DataFrame, pd.Series)):
                    kwargs_oos[k] = v.iloc[oos_start:oos_end]
                else:
                    kwargs_oos[k] = v

            if self.base_config is not None:
                config_dict = self.base_config.model_dump()
                config_dict.update(best_combo)
                config_oos = type(self.base_config)(**config_dict)
            else:
                config_oos = best_combo

            try:
                strategy_oos = self.strategy_cls(config_oos)
                portfolio_oos = strategy_oos.run(close_oos, volume_oos, **kwargs_oos)
                oos_stats = safe_portfolio_stats(portfolio_oos)
                oos_sharpe = oos_stats.get("Sharpe Ratio", 0.0)
                if np.isnan(oos_sharpe):
                    oos_sharpe = 0.0
                oos_max_dd = oos_stats.get("Max Drawdown [%]", 0.0) / 100.0

                # Append returns for continuous out-of-sample series
                oos_returns_list.append(portfolio_oos.returns())

                # Robustness check
                is_degraded = (
                    oos_sharpe < (0.5 * best_sharpe) if best_sharpe > 0 else oos_sharpe < 0
                )
                status = "degraded" if is_degraded else "robust"

                # Persist to database
                self._save_split_record(
                    db,
                    window_id,
                    best_combo,
                    best_sharpe,
                    oos_sharpe,
                    oos_max_dd,
                    status,
                )

                self.results.append(
                    {
                        "window_id": window_id,
                        "params": best_combo,
                        "in_sample_sharpe": best_sharpe if best_sharpe != -np.inf else 0.0,
                        "oos_sharpe": oos_sharpe,
                        "oos_max_dd": oos_max_dd,
                        "status": status,
                    }
                )
            except Exception as e:
                logger.error(f"Out-of-sample evaluation failed for window {window_id}: {e}")
                continue

        if oos_returns_list:
            self.continuous_returns = pd.concat(oos_returns_list)
            return pd.DataFrame({"returns": self.continuous_returns})
        else:
            return pd.DataFrame(columns=["returns"])

    def _save_split_record(
        self,
        db: ResultDBManager,
        window_id: int,
        params: dict,
        is_sharpe: float,
        oos_sharpe: float,
        oos_max_dd: float,
        status: str,
    ) -> None:
        """Persist a single window record into TimescaleDB."""
        query = text(
            """
            INSERT INTO wfo_splits (
                run_id, window_id, params, in_sample_sharpe, oos_sharpe, oos_max_dd, status
            )
            VALUES (
                :run_id, :window_id, CAST(:params AS JSONB),
                :is_sharpe, :oos_sharpe, :oos_max_dd, :status
            )
            ON CONFLICT (run_id, window_id) DO UPDATE SET
                params = EXCLUDED.params,
                in_sample_sharpe = EXCLUDED.in_sample_sharpe,
                oos_sharpe = EXCLUDED.oos_sharpe,
                oos_max_dd = EXCLUDED.oos_max_dd,
                status = EXCLUDED.status
            """
        )
        try:
            with db.engine.begin() as conn:
                conn.execute(
                    query,
                    {
                        "run_id": self.run_id,
                        "window_id": window_id,
                        "params": db._safe_json_dumps(params),
                        "is_sharpe": float(is_sharpe) if is_sharpe != -np.inf else None,
                        "oos_sharpe": float(oos_sharpe),
                        "oos_max_dd": float(oos_max_dd),
                        "status": status,
                    },
                )
        except Exception as e:
            logger.warning(f"Could not persist split record to DB: {e}")

    def summary(self) -> dict[str, Any]:
        """Aggregate stats over all out-of-sample windows.

        Returns:
            dict containing mean_oos_sharpe, std_oos_sharpe, max_oos_dd,
            n_degraded_windows, and deploy_ready (bool).
        """
        if not self.results:
            return {
                "mean_oos_sharpe": 0.0,
                "std_oos_sharpe": 0.0,
                "max_oos_dd": 0.0,
                "n_degraded_windows": 0,
                "deploy_ready": False,
            }

        oos_sharpes = [r["oos_sharpe"] for r in self.results]
        oos_max_dds = [r["oos_max_dd"] for r in self.results]
        n_degraded = sum(1 for r in self.results if r["status"] == "degraded")

        mean_sharpe = float(np.mean(oos_sharpes))
        std_sharpe = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
        max_dd = float(np.max(oos_max_dds))

        deploy_ready = (
            mean_sharpe > self.wfo_config.deployment_sharpe_threshold
            and max_dd < self.wfo_config.deployment_max_dd_threshold
        )

        return {
            "mean_oos_sharpe": mean_sharpe,
            "std_oos_sharpe": std_sharpe,
            "max_oos_dd": max_dd,
            "n_degraded_windows": n_degraded,
            "deploy_ready": deploy_ready,
        }

    def plot_robustness(self, save_path: Optional[str] = None) -> Any:
        """Plot OOS Sharpe vs. In-Sample Sharpe across all windows.

        Args:
            save_path: Optional file path to save the generated plot.
        """
        import matplotlib.pyplot as plt

        if not self.results:
            logger.warning("No results to plot.")
            return None

        is_sharpes = [r["in_sample_sharpe"] for r in self.results]
        oos_sharpes = [r["oos_sharpe"] for r in self.results]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(is_sharpes, oos_sharpes, alpha=0.7, color="blue", edgecolors="k")
        ax.set_xlabel("In-Sample Sharpe")
        ax.set_ylabel("Out-of-Sample Sharpe")
        ax.set_title("WFO Robustness Check: IS vs OOS Sharpe")
        ax.grid(True, linestyle="--", alpha=0.5)

        if len(is_sharpes) > 1:
            corr = pd.Series(oos_sharpes).corr(pd.Series(is_sharpes))
            if np.isnan(corr):
                corr = 0.0
            ax.text(
                0.05,
                0.95,
                f"Correlation: {corr:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved robustness plot to {save_path}")

        return fig
