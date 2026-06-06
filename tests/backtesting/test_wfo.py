"""Unit tests for Strategy 1 Walk-Forward Optimization pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from ggTrader.backtesting.wfo import WalkForwardOptimizer, WFOConfig
from ggTrader.strategies.momentum.config import MomentumConfig
from ggTrader.strategies.momentum.cross_sectional import CrossSectionalMomentum
from ggTrader.utils.result_db_manager import ResultDBManager


def generate_synthetic_wfo_data(
    symbols: list[str], n_bars: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic price and volume data for WFO testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="D", tz="UTC")

    close_dict = {}
    vol_dict = {}

    for sym in symbols:
        prices = 100.0 + np.cumsum(np.random.normal(0.05, 0.8, n_bars))
        prices = np.clip(prices, 5.0, 500.0)
        close_dict[sym] = prices

        vols = np.random.uniform(50000, 500000, n_bars)
        vol_dict[sym] = vols

    close_df = pd.DataFrame(close_dict, index=dates)
    volume_df = pd.DataFrame(vol_dict, index=dates)

    return close_df, volume_df


def test_wfo_pipeline_e2e() -> None:
    """Run walk-forward optimization e2e, asserting correct window output and metrics."""
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    close_df, volume_df = generate_synthetic_wfo_data(symbols, 400)

    # Param grid for optimization
    param_grid = {
        "formation_window": [20, 30],
        "exclusion_gap": [2, 4],
    }

    # Smaller bars configuration so it runs fast and fits in 400 bars
    wfo_config = WFOConfig(
        in_sample_bars=200,
        out_of_sample_bars=50,
        step_bars=50,
        chunk_size=10,
        deployment_sharpe_threshold=0.1,  # Lowered for synthetic test pass gate
        deployment_max_dd_threshold=0.5,
    )

    base_config = MomentumConfig.for_equities()

    optimizer = WalkForwardOptimizer(
        strategy_cls=CrossSectionalMomentum,
        param_grid=param_grid,
        wfo_config=wfo_config,
        base_config=base_config,
    )

    sector_map = {s: "Tech" for s in symbols}

    # Run optimizer
    oos_df = optimizer.run(close_df, volume_df, sector_map=sector_map, hmm_filter_enabled=False)

    # Assert returns dataframe is correct
    assert isinstance(oos_df, pd.DataFrame)
    assert "returns" in oos_df.columns
    # With 400 bars: splits:
    # Split 0: IS [0:200], OOS [200:250]
    # Split 1: IS [50:250], OOS [250:300]
    # Split 2: IS [100:300], OOS [300:350]
    # Split 3: IS [150:350], OOS [350:400]
    # Total OOS bars should be 4 * 50 = 200
    assert len(oos_df) > 0

    # Assert results list contains 4 entries
    assert len(optimizer.results) == 4

    # Assert result structure
    for res in optimizer.results:
        assert "window_id" in res
        assert "params" in res
        assert "in_sample_sharpe" in res
        assert "oos_sharpe" in res
        assert "oos_max_dd" in res
        assert "status" in res
        assert res["status"] in ["robust", "degraded"]

    # Verify summary
    summary = optimizer.summary()
    assert "mean_oos_sharpe" in summary
    assert "std_oos_sharpe" in summary
    assert "max_oos_dd" in summary
    assert "n_degraded_windows" in summary
    assert "deploy_ready" in summary
    assert isinstance(summary["deploy_ready"], bool)

    # Verify plotting returns a fig and works
    fig = optimizer.plot_robustness()
    assert fig is not None

    # Clean up DB entries created by this test run
    db = ResultDBManager()
    with db.engine.connect() as conn:
        with conn.begin():
            conn.execute(
                text("DELETE FROM wfo_splits WHERE run_id = :rid"),
                {"rid": optimizer.run_id},
            )
