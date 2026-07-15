"""Tests for the leveraged-rotation research orchestration script."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestUniversesMapping:
    def test_maps_all_three_universes(self):
        from ggTrader.lab.strategies.leveraged_rotation import (
            LeveragedRotationNasdaq100,
            LeveragedRotationRussell2000,
            LeveragedRotationSp500,
        )
        from scripts.leveraged_rotation_research import UNIVERSES

        assert UNIVERSES["sp500"] is LeveragedRotationSp500
        assert UNIVERSES["nasdaq100"] is LeveragedRotationNasdaq100
        assert UNIVERSES["russell2000"] is LeveragedRotationRussell2000


class TestRunUniverse:
    @patch("scripts.leveraged_rotation_research.run_wfo")
    @patch("scripts.leveraged_rotation_research.load_ohlcv")
    @patch("scripts.leveraged_rotation_research.equity_universe_between")
    def test_calls_run_wfo_with_fixed_universe_fn(self, mock_members, mock_load, mock_run_wfo):
        import pandas as pd

        from ggTrader.lab.strategy import LabConfig
        from ggTrader.lab.wfo import WfoResult
        from scripts.leveraged_rotation_research import run_universe

        mock_members.return_value = ["AAPL", "MSFT"]
        ohlcv = MagicMock()
        ohlcv.__getitem__.return_value = MagicMock()
        mock_load.return_value = ohlcv
        mock_run_wfo.return_value = WfoResult(
            oos_equity=pd.Series(dtype=float),
            fold_results=[],
            live_params={},
            table="fake table",
        )

        result = run_universe("sp500", "2010-06-30", "2020-01-01", LabConfig())

        assert "sp500" in result
        mock_run_wfo.assert_called_once()
        call_kwargs = mock_run_wfo.call_args
        universe_fn = call_kwargs.kwargs["universe_fn"]
        # Fixed regardless of asof/past -- always all 4 ETF tickers for sp500.
        eligible = universe_fn(pd.Timestamp("2015-01-01", tz="UTC"), None)
        assert set(eligible) == {"UPRO", "SPXU", "SSO", "SDS"}

    @patch("scripts.leveraged_rotation_research.run_wfo")
    @patch("scripts.leveraged_rotation_research.load_ohlcv")
    @patch("scripts.leveraged_rotation_research.equity_universe_between")
    def test_no_valid_folds_reports_gracefully(self, mock_members, mock_load, mock_run_wfo):
        from ggTrader.lab.strategy import LabConfig
        from scripts.leveraged_rotation_research import run_universe

        mock_members.return_value = ["AAPL"]
        mock_load.return_value = MagicMock()
        mock_run_wfo.return_value = "WFO: leveraged_rotation_sp500 | no valid folds"

        result = run_universe("sp500", "2010-06-30", "2010-07-01", LabConfig())
        assert "no valid folds" in result
