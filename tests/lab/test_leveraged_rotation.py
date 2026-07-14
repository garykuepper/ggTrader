"""Tests for the leveraged/inverse index rotation strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _idx(n, start="2020-01-31", freq="ME"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


class TestComputeBreadth:
    def test_fraction_of_true_columns(self):
        from ggTrader.lab.strategies.leveraged_rotation import compute_breadth

        entries = pd.DataFrame(
            {
                "A": [True, False, True],
                "B": [True, False, False],
                "C": [False, False, False],
                "D": [True, False, False],
            },
            index=_idx(3),
        )
        breadth = compute_breadth(entries)
        assert list(breadth) == [0.75, 0.0, 0.25]

    def test_empty_columns_returns_empty_series(self):
        from ggTrader.lab.strategies.leveraged_rotation import compute_breadth

        entries = pd.DataFrame(index=_idx(3))
        breadth = compute_breadth(entries)
        assert len(breadth) == 3
        assert breadth.isna().all() or (breadth == 0.0).all()


class TestRotatePositions:
    def test_first_state_takes_effect_immediately(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7], index=_idx(1))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3
        )
        assert states.iloc[0] == "long"

    def test_min_hold_one_flips_immediately(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7, 0.3, 0.7], index=_idx(3))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=1
        )
        assert list(states) == ["long", "inverse", "long"]

    def test_min_hold_three_requires_confirmation(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        # Starts long, dips to inverse-territory for only 2 straight readings,
        # then bounces back to long-territory -- the flip to inverse should
        # never actually take effect (never held 3 consecutive readings).
        breadth = pd.Series([0.7, 0.3, 0.3, 0.7, 0.7], index=_idx(5))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3
        )
        assert list(states) == ["long", "long", "long", "long", "long"]

    def test_min_hold_three_flips_after_three_consecutive(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7, 0.3, 0.3, 0.3, 0.3], index=_idx(5))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3
        )
        assert list(states) == ["long", "long", "long", "inverse", "inverse"]

    def test_between_thresholds_is_cash(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.5], index=_idx(1))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=1
        )
        assert states.iloc[0] == "cash"


def _ohlcv_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    frames = {}
    for col in returns.columns:
        close = 100.0 * (1.0 + returns[col].fillna(0.0)).cumprod()
        frames[col] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=returns.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def _daily_returns(symbols, n=500, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    data = {s: rng.normal(0.0003, 0.01, n) for s in symbols}
    return pd.DataFrame(data, index=idx)


def _concrete_cls():
    """A concrete subclass for testing the base class directly."""
    from ggTrader.lab.strategies.leveraged_rotation import _LeveragedRotationBase

    class _Concrete(_LeveragedRotationBase):
        name = "leveraged_rotation_test"
        PAIR_3X = ("LONG3X", "INV3X")
        PAIR_2X = ("LONG2X", "INV2X")

    return _Concrete


class TestLeveragedRotationBaseSelect:
    def test_select_returns_active_tier_pair_only(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig(), leverage_tier="3x")
        eligible = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        plan = strat.select(pd.Timestamp("2020-06-30", tz="UTC"), pd.DataFrame(), eligible)
        symbols = {s["symbol"] for s in plan}
        assert symbols == {"LONG3X", "INV3X"}

    def test_select_2x_tier(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig(), leverage_tier="2x")
        eligible = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        plan = strat.select(pd.Timestamp("2020-06-30", tz="UTC"), pd.DataFrame(), eligible)
        symbols = {s["symbol"] for s in plan}
        assert symbols == {"LONG2X", "INV2X"}


class TestLeveragedRotationBaseToTargets:
    def test_to_targets_shape_and_columns(self):
        from ggTrader.lab.strategy import LabConfig

        stocks = [f"S{i}" for i in range(20)]
        etfs = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        returns = _daily_returns(stocks + etfs, n=400, seed=1)
        ohlcv = _ohlcv_from_returns(returns)

        strat = _concrete_cls()(
            LabConfig(min_history_bars=60),
            leverage_tier="3x",
            upper_threshold=0.6,
            lower_threshold=0.4,
            min_hold_months=1,
        )
        rebalance_dates = ohlcv.index[[200, 260, 320]]
        eligible = etfs
        plans = {d: strat.select(d, ohlcv.loc[:d], eligible) for d in rebalance_dates}
        targets = strat.to_targets(plans, ohlcv)

        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"LONG3X", "INV3X"}
        assert targets.index.equals(ohlcv.index)

    def test_to_targets_never_selects_inactive_tier(self):
        from ggTrader.lab.strategy import LabConfig

        stocks = [f"S{i}" for i in range(20)]
        etfs = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        returns = _daily_returns(stocks + etfs, n=400, seed=2)
        ohlcv = _ohlcv_from_returns(returns)

        strat = _concrete_cls()(LabConfig(min_history_bars=60), leverage_tier="2x")
        rebalance_dates = ohlcv.index[[200, 260]]
        plans = {d: strat.select(d, ohlcv.loc[:d], etfs) for d in rebalance_dates}
        targets = strat.to_targets(plans, ohlcv)

        assert set(targets.columns) == {"LONG2X", "INV2X"}

    def test_empty_plans_returns_empty_frame(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig())
        targets = strat.to_targets({}, pd.DataFrame())
        assert isinstance(targets, pd.DataFrame)
        assert len(targets) == 0

    def test_sweep_params_grid(self):
        params = _concrete_cls().sweep_params()
        assert "upper_threshold" in params
        assert "lower_threshold" in params
        assert "min_hold_months" in params
        assert "leverage_tier" in params
        assert set(params["leverage_tier"]) == {"2x", "3x"}


class TestPerUniverseSubclasses:
    def test_sp500_pairs(self):
        from ggTrader.lab.strategies.leveraged_rotation import LeveragedRotationSp500

        assert LeveragedRotationSp500.PAIR_3X == ("UPRO", "SPXU")
        assert LeveragedRotationSp500.PAIR_2X == ("SSO", "SDS")
        assert LeveragedRotationSp500.BREADTH_UNIVERSE == "sp500"
        assert LeveragedRotationSp500.name == "leveraged_rotation_sp500"

    def test_nasdaq100_pairs(self):
        from ggTrader.lab.strategies.leveraged_rotation import LeveragedRotationNasdaq100

        assert LeveragedRotationNasdaq100.PAIR_3X == ("TQQQ", "SQQQ")
        assert LeveragedRotationNasdaq100.PAIR_2X == ("QLD", "QID")
        assert LeveragedRotationNasdaq100.BREADTH_UNIVERSE == "nasdaq100"

    def test_russell2000_pairs(self):
        from ggTrader.lab.strategies.leveraged_rotation import LeveragedRotationRussell2000

        assert LeveragedRotationRussell2000.PAIR_3X == ("TNA", "TZA")
        assert LeveragedRotationRussell2000.PAIR_2X == ("UWM", "TWM")
        assert LeveragedRotationRussell2000.BREADTH_UNIVERSE == "russell2000"

    def test_bare_construction_works_without_extra_args(self):
        """wfo.py calls strategy_cls(cfg) with no extra args in some paths
        (anchor-set computation) -- every subclass must support this."""
        from ggTrader.lab.strategies.leveraged_rotation import (
            LeveragedRotationNasdaq100,
            LeveragedRotationRussell2000,
            LeveragedRotationSp500,
        )
        from ggTrader.lab.strategy import LabConfig

        for cls in (
            LeveragedRotationSp500,
            LeveragedRotationNasdaq100,
            LeveragedRotationRussell2000,
        ):
            strat = cls(LabConfig())
            assert strat.leverage_tier == "3x"


def test_all_three_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.leveraged_rotation import (
        LeveragedRotationNasdaq100,
        LeveragedRotationRussell2000,
        LeveragedRotationSp500,
    )

    assert STRATEGY_REGISTRY["leveraged_rotation_sp500"] is LeveragedRotationSp500
    assert STRATEGY_REGISTRY["leveraged_rotation_nasdaq100"] is LeveragedRotationNasdaq100
    assert STRATEGY_REGISTRY["leveraged_rotation_russell2000"] is LeveragedRotationRussell2000
