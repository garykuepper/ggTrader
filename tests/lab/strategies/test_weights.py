"""Weight strategies (xs_momentum, cross_asset_sleeve, retail_attention) vs frozen legacy output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.data.bars import Bars
from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.lab.data import rebalance_dates
from ggTrader.lab.strategies.contract import ParamGrid, tile_columns
from ggTrader.lab.strategies.cross_asset_sleeve import UNIVERSE, CrossAssetSleeve
from ggTrader.lab.strategies.retail_attention import RetailAttention
from ggTrader.lab.strategies.xs_momentum import XsMomentum, momentum_scores
from tests.lab.strategies.helpers import (
    frozen_array,
    frozen_frame,
    legacy_grids,
    make_bars,
)

CFG = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0005, "FREQ": "1D"}


def _rebal(bars: Bars, start: str = "2020-06-01") -> list[pd.Timestamp]:
    index = bars.close.index
    return rebalance_dates(index, pd.Timestamp(start, tz="UTC"), index[-1])


def _block(frame: pd.DataFrame, i: int, symbols: list[str]) -> pd.DataFrame:
    """The ``i``-th combination's block of a stacked weight frame, restricted to ``symbols``."""
    syms = list(dict.fromkeys(frame.columns.get_level_values(-1)))
    block = frame.iloc[:, i * len(syms) : (i + 1) * len(syms)].set_axis(syms, axis=1)
    return block[symbols]


# ----------------------------------------------------------------------------- xs_momentum


@pytest.fixture(scope="module")
def mom_bars() -> Bars:
    return make_bars(seed=41, n_bars=900, n_syms=14, holes=True)


@pytest.fixture(scope="module")
def eligible(mom_bars) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    mask = rng.random(mom_bars.close.shape) < 0.8
    return pd.DataFrame(mask, index=mom_bars.close.index, columns=mom_bars.close.columns)


def test_xs_momentum_grid_equals_the_frozen_legacy_grid() -> None:
    assert XsMomentum.grid.combos() == legacy_grids()["xs_momentum"]["grid"]
    assert len(XsMomentum.grid) == 12


def test_momentum_scores_use_each_symbols_valid_closes() -> None:
    close = pd.DataFrame(
        {"A": [1.0, np.nan, 2.0, 3.0, 4.0, 6.0], "B": np.arange(1.0, 7.0)},
        index=pd.bdate_range("2024-01-01", periods=6, tz="UTC"),
    )
    score = momentum_scores(close, lookback=2, skip=0)
    # A valid closes [1,2,3,4,6]: at the last bar 6/3-1 = 1.0 (hole skipped, not counted)
    assert score["A"].iloc[-1] == pytest.approx(1.0)
    assert np.isnan(score["A"].iloc[1]) and score["B"].iloc[-1] == pytest.approx(6 / 4 - 1)


def test_xs_momentum_targets_equal_frozen_legacy_for_every_combination(mom_bars, eligible) -> None:
    rebal = _rebal(mom_bars)
    w = XsMomentum().build(mom_bars, rebal=pd.DatetimeIndex(rebal), eligible=eligible)
    for i, _combo in enumerate(XsMomentum.grid.combos()):
        legacy = frozen_frame(f"xs_momentum|{i}")
        new = _block(w, i, list(legacy.columns))
        pd.testing.assert_frame_equal(new, legacy, check_freq=False, check_names=False)
        # symbols this combination never selected exist only as zero or NaN rows
        syms = list(dict.fromkeys(w.columns.get_level_values(-1)))
        extra = [s for s in syms if s not in legacy.columns]
        assert np.nan_to_num(_block(w, i, extra).to_numpy()).sum() == 0.0


def test_xs_momentum_equity_equals_frozen_legacy_simulator(mom_bars, eligible) -> None:
    rebal = _rebal(mom_bars)
    w = XsMomentum().build(mom_bars, rebal=pd.DatetimeIndex(rebal), eligible=eligible)
    new = engine_portfolio.run_weights(
        w,
        tile_columns(mom_bars.close, w.columns),
        CFG,
        groups=[k[:-1] for k in w.columns.to_list()],
    )
    assert np.array_equal(new.value.to_numpy(), frozen_array("xs_momentum|equity"))


def test_xs_momentum_requires_rebalance_dates_and_respects_eligibility(mom_bars) -> None:
    with pytest.raises(ValueError, match="rebalance"):
        XsMomentum().build(mom_bars)
    none_ok = pd.DataFrame(False, index=mom_bars.close.index, columns=mom_bars.close.columns)
    w = XsMomentum().build(mom_bars, rebal=pd.DatetimeIndex(_rebal(mom_bars)[:3]), eligible=none_ok)
    assert w.shape[1] == 0 or np.nan_to_num(w.to_numpy()).sum() == 0.0


# ----------------------------------------------------------------------------- cross_asset_sleeve


@pytest.fixture(scope="module")
def sleeve_bars() -> Bars:
    base = make_bars(seed=7, n_bars=700, n_syms=5, symbols=sorted(UNIVERSE))
    return base


def test_sleeve_grid_equals_the_frozen_legacy_grid() -> None:
    assert CrossAssetSleeve.grid.combos() == legacy_grids()["cross_asset_sleeve"]["grid"]
    assert len(CrossAssetSleeve.grid) == 3


@pytest.mark.parametrize("trend", [True, False])
def test_sleeve_targets_equal_frozen_legacy(sleeve_bars, trend) -> None:
    rebal = _rebal(sleeve_bars, "2020-05-01")
    grid = CrossAssetSleeve.grid if trend else ParamGrid.single({"trend": False})
    w = CrossAssetSleeve().build(sleeve_bars, grid, pd.DatetimeIndex(rebal))
    for i, _combo in enumerate(grid.combos()):
        legacy = frozen_frame(f"sleeve|{trend}|{i}")
        new = _block(w, i, list(legacy.columns))
        pd.testing.assert_frame_equal(new, legacy, check_freq=False, check_names=False)


def test_sleeve_trend_arm_moves_a_falling_leg_to_cash(sleeve_bars) -> None:
    close = sleeve_bars.close.copy()
    close["TLT"] = np.linspace(100, 60, len(close))  # falling: off
    close["GLD"] = np.linspace(100, 130, len(close))  # rising: on
    close["PDBC"] = np.linspace(100, 110, len(close))  # rising: on
    from dataclasses import replace

    bars = replace(sleeve_bars, close=close)
    rebal = pd.DatetimeIndex([close.index[400]])
    w = CrossAssetSleeve().build(bars, ParamGrid.single({"lookback": 252}), rebal)
    row = w.iloc[401]
    row.index = row.index.get_level_values(-1)
    assert row["TLT"] == 0.0 and row["BIL"] == pytest.approx(0.2 / 3) and row["SPY"] == 0.8
    assert row.sum() == pytest.approx(1.0)


def test_sleeve_leg_with_a_flat_trailing_return_is_off(sleeve_bars) -> None:
    """Strictly positive is required: a zero trailing return counts as off."""
    from dataclasses import replace

    close = sleeve_bars.close.copy()
    close["GLD"] = 100.0
    bars = replace(sleeve_bars, close=close)
    w = CrossAssetSleeve().build(
        bars, ParamGrid.single({"lookback": 126}), pd.DatetimeIndex([close.index[400]])
    )
    row = w.iloc[401]
    row.index = row.index.get_level_values(-1)
    assert row["GLD"] == 0.0 and row["BIL"] >= 0.2 / 3 - 1e-12


def test_sleeve_needs_all_etfs(mom_bars) -> None:
    with pytest.raises(ValueError, match="needs bars"):
        CrossAssetSleeve().build(mom_bars, rebal=pd.DatetimeIndex(_rebal(mom_bars)[:2]))


# ----------------------------------------------------------------------------- retail_attention


def _synthetic_interest():
    """Deterministic monthly search interest that differs by symbol and month."""
    months = pd.date_range("2018-01-01", "2024-12-31", freq="MS")

    def loader(syms, start, end):
        rows = [
            (s, m, 20.0 + (int(s[1:]) * 7 + m.month * 3 + m.year) % 29)
            for s in syms
            for m in months
        ]
        df = pd.DataFrame(rows, columns=["symbol", "date", "search_interest"])
        return df[(df["date"] >= start) & (df["date"] <= end)]

    return loader


def test_retail_attention_grid_equals_the_frozen_legacy_grid() -> None:
    assert RetailAttention.grid.combos() == legacy_grids()["retail_attention"]["grid"]
    assert len(RetailAttention.grid) == 6


def test_retail_attention_targets_equal_frozen_legacy_on_synthetic_interest(mom_bars) -> None:
    rebal = _rebal(mom_bars, "2021-06-01")
    loader = _synthetic_interest()
    w = RetailAttention(interest_loader=loader, min_history_bars=1).build(
        mom_bars, rebal=pd.DatetimeIndex(rebal)
    )
    for i, _combo in enumerate(RetailAttention.grid.combos()):
        legacy = frozen_frame(f"retail|{i}")
        assert legacy.shape[1] > 0, "synthetic interest must select something"
        new = _block(w, i, list(legacy.columns))
        pd.testing.assert_frame_equal(new, legacy, check_freq=False, check_names=False)


def test_retail_attention_is_flat_cash_without_interest_data(mom_bars) -> None:
    empty = lambda syms, start, end: pd.DataFrame(columns=["symbol", "date", "search_interest"])  # noqa: E731
    w = RetailAttention(interest_loader=empty, min_history_bars=1).build(
        mom_bars, rebal=pd.DatetimeIndex(_rebal(mom_bars, "2021-06-01"))
    )
    assert w.shape[1] == 0
