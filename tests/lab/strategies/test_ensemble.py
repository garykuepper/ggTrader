"""Ensemble on the engine contract against the legacy ``EnsembleSignal``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.contract import ParamGrid, simulate_signal_out, tile_columns
from ggTrader.lab.strategies.ensemble import ENSEMBLE_DEFAULTS, Ensemble
from ggTrader.lab.strategies.voters import voter_arrays
from ggTrader.lab.wfo.inputs import combo_name
from tests.lab.strategies.helpers import (
    diff_after_burn_in,
    frozen_cells,
    legacy_grids,
    make_bars,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def bars():
    return make_bars(seed=21, n_bars=900, n_syms=10, holes=True)


@pytest.fixture(scope="module")
def built(bars):
    return Ensemble().build(bars)


def test_grid_is_the_48_combination_wfo_grid_with_legacy_values() -> None:
    legacy = legacy_grids()["ensemble"]["grid"]
    assert len(Ensemble.grid) == len(legacy) == 48
    assert Ensemble.grid.combos() == legacy
    assert Ensemble.grid.swept() == [
        "bb_std",
        "ema_fast",
        "min_agree",
        "min_agree_exit",
        "rsi_oversold",
    ]


def test_live_defaults_are_the_legacy_constructor_defaults_and_outside_the_grid() -> None:
    legacy = legacy_grids()["ensemble"]["constructor_attributes"]
    for key in ("bb_period", "bb_std", "rsi_period", "rsi_oversold", "rsi_exit", "ema_fast"):
        assert Ensemble.live[key] == legacy[key]
    assert Ensemble.live["divergence_window"] == legacy["divergence_window"] == 20
    assert Ensemble.live["vol_mult"] == legacy["vol_mult"] == 2.0
    assert Ensemble.live["min_agree_exit"] == legacy["min_agree_exit"] == 2
    assert 20 not in Ensemble.grid.values["divergence_window"]  # OD14: control row, not a grid cell
    assert 2.0 not in Ensemble.grid.values["vol_mult"]
    assert Ensemble.live_grid().combos() == [Ensemble.live]


def test_every_combination_equals_the_frozen_legacy_sweep_after_burn_in(bars, built) -> None:
    n = len(bars.symbols)
    grid_keys = list(Ensemble.grid.values)
    assert len(built.combos) == 48
    for i, combo in enumerate(built.combos):
        key = f"ensemble|{combo_name('ensemble', {k: combo[k] for k in grid_keys})}"
        sl = slice(i * n, (i + 1) * n)
        entries = diff_after_burn_in(
            built.entries.to_numpy()[:, sl], frozen_cells(key, "entries"), bars.close
        )
        exits = diff_after_burn_in(
            built.exits.to_numpy()[:, sl], frozen_cells(key, "exits"), bars.close
        )
        assert entries == 0 and exits == 0


def test_live_control_config_equals_the_frozen_legacy_defaults_after_burn_in(bars) -> None:
    out = Ensemble().build(bars, Ensemble.live_grid())
    assert out.entries.shape == bars.close.shape
    old_entries = frozen_cells("ensemble_live_control", "entries")
    old_exits = frozen_cells("ensemble_live_control", "exits")
    assert diff_after_burn_in(out.entries.to_numpy(), old_entries, bars.close) == 0
    assert diff_after_burn_in(out.exits.to_numpy(), old_exits, bars.close) == 0
    assert list(out.entries.columns.names) == ["combo", "symbol"]


def test_columns_carry_the_swept_levels_and_symbols(built, bars) -> None:
    assert list(built.entries.columns.names) == [*Ensemble.grid.swept(), "symbol"]
    assert built.entries.shape == (len(bars.close), 48 * len(bars.symbols))
    assert built.entries.dtypes.eq(bool).all()
    assert built.size is None and built.td_stop is None


def test_one_spike_can_meet_min_agree_two_through_the_double_vote(bars) -> None:
    """OD3a: bb and vbb both fire on a spike entry, so min_agree=2 is met by bb's own event."""
    combo = {**ENSEMBLE_DEFAULTS, "vol_mult": 1.0, "min_agree": 2, "min_agree_exit": 2}
    only_bb_family = Ensemble(voters=("bb", "vbb"))
    out = only_bb_family.build(bars, ParamGrid.single(combo))
    bb = voter_arrays("bb", bars, [combo]).pick(combo)[0]
    vbb = voter_arrays("vbb", bars, [combo]).pick(combo)[0]
    assert vbb.sum() > 0
    # ensemble entry == vbb entry (bb alone is one vote, bb+spike is two)
    assert np.array_equal(out.entries.to_numpy(), vbb)
    assert out.entries.to_numpy().sum() < bb.sum()


def test_rsi_exit_fires_without_extra_votes(bars) -> None:
    combo = {**ENSEMBLE_DEFAULTS, "min_agree": 4, "min_agree_exit": 4}
    out = Ensemble().build(bars, ParamGrid.single(combo))
    rsi_exit = voter_arrays("rsi", bars, [combo]).pick(combo)[1]
    assert rsi_exit.any()
    assert out.exits.to_numpy()[rsi_exit].all()


def test_min_agree_exit_none_defaults_to_min_agree(bars) -> None:
    a = Ensemble().build(
        bars, ParamGrid.single({**ENSEMBLE_DEFAULTS, "min_agree": 3, "min_agree_exit": None})
    )
    b = Ensemble().build(
        bars, ParamGrid.single({**ENSEMBLE_DEFAULTS, "min_agree": 3, "min_agree_exit": 3})
    )
    assert a.exits.equals(b.exits)


def test_mtf_and_unknown_voters_are_rejected() -> None:
    with pytest.raises(ValueError, match="legacy-only"):
        Ensemble(voters=("bb", "mtf"))
    with pytest.raises(ValueError):
        Ensemble(voters=())


def test_voter_subset_changes_the_vote(bars) -> None:
    three = Ensemble(voters=("bb", "rsi", "ema")).build(bars, Ensemble.live_grid())
    five = Ensemble().build(bars, Ensemble.live_grid())
    assert not three.entries.equals(five.entries)


def test_td_stop_is_carried_for_the_engine_and_not_applied_in_build(bars) -> None:
    plain = Ensemble().build(bars, Ensemble.live_grid())
    timed = Ensemble(td_stop=5).build(bars, Ensemble.live_grid())
    assert timed.td_stop == 5
    assert timed.exits.equals(plain.exits)  # engine adds the time stop after the PIT mask


def test_pit_mask_is_applied_by_the_engine_before_the_time_stop(bars) -> None:
    out = Ensemble(td_stop=3).build(bars, Ensemble.live_grid())
    mask = pd.DataFrame(True, index=bars.close.index, columns=bars.close.columns)
    mask.iloc[:, 0] = False  # first symbol never eligible: no trades, so no time-stop trades
    cfg = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1D",
        "SIGNAL_POSITION_SIZE": 0.03,
    }
    res = simulate_signal_out(out, bars.close, cfg, mask=mask)
    tiled = tile_columns(mask, out.entries.columns)
    assert (out.entries & ~tiled).to_numpy().any()  # raw signals do fire on the masked symbol
    trades = res.trades
    assert (trades["symbol"] != bars.symbols[0]).all()
    assert np.isfinite(res.value.to_numpy()).all()
