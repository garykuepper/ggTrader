"""ParamGrid, group columns, tiling and weight stacking of the engine contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.contract import (
    COMBO_LEVEL,
    Lt,
    ParamGrid,
    SignalOut,
    group_columns,
    next_bar_positions,
    resolve_combos,
    simulate_signal_out,
    stack_weight_plans,
    tile_columns,
)


def test_combos_are_the_sorted_key_product_in_legacy_order() -> None:
    grid = ParamGrid.from_sweep({"b": [1, 2], "a": [10, 20]})
    assert grid.combos() == [
        {"a": 10, "b": 1},
        {"a": 10, "b": 2},
        {"a": 20, "b": 1},
        {"a": 20, "b": 2},
    ]
    assert len(grid) == 4


def test_fast_below_slow_rule_is_data() -> None:
    grid = ParamGrid.from_sweep(
        {"ema_fast": [5, 10, 20], "ema_slow": [10, 20]}, rules=(Lt("ema_fast", "ema_slow"),)
    )
    assert all(c["ema_fast"] < c["ema_slow"] for c in grid.combos())
    assert len(grid) == 3  # (5,10) (5,20) (10,20)


def test_rule_with_a_missing_key_is_skipped() -> None:
    assert Lt("x", "y").ok({"x": 5})


def test_swept_and_used_values_follow_the_valid_combos() -> None:
    grid = ParamGrid.from_sweep(
        {"fast": [5, 50], "slow": [20], "k": [1]}, rules=(Lt("fast", "slow"),)
    )
    assert grid.swept() == []  # fast=50 never occurs in a valid combination
    assert grid.used_values("fast") == [5]


def test_single_and_override() -> None:
    one = ParamGrid.single({"a": 1, "b": 2.0})
    assert one.combos() == [{"a": 1, "b": 2.0}]
    wide = one.override({"a": [1, 2], "c": [9]})
    assert len(wide) == 2 and wide.swept() == ["a"]
    assert one.values["a"] == (1,)  # the original is untouched


def test_resolve_combos_fills_defaults_without_overriding() -> None:
    grid = ParamGrid.from_sweep({"a": [1, 2]})
    assert resolve_combos(grid, {"a": 99, "z": 7}) == [{"z": 7, "a": 1}, {"z": 7, "a": 2}]


def test_group_columns_blocks_and_names() -> None:
    combos = [{"a": 1, "b": 5}, {"a": 2, "b": 5}]
    cols = group_columns(combos, ["a"], ["X", "Y"])
    assert list(cols.names) == ["a", "symbol"]
    assert cols.to_list() == [(1, "X"), (1, "Y"), (2, "X"), (2, "Y")]


def test_group_columns_without_a_swept_level_uses_the_combo_level() -> None:
    cols = group_columns([{"a": 1}], [], ["X", "Y"])
    assert list(cols.names) == [COMBO_LEVEL, "symbol"]
    assert cols.to_list() == [(0, "X"), (0, "Y")]


def test_tile_columns_repeats_symbols_and_rejects_unknown_ones() -> None:
    frame = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
    cols = group_columns([{"a": 1}, {"a": 2}], ["a"], ["Y", "X"])
    tiled = tile_columns(frame, cols)
    assert tiled.columns.equals(cols)
    assert tiled.iloc[:, 0].tolist() == [3.0, 4.0] and tiled.iloc[:, 3].tolist() == [1.0, 2.0]
    with pytest.raises(KeyError):
        tile_columns(frame, group_columns([{"a": 1}], ["a"], ["Z"]))


def test_next_bar_positions_is_strictly_after_and_drops_the_last_bar() -> None:
    index = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    pos = next_bar_positions(index, [index[0], index[2], index[4]])
    assert pos == [1, 3, -1]


def test_stack_weight_plans_zeroes_unselected_and_holds_in_between() -> None:
    index = pd.bdate_range("2024-01-01", periods=6, tz="UTC")
    rebal = [index[0], index[2]]
    plans = [
        {index[0]: {"A": 0.5, "B": 0.5}, index[2]: {"A": 1.0}},
        {index[0]: {"B": 1.0}, index[2]: {}},
    ]
    combos = [{"k": 1}, {"k": 2}]
    w = stack_weight_plans(index, rebal, plans, combos, ["k"])
    assert w.columns.to_list() == [(1, "A"), (1, "B"), (2, "A"), (2, "B")]
    assert w.iloc[1].tolist() == [0.5, 0.5, 0.0, 1.0]
    assert w.iloc[3].tolist() == [1.0, 0.0, 0.0, 0.0]  # B exits, empty plan exits all
    assert w.iloc[[0, 2, 4, 5]].isna().all().all()


def test_signal_out_group_slices_and_labels() -> None:
    index = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    cols = group_columns([{"a": 1}, {"a": 2}], ["a"], ["X", "Y"])
    e = pd.DataFrame(np.eye(3, 4, dtype=bool), index=index, columns=cols)
    out = SignalOut(entries=e, exits=~e, combos=({"a": 1}, {"a": 2}))
    assert out.symbols == ["X", "Y"]
    assert out.group_labels() == [(1,), (1,), (2,), (2,)]
    ent, ext = out.group(1)
    assert ent.columns.to_list() == ["X", "Y"] and ent.shape == (3, 2)
    assert (ent.to_numpy() == e.to_numpy()[:, 2:]).all() and (
        ext.to_numpy() == ~ent.to_numpy()
    ).all()


def _toy_out(size=None, **kwargs) -> tuple[SignalOut, pd.DataFrame]:
    index = pd.bdate_range("2024-01-01", periods=60, tz="UTC")
    rng = np.random.default_rng(3)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (60, 2)), axis=0)), index, ["X", "Y"]
    )
    cols = group_columns([{"a": 1}], ["a"], ["X", "Y"])
    entries = pd.DataFrame(False, index=index, columns=cols)
    entries.iloc[5, :] = True
    exits = pd.DataFrame(False, index=index, columns=cols)
    exits.iloc[40, :] = True
    return SignalOut(entries=entries, exits=exits, combos=({"a": 1},), size=size, **kwargs), close


CFG = {
    "START_CASH": 10000.0,
    "FEES": 0.0,
    "SLIPPAGE": 0.0,
    "FREQ": "1D",
    "SIGNAL_POSITION_SIZE": 0.03,
}


def test_simulate_signal_out_maps_stop_metadata_to_the_engine_config() -> None:
    from ggTrader.engine import portfolio

    out, close = _toy_out(sl_stop=0.01, sl_trail=True, tp_stop=0.5)
    got = simulate_signal_out(out, close, CFG)
    want = portfolio.run_signals(
        out.entries,
        out.exits,
        tile_columns(close, out.entries.columns),
        {**CFG, "ts_stop": 0.01, "tp_stop": 0.5},
        groups=out.group_labels(),
    )
    assert np.array_equal(got.value.to_numpy(), want.value.to_numpy())
    plain = simulate_signal_out(_toy_out()[0], close, CFG)
    assert not np.array_equal(got.value.to_numpy(), plain.value.to_numpy())  # the stop bites


def test_simulate_signal_out_fills_nan_sizes_with_the_base_size() -> None:
    out, close = _toy_out()
    sizes = pd.DataFrame(np.nan, index=out.entries.index, columns=out.entries.columns)
    with_nan = SignalOut(entries=out.entries, exits=out.exits, combos=out.combos, size=sizes)
    a = simulate_signal_out(with_nan, close, CFG)
    b = simulate_signal_out(out, close, CFG)  # size None -> SIGNAL_POSITION_SIZE
    assert np.array_equal(a.value.to_numpy(), b.value.to_numpy())
    big = sizes.copy()
    big.iloc[5, :] = 0.1
    c = simulate_signal_out(
        SignalOut(entries=out.entries, exits=out.exits, combos=out.combos, size=big), close, CFG
    )
    assert not np.array_equal(c.value.to_numpy(), b.value.to_numpy())
