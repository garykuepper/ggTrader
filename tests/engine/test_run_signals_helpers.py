"""Unit tests for the private helpers extracted from ``run_signals`` (engine chain, Tasks 20-22)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine.portfolio import _halt_positions

DATES = pd.date_range("2024-01-01", periods=5, freq="B")


def _close_flat(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=DATES, columns=columns)


def _close_grid(symbols: list[str], combos: list[int]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([combos, symbols], names=["combo", "symbol"])
    return pd.DataFrame(1.0, index=DATES, columns=columns)


def _mask(columns: list[str], flags: dict[str, list[int]]) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=DATES, columns=columns)
    for name, bars in flags.items():
        mask.iloc[bars, mask.columns.get_loc(name)] = True
    return mask


def test_halt_positions_none_returns_empty() -> None:
    positions, columns = _halt_positions(None, _close_flat(["A", "B"]))
    assert positions == {}
    assert isinstance(columns, np.ndarray)
    assert columns.shape == (0,)
    assert columns.dtype == np.intp


def test_halt_positions_grid_aligned_maps_columns_to_themselves() -> None:
    close = _close_flat(["A", "B", "C"])
    positions, columns = _halt_positions(_mask(["A", "B", "C"], {"B": [1, 3]}), close)
    assert columns.tolist() == [0, 1, 2]
    assert list(positions) == [1]
    assert positions[1].tolist() == [1, 3]


def test_halt_positions_compact_mask_maps_symbols_of_a_multiindex_close() -> None:
    close = _close_grid(["A", "B"], [10, 20])
    # The compact mask has one column per symbol, in a different order than close.
    positions, columns = _halt_positions(_mask(["B", "A"], {"A": [0, 4]}), close)
    assert columns.tolist() == [1, 0, 1, 0]
    assert list(positions) == [1]
    assert positions[1].tolist() == [0, 4]
    assert all(isinstance(key, int) for key in positions)


def test_halt_positions_only_lists_columns_with_a_flag() -> None:
    close = _close_flat(["A", "B", "C"])
    positions, _ = _halt_positions(_mask(["A", "B", "C"], {"A": [2], "C": [0, 1]}), close)
    assert list(positions) == [0, 2]
    assert positions[0].tolist() == [2]
    assert positions[2].tolist() == [0, 1]
    none_flagged, _ = _halt_positions(_mask(["A", "B", "C"], {}), close)
    assert none_flagged == {}


def test_halt_positions_error_messages_are_unchanged() -> None:
    close = _close_flat(["A", "B"])
    grid = _close_grid(["A", "B"], [1, 2])
    base = "halt_mask must have matching dates and known boolean states"

    with pytest.raises(ValueError) as shape:
        _halt_positions(_mask(["A", "B"], {}).iloc[1:], close)
    assert str(shape.value) == base

    with pytest.raises(ValueError) as not_frame:
        _halt_positions(np.zeros((5, 2), dtype=bool), close)  # type: ignore[arg-type]
    assert str(not_frame.value) == base

    with pytest.raises(ValueError) as cover:
        _halt_positions(_mask(["A", "Z"], {}), grid)
    assert str(cover.value) == "halt_mask must cover every symbol"

    with pytest.raises(ValueError) as match:
        _halt_positions(_mask(["A", "C"], {}), close)
    assert str(match.value) == "halt_mask must match close columns"
