"""Unit tests for the private helpers extracted from ``run_signals`` (engine chain, Tasks 20-22)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine.portfolio import (
    SimResult,
    _halt_positions,
    _reject_overlaps,
    _validate_intraday_stops,
)

DATES = pd.date_range("2024-01-01", periods=5, freq="B")


def _close_flat(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=DATES, columns=columns)


def _close_grid(symbols: list[str], combos: list[int] | list[str]) -> pd.DataFrame:
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


STOP_MESSAGE = "stop prices must have the same index and columns as close"


def _stop_frames(columns: pd.Index | list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(1.0, index=DATES, columns=columns)
    return frame.copy(), frame.copy(), frame.copy()


def test_validate_intraday_stops_none_when_not_requested() -> None:
    # Nothing else is checked when no intraday stops were asked for.
    assert _validate_intraday_stops(None, None, None, _close_flat(["A"]), {}, 5) is None
    assert (
        _validate_intraday_stops(None, None, None, _close_flat(["A"]), {"atr_mult": 2}, None)
        is None
    )


def test_validate_intraday_stops_returns_none_for_shared_columns() -> None:
    close = _close_flat(["A", "B"])
    stop_open, high, low = _stop_frames(["A", "B"])
    assert _validate_intraday_stops(stop_open, high, low, close, {"ts_stop": 0.1}, None) is None


def test_validate_intraday_stops_returns_indexer_for_compact_prices() -> None:
    close = _close_grid(["A", "B"], [10, 20])
    stop_open, high, low = _stop_frames(["B", "A"])
    result = _validate_intraday_stops(stop_open, high, low, close, {"tp_stop": 0.1}, None)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [1, 0, 1, 0]


def test_validate_intraday_stops_rejects_a_time_stop() -> None:
    close = _close_flat(["A"])
    stop_open, high, low = _stop_frames(["A"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(stop_open, high, low, close, {"ts_stop": 0.1}, 3)
    assert str(err.value) == "td_stop is unreliable and cannot be combined with intraday stops"


def test_validate_intraday_stops_rejects_atr_without_a_trailing_stop() -> None:
    close = _close_flat(["A"])
    stop_open, high, low = _stop_frames(["A"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(
            stop_open, high, low, close, {"atr_mult": 2.0, "tp_stop": 0.1}, None
        )
    assert str(err.value) == "ATR stops are not supported with intraday stop prices"
    # With a trailing stop as well, the ATR check passes.
    assert (
        _validate_intraday_stops(
            stop_open, high, low, close, {"atr_mult": 2.0, "ts_stop": 0.1}, None
        )
        is None
    )


def test_validate_intraday_stops_requires_a_price_stop() -> None:
    close = _close_flat(["A"])
    stop_open, high, low = _stop_frames(["A"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(stop_open, high, low, close, {}, None)
    assert str(err.value) == "intraday stop prices require a configured price stop"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ts_stop", "abc"),
        ("ts_stop", [0.1]),
        ("ts_stop", float("nan")),
        ("ts_stop", float("inf")),
        ("ts_stop", 0.0),
        ("ts_stop", -0.1),
        ("ts_stop", 1.0),
        ("tp_stop", "abc"),
        ("tp_stop", float("nan")),
        ("tp_stop", float("inf")),
        ("tp_stop", 0.0),
        ("tp_stop", -0.5),
    ],
)
def test_validate_intraday_stops_rejects_an_invalid_fraction(key: str, value: object) -> None:
    close = _close_flat(["A"])
    stop_open, high, low = _stop_frames(["A"])
    config = {"ts_stop": 0.1, "tp_stop": 0.1, key: value}
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(stop_open, high, low, close, config, None)
    assert str(err.value) == f"invalid {key}: expected a finite positive fraction"


def test_validate_intraday_stops_checks_ts_stop_before_tp_stop() -> None:
    close = _close_flat(["A"])
    stop_open, high, low = _stop_frames(["A"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(
            stop_open, high, low, close, {"ts_stop": 2.0, "tp_stop": -1.0}, None
        )
    assert str(err.value) == "invalid ts_stop: expected a finite positive fraction"


def test_validate_intraday_stops_allows_a_take_profit_above_one() -> None:
    close = _close_flat(["A"])
    stop_open, high, low = _stop_frames(["A"])
    assert _validate_intraday_stops(stop_open, high, low, close, {"tp_stop": 2.0}, None) is None


def test_validate_intraday_stops_requires_high_and_low() -> None:
    close = _close_flat(["A"])
    stop_open, high, low = _stop_frames(["A"])
    for missing_high, missing_low in ((None, low), (high, None), (None, None)):
        with pytest.raises(ValueError) as err:
            _validate_intraday_stops(
                stop_open, missing_high, missing_low, close, {"ts_stop": 0.1}, None
            )
        assert str(err.value) == "stop_open requires high and low"


def test_validate_intraday_stops_config_is_checked_before_missing_frames() -> None:
    close = _close_flat(["A"])
    stop_open, _, _ = _stop_frames(["A"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(stop_open, None, None, close, {}, None)
    assert str(err.value) == "intraday stop prices require a configured price stop"


def test_validate_intraday_stops_rejects_different_columns_on_a_flat_close() -> None:
    close = _close_flat(["A", "B"])
    stop_open, high, low = _stop_frames(["A", "C"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(stop_open, high, low, close, {"ts_stop": 0.1}, None)
    assert str(err.value) == STOP_MESSAGE


def test_validate_intraday_stops_rejects_duplicate_compact_columns() -> None:
    close = _close_grid(["A", "B"], [1, 2])
    stop_open, high, low = _stop_frames(["A", "A"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(stop_open, high, low, close, {"ts_stop": 0.1}, None)
    assert str(err.value) == STOP_MESSAGE


def test_validate_intraday_stops_rejects_a_missing_symbol() -> None:
    close = _close_grid(["A", "B"], [1, 2])
    stop_open, high, low = _stop_frames(["A", "C"])
    with pytest.raises(ValueError) as err:
        _validate_intraday_stops(stop_open, high, low, close, {"ts_stop": 0.1}, None)
    assert str(err.value) == STOP_MESSAGE


def test_validate_intraday_stops_rejects_frames_that_do_not_match() -> None:
    close = _close_flat(["A", "B"])
    stop_open, high, low = _stop_frames(["A", "B"])
    config = {"ts_stop": 0.1}
    cases = {
        "high index": (stop_open, high.iloc[1:], low),
        "low index": (stop_open, high, low.iloc[1:]),
        "high columns": (stop_open, high[["B", "A"]], low),
        "low columns": (stop_open, high, low[["B", "A"]]),
        "stop_open index": (stop_open.iloc[1:], high, low),
    }
    for name, (o, h, lo) in cases.items():
        with pytest.raises(ValueError) as err:
            _validate_intraday_stops(o, h, lo, close, config, None)
        assert str(err.value) == STOP_MESSAGE, name


def _sim_result(trades: list[tuple[int, int, int]]) -> SimResult:
    """A SimResult holding only the trade columns ``_reject_overlaps`` reads."""
    frame = pd.DataFrame(trades, columns=["col", "entry_idx", "exit_idx"], dtype="int64")
    empty = pd.DataFrame()
    return SimResult(
        value=empty,
        returns=empty,
        trades=frame,
        n_trades=pd.Series(dtype="int64"),
        engine="test",
        n_chunks=1,
    )


def test_reject_overlaps_noop_without_flags() -> None:
    close = _close_flat(["A", "B"])
    result = _sim_result([(0, 0, 4), (1, 1, 3)])
    outcome = _reject_overlaps(result, close, halt_positions={}, halt_columns=np.empty(0), gaps={})
    assert outcome is None


def test_reject_overlaps_noop_when_flags_miss_every_trade() -> None:
    close = _close_flat(["A", "B"])
    result = _sim_result([(0, 1, 2), (1, 0, 1)])
    halts = {0: np.array([0, 3]), 1: np.array([4])}
    gaps = {0: np.array([4]), 1: np.array([2, 3])}
    _reject_overlaps(result, close, halt_positions=halts, halt_columns=np.arange(2), gaps=gaps)


def test_reject_overlaps_bounds_are_inclusive() -> None:
    close = _close_flat(["A"])
    for bar in (1, 3):
        with pytest.raises(ValueError, match="halt during an open position"):
            _reject_overlaps(
                _sim_result([(0, 1, 3)]),
                close,
                halt_positions={0: np.array([bar])},
                halt_columns=np.arange(1),
                gaps={},
            )
        with pytest.raises(ValueError, match="missing price bar during an open position"):
            _reject_overlaps(
                _sim_result([(0, 1, 3)]),
                close,
                halt_positions={},
                halt_columns=np.empty(0),
                gaps={0: np.array([bar])},
            )


def test_reject_overlaps_halt_before_gap() -> None:
    close = _close_flat(["A", "B"])
    result = _sim_result([(0, 0, 4), (1, 0, 4)])
    # The gap sits on the earlier trade, but the halt loop runs first.
    with pytest.raises(ValueError) as err:
        _reject_overlaps(
            result,
            close,
            halt_positions={1: np.array([2])},
            halt_columns=np.arange(2),
            gaps={0: np.array([1])},
        )
    assert str(err.value) == f"halt during an open position at {DATES[2]}, B"


def test_reject_overlaps_reports_first_trade_by_column() -> None:
    close = _close_flat(["A", "B"])
    result = _sim_result([(0, 0, 1), (0, 2, 4), (1, 0, 4)])
    halts = {0: np.array([3, 4]), 1: np.array([1])}
    with pytest.raises(ValueError) as err:
        _reject_overlaps(result, close, halt_positions=halts, halt_columns=np.arange(2), gaps={})
    # Column 0's second trade comes before column 1's trade; its first flagged bar is 3.
    assert str(err.value) == f"halt during an open position at {DATES[3]}, A"
    with pytest.raises(ValueError) as err:
        _reject_overlaps(
            result,
            close,
            halt_positions={},
            halt_columns=np.empty(0),
            gaps={0: np.array([3, 4]), 1: np.array([1])},
        )
    assert str(err.value) == f"missing price bar during an open position at {DATES[3]}, A"


def test_reject_overlaps_maps_columns_through_halt_columns() -> None:
    close = _close_grid(["A", "B"], ["x", "y"])
    # The halt mask has one column per symbol (B, A); close has (x,A) (x,B) (y,A) (y,B).
    halt_columns = np.array([1, 0, 1, 0])
    halts = {1: np.array([2])}  # only symbol A is flagged, at bar 2
    result = _sim_result([(1, 0, 4), (2, 0, 4)])
    with pytest.raises(ValueError) as err:
        _reject_overlaps(result, close, halt_positions=halts, halt_columns=halt_columns, gaps={})
    # Close column 1 is (x, B), not flagged; close column 2 is (y, A), flagged.
    assert str(err.value) == f"halt during an open position at {DATES[2]}, ('y', 'A')"


def test_reject_overlaps_gap_loop_uses_the_column_directly() -> None:
    close = _close_grid(["A", "B"], ["x", "y"])
    result = _sim_result([(0, 0, 4), (3, 0, 4)])
    # Gaps are keyed by close column position, never through halt_columns.
    with pytest.raises(ValueError) as err:
        _reject_overlaps(
            result,
            close,
            halt_positions={},
            halt_columns=np.array([1, 0, 1, 0]),
            gaps={3: np.array([1])},
        )
    assert str(err.value) == f"missing price bar during an open position at {DATES[1]}, ('y', 'B')"


def test_reject_overlaps_message_formats_timestamp_and_symbol_label() -> None:
    close = _close_grid(["A", "B"], ["x", "y"])
    result = _sim_result([(1, 1, 3)])
    with pytest.raises(ValueError) as err:
        _reject_overlaps(
            result,
            close,
            halt_positions={0: np.array([2])},
            halt_columns=np.array([0, 0, 1, 1]),
            gaps={},
        )
    assert str(err.value) == "halt during an open position at 2024-01-03 00:00:00, ('x', 'B')"
    with pytest.raises(ValueError) as err:
        _reject_overlaps(
            result,
            close,
            halt_positions={},
            halt_columns=np.empty(0),
            gaps={1: np.array([3])},
        )
    assert (
        str(err.value)
        == "missing price bar during an open position at 2024-01-04 00:00:00, ('x', 'B')"
    )
