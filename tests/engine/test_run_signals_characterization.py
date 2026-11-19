"""Pin run_signals: exact messages, error order, warning attribution and halt/intraday results.

Every case builds tiny closed-form inputs (at most 8 business days and 3 columns, no
randomness), calls ``run_signals`` on the Rust engine and records either the full result or
the exact exception, together with the warnings the call raised. The records live in
``tests/engine/characterization/run_signals.json``. These pins must pass before and after the
helper extractions of the engine chain, so they name no helper.

Input shapes and values are reused from ``test_intraday_stop_inputs.py`` (cited per case),
``test_time_stop_entry_reuse.py`` and ``test_portfolio.py``.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ggTrader.engine import config as engine_config
from ggTrader.engine.portfolio import run_signals
from tests.characterization import RTOL, check_golden, compare, dumps, encode

GOLDEN = Path(__file__).parent / "characterization" / "run_signals.json"
CFG: dict[str, Any] = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
NAN = float("nan")
INF = float("inf")

CASES: dict[str, Callable[[], dict]] = {}


@pytest.fixture(autouse=True)
def _rust_engine() -> Iterator[None]:
    """Run every case on the Rust engine (tests/engine/conftest.py restores the setting)."""
    engine_config.set_engine("rust")
    yield


# --- recording -----------------------------------------------------------------------------


def _warning_record(item: warnings.WarningMessage) -> dict:
    """One recorded warning: category, text and whether the caller (this module) owns it."""
    return {
        "category": item.category.__name__,
        "message": str(item.message),
        "from_caller": Path(item.filename).resolve() == Path(__file__).resolve(),
    }


def _record(call: Callable[[], Any]) -> dict:
    """Run ``call`` and record its SimResult or exact exception plus its warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = call()
        except Exception as exc:
            outcome: dict = {
                "outcome": "raises",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        else:
            outcome = {
                "outcome": "result",
                "result": {
                    "engine": out.engine,
                    "n_chunks": out.n_chunks,
                    "value": out.value,
                    "returns": out.returns,
                    "trades": out.trades,
                    "n_trades": out.n_trades,
                },
            }
    outcome["warnings"] = [_warning_record(item) for item in caught]
    return outcome


def _numbers(record: dict) -> dict:
    """The part of a result that must not change when only chunking or an empty mask does."""
    result = record["result"]
    return {key: result[key] for key in ("value", "returns", "trades", "n_trades")}


def _same(first: dict, second: dict) -> bool:
    """Exact equality of two encoded trees, including dtypes and signed zeros."""
    return dumps(encode(first)) == dumps(encode(second))


def case(func: Callable[[], dict]) -> Callable[[], dict]:
    """Register a case under its function name, in definition order."""
    CASES[func.__name__] = func
    return func


# --- inputs --------------------------------------------------------------------------------


def _frame(index_len: int = 4, **columns: Any) -> pd.DataFrame:
    """A frame on business days with one named column per keyword."""
    index = pd.date_range("2024-01-02", periods=index_len, freq="B")
    return pd.DataFrame(columns, index=index)


def _cfg(**extra: Any) -> dict[str, Any]:
    return {**CFG, **extra}


def _close() -> pd.DataFrame:
    return _frame(A=[100.0] * 4)


def _entry0() -> pd.DataFrame:
    return _frame(A=[True, False, False, False])


def _off(frame: pd.DataFrame) -> pd.DataFrame:
    """All-false exits shaped like ``frame``."""
    return frame & False


def _wrong_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"A": "other"})


def _go(
    entries: pd.DataFrame,
    close: pd.DataFrame,
    config: dict[str, Any] | None = None,
    exits: pd.DataFrame | None = None,
    **kwargs: Any,
) -> dict:
    """Record ``run_signals(entries, exits, close, config, **kwargs)`` (exits default: none)."""
    exits = _off(entries) if exits is None else exits
    config = _cfg() if config is None else config
    return _record(lambda: run_signals(entries, exits, close, config, **kwargs))


def _bars(close: pd.DataFrame, **overrides: pd.DataFrame | None) -> dict[str, Any]:
    """Intraday price keywords: open, high and low all equal to close unless overridden."""
    prices: dict[str, Any] = {"stop_open": close, "high": close, "low": close}
    prices.update(overrides)
    return prices


def _two_flat() -> pd.DataFrame:
    return _frame(A=[100.0] * 4, B=[100.0] * 4)


def _two_entries(first: bool = True, second: bool = True) -> pd.DataFrame:
    return _frame(A=[first, False, False, False], B=[second, False, False, False])


def _multi(*symbols: tuple[str, str], values: list[float] | None = None) -> pd.DataFrame:
    """Close on a (group, symbol) MultiIndex, flat at 100 unless values are given."""
    columns = pd.MultiIndex.from_tuples(list(symbols))
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    return pd.DataFrame(values or 100.0, index=index, columns=columns)


# --- config order --------------------------------------------------------------------------


@case
def unsupported_key() -> dict:
    return _go(_entry0(), _close(), _cfg(vol_target=0.1))


@case
def bad_order_sequence() -> dict:
    return _go(_entry0(), _close(), order_sequence="random")


@case
def bad_stop_exit_price() -> dict:
    return _go(_entry0(), _close(), stop_exit_price="unknown")


@case
def bad_stop_entry_price() -> dict:
    return _go(_entry0(), _close(), _cfg(stop_entry_price="unknown"))


@case
def shape_mismatch_entries() -> dict:
    return _go(_wrong_columns(_entry0()), _close())


# --- precedence: every case makes both of its inputs invalid -------------------------------


@case
def precedence_unsupported_key_over_order_sequence() -> dict:
    return _go(_entry0(), _close(), _cfg(vol_target=0.1), order_sequence="random")


@case
def precedence_order_sequence_over_stop_exit_price() -> dict:
    return _go(_entry0(), _close(), order_sequence="random", stop_exit_price="unknown")


@case
def precedence_stop_exit_price_over_stop_entry_price() -> dict:
    return _go(_entry0(), _close(), _cfg(stop_entry_price="unknown"), stop_exit_price="unknown")


@case
def precedence_stop_entry_price_over_shapes() -> dict:
    return _go(_wrong_columns(_entry0()), _close(), _cfg(stop_entry_price="unknown"))


@case
def precedence_entries_shape_over_exits_shape() -> dict:
    entries = _wrong_columns(_entry0())
    return _go(entries, _close(), exits=_wrong_columns(_off(_entry0())))


@case
def precedence_shapes_over_halt_mask() -> dict:
    return _go(_wrong_columns(_entry0()), _close(), halt_mask="not a frame")


@case
def precedence_halt_bad_frame_over_missing_symbol() -> dict:
    # A NaN state and a symbol (B) the mask does not cover, on a MultiIndex close.
    close = _multi(("first", "A"), ("second", "B"))
    entries = close < 0
    halt_mask = _frame(A=[False, NAN, False, False])
    return _go(entries, close, halt_mask=halt_mask, groups=["first", "second"])


@case
def precedence_halt_mask_over_intraday_config() -> dict:
    close = _close()
    return _go(
        _entry0(), close, _cfg(ts_stop=0.1), halt_mask=[False] * 4, td_stop=2, **_bars(close)
    )


@case
def precedence_td_stop_over_atr_without_trailing() -> dict:
    close = _close()
    return _go(_entry0(), close, _cfg(atr_mult=2.0), td_stop=2, **_bars(close))


@case
def precedence_atr_without_trailing_over_no_price_stop() -> dict:
    close = _close()
    return _go(_entry0(), close, _cfg(atr_mult=2.0), **_bars(close))


@case
def precedence_trailing_over_target_both_invalid() -> dict:
    close = _close()
    return _go(_entry0(), close, _cfg(ts_stop=NAN, tp_stop=0), **_bars(close))


@case
def precedence_trailing_over_target_both_invalid_text_inf() -> dict:
    close = _close()
    return _go(_entry0(), close, _cfg(ts_stop="x", tp_stop=INF), **_bars(close))


@case
def precedence_invalid_target_over_missing_high_low() -> dict:
    close = _close()
    return _go(_entry0(), close, _cfg(tp_stop=0), **_bars(close, high=None))


@case
def precedence_missing_high_low_over_stop_columns_mismatch() -> dict:
    close = _close()
    return _go(
        _entry0(),
        close,
        _cfg(ts_stop=0.1),
        **_bars(close, stop_open=_wrong_columns(close), high=None),
    )


@case
def precedence_intraday_config_over_time_stop_limit() -> dict:
    close = _close()
    return _go(_entry0(), close, _cfg(ts_stop=0.1), td_stop=1.9, **_bars(close))


@case
def precedence_time_stop_limit_over_atr_arguments() -> dict:
    return _go(_entry0(), _close(), _cfg(atr_mult=2.0, atr_period=0), td_stop=-1)


@case
def precedence_atr_shapes_over_size_shape() -> dict:
    close = _close()
    wrong = _wrong_columns(close)
    return _go(_entry0(), close, _cfg(atr_mult=2.0), high=wrong, low=close, size=wrong)


@case
def precedence_stop_bars_invalid_price_over_entry_on_missing_bar() -> dict:
    close = _frame(A=[100.0, 100.0, NAN, 100.0])
    low = close.copy()
    low.iloc[0, 0] = NAN  # a partial bar: invalid, not missing
    entries = _frame(A=[False, False, True, False])  # entry on the wholly missing bar 2
    return _go(entries, close, _cfg(ts_stop=0.1), **_bars(close, low=low))


@case
def precedence_stop_bars_entry_on_missing_bar_over_invalid_ohlc() -> dict:
    close = _frame(A=[100.0, NAN, 100.0, 100.0])
    high = close.copy()
    high.iloc[3, 0] = 90.0  # high below the close: invalid OHLC
    entries = _frame(A=[False, True, False, False])
    return _go(entries, close, _cfg(ts_stop=0.1), **_bars(close, high=high))


@case
def precedence_intraday_chunk_check_over_halt_overlap() -> dict:
    close = _two_flat()
    high = close.copy()
    high.iloc[2, 0] = 90.0  # invalid OHLC in chunk 1 (group a)
    halt = _frame(A=[False] * 4, B=[False, True, False, False])  # a halted position in chunk 2
    return _go(
        _two_entries(),
        close,
        _cfg(ts_stop=0.1),
        halt_mask=halt,
        groups=["a", "b"],
        max_bytes=1,
        **_bars(close, high=high),
    )


@case
def precedence_halt_overlap_in_chunk_two_over_gap_in_chunk_one() -> dict:
    close = _frame(A=[100.0, NAN, 100.0, 100.0], B=[100.0] * 4)
    halt = _frame(A=[False] * 4, B=[False, True, False, False])
    return _go(
        _two_entries(),
        close,
        _cfg(ts_stop=0.1),
        halt_mask=halt,
        groups=["a", "b"],
        max_bytes=1,
        **_bars(close),
    )


# --- halt mask -----------------------------------------------------------------------------


def _halt_inputs() -> dict[str, Any]:
    """A compact halt mask on a MultiIndex grid that overlaps no position (test_intraday_stop_
    inputs.py: test_compact_halt_mask_catches_open_position_in_later_chunk, mask reordered)."""
    close = _multi(("first", "A"), ("second", "B"))
    entries = pd.DataFrame(
        [[True, True], [False, False], [False, False], [False, False]],
        index=close.index,
        columns=close.columns,
    )
    exits = pd.DataFrame(
        [[False, False], [True, True], [False, False], [False, False]],
        index=close.index,
        columns=close.columns,
    )
    halt = _frame(B=[False, False, False, True], A=[False, False, False, True])
    return {
        "entries": entries,
        "exits": exits,
        "close": close,
        "config": _cfg(),
        "kwargs": {"groups": ["first", "second"], "halt_mask": halt},
    }


def _call(inputs: dict[str, Any]) -> Callable[[], Any]:
    return lambda: run_signals(
        inputs["entries"], inputs["exits"], inputs["close"], inputs["config"], **inputs["kwargs"]
    )


@case
def halt_mask_not_a_frame() -> dict:
    return _go(_entry0(), _close(), halt_mask=[False] * 4)


@case
def halt_mask_index_mismatch() -> dict:
    return _go(_entry0(), _close(), halt_mask=pd.DataFrame({"A": [False] * 4}))


@case
def halt_mask_duplicate_columns() -> dict:
    index = _close().index
    halt = pd.DataFrame([[False, False]] * 4, index=index, columns=["A", "A"])
    return _go(_entry0(), _close(), halt_mask=halt)


@case
def halt_mask_nan_state() -> dict:
    return _go(_entry0(), _close(), halt_mask=_frame(A=[False, NAN, False, False]))


@case
def halt_mask_int_dtype() -> dict:
    return _go(_entry0(), _close(), halt_mask=_frame(A=[0, 0, 1, 0]))


@case
def halt_mask_grid_aligned_no_overlap() -> dict:
    exits = _frame(A=[False, True, False, False])
    halt = _frame(A=[False, False, False, True])
    return _go(_entry0(), _close(), exits=exits, halt_mask=halt)


@case
def halt_mask_compact_multiindex_no_overlap() -> dict:
    return _record(_call(_halt_inputs()))


@case
def halt_mask_missing_symbol() -> dict:
    # test_halt_mask_must_cover_every_grid_symbol
    close = _multi(("first", "A"), ("second", "B"))
    halt = _frame(A=[False] * 4)
    return _go(close < 0, close, halt_mask=halt, groups=["first", "second"])


@case
def halt_mask_columns_differ_flat_close() -> dict:
    return _go(_entry0(), _close(), halt_mask=_frame(B=[False] * 4))


def _held_a_halted(bar: int) -> dict:
    """One position held from bar 0 to bar 2; a halt on ``bar``."""
    exits = _frame(A=[False, False, True, False])
    halt = _frame(A=[i == bar for i in range(4)])
    return _go(_entry0(), _close(), exits=exits, halt_mask=halt)


@case
def halt_overlap_on_entry_bar() -> dict:
    return _held_a_halted(0)


@case
def halt_overlap_on_exit_bar() -> dict:
    return _held_a_halted(2)


@case
def halt_overlap_in_second_chunk() -> dict:
    close = _multi(("first", "A"), ("second", "B"))
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    entries.iloc[0] = True
    halt = _frame(B=[False, True, False, False], A=[False] * 4)
    return _go(entries, close, halt_mask=halt, groups=["first", "second"], max_bytes=1)


@case
def two_halted_positions_first_by_column() -> dict:
    # B is halted earlier in time than A, yet the first column (A) is the one reported.
    halt = _frame(A=[False, False, False, True], B=[False, True, False, False])
    return _go(_two_entries(), _two_flat(), halt_mask=halt)


@case
def halt_flagged_but_flat() -> dict:
    exits = _frame(A=[False, True, False, False], B=[False] * 4)
    halt = _frame(A=[False, False, False, True], B=[False, True, False, False])
    return _go(_two_entries(second=False), _two_flat(), exits=exits, halt_mask=halt)


@case
def all_false_halt_mask_equals_none() -> dict:
    entries, close = _entry0(), _close()
    exits = _frame(A=[False, True, False, False])
    without = _go(entries, close, exits=exits)
    with_mask = _go(entries, close, exits=exits, halt_mask=_off(entries))
    assert _same(without, with_mask)
    return {"outcome": "pair", "equal": True, "first": without, "second": with_mask}


# --- intraday stops ------------------------------------------------------------------------


@case
def intraday_with_time_stop() -> dict:
    # test_intraday_stop_refuses_unreliable_time_stop
    close = _close()
    return _go(_entry0(), close, _cfg(ts_stop=0.1), td_stop=2, **_bars(close))


@case
def intraday_atr_without_trailing_stop() -> dict:
    # test_intraday_fixed_stop_takes_precedence_over_unused_atr
    close = _close()
    return _go(_entry0(), close, _cfg(atr_mult=2.0), **_bars(close))


@case
def intraday_without_any_price_stop() -> dict:
    # test_intraday_mode_requires_an_actual_price_stop
    close = _close()
    return _go(_entry0(), close, **_bars(close))


def _bad_fraction(key: str, value: Any) -> Callable[[], dict]:
    """A case: an intraday run whose ``key`` is an invalid fraction (the loss stop is valid)."""

    # Input shapes follow test_intraday_stop_requires_complete_matching_price_bars.

    def build() -> dict:
        close = _close()
        extra = {"tp_stop": 0.1} if key == "ts_stop" else {"ts_stop": 0.1}
        return _go(_entry0(), close, _cfg(**{**extra, key: value}), **_bars(close))

    return build


for _name, _key, _value in (
    ("intraday_invalid_trailing_fraction_nan", "ts_stop", NAN),
    ("intraday_invalid_trailing_fraction_zero", "ts_stop", 0),
    ("intraday_invalid_trailing_fraction_one", "ts_stop", 1.0),
    ("intraday_invalid_trailing_fraction_text", "ts_stop", "x"),
    ("intraday_invalid_target_fraction_zero", "tp_stop", 0),
    ("intraday_invalid_target_fraction_negative", "tp_stop", -1),
    ("intraday_invalid_target_fraction_inf", "tp_stop", INF),
):
    _build = _bad_fraction(_key, _value)
    _build.__name__ = _name
    case(_build)


@case
def intraday_missing_high_low() -> dict:
    # test_intraday_stop_requires_complete_matching_price_bars
    close = _close()
    return _go(_entry0(), close, _cfg(ts_stop=0.1), **_bars(close, high=None))


@case
def intraday_stop_columns_mismatch_flat_close() -> dict:
    # test_intraday_stop_requires_complete_matching_price_bars
    close = _close()
    return _go(_entry0(), close, _cfg(ts_stop=0.1), **_bars(close, stop_open=_wrong_columns(close)))


@case
def intraday_frame_index_mismatch() -> dict:
    # test_intraday_stop_requires_complete_matching_price_bars
    close = _close()
    high = close.copy()
    high.index = range(4)
    return _go(_entry0(), close, _cfg(ts_stop=0.1), **_bars(close, high=high))


def _compact_inputs() -> dict[str, Any]:
    """Compact per-symbol prices on a MultiIndex grid with a mask and per-bar sizes; the A stop
    fires on bar 1 (test_compact_stop_prices_match_expanded_prices_with_chunking)."""
    close = _multi(("first", "A"), ("second", "B"))
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    entries.iloc[0] = True
    flat = _frame(A=[100.0] * 4, B=[100.0] * 4)
    low = flat.copy()
    low.iloc[1, 0] = 80.0
    return {
        "entries": entries,
        "exits": entries & False,
        "close": close,
        "config": _cfg(ts_stop=0.1, SLIPPAGE=0.01),
        "kwargs": {
            "groups": ["first", "second"],
            "mask": pd.DataFrame(True, index=close.index, columns=close.columns),
            "size": pd.DataFrame(0.05, index=close.index, columns=close.columns),
            "stop_open": flat,
            "high": flat,
            "low": low,
        },
    }


@case
def intraday_compact_prices_multiindex() -> dict:
    # test_compact_stop_prices_match_expanded_prices_with_chunking
    return _record(_call(_compact_inputs()))


@case
def intraday_compact_prices_missing_symbol() -> dict:
    # test_compact_stop_prices_validate_each_combination_close
    inputs = _compact_inputs()
    kwargs = inputs["kwargs"]
    for key in ("stop_open", "high", "low"):
        kwargs[key] = kwargs[key][["A"]]
    return _record(_call(inputs))


@case
def intraday_gap_in_second_chunk() -> dict:
    # test_missing_open_position_is_detected_in_later_chunk
    close = _frame(A=[100.0] * 4, B=[100.0, NAN, 100.0, 100.0])
    entries = _two_entries(first=False)
    return _go(entries, close, _cfg(ts_stop=0.1), groups=["a", "b"], max_bytes=1, **_bars(close))


@case
def intraday_entry_on_missing_bar() -> dict:
    # test_intraday_mode_refuses_entry_on_a_missing_bar
    close = _frame(A=[100.0, NAN, 100.0, 100.0])
    entries = _frame(A=[False, True, False, False])
    return _go(entries, close, _cfg(ts_stop=0.1), **_bars(close))


@case
def intraday_invalid_ohlc() -> dict:
    # test_invalid_price_reports_location
    close = _close()
    high = close.copy()
    high.iloc[1, 0] = 90.0
    return _go(_entry0(), close, _cfg(ts_stop=0.1), **_bars(close, high=high))


@case
def intraday_trailing_stop_result() -> dict:
    # test_trailing_stop_ratchets_after_prior_bar_high: the stop fires on bar 2 at 108.
    close = _frame(A=[100.0, 110.0, 109.0, 109.0])
    prices = _bars(
        close,
        stop_open=_frame(A=[100.0, 100.0, 110.0, 109.0]),
        high=_frame(A=[100.0, 120.0, 110.0, 109.0]),
        low=_frame(A=[100.0, 95.0, 107.0, 109.0]),
    )
    return _go(_entry0(), close, _cfg(ts_stop=0.1), **prices)


@case
def intraday_take_profit_only_result() -> dict:
    # test_intraday_mode_accepts_a_take_profit_without_a_loss_stop: the target fires at 110.
    close = _close()
    high = _frame(A=[100.0, 120.0, 100.0, 100.0])
    return _go(_entry0(), close, _cfg(tp_stop=0.1), **_bars(close, high=high))


@case
def intraday_chunked_equals_unchunked() -> dict:
    # test_intraday_stop_chunking_preserves_results
    close = _two_flat()
    low = close.copy()
    low.iloc[1] = 80.0
    kwargs = {"groups": ["a", "b"], **_bars(close, low=low)}
    whole = _go(_two_entries(), close, _cfg(ts_stop=0.1), **kwargs)
    chunked = _go(_two_entries(), close, _cfg(ts_stop=0.1), max_bytes=1, **kwargs)
    assert _same(_numbers(whole), _numbers(chunked))
    return {"outcome": "pair", "equal": True, "first": whole, "second": chunked}


# --- halt and gap together -----------------------------------------------------------------


@case
def halt_and_gap_both_overlap() -> dict:
    # One open position has a missing bar (1) and a halt (2): the halt message wins (risk 4).
    close = _frame(A=[100.0, NAN, 100.0, 100.0])
    halt = _frame(A=[False, False, True, False])
    return _go(_entry0(), close, _cfg(ts_stop=0.1), halt_mask=halt, **_bars(close))


@case
def gap_only_overlap() -> dict:
    # Two gapped positions; the later gap sits in the first column, which is the one reported.
    close = _frame(A=[100.0, 100.0, 100.0, NAN], B=[100.0, NAN, 100.0, 100.0])
    return _go(_two_entries(), close, _cfg(ts_stop=0.1), **_bars(close))


# --- time stop and defaults ----------------------------------------------------------------


@case
def time_stop_result() -> dict:
    return _go(_entry0(), _close(), td_stop=2)


@case
def plain_result() -> dict:
    exits = _frame(A=[False, False, True, False])
    return _go(_entry0(), _close(), exits=exits)


# --- what each case must show --------------------------------------------------------------

RESULT = ("result", "")
EQUAL = ("pair", "")
WHAT_THE_NAME_CLAIMS: dict[str, tuple[str, str]] = {
    "unsupported_key": ("raises", "vol-target sizing is not ported"),
    "bad_order_sequence": ("raises", "order_sequence must be"),
    "bad_stop_exit_price": ("raises", "stop_exit_price must be"),
    "bad_stop_entry_price": ("raises", "stop_entry_price must be"),
    "shape_mismatch_entries": ("raises", "entries must have the same index"),
    "precedence_unsupported_key_over_order_sequence": ("raises", "vol-target sizing is not ported"),
    "precedence_order_sequence_over_stop_exit_price": ("raises", "order_sequence must be"),
    "precedence_stop_exit_price_over_stop_entry_price": ("raises", "stop_exit_price must be"),
    "precedence_stop_entry_price_over_shapes": ("raises", "stop_entry_price must be"),
    "precedence_entries_shape_over_exits_shape": ("raises", "entries must have the same index"),
    "precedence_shapes_over_halt_mask": ("raises", "entries must have the same index"),
    "precedence_halt_bad_frame_over_missing_symbol": ("raises", "known boolean states"),
    "precedence_halt_mask_over_intraday_config": ("raises", "known boolean states"),
    "precedence_td_stop_over_atr_without_trailing": ("raises", "td_stop is unreliable"),
    "precedence_atr_without_trailing_over_no_price_stop": ("raises", "ATR stops are not supported"),
    "precedence_trailing_over_target_both_invalid": ("raises", "invalid ts_stop"),
    "precedence_trailing_over_target_both_invalid_text_inf": ("raises", "invalid ts_stop"),
    "precedence_invalid_target_over_missing_high_low": ("raises", "invalid tp_stop"),
    "precedence_missing_high_low_over_stop_columns_mismatch": (
        "raises",
        "stop_open requires high and low",
    ),
    "precedence_intraday_config_over_time_stop_limit": ("raises", "td_stop is unreliable"),
    "precedence_time_stop_limit_over_atr_arguments": ("raises", "n must be a non-negative int"),
    "precedence_atr_shapes_over_size_shape": ("raises", "high must have the same index"),
    "precedence_stop_bars_invalid_price_over_entry_on_missing_bar": (
        "raises",
        "stop prices must be finite and positive",
    ),
    "precedence_stop_bars_entry_on_missing_bar_over_invalid_ohlc": (
        "raises",
        "entry requested on a missing price bar",
    ),
    "precedence_intraday_chunk_check_over_halt_overlap": ("raises", "invalid OHLC prices at"),
    "precedence_halt_overlap_in_chunk_two_over_gap_in_chunk_one": (
        "raises",
        "halt during an open position at",
    ),
    "halt_mask_not_a_frame": ("raises", "known boolean states"),
    "halt_mask_index_mismatch": ("raises", "known boolean states"),
    "halt_mask_duplicate_columns": ("raises", "known boolean states"),
    "halt_mask_nan_state": ("raises", "known boolean states"),
    "halt_mask_int_dtype": ("raises", "known boolean states"),
    "halt_mask_grid_aligned_no_overlap": RESULT,
    "halt_mask_compact_multiindex_no_overlap": RESULT,
    "halt_mask_missing_symbol": ("raises", "halt_mask must cover every symbol"),
    "halt_mask_columns_differ_flat_close": ("raises", "halt_mask must match close columns"),
    "halt_overlap_on_entry_bar": ("raises", "halt during an open position at 2024-01-02"),
    "halt_overlap_on_exit_bar": ("raises", "halt during an open position at 2024-01-04"),
    "halt_overlap_in_second_chunk": ("raises", "halt during an open position at"),
    "two_halted_positions_first_by_column": ("raises", ", A"),
    "halt_flagged_but_flat": RESULT,
    "all_false_halt_mask_equals_none": EQUAL,
    "intraday_with_time_stop": ("raises", "td_stop is unreliable"),
    "intraday_atr_without_trailing_stop": ("raises", "ATR stops are not supported"),
    "intraday_without_any_price_stop": ("raises", "intraday stop prices require"),
    "intraday_invalid_trailing_fraction_nan": ("raises", "invalid ts_stop"),
    "intraday_invalid_trailing_fraction_zero": ("raises", "invalid ts_stop"),
    "intraday_invalid_trailing_fraction_one": ("raises", "invalid ts_stop"),
    "intraday_invalid_trailing_fraction_text": ("raises", "invalid ts_stop"),
    "intraday_invalid_target_fraction_zero": ("raises", "invalid tp_stop"),
    "intraday_invalid_target_fraction_negative": ("raises", "invalid tp_stop"),
    "intraday_invalid_target_fraction_inf": ("raises", "invalid tp_stop"),
    "intraday_missing_high_low": ("raises", "stop_open requires high and low"),
    "intraday_stop_columns_mismatch_flat_close": ("raises", "stop prices must have the same"),
    "intraday_frame_index_mismatch": ("raises", "stop prices must have the same"),
    "intraday_compact_prices_multiindex": RESULT,
    "intraday_compact_prices_missing_symbol": ("raises", "stop prices must have the same"),
    "intraday_gap_in_second_chunk": ("raises", "missing price bar during an open position"),
    "intraday_entry_on_missing_bar": ("raises", "entry requested on a missing price bar"),
    "intraday_invalid_ohlc": ("raises", "invalid OHLC prices at"),
    "intraday_trailing_stop_result": RESULT,
    "intraday_take_profit_only_result": RESULT,
    "intraday_chunked_equals_unchunked": EQUAL,
    "halt_and_gap_both_overlap": ("raises", "halt during an open position at"),
    "gap_only_overlap": ("raises", "missing price bar during an open position at"),
    "time_stop_result": RESULT,
    "plain_result": RESULT,
}

#: Frozen ordering (Task 20 step or Task 21 check) -> the cases that fix it.
FROZEN_ORDERINGS: list[tuple[str, list[str]]] = [
    ("1 over 2", ["precedence_unsupported_key_over_order_sequence"]),
    ("2 over 3", ["precedence_order_sequence_over_stop_exit_price"]),
    ("3 over 4", ["precedence_stop_exit_price_over_stop_entry_price"]),
    ("4 over 5", ["precedence_stop_entry_price_over_shapes"]),
    ("entries over exits shape", ["precedence_entries_shape_over_exits_shape"]),
    ("5 over 6", ["precedence_shapes_over_halt_mask"]),
    ("inside 6", ["precedence_halt_bad_frame_over_missing_symbol"]),
    ("6 over 9", ["precedence_halt_mask_over_intraday_config"]),
    (
        "b over c; c over d",
        [
            "precedence_td_stop_over_atr_without_trailing",
            "precedence_atr_without_trailing_over_no_price_stop",
        ],
    ),
    (
        "e: ts_stop over tp_stop",
        [
            "precedence_trailing_over_target_both_invalid",
            "precedence_trailing_over_target_both_invalid_text_inf",
        ],
    ),
    (
        "e over f; f over g",
        [
            "precedence_invalid_target_over_missing_high_low",
            "precedence_missing_high_low_over_stop_columns_mismatch",
        ],
    ),
    ("9 over 10", ["precedence_intraday_config_over_time_stop_limit"]),
    ("10 over 11", ["precedence_time_stop_limit_over_atr_arguments"]),
    ("11 (ATR shapes) over size shape", ["precedence_atr_shapes_over_size_shape"]),
    (
        "inside _validate_stop_bars",
        [
            "precedence_stop_bars_invalid_price_over_entry_on_missing_bar",
            "precedence_stop_bars_entry_on_missing_bar_over_invalid_ohlc",
        ],
    ),
    ("12 over 13", ["precedence_intraday_chunk_check_over_halt_overlap"]),
    (
        "13 over 14",
        [
            "precedence_halt_overlap_in_chunk_two_over_gap_in_chunk_one",
            "halt_and_gap_both_overlap",
        ],
    ),
]

SIGNATURE: list[tuple[str, str, Any]] = [
    ("entries", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("exits", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("close", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("base_config", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("groups", "KEYWORD_ONLY", None),
    ("all_groups", "KEYWORD_ONLY", None),
    ("mask", "KEYWORD_ONLY", None),
    ("td_stop", "KEYWORD_ONLY", None),
    ("size", "KEYWORD_ONLY", None),
    ("high", "KEYWORD_ONLY", None),
    ("low", "KEYWORD_ONLY", None),
    ("stop_open", "KEYWORD_ONLY", None),
    ("halt_mask", "KEYWORD_ONLY", None),
    ("order_sequence", "KEYWORD_ONLY", None),
    ("stop_exit_price", "KEYWORD_ONLY", None),
    ("max_bytes", "KEYWORD_ONLY", 5368709120),
]


# --- tests ---------------------------------------------------------------------------------


def _entries(tree: Any) -> dict[str, Any]:
    """The key -> encoded value mapping of an encoded dict (the golden stores tagged trees)."""
    return {key: value for key, value in tree["$dict"]}


def _golden_cases() -> dict[str, Any]:
    """Each case's encoded record, read from the golden file."""
    return _entries(_entries(json.loads(GOLDEN.read_text(encoding="utf-8")))["cases"])


def _trade_rows(record: dict[str, Any]) -> int:
    """How many trades an encoded result record holds."""
    result = _entries(_entries(record)["result"])
    return len(result["trades"]["$frame"]["rows"])


@pytest.mark.parametrize("name", list(CASES))
def test_run_signals_case_matches_golden(name: str) -> None:
    if os.environ.get("GGT_REGEN_CHARACTERIZATION") == "1":
        check_golden(GOLDEN, {"cases": {key: build() for key, build in CASES.items()}})
    actual = encode(CASES[name]())
    golden = _golden_cases()[name]
    exact = os.environ.get("GGT_CHARACTERIZATION_EXACT") == "1"
    paths = compare(actual, golden, rtol=RTOL, exact=exact)
    assert not paths, f"{name} differs at {paths}\n{dumps(actual)[:3000]}"


def test_every_case_has_a_golden_entry() -> None:
    assert not [name for name in CASES if name not in _golden_cases()]


def test_no_orphan_golden_entries() -> None:
    assert not [name for name in _golden_cases() if name not in CASES]


def test_what_the_name_claims_covers_every_case() -> None:
    assert list(WHAT_THE_NAME_CLAIMS) == list(CASES)


@pytest.mark.parametrize("name", list(CASES))
def test_case_shows_what_its_name_claims(name: str) -> None:
    record = _entries(_golden_cases()[name])
    kind, text = WHAT_THE_NAME_CLAIMS[name]
    if kind == "raises":
        assert record["outcome"] == "raises"
        error = _entries(record["error"])
        assert error["type"] == "ValueError"
        assert text in error["message"]
    elif kind == "result":
        assert record["outcome"] == "result"
        assert _trade_rows(_golden_cases()[name]) >= 1
    else:
        assert record["outcome"] == "pair"
        assert record["equal"] is True
        for part in (record["first"], record["second"]):
            assert _entries(part)["outcome"] == "result"
            assert _trade_rows(part) >= 1


def test_td_stop_warning_is_attributed_to_the_caller() -> None:
    close = _close()
    entries = _entry0()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_signals(entries, _off(entries), close, _cfg(), td_stop=2)
    user = [item for item in caught if item.category is UserWarning]
    assert len(user) == 1
    assert Path(user[0].filename).resolve() == Path(__file__).resolve()
    assert str(user[0].message).startswith("td_stop is based on entry signals")


def _frames_equal(before: Any, after: Any) -> None:
    if isinstance(before, pd.DataFrame):
        pd.testing.assert_frame_equal(before, after, check_freq=True)
        assert before.index.equals(after.index) and before.index.names == after.index.names
        assert before.columns.equals(after.columns) and before.columns.names == after.columns.names
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            _frames_equal(before[key], after[key])
    else:
        assert before == after


@pytest.mark.parametrize("inputs", [_halt_inputs, _compact_inputs], ids=["halt", "intraday"])
def test_inputs_are_not_mutated(inputs: Callable[[], dict[str, Any]]) -> None:
    live = inputs()
    before = copy.deepcopy(live)
    result = _call(live)()
    assert len(result.trades) >= 1
    _frames_equal(before, live)


def test_every_frozen_ordering_has_a_case() -> None:
    golden = _golden_cases()
    for ordering, names in FROZEN_ORDERINGS:
        assert names, ordering
        for name in names:
            assert name in CASES, f"{ordering}: {name} is not a case"
            assert name in golden, f"{ordering}: {name} has no golden entry"
            assert _entries(golden[name])["outcome"] == "raises", (
                f"{ordering}: {name} does not raise"
            )


def test_run_signals_signature_is_frozen() -> None:
    actual = [
        (p.name, p.kind.name, p.default) for p in inspect.signature(run_signals).parameters.values()
    ]
    assert actual == SIGNATURE
