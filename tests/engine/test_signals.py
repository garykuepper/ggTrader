"""L1/L3: signal wrappers, including the int-threshold trap that raises on Rust."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from ggTrader.engine import config
from ggTrader.engine import signals as sig


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"x": values}, index=pd.bdate_range("2024-01-01", periods=len(values)))


def test_cross_above_marks_only_the_crossing_bar(engine: str) -> None:
    out = sig.cross_above(_frame([1, 2, 3, 2, 1, 2]), 1.5)
    assert out["x"].tolist() == [False, True, False, False, False, True]
    assert out.dtypes.iloc[0] == np.bool_


def test_cross_below_marks_only_the_crossing_bar(engine: str) -> None:
    out = sig.cross_below(_frame([2, 1, 0, 1, 2, 1]), 1.5)
    assert out["x"].tolist() == [False, True, False, False, False, True]


def test_integer_threshold_is_coerced_to_float64_and_works_on_rust() -> None:
    config.set_engine("rust")
    assert sig.cross_below(_frame([40.0, 25.0, 31.0, 29.0]), 30)["x"].tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_the_raw_int_threshold_really_raises_on_rust() -> None:
    """The trap this wrapper exists for: if vectorbt ever fixes it, this test says so."""
    config.set_engine("rust")
    with pytest.raises(ValueError, match="float64"):
        _frame([40.0, 25.0]).vbt.crossed_below(30)


def test_integer_dtype_inputs_are_coerced() -> None:
    config.set_engine("rust")
    ints = pd.DataFrame({"x": [1, 2, 3, 2, 1, 2]})
    out = sig.cross_above(ints, pd.DataFrame({"x": [2, 2, 2, 2, 2, 2]}))
    assert out["x"].tolist() == [False, False, True, False, False, False]


def test_tie_is_not_a_crossing_documented_difference_from_legacy(engine: str) -> None:
    """Legacy ``prev <= b & now > b`` fires after a tie; vectorbt needs a strict dip first."""
    values = _frame([2.0, 1.5, 2.0])  # touches the threshold exactly, then rises
    assert sig.cross_above(values, 1.5)["x"].tolist() == [False, False, False]


def test_nan_resets_the_crossing_state(engine: str) -> None:
    assert sig.cross_above(_frame([1.0, np.nan, 2.0]), 1.5)["x"].tolist() == [False] * 3


def test_series_input_gives_series_output(engine: str) -> None:
    out = sig.cross_above(pd.Series([1.0, 2.0, 1.0, 2.0]), 1.5)
    assert isinstance(out, pd.Series) and out.tolist() == [False, True, False, True]


def test_array_valued_threshold(engine: str) -> None:
    close = _frame([1.0, 3.0, 1.0, 3.0])
    line = pd.DataFrame({"x": [2, 2, 2, 2]}, index=close.index)
    assert sig.cross_above(close, line)["x"].tolist() == [False, True, False, True]


@pytest.mark.parametrize("n", [0, 1, 3, 7])
def test_time_stop_equals_a_forward_shift_of_entries(engine: str, n: int) -> None:
    rng = np.random.default_rng(5)
    entries = pd.DataFrame(
        rng.random((120, 6)) < 0.1,
        index=pd.bdate_range("2024-01-01", periods=120),
        columns=list("abcdef"),
    )
    out = sig.time_stop(entries, n)
    expected = entries.shift(n, fill_value=False).astype(bool)
    pd.testing.assert_frame_equal(out, expected)
    assert out.dtypes.eq(np.bool_).all()


def test_time_stop_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        sig.time_stop(_frame([1.0, 2.0]).astype(bool), -1)


def test_time_stop_does_not_use_the_bool_dtype_that_rust_rejects() -> None:
    """vectorbt's fshift on a bool frame raises on Rust; the wrapper must work around it."""
    config.set_engine("rust")
    entries = _frame([1.0, 0.0, 0.0, 0.0]).astype(bool)
    with pytest.raises(ValueError):
        entries.vbt.fshift(2, fill_value=False)
    assert sig.time_stop(entries, 2)["x"].tolist() == [False, False, True, False]


def test_vbt_accessor_still_registered() -> None:
    assert hasattr(vbt, "Portfolio")
