"""Tests for the live-frame to `Bars` conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.paper.live_bars import bars_from_legacy_frame


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for k, sym in enumerate(["MSFT", "AAPL"]):
        px = np.linspace(10 + k, 12 + k, len(index))
        cols[sym] = pd.DataFrame(
            {"open": px, "high": px, "low": px, "close": px, "volume": 1e6}, index=index
        )
    df = pd.concat(cols, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


def test_fields_become_frames_with_sorted_symbols_and_utc_midnight():
    index = pd.bdate_range("2026-06-10", periods=5, tz="UTC") + pd.Timedelta(hours=4)
    bars = bars_from_legacy_frame(_frame(index), ["AAPL", "MSFT", "ZZZ"])
    assert bars.symbols == ["AAPL", "MSFT"]
    assert bars.missing_symbols == ("ZZZ",)
    assert str(bars.close.index.tz) == "UTC"
    assert (bars.close.index == bars.close.index.normalize()).all()
    assert bars.close["AAPL"].iloc[0] == pytest.approx(11.0)  # AAPL is k=1
    assert bars.last_valid_bar["MSFT"] == bars.close.index[-1]


def test_naive_index_is_read_as_utc_and_duplicates_are_kept():
    index = pd.DatetimeIndex(["2026-06-10", "2026-06-11", "2026-06-11"])
    bars = bars_from_legacy_frame(_frame(index))
    assert str(bars.close.index.tz) == "UTC"
    assert not bars.close.index.is_unique  # left for the sanity gate to refuse


def test_empty_frame_gives_empty_bars():
    bars = bars_from_legacy_frame(pd.DataFrame(), ["A"])
    assert bars.close.empty and bars.missing_symbols == ("A",)


def test_missing_field_is_rejected():
    frame = _frame(pd.bdate_range("2026-06-10", periods=3, tz="UTC"))
    frame = frame.drop(columns="volume", level=1)
    with pytest.raises(ValueError, match="volume"):
        bars_from_legacy_frame(frame)


def test_flat_frame_is_rejected():
    with pytest.raises(ValueError, match="MultiIndex"):
        bars_from_legacy_frame(pd.DataFrame({"a": [1.0]}, index=pd.DatetimeIndex(["2026-06-10"])))
