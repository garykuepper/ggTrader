"""Tests for metadata-driven terminal exits."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.dead_stocks import (
    apply_signal_rule,
    apply_weight_rule,
    forced_exit_frames,
    haircut_close,
)


@pytest.fixture
def data():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], "B": 20.0}, index=dates)
    spans = pd.DataFrame({"symbol": ["A"], "last_date": [dates[2]]})
    return close, spans


def test_exit_on_last_valid_bar_and_no_entries_after(data):
    close, spans = data
    f = forced_exit_frames(close, spans)
    assert f.exits.A.tolist() == [False, False, True, False, False, False]
    assert f.entry_ok.A.tolist() == [True, True, False, False, False, False]
    assert f.weight_ok.A.tolist() == [True, True, True, False, False, False]


def test_weights_zero_then_no_order(data):
    close, spans = data
    weights = close.copy()
    weights.iloc[2, 0] = np.nan
    result = apply_weight_rule(weights, forced_exit_frames(close, spans))
    assert result.A.iloc[:3].tolist() == [10.0, 11.0, 0.0]
    assert result.A.iloc[3:].isna().all()
    pd.testing.assert_series_equal(result.B, weights.B)


def test_window_end_independence():
    close = pd.DataFrame({"A": 1.0}, index=pd.date_range("2024-01-01", periods=100))
    spans = pd.DataFrame({"symbol": ["A"], "last_date": [close.index[9]]})
    a, b = forced_exit_frames(close.iloc[:40], spans), forced_exit_frames(close, spans)
    for name in ("exits", "entry_ok", "weight_ok"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name).iloc[:40])


def test_temporary_outage_is_not_dead(data):
    close, spans = data
    close.iloc[1, 0] = np.nan
    assert forced_exit_frames(close, spans).exits.A.tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]


def test_series_not_ended_in_window_has_no_exit(data):
    close, spans = data
    spans.loc[0, "last_date"] = close.index[-1] + pd.Timedelta(days=1)
    assert not forced_exit_frames(close, spans).exits.to_numpy().any()


def test_last_date_on_weekend_uses_previous_session(data):
    close, spans = data
    spans.loc[0, "last_date"] = pd.Timestamp("2024-01-06")
    assert forced_exit_frames(close, spans).exits.A.tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]


def test_last_date_equal_to_window_end_still_exits(data):
    close, spans = data
    spans.loc[0, "last_date"] = close.index[-1]
    assert forced_exit_frames(close, spans).exits.A.iloc[-1]


def test_unlisted_and_unknown_symbols_are_ignored(data):
    close, spans = data
    spans.loc[1] = ["unknown", close.index[1]]
    f = forced_exit_frames(close, spans)
    assert not f.exits.B.any() and f.entry_ok.B.all() and f.weight_ok.B.all()


@pytest.mark.parametrize("loss,expected", [(0, 12.0), (0.5, 6.0), (1, 0.0)])
def test_haircut_scales_only_the_last_bar(data, loss, expected):
    close, spans = data
    close["C"] = np.nan
    close["D"] = 10.0
    spans = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "last_date": [close.index[2]] * 4,
            "has_successor": [False, True, False, True],
            "total_loss": [False, False, False, True],
        }
    )
    original = close.copy()
    result = haircut_close(close, spans, loss)
    assert result.iloc[2].tolist()[:2] == [expected, 20.0]
    assert np.isnan(result.C.iloc[2]) and result.D.iloc[2] == 0.0
    pd.testing.assert_frame_equal(result.drop(close.index[2]), close.drop(close.index[2]))
    pd.testing.assert_frame_equal(close, original)


def test_haircut_rejects_bad_loss(data):
    for loss in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            haircut_close(*data, loss)


def test_apply_signal_rule_counts_removed_entries(data):
    close, spans = data
    entries = close.notna()
    kept, exits, count = apply_signal_rule(
        entries, entries & False, forced_exit_frames(close, spans)
    )
    assert count == 4 and not kept.A.iloc[2:].any() and exits.A.iloc[2]


def test_duplicate_symbols_in_spans_raise(data):
    close, spans = data
    with pytest.raises(ValueError):
        forced_exit_frames(close, pd.concat([spans, spans]))


def test_meta_lists_biases_and_count(data):
    f = forced_exit_frames(*data)
    assert f.meta["n_ended"] == 1 and len(f.meta["biases"]) == 2
