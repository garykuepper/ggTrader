"""Expose an earlier entry signal's timer closing a later position."""

import warnings

import pandas as pd
import pytest

from ggTrader.engine.portfolio import run_signals


@pytest.mark.parametrize("limit", [1.9, -0.5, 3.0, True, "5"])
def test_invalid_time_limit_is_not_coerced(limit) -> None:
    index = pd.date_range("2020-01-01", periods=4)
    close = pd.DataFrame({"stock": [100.0] * len(index)}, index=index)
    signals = pd.DataFrame(False, index=index, columns=close.columns)
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="non-negative int"):
            run_signals(signals, signals, close, config, td_stop=limit)


def test_time_stop_warns_that_signal_timers_can_close_later_positions() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    close = pd.DataFrame({"stock": [100.0] * len(index)}, index=index)
    signals = pd.DataFrame(False, index=index, columns=close.columns)
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.warns(UserWarning, match="earlier entry signal"):
        run_signals(signals, signals, close, config, td_stop=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_signals(signals, signals, close, config)
    assert not any("earlier entry signal" in str(item.message) for item in caught)


@pytest.mark.xfail(strict=True, reason="time stop currently follows entry signals, not fills")
def test_time_stop_does_not_carry_over_to_a_new_position() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    close = pd.DataFrame({"stock": [100.0] * len(index)}, index=index)
    entries = pd.DataFrame(False, index=index, columns=close.columns)
    entries.iloc[[0, 3]] = True
    exits = pd.DataFrame(False, index=index, columns=close.columns)
    exits.iloc[2] = True
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}

    result = run_signals(entries, exits, close, config, td_stop=5)

    # The second position opens on bar 3, so its own five-bar limit is bar 8.
    assert result.trades["entry_idx"].tolist() == [0, 3]
    assert result.trades["exit_idx"].tolist() == [2, 8]


@pytest.mark.xfail(strict=True, reason="time stop currently follows entry signals, not fills")
def test_old_timer_does_not_skip_new_entry() -> None:
    index = pd.RangeIndex(7)
    close = pd.DataFrame({"stock": [100.0] * len(index)}, index=index)
    entries = pd.DataFrame(False, index=index, columns=close.columns)
    entries.iloc[[0, 3]] = True
    exits = pd.DataFrame(False, index=index, columns=close.columns)
    exits.iloc[1] = True
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    result = run_signals(entries, exits, close, config, td_stop=3)
    assert result.trades["entry_idx"].tolist() == [0, 3]


@pytest.mark.xfail(strict=True, reason="time stop currently follows entry signals, not fills")
def test_new_signal_does_not_cancel_current_time_exit() -> None:
    index = pd.RangeIndex(12)
    close = pd.DataFrame({"stock": [100.0] * len(index)}, index=index)
    entries = pd.DataFrame(False, index=index, columns=close.columns)
    entries.iloc[[0, 5]] = True
    exits = pd.DataFrame(False, index=index, columns=close.columns)
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    result = run_signals(entries, exits, close, config, td_stop=5)
    assert result.trades["exit_idx"].iloc[0] == 5
