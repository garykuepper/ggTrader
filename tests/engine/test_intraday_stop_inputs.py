"""Synthetic checks for explicitly supplied intraday stop prices."""

import pandas as pd
import pytest

from ggTrader.engine.portfolio import run_signals


@pytest.mark.parametrize(("opening", "expected"), [(100.0, 90.0), (80.0, 80.0)])
def test_stop_uses_intraday_prices_and_gap_open(engine, opening, expected):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    result = run_signals(
        pd.DataFrame({"stock": [True, False, False]}),
        pd.DataFrame({"stock": [False, False, False]}),
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=pd.DataFrame({"stock": [100.0, opening, 100.0]}),
        high=pd.DataFrame({"stock": [100.0, 101.0, 100.0]}),
        low=pd.DataFrame({"stock": [100.0, 80.0, 100.0]}),
    )
    trade = result.trades.iloc[0]
    assert result.engine == engine
    assert (trade.status, trade.exit_idx, trade.exit_price) == (1, 1, expected)


def test_daily_bar_cannot_distinguish_an_intraday_halt_gap(engine):
    # The second path jumps from 99 to 75 after a hypothetical halt; neither is real data.
    continuous = [100.0, 99.0, 90.0, 75.0, 80.0]
    halted = [100.0, 99.0, 75.0, 80.0]

    def bar(path):
        return (path[0], max(path), min(path), path[-1])

    assert bar(continuous) == bar(halted) == (100.0, 100.0, 75.0, 80.0)
    close = pd.DataFrame({"stock": [100.0, 80.0]})
    entry = pd.DataFrame({"stock": [True, False]})
    result = run_signals(
        entry,
        entry & False,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=pd.DataFrame({"stock": [100.0, 100.0]}),
        high=pd.DataFrame({"stock": [100.0, 100.0]}),
        low=pd.DataFrame({"stock": [100.0, 75.0]}),
    )
    assert result.engine == engine
    assert result.trades.iloc[0].exit_price == pytest.approx(90.0)
    assert halted[2] == 75.0  # First observed price after resumption, not a guaranteed fill.


def test_daily_bar_alone_cannot_block_signal_exit_during_a_halt(engine):
    """A carried flat bar has no halt flag, so it permits an unverified exit."""
    close = pd.DataFrame({"stock": [100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False]})
    exits = pd.DataFrame({"stock": [False, True]})
    result = run_signals(
        entries,
        exits,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=close,
        low=close,
    )
    assert result.engine == engine
    assert (result.trades.iloc[0].status, result.trades.iloc[0].exit_idx) == (1, 1)
    assert result.trades.iloc[0].exit_price == pytest.approx(100.0)


def test_known_halt_refuses_position_instead_of_reporting_a_fill(engine):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    exits = pd.DataFrame({"stock": [False, True, False]})
    halt_mask = pd.DataFrame({"stock": [False, True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match="halt.*1.*stock"):
        run_signals(entries, exits, close, config, halt_mask=halt_mask, max_bytes=1)


def test_known_halt_without_a_position_does_not_refuse(engine):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entries = pd.DataFrame({"stock": [False, False, True]})
    halt_mask = pd.DataFrame({"stock": [False, True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    result = run_signals(entries, entries & False, close, config, halt_mask=halt_mask)
    assert result.engine == engine
    assert len(result.trades) == 1


@pytest.mark.parametrize(("halt_bar", "signal_exit"), [(0, True), (2, False)])
def test_halt_guard_includes_entry_bar_and_open_position_at_end(engine, halt_bar, signal_exit):
    """Conservatively reject a halt even on entry or the last open-position bar."""
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    exits = pd.DataFrame({"stock": [False, True, False] if signal_exit else [False] * 3})
    halt_mask = pd.DataFrame({"stock": [i == halt_bar for i in range(3)]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match=rf"halt.*{halt_bar}.*stock"):
        run_signals(entries, exits, close, config, halt_mask=halt_mask)


@pytest.mark.parametrize("intraday", [False, True])
def test_clear_halt_mask_preserves_simulation(engine, intraday):
    close = pd.DataFrame({"stock": [100.0, 90.0, 110.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    options = {"stop_open": close, "high": close, "low": close} if intraday else {}
    baseline = run_signals(entries, entries & False, close, config, **options)
    guarded = run_signals(
        entries, entries & False, close, config, halt_mask=entries & False, **options
    )
    assert baseline.engine == guarded.engine == engine
    pd.testing.assert_frame_equal(baseline.value, guarded.value)
    pd.testing.assert_frame_equal(baseline.trades, guarded.trades)


def test_halt_mask_rejects_unknown_state():
    close = pd.DataFrame({"stock": [100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match="halt_mask"):
        run_signals(entries, entries & False, close, config, halt_mask=close.isna().mask(entries))


def test_compact_halt_mask_catches_open_position_in_later_chunk(engine):
    columns = pd.MultiIndex.from_tuples([("first", "A"), ("second", "B")])
    close = pd.DataFrame(100.0, index=range(3), columns=columns)
    entries = pd.DataFrame([[True, True], [False, False], [False, False]], columns=columns)
    halt_mask = pd.DataFrame({"B": [False, True, False], "A": [False] * 3})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match="halt.*B"):
        run_signals(
            entries,
            entries & False,
            close,
            config,
            groups=["first", "second"],
            halt_mask=halt_mask,
            max_bytes=1,
        )


def test_halt_mask_must_cover_every_grid_symbol():
    columns = pd.MultiIndex.from_tuples([("first", "A"), ("second", "B")])
    close = pd.DataFrame(100.0, index=range(2), columns=columns)
    entries = pd.DataFrame(False, index=close.index, columns=columns)
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match="halt_mask.*every symbol"):
        run_signals(entries, entries, close, config, halt_mask=pd.DataFrame({"A": [False, False]}))


@pytest.mark.parametrize(("opening", "expected"), [(100.0, 89.1), (80.0, 79.2)])
def test_intraday_stop_and_gap_exits_pay_configured_slippage(engine, opening, expected):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    result = run_signals(
        pd.DataFrame({"stock": [True, False, False]}),
        pd.DataFrame({"stock": [False, False, False]}),
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.01, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=pd.DataFrame({"stock": [100.0, opening, 100.0]}),
        high=pd.DataFrame({"stock": [100.0, 101.0, 100.0]}),
        low=pd.DataFrame({"stock": [100.0, 80.0, 100.0]}),
    )
    assert result.engine == engine
    assert result.trades.iloc[0].exit_price == pytest.approx(expected)


@pytest.mark.parametrize("setting", [{}, {"stop_entry_price": None}])
def test_intraday_stop_uses_entry_close_not_slipped_entry_fill(engine, setting):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    low = pd.DataFrame({"stock": [100.0, 90.5, 89.0]})
    result = run_signals(
        entries,
        entries & False,
        close,
        {
            "START_CASH": 10000.0,
            "FEES": 0.0,
            "SLIPPAGE": 0.01,
            "FREQ": "1D",
            "ts_stop": 0.1,
            **setting,
        },
        stop_open=close,
        high=close,
        low=low,
    )
    trade = result.trades.iloc[0]
    assert result.engine == engine
    assert trade.entry_price == pytest.approx(101.0)
    assert (trade.status, trade.exit_idx) == (1, 2)


def test_intraday_stop_can_use_slipped_entry_fill(engine):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    low = pd.DataFrame({"stock": [100.0, 90.5, 89.0]})
    result = run_signals(
        entries,
        entries & False,
        close,
        {
            "START_CASH": 10000.0,
            "FEES": 0.0,
            "SLIPPAGE": 0.01,
            "FREQ": "1D",
            "ts_stop": 0.1,
            "stop_entry_price": "fillprice",
        },
        stop_open=close,
        high=close,
        low=low,
    )
    trade = result.trades.iloc[0]
    assert result.engine == engine
    assert trade.entry_price == pytest.approx(101.0)
    assert (trade.status, trade.exit_idx) == (1, 1)


@pytest.mark.parametrize("bad_value", ["unknown", "", 1])
def test_unknown_stop_entry_price_is_rejected(bad_value):
    close = pd.DataFrame({"stock": [100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False]})
    config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1D",
        "ts_stop": 0.1,
        "stop_entry_price": bad_value,
    }
    with pytest.raises(ValueError, match="stop_entry_price"):
        run_signals(entries, entries & False, close, config)


def test_intraday_stop_requires_complete_matching_price_bars():
    close = pd.DataFrame({"stock": [100.0, 100.0]})
    entry = pd.DataFrame({"stock": [True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    with pytest.raises(ValueError, match="high and low"):
        run_signals(entry, entry & False, close, config, stop_open=close)
    with pytest.raises(ValueError, match="same index and columns"):
        run_signals(
            entry,
            entry & False,
            close,
            config,
            stop_open=close,
            high=close,
            low=close.rename(columns={"stock": "other"}),
        )
    with pytest.raises(ValueError, match="finite"):
        run_signals(
            entry, entry & False, close, config, stop_open=close, high=close, low=close.mask(entry)
        )
    with pytest.raises(ValueError, match="invalid OHLC"):
        run_signals(
            entry, entry & False, close, config, stop_open=close, high=close * 0.9, low=close
        )
    with pytest.raises(ValueError, match="ATR"):
        run_signals(
            entry,
            entry & False,
            close,
            {**config, "ts_stop": None, "atr_mult": 2.0},
            stop_open=close,
            high=close,
            low=close,
        )


def test_intraday_stop_refuses_unreliable_time_stop():
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    with pytest.raises(ValueError, match="td_stop.*intraday"):
        run_signals(
            entries,
            entries & False,
            close,
            config,
            stop_open=close,
            high=close,
            low=close,
            td_stop=2,
        )


def test_intraday_fixed_stop_takes_precedence_over_unused_atr(engine):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entry = pd.DataFrame({"stock": [True, False, False]})
    low = pd.DataFrame({"stock": [100.0, 85.0, 100.0]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    kwargs = {"stop_open": close, "high": close, "low": low}
    expected = run_signals(entry, entry & False, close, config, **kwargs)
    actual = run_signals(entry, entry & False, close, {**config, "atr_mult": 2.0}, **kwargs)
    assert actual.engine == engine
    pd.testing.assert_frame_equal(actual.trades, expected.trades)


def test_intraday_stop_sale_funds_close_entry_in_either_column_order(engine):
    close = pd.DataFrame({"A": [100.0] * 3, "B": [100.0] * 3})
    low = close.copy()
    low.loc[1, "B"] = 80.0
    entries = pd.DataFrame({"A": [False, True, False], "B": [True, False, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    first = run_signals(
        entries, entries & False, close, config, stop_open=close, high=close, low=low, size=1.0
    )
    order = ["B", "A"]
    second = run_signals(
        entries[order],
        entries[order] & False,
        close[order],
        config,
        stop_open=close[order],
        high=close[order],
        low=low[order],
        size=1.0,
    )
    assert first.engine == second.engine == engine
    assert set(first.trades.symbol) == {"A", "B"}
    assert set(second.trades.symbol) == {"A", "B"}
    pd.testing.assert_frame_equal(first.value, second.value)


def test_matching_sequence_compares_modes_without_a_stop_firing(engine):
    close = pd.DataFrame({"A": [100.0, 100.0, 110.0], "B": [100.0] * 3})
    entries = pd.DataFrame({"A": [False, True, False], "B": [True, False, False]})
    exits = pd.DataFrame({"A": [False] * 3, "B": [False, True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    baseline = run_signals(entries, exits, close, config, size=1.0, order_sequence="auto")
    intraday = run_signals(
        entries,
        exits,
        close,
        config,
        size=1.0,
        stop_open=close,
        high=close,
        low=close,
        order_sequence="auto",
    )
    assert baseline.engine == intraday.engine == engine
    pd.testing.assert_frame_equal(baseline.value, intraday.value)
    pd.testing.assert_frame_equal(baseline.trades, intraday.trades)


def test_matching_stop_fill_applies_slippage_in_both_modes(engine):
    close = pd.DataFrame({"stock": [100.0, 85.0, 85.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.01, "FREQ": "1D", "ts_stop": 0.1}
    kwargs = {"order_sequence": "auto", "stop_exit_price": "stopmarket"}
    baseline = run_signals(entries, entries & False, close, config, **kwargs)
    intraday = run_signals(
        entries, entries & False, close, config, stop_open=close, high=close, low=close, **kwargs
    )
    assert baseline.engine == intraday.engine == engine
    assert baseline.trades.iloc[0].exit_price == pytest.approx(84.15)
    pd.testing.assert_frame_equal(baseline.trades, intraday.trades)


def test_matched_stop_modes_expose_intraday_rebound_tradeoff(engine):
    """A low-only stop can exit before a rebound that close-only simulation holds."""
    close = pd.DataFrame({"stock": [100.0, 100.0, 120.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    low = pd.DataFrame({"stock": [100.0, 80.0, 120.0]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.01, "FREQ": "1D", "ts_stop": 0.1}
    common = {"size": 1.0, "order_sequence": "auto", "stop_exit_price": "stopmarket"}
    close_only = run_signals(entries, entries & False, close, config, **common)
    intraday = run_signals(
        entries, entries & False, close, config, stop_open=close, high=close, low=low, **common
    )
    assert close_only.engine == intraday.engine == engine
    assert intraday.trades.iloc[0].exit_price == pytest.approx(89.1)
    assert intraday.value.iloc[-1, 0] == pytest.approx(10000 / 101 * 89.1)
    assert close_only.value.iloc[-1, 0] == pytest.approx(10000 / 101 * 120)


@pytest.mark.parametrize(
    ("high_value", "low_value", "fill_rule", "expected"),
    [
        (100.0, 88.0, "stoplimit", 90.0),
        (112.0, 100.0, "stoplimit", 110.0),
        (112.0, 100.0, "stopmarket", 108.9),
    ],
)
def test_explicit_intraday_fill_rule_applies_to_stop_and_target(
    engine, high_value, low_value, fill_rule, expected
):
    close = pd.DataFrame({"stock": [100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False]})
    high = pd.DataFrame({"stock": [100.0, high_value]})
    low = pd.DataFrame({"stock": [100.0, low_value]})
    config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.01,
        "FREQ": "1D",
        "ts_stop": 0.1,
        "tp_stop": 0.1,
    }
    result = run_signals(
        entries,
        entries & False,
        close,
        config,
        stop_open=close,
        high=high,
        low=low,
        stop_exit_price=fill_rule,
    )
    assert result.engine == engine
    assert result.trades.iloc[0].exit_price == pytest.approx(expected)


def test_unknown_order_sequence_is_rejected():
    close = pd.DataFrame({"stock": [100.0, 100.0]})
    entries = close > 0
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match="order_sequence"):
        run_signals(entries, entries & False, close, config, order_sequence="random")
    with pytest.raises(ValueError, match="stop_exit_price"):
        run_signals(entries, entries & False, close, config, stop_exit_price="unknown")


def test_invalid_price_reports_location():
    close = pd.DataFrame({"stock": [100.0, 100.0]}, index=["first", "second"])
    entries = pd.DataFrame({"stock": [True, False]}, index=close.index)
    bad_high = close.copy()
    bad_high.loc["second", "stock"] = 90.0
    with pytest.raises(ValueError, match="invalid OHLC.*second.*stock"):
        run_signals(
            entries,
            entries & False,
            close,
            {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
            stop_open=close,
            high=bad_high,
            low=close,
        )


def test_rounding_scale_ohlc_discrepancy_is_accepted():
    close = pd.DataFrame({"stock": [100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False]})
    high = close.copy()
    high.iloc[1, 0] -= 1e-12
    run_signals(
        entries,
        entries & False,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=high,
        low=close,
    )


def test_trailing_stop_does_not_ratchet_from_same_bar_open(engine):
    close = pd.DataFrame({"stock": [100.0, 100.0, 101.0]})
    result = run_signals(
        pd.DataFrame({"stock": [True, False, False]}),
        pd.DataFrame({"stock": [False, False, False]}),
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=pd.DataFrame({"stock": [100.0, 120.0, 101.0]}),
        high=pd.DataFrame({"stock": [100.0, 120.0, 101.0]}),
        low=pd.DataFrame({"stock": [100.0, 100.0, 101.0]}),
    )
    assert result.engine == engine
    assert (result.trades.iloc[0].exit_idx, result.trades.iloc[0].exit_price) == (2, 101.0)


def test_intraday_stop_chunking_preserves_results():
    close = pd.DataFrame({"a": [100.0, 100.0, 100.0], "b": [100.0, 100.0, 100.0]})
    opening = close.copy()
    low = close.copy()
    low.iloc[1] = 80.0
    entries = pd.DataFrame({"a": [True, False, False], "b": [True, False, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    kwargs = {"stop_open": opening, "high": close, "low": low, "groups": ["a", "b"]}
    whole = run_signals(entries, entries & False, close, config, **kwargs)
    chunked = run_signals(entries, entries & False, close, config, max_bytes=1, **kwargs)
    pd.testing.assert_frame_equal(whole.value, chunked.value)
    pd.testing.assert_frame_equal(whole.trades, chunked.trades)


def test_compact_stop_prices_match_expanded_prices_with_chunking(engine):
    from ggTrader.lab.strategies.contract import group_columns, tile_columns

    cols = group_columns([{"a": 1}, {"a": 2}], ["a"], ["A", "B"])
    close = pd.DataFrame({"A": [100.0, 100.0, 100.0], "B": [50.0, 50.0, 50.0]})
    low = close.copy()
    low.loc[1, "A"] = 80.0
    entries = pd.DataFrame(False, index=close.index, columns=cols)
    entries.iloc[0] = True
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.01, "FREQ": "1D", "ts_stop": 0.1}
    kwargs = {"groups": [(1,), (1,), (2,), (2,)], "max_bytes": 1}
    expanded = run_signals(
        entries,
        entries & False,
        tile_columns(close, cols),
        config,
        stop_open=tile_columns(close, cols),
        high=tile_columns(close, cols),
        low=tile_columns(low, cols),
        **kwargs,
    )
    compact = run_signals(
        entries,
        entries & False,
        tile_columns(close, cols),
        config,
        stop_open=close,
        high=close,
        low=low,
        **kwargs,
    )
    assert compact.engine == engine
    pd.testing.assert_frame_equal(compact.value, expanded.value)
    pd.testing.assert_frame_equal(compact.trades, expanded.trades)


def test_compact_stop_prices_reject_held_gap_in_later_chunk():
    import numpy as np

    from ggTrader.lab.strategies.contract import group_columns, tile_columns

    cols = group_columns([{"a": 1}, {"a": 2}], ["a"], ["A"])
    close = pd.DataFrame({"A": [100.0, np.nan, 100.0]})
    entries = pd.DataFrame(False, index=close.index, columns=cols)
    entries.iloc[0, 1] = True
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    with pytest.raises(ValueError, match="missing.*open position.*1.*A"):
        run_signals(
            entries,
            entries & False,
            tile_columns(close, cols),
            config,
            stop_open=close,
            high=close,
            low=close,
            groups=[1, 2],
            max_bytes=1,
        )


def test_compact_stop_prices_validate_each_combination_close():
    from ggTrader.lab.strategies.contract import group_columns, tile_columns

    cols = group_columns([{"a": 1}, {"a": 2}], ["a"], ["A"])
    close = pd.DataFrame({"A": [100.0, 100.0]})
    grid_close = tile_columns(close, cols)
    grid_close.iloc[1, 1] = 110.0
    entries = pd.DataFrame(False, index=close.index, columns=cols)
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1}
    with pytest.raises(ValueError, match="invalid OHLC.*1.*A"):
        run_signals(
            entries,
            entries,
            grid_close,
            config,
            stop_open=close,
            high=close,
            low=close,
            groups=[1, 2],
            max_bytes=1,
        )


def test_trailing_stop_ratchets_after_prior_bar_high(engine):
    """The next bar uses the prior completed bar's high to trail the stop."""

    def frame(values):
        return pd.DataFrame({"stock": values})

    close = frame([100.0, 110.0, 109.0, 109.0])
    result = run_signals(
        frame([True, False, False, False]),
        frame([False, False, False, False]),
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=frame([100.0, 100.0, 110.0, 109.0]),
        high=frame([100.0, 120.0, 110.0, 109.0]),
        low=frame([100.0, 95.0, 107.0, 109.0]),
    )
    trade = result.trades.iloc[0]
    assert result.engine == engine
    assert (trade.status, trade.exit_idx, trade.exit_price) == (1, 2, 108.0)


def test_entry_bar_high_does_not_raise_trailing_stop(engine):
    """An entry at the close cannot benefit from the preceding intraday high."""

    def frame(values):
        return pd.DataFrame({"stock": values})

    close = frame([100.0, 100.0, 100.0])
    result = run_signals(
        frame([True, False, False]),
        frame([False, False, False]),
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=frame([150.0, 100.0, 100.0]),
        low=frame([80.0, 95.0, 89.0]),
    )
    trade = result.trades.iloc[0]
    assert result.engine == engine
    assert (trade.status, trade.exit_idx, trade.exit_price) == (1, 2, 90.0)


def test_same_bar_stop_and_target_uses_stop_first(engine):
    """OHLC cannot tell which threshold was first; this pins the simulator convention."""

    def frame(values):
        return pd.DataFrame({"stock": values})

    close = frame([100.0, 100.0, 100.0])
    result = run_signals(
        frame([True, False, False]),
        frame([False, False, False]),
        close,
        {
            "START_CASH": 10000.0,
            "FEES": 0.0,
            "SLIPPAGE": 0.0,
            "FREQ": "1D",
            "ts_stop": 0.1,
            "tp_stop": 0.1,
        },
        stop_open=close,
        high=frame([100.0, 120.0, 100.0]),
        low=frame([100.0, 80.0, 100.0]),
    )
    trade = result.trades.iloc[0]
    assert result.engine == engine
    assert (trade.status, trade.exit_idx, trade.exit_price) == (1, 1, 90.0)


def test_late_listing_with_masked_prelisting_signal_has_no_fabricated_trade(engine):
    import numpy as np

    close = pd.DataFrame({"A": [100.0] * 4, "B": [np.nan, 100.0, 100.0, 100.0]})
    low = close.copy()
    low.loc[2, "B"] = 80.0
    entries = pd.DataFrame({"A": [False] * 4, "B": [True, True, False, False]})
    mask = pd.DataFrame({"A": [True] * 4, "B": [False, True, True, True]})
    result = run_signals(
        entries,
        entries & False,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=close,
        low=low,
        mask=mask,
    )
    trade = result.trades.iloc[0]
    assert result.engine == engine
    assert (trade.symbol, trade.entry_idx, trade.exit_idx, trade.exit_price) == ("B", 1, 2, 90.0)


def test_intraday_mode_accepts_missing_bar_after_actual_exit(engine):
    import numpy as np

    close = pd.DataFrame({"A": [100.0, 100.0, 100.0, np.nan]})
    entries = pd.DataFrame({"A": [True, False, False, False]})
    exits = pd.DataFrame({"A": [False, True, False, False]})
    result = run_signals(
        entries,
        exits,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=close,
        low=close,
    )
    assert result.engine == engine
    assert (result.trades.iloc[0].entry_idx, result.trades.iloc[0].exit_idx) == (0, 1)


@pytest.mark.parametrize("exit_on_gap", [False, True])
def test_intraday_mode_refuses_gap_while_position_is_open(engine, exit_on_gap):
    import numpy as np

    close = pd.DataFrame({"A": [100.0, np.nan, 100.0]})
    entries = pd.DataFrame({"A": [True, False, False]})
    exits = pd.DataFrame({"A": [False, exit_on_gap, not exit_on_gap]})
    with pytest.raises(ValueError, match="missing.*open position"):
        run_signals(
            entries,
            exits,
            close,
            {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
            stop_open=close,
            high=close,
            low=close,
        )


def test_intraday_mode_refuses_entry_on_a_missing_bar():
    import numpy as np

    close = pd.DataFrame({"A": [100.0, np.nan, 100.0]})
    entries = pd.DataFrame({"A": [False, True, False]})
    with pytest.raises(ValueError, match="entry.*missing"):
        run_signals(
            entries,
            entries & False,
            close,
            {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
            stop_open=close,
            high=close,
            low=close,
        )


def test_missing_entire_never_eligible_stock_does_not_create_a_trade():
    import numpy as np

    close = pd.DataFrame({"A": [100.0, 101.0], "B": [np.nan, np.nan]})
    entries = pd.DataFrame({"A": [True, False], "B": [False, False]})
    result = run_signals(
        entries,
        entries & False,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=close,
        low=close,
    )
    assert result.trades.symbol.tolist() == ["A"]


def test_partial_price_bar_is_rejected_even_before_any_entry():
    close = pd.DataFrame({"A": [100.0, 100.0]})
    low = close.copy()
    low.iloc[0, 0] = float("nan")
    entries = pd.DataFrame({"A": [False, True]})
    with pytest.raises(ValueError, match="finite"):
        run_signals(
            entries,
            entries & False,
            close,
            {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
            stop_open=close,
            high=close,
            low=low,
        )


def test_intraday_mode_requires_an_actual_price_stop():
    close = pd.DataFrame({"A": [100.0, 100.0]})
    entries = pd.DataFrame({"A": [True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"}
    with pytest.raises(ValueError, match="require.*stop"):
        run_signals(entries, entries & False, close, config, stop_open=close, high=close, low=close)


def test_intraday_mode_accepts_a_take_profit_without_a_loss_stop():
    close = pd.DataFrame({"A": [100.0, 100.0]})
    high = pd.DataFrame({"A": [100.0, 120.0]})
    entries = pd.DataFrame({"A": [True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "tp_stop": 0.1}
    result = run_signals(
        entries, entries & False, close, config, stop_open=close, high=high, low=close
    )
    assert result.trades.iloc[0].exit_price == pytest.approx(110.0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ts_stop", float("nan")),
        ("ts_stop", float("inf")),
        ("ts_stop", 0.0),
        ("ts_stop", 1.0),
        ("ts_stop", -0.1),
        ("ts_stop", "invalid"),
        ("tp_stop", float("nan")),
        ("tp_stop", float("inf")),
        ("tp_stop", 0.0),
        ("tp_stop", -0.1),
    ],
)
def test_intraday_mode_rejects_invalid_stop_values(key, value):
    close = pd.DataFrame({"A": [100.0, 100.0]})
    entries = pd.DataFrame({"A": [True, False]})
    config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", key: value}
    with pytest.raises(ValueError, match="stop"):
        run_signals(entries, entries & False, close, config, stop_open=close, high=close, low=close)


def test_intraday_mode_rejects_invalid_target_even_with_valid_loss_stop():
    close = pd.DataFrame({"A": [100.0, 100.0]})
    entries = pd.DataFrame({"A": [True, False]})
    config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1D",
        "ts_stop": 0.1,
        "tp_stop": float("nan"),
    }
    with pytest.raises(ValueError, match="tp_stop"):
        run_signals(entries, entries & False, close, config, stop_open=close, high=close, low=close)


def test_missing_gap_between_two_closed_trades_is_allowed(engine):
    import numpy as np

    close = pd.DataFrame({"A": [100.0, 100.0, np.nan, 100.0, 100.0]})
    entries = pd.DataFrame({"A": [True, False, False, True, False]})
    exits = pd.DataFrame({"A": [False, True, False, False, True]})
    result = run_signals(
        entries,
        exits,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=close,
        low=close,
    )
    assert result.engine == engine
    assert list(zip(result.trades.entry_idx, result.trades.exit_idx)) == [(0, 1), (3, 4)]


def test_gap_after_cash_rejected_entry_is_allowed(engine):
    import numpy as np

    close = pd.DataFrame({"A": [100.0, 100.0, 100.0], "B": [100.0, np.nan, 100.0]})
    entries = pd.DataFrame({"A": [True, False, False], "B": [True, False, False]})
    result = run_signals(
        entries,
        entries & False,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=close,
        low=close,
        size=1.0,
    )
    assert result.engine == engine
    assert result.trades.symbol.tolist() == ["A"]


def test_missing_open_position_is_detected_in_later_chunk():
    import numpy as np

    close = pd.DataFrame({"A": [100.0, 100.0, 100.0], "B": [100.0, np.nan, 100.0]})
    entries = pd.DataFrame({"A": [False, False, False], "B": [True, False, False]})
    with pytest.raises(ValueError, match="missing.*open position"):
        run_signals(
            entries,
            entries & False,
            close,
            {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
            stop_open=close,
            high=close,
            low=close,
            groups=["a", "b"],
            max_bytes=1,
        )


def test_open_trade_with_missing_final_bar_is_rejected(engine):
    import numpy as np

    close = pd.DataFrame({"A": [100.0, 100.0, np.nan]})
    entries = pd.DataFrame({"A": [True, False, False]})
    with pytest.raises(ValueError, match="missing.*open position"):
        run_signals(
            entries,
            entries & False,
            close,
            {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
            stop_open=close,
            high=close,
            low=close,
        )


def test_missing_unheld_stock_does_not_distort_a_winner_or_later_entry(engine):
    import numpy as np

    close = pd.DataFrame({"A": [100.0, 110.0, 120.0, 120.0], "B": [100.0, np.nan, 100.0, 110.0]})
    entries = pd.DataFrame({"A": [True, False, False, False], "B": [True, False, True, False]})
    exits = pd.DataFrame({"A": [False, False, True, False], "B": [False, False, False, True]})
    result = run_signals(
        entries,
        exits,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        stop_open=close,
        high=close,
        low=close,
        size=1.0,
    )
    assert result.engine == engine
    assert result.value["all"].tolist() == pytest.approx([10000.0, 11000.0, 12000.0, 13200.0])
    assert result.trades.symbol.tolist() == ["A", "B"]
    assert result.trades.entry_idx.tolist() == [0, 2]
