"""Exercise a position-aware time-exit prototype on synthetic inputs only.

This uses vectorbt's Numba callback path, not ggTrader's production Rust path.
"""

import inspect

import numba as nb
import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
from vectorbt.portfolio import base, dispatch

pytestmark = pytest.mark.numba


@nb.njit
def _position_time_signal(context, entries, exits, first_bar, pending_entry, limit):
    """Use observed filled position state to time a synthetic long-only trade."""
    row, column = context.i, context.col
    if context.position_now == 0:
        first_bar[column] = -1
        pending_entry[column] = -1
        if entries[row, column] and not exits[row, column]:
            pending_entry[column] = row
    elif first_bar[column] == -1:
        first_bar[column] = pending_entry[column]
    if context.position_now > 0 and row - first_bar[column] >= limit:
        return False, True, False, False
    return entries[row, column], exits[row, column], False, False


@nb.njit
def _unchanged_stop(context):
    return context.curr_stop, context.curr_trail


@pytest.mark.parametrize(
    ("entry_bars", "exit_bars", "limit", "expected"),
    [
        ([0, 3], [2], 5, [(0, 2), (3, 8)]),
        ([0, 3], [1], 3, [(0, 1), (3, 6)]),
        ([0, 5], [], 5, [(0, 5)]),
    ],
)
def test_position_aware_synthetic_examples(entry_bars, exit_bars, limit, expected):
    close = pd.DataFrame({"stock": np.full(12, 100.0)})
    entries = np.zeros((12, 1), dtype=np.bool_)
    exits = np.zeros((12, 1), dtype=np.bool_)
    entries[entry_bars, 0] = True
    exits[exit_bars, 0] = True
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            limit,
        ),
        init_cash=10000.0,
        size=0.03,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    records = portfolio.trades.records
    assert list(zip(records.entry_idx, records.exit_idx)) == expected


def test_rejected_entry_does_not_start_a_timer_for_later_fill():
    close = pd.DataFrame(np.full((9, 2), 100.0), columns=["a", "b"])
    entries = np.zeros((9, 2), dtype=np.bool_)
    exits = np.zeros((9, 2), dtype=np.bool_)
    entries[0, :] = True  # A uses all shared cash; B's first entry cannot fill.
    exits[1, 0] = True
    entries[2, 1] = True
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(2, -1, dtype=np.int64),
            np.full(2, -1, dtype=np.int64),
            3,
        ),
        init_cash=10000.0,
        cash_sharing=True,
        group_by=True,
        size=1.0,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    records = portfolio.trades.records
    assert list(zip(records.col, records.entry_idx, records.exit_idx)) == [(0, 0, 1), (1, 2, 5)]


def test_prior_price_stop_does_not_transfer_its_timer_to_a_new_trade():
    close = pd.DataFrame({"stock": [100.0, 100.0, 85.0] + [100.0] * 9})
    entries = np.zeros((12, 1), dtype=np.bool_)
    exits = np.zeros((12, 1), dtype=np.bool_)
    entries[[0, 3], 0] = True
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            5,
        ),
        sl_stop=0.1,
        init_cash=10000.0,
        size=0.03,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    records = portfolio.trades.records
    assert list(zip(records.entry_idx, records.exit_idx)) == [(0, 2), (3, 8)]


def test_partial_price_stop_keeps_original_entry_time():
    close = pd.DataFrame({"stock": [100.0, 80.0] + [100.0] * 6})
    entries = np.zeros((8, 1), dtype=np.bool_)
    exits = np.zeros((8, 1), dtype=np.bool_)
    entries[0, 0] = True
    max_size = np.full((8, 1), np.inf)
    max_size[1, 0] = 4.0
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            5,
        ),
        sl_stop=0.1,
        max_size=max_size,
        init_cash=10000.0,
        size=10.0,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    assert portfolio.orders.records.idx.tolist() == [0, 1, 5]


def test_rejected_price_stop_keeps_original_entry_time():
    close = pd.DataFrame({"stock": [100.0, 80.0] + [100.0] * 6})
    entries = np.zeros((8, 1), dtype=np.bool_)
    exits = np.zeros((8, 1), dtype=np.bool_)
    entries[0, 0] = True
    min_size = np.zeros((8, 1))
    min_size[1, 0] = 20.0
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            5,
        ),
        sl_stop=0.1,
        min_size=min_size,
        init_cash=10000.0,
        size=10.0,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    assert portfolio.orders.records.idx.tolist() == [0, 5]


def test_rejected_time_exit_is_retried_while_position_is_open():
    close = pd.DataFrame({"stock": [100.0] * 7})
    entries = np.zeros((7, 1), dtype=np.bool_)
    exits = np.zeros((7, 1), dtype=np.bool_)
    entries[0, 0] = True
    min_size = np.zeros((7, 1))
    min_size[3, 0] = 20.0
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            3,
        ),
        init_cash=10000.0,
        size=10.0,
        size_type="amount",
        min_size=min_size,
        log=True,
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    assert portfolio.orders.records.idx.tolist() == [0, 4]
    assert (portfolio.logs.records_readable["Result Status"] == "Rejected").any()


def test_gap_stop_takes_priority_when_time_exit_is_due():
    """A gap through the price stop exits at the open, before the timed close."""
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0, 85.0, 85.0]})
    opened = pd.DataFrame({"stock": [100.0, 100.0, 100.0, 80.0, 85.0]})
    high = pd.DataFrame({"stock": [100.0, 100.0, 100.0, 86.0, 85.0]})
    low = pd.DataFrame({"stock": [100.0, 100.0, 100.0, 79.0, 85.0]})
    entries = np.zeros((5, 1), dtype=np.bool_)
    entries[0, 0] = True
    exits = np.zeros_like(entries)
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(entries, exits, np.full(1, -1), np.full(1, -1), 3),
        open=opened,
        high=high,
        low=low,
        sl_stop=0.1,
        init_cash=10000.0,
        size=10.0,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    assert portfolio.orders.records.idx.tolist() == [0, 3]
    assert portfolio.orders.records.price.tolist() == [100.0, 80.0]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Accumulation is not supported")
def test_prototype_does_not_yet_close_accumulated_positions():
    close = pd.DataFrame({"stock": [100.0] * 12})
    entries = np.zeros((12, 1), dtype=np.bool_)
    exits = np.zeros((12, 1), dtype=np.bool_)
    entries[[0, 3], 0] = True
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            5,
        ),
        accumulate=True,
        init_cash=10000.0,
        size=0.03,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    assert portfolio.trades.records.status.tolist() == [1]


@pytest.mark.parametrize("limit", [1, 3, 5])
def test_repeated_signals_hold_each_filled_trade_for_exactly_the_limit(limit):
    close = pd.DataFrame(np.full((20, 3), 100.0), columns=["a", "b", "c"])
    entries = np.ones((20, 3), dtype=np.bool_)
    exits = np.zeros((20, 3), dtype=np.bool_)
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(3, -1, dtype=np.int64),
            np.full(3, -1, dtype=np.int64),
            limit,
        ),
        init_cash=10000.0,
        cash_sharing=True,
        group_by=True,
        size=0.3,
        size_type="percent",
        fees=0.001,
        slippage=0.001,
        freq="1D",
        engine="numba",
    )
    records = portfolio.trades.records
    closed = records[records.status == 1]
    assert not closed.empty
    assert (closed.exit_idx - closed.entry_idx == limit).all()


@pytest.mark.parametrize("limit", [1, 3, 5])
def test_matches_independent_reference_for_many_overlapping_signals(limit):
    rng = np.random.default_rng(128)
    rows, columns = 250, 8
    entries = rng.random((rows, columns)) < 0.2
    exits = rng.random((rows, columns)) < 0.18
    close = pd.DataFrame(np.full((rows, columns), 100.0))
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(columns, -1, dtype=np.int64),
            np.full(columns, -1, dtype=np.int64),
            limit,
        ),
        init_cash=10000.0,
        cash_sharing=True,
        group_by=True,
        size=0.03,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    expected = []
    for col in range(columns):
        opened = None
        for row in range(rows):
            if opened is None:
                if entries[row, col] and not exits[row, col]:
                    opened = row
            elif row - opened >= limit or (exits[row, col] and not entries[row, col]):
                expected.append((col, opened, row))
                opened = None
    records = portfolio.trades.records
    closed = records[records.status == 1]
    actual = list(zip(closed.col, closed.entry_idx, closed.exit_idx))
    assert actual == expected


def test_custom_callback_selects_numba_even_when_rust_is_requested(monkeypatch):
    original = base.nb.simulate_from_signal_func_nb
    calls = []

    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(base.nb, "simulate_from_signal_func_nb", record_call)
    entries = np.array([[True], [False], [False]])
    exits = np.zeros((3, 1), dtype=np.bool_)
    portfolio = vbt.Portfolio.from_signals(
        pd.DataFrame({"stock": [100.0, 100.0, 100.0]}),
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            1,
        ),
        init_cash=10000.0,
        size=0.03,
        size_type="percent",
        freq="1D",
        engine="rust",
    )
    assert len(calls) == 1
    assert portfolio.trades.records.exit_idx.tolist() == [1]


def test_static_rust_dispatch_has_no_native_duration_stop_parameter():
    assert "td_stop" not in inspect.signature(dispatch.simulate_from_signals).parameters


def test_custom_stop_callback_selects_numba_even_when_rust_is_requested(monkeypatch):
    original = base.nb.simulate_from_signal_func_nb
    calls = []

    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(base.nb, "simulate_from_signal_func_nb", record_call)
    vbt.Portfolio.from_signals(
        pd.DataFrame({"stock": [100.0, 100.0]}),
        entries=np.array([[True], [False]]),
        exits=np.zeros((2, 1), dtype=bool),
        sl_stop=0.1,
        adjust_sl_func_nb=_unchanged_stop,
        init_cash=10000.0,
        size=0.03,
        size_type="percent",
        freq="1D",
        engine="rust",
    )
    assert calls == [True]


@pytest.mark.parametrize("seed", [11, 29, 47])
def test_filled_orders_respect_limit_with_price_stops_and_shared_cash(seed):
    """Check actual fills, including price-stop exits and cash-constrained entries."""
    rng = np.random.default_rng(seed)
    rows, columns, limit = 300, 12, 5
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.025, (rows, columns)), axis=0)))
    entries = rng.random((rows, columns)) < 0.22
    exits = rng.random((rows, columns)) < 0.08
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(columns, -1, dtype=np.int64),
            np.full(columns, -1, dtype=np.int64),
            limit,
        ),
        init_cash=10000.0,
        cash_sharing=True,
        group_by=True,
        size=3000.0,
        size_type="value",
        allow_partial=False,
        log=True,
        sl_stop=0.05,
        tp_stop=0.08,
        fees=0.001,
        slippage=0.001,
        freq="1D",
        engine="numba",
    )
    orders = portfolio.orders.records
    assert len(orders) > 100
    assert (portfolio.logs.records_readable["Result Status"] == "Rejected").any()
    early_exits = 0
    for col in range(columns):
        opened = None
        entry_price = None
        for order in orders[orders.col == col].itertuples():
            if order.side == 0:
                assert opened is None
                opened = order.idx
                entry_price = close.iloc[opened, col]
            else:
                assert opened is not None
                age = order.idx - opened
                assert age <= limit
                if age < limit:
                    early_exits += 1
                    price = close.iloc[order.idx, col]
                    signal_exit = exits[order.idx, col] and not entries[order.idx, col]
                    price_stop = price <= entry_price * 0.95 or price >= entry_price * 1.08
                    assert signal_exit or price_stop
                opened = None
                entry_price = None
        if opened is not None:
            assert rows - 1 - opened < limit
    assert early_exits > 0


def test_synthetic_rising_trade_is_not_closed_before_its_time_limit():
    """An uninterrupted winner stays open until the position's own limit."""
    close = pd.DataFrame({"stock": [100.0, 106.0, 112.0, 118.0, 124.0, 130.0]})
    entries = np.array([[True], [False], [False], [False], [False], [False]])
    exits = np.zeros((6, 1), dtype=np.bool_)
    portfolio = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            4,
        ),
        init_cash=10000.0,
        size=1.0,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    trade = portfolio.trades.records.iloc[0]
    assert (trade.entry_idx, trade.exit_idx) == (0, 4)
    assert trade.entry_price == 100.0
    assert trade.exit_price == 124.0


def test_stop_loss_after_gap_fills_at_open_not_at_stop_price():
    """A stop is a trigger, not a guarantee of the requested exit price."""
    close = pd.DataFrame({"stock": [100.0, 84.0, 86.0]})
    entries = np.array([[True], [False], [False]])
    exits = np.zeros((3, 1), dtype=np.bool_)
    portfolio = vbt.Portfolio.from_signals(
        close,
        open=pd.DataFrame({"stock": [100.0, 80.0, 84.0]}),
        high=pd.DataFrame({"stock": [100.0, 86.0, 88.0]}),
        low=pd.DataFrame({"stock": [100.0, 78.0, 83.0]}),
        signal_func_nb=_position_time_signal,
        signal_args=(
            entries,
            exits,
            np.full(1, -1, dtype=np.int64),
            np.full(1, -1, dtype=np.int64),
            5,
        ),
        sl_stop=0.1,
        init_cash=10000.0,
        size=1.0,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="numba",
    )
    trade = portfolio.trades.records.iloc[0]
    assert (trade.entry_idx, trade.exit_idx) == (0, 1)
    assert trade.exit_price == 80.0


@pytest.mark.parametrize("engine", ["rust", "numba"])
def test_native_stop_can_use_actual_entry_fill(engine):
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    low = pd.DataFrame({"stock": [100.0, 90.5, 89.0]})
    common = dict(
        close=close,
        entries=entries,
        exits=entries & False,
        open=close,
        high=close,
        low=low,
        sl_stop=0.1,
        init_cash=10000.0,
        size=1.0,
        size_type="percent",
        slippage=0.01,
        freq="1D",
        engine=engine,
    )
    close_based = vbt.Portfolio.from_signals(**common, stop_entry_price="close")
    fill_based = vbt.Portfolio.from_signals(**common, stop_entry_price="fillprice")
    assert close_based.trades.records.entry_price.iloc[0] == pytest.approx(101.0)
    assert fill_based.trades.records.entry_price.iloc[0] == pytest.approx(101.0)
    assert close_based.trades.records.exit_idx.iloc[0] == 2
    assert fill_based.trades.records.exit_idx.iloc[0] == 1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Lab wrapper does not forward OHLC to stop simulation",
)
def test_lab_trailing_stop_should_respond_to_intraday_low():
    """A fall below a stop must not disappear just because the close recovered."""
    from ggTrader.engine.portfolio import run_signals

    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    high = pd.DataFrame({"stock": [100.0, 101.0, 100.0]})
    low = pd.DataFrame({"stock": [100.0, 80.0, 100.0]})
    entries = pd.DataFrame({"stock": [True, False, False]})
    exits = pd.DataFrame({"stock": [False, False, False]})
    result = run_signals(
        entries,
        exits,
        close,
        {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D", "ts_stop": 0.1},
        high=high,
        low=low,
    )
    trade = result.trades.iloc[0]
    assert trade["status"] == 1
    assert trade["exit_idx"] == 1


def test_rust_stop_with_ohlc_detects_intraday_drop():
    """The Rust simulator can detect an intraday breach when given OHLC."""
    close = pd.DataFrame({"stock": [100.0, 100.0, 100.0]})
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=pd.DataFrame({"stock": [True, False, False]}),
        exits=pd.DataFrame({"stock": [False, False, False]}),
        open=pd.DataFrame({"stock": [100.0, 100.0, 100.0]}),
        high=pd.DataFrame({"stock": [100.0, 101.0, 100.0]}),
        low=pd.DataFrame({"stock": [100.0, 80.0, 100.0]}),
        sl_stop=0.1,
        sl_trail=True,
        init_cash=10000.0,
        size=1.0,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="rust",
    )
    trade = portfolio.trades.records.iloc[0]
    assert (trade.entry_idx, trade.exit_idx, trade.status) == (0, 1, 1)
    assert trade.exit_price == 90.0


def test_rust_stop_with_ohlc_fills_gap_at_open():
    """A gap through the stop exits at the observed open in this simulation."""
    close = pd.DataFrame({"stock": [100.0, 84.0, 86.0]})
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=pd.DataFrame({"stock": [True, False, False]}),
        exits=pd.DataFrame({"stock": [False, False, False]}),
        open=pd.DataFrame({"stock": [100.0, 80.0, 84.0]}),
        high=pd.DataFrame({"stock": [100.0, 86.0, 88.0]}),
        low=pd.DataFrame({"stock": [100.0, 78.0, 83.0]}),
        sl_stop=0.1,
        sl_trail=True,
        init_cash=10000.0,
        size=1.0,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="rust",
    )
    trade = portfolio.trades.records.iloc[0]
    assert (trade.entry_idx, trade.exit_idx, trade.status) == (0, 1, 1)
    assert trade.exit_price == 80.0


def test_synthetic_rust_reruns_match_shared_cash_reference():
    """Probe reruns only; this does not establish a production time-stop method."""
    for seed in range(40):
        rng = np.random.default_rng(seed)
        entries = rng.random((15, 3)) < 0.25
        exits = rng.random((15, 3)) < 0.12
        forced = np.zeros_like(entries)
        close = pd.DataFrame(np.full(entries.shape, 100.0))
        limit = 3

        for _ in range(entries.size + 1):
            portfolio = vbt.Portfolio.from_signals(
                close,
                entries=entries & ~forced,
                exits=exits | forced,
                init_cash=10000.0,
                cash_sharing=True,
                group_by=True,
                size=1.0,
                size_type="percent",
                fees=0.0,
                slippage=0.0,
                freq="1D",
                engine="rust",
            )
            due = [
                (int(trade.entry_idx) + limit, int(trade.col))
                for trade in portfolio.trades.records.itertuples()
                if trade.entry_idx + limit < len(close)
                and (trade.status == 0 or trade.exit_idx > trade.entry_idx + limit)
                and not forced[int(trade.entry_idx) + limit, int(trade.col)]
            ]
            if not due:
                break
            row, col = min(due)
            forced[row, col] = True
        else:
            pytest.fail("Timed-exit reruns exceeded the iteration cap")

        cash = True
        opened = [None] * entries.shape[1]
        expected = []
        for row in range(entries.shape[0]):
            for col in range(entries.shape[1]):
                timed = opened[col] is not None and row - opened[col] >= limit
                if opened[col] is not None and (
                    timed or (exits[row, col] and not entries[row, col])
                ):
                    opened[col], cash = None, True
                    expected.append((row, col, 1))
                elif opened[col] is None and entries[row, col] and not exits[row, col] and cash:
                    opened[col], cash = row, False
                    expected.append((row, col, 0))
        actual = [
            (int(o.idx), int(o.col), int(o.side)) for o in portfolio.orders.records.itertuples()
        ]
        assert actual == expected, f"seed {seed}"


@pytest.mark.parametrize("call_seq", ["default", "auto"])
@pytest.mark.parametrize("seed", [13, 31, 59])
def test_synthetic_batch_reruns_preserve_orders_when_future_bars_are_added(seed, call_seq):
    """A later bar must not revise any earlier fill in the synthetic prototype."""
    rng = np.random.default_rng(seed)
    rows, cols, limit = 80, 4, 5
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (rows, cols)), axis=0)))
    entries = rng.random((rows, cols)) < 0.22
    exits = rng.random((rows, cols)) < 0.09
    max_size = np.where(rng.random((rows, cols)) < 0.15, 3.0, np.inf)
    min_size = np.where(rng.random((rows, cols)) < 0.08, 12.0, 0.0)

    def simulate(length):
        forced = np.zeros((length, cols), dtype=bool)
        seen = set()
        for _ in range(length * cols + 1):
            pf = vbt.Portfolio.from_signals(
                close.iloc[:length],
                entries=entries[:length] & ~forced,
                exits=exits[:length] | forced,
                init_cash=1200.0,
                cash_sharing=True,
                group_by=True,
                size=10.0,
                size_type="amount",
                max_size=max_size[:length],
                min_size=min_size[:length],
                fees=0.001,
                slippage=0.001,
                sl_stop=0.05,
                call_seq=call_seq,
                freq="1D",
                engine="rust",
            )
            updated = np.zeros_like(forced)
            for trade in pf.trades.records.itertuples():
                start, end, col = int(trade.entry_idx) + limit, int(trade.exit_idx), int(trade.col)
                if start <= end:
                    updated[start : end + 1, col] = True
            if np.array_equal(updated, forced):
                return pf.orders.records[["idx", "col", "side", "size", "price", "fees"]]
            key = updated.tobytes()
            if key in seen:
                pytest.fail(f"synthetic reruns cycled: seed={seed}, length={length}")
            seen.add(key)
            forced = updated
        pytest.fail(f"synthetic reruns exceeded the iteration cap: seed={seed}, length={length}")

    full = simulate(rows)
    for cutoff in (17, 37, 61):
        short = simulate(cutoff).reset_index(drop=True)
        prefix = full[full.idx < cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(short, prefix, check_exact=False, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("rows", [20, 100])
def test_synthetic_batch_reruns_need_one_pass_per_bar_for_dense_entries(rows):
    """Characterize a costly case for the experimental rerun method."""
    close = pd.DataFrame({"stock": np.full(rows, 100.0)})
    entries = np.ones((rows, 1), dtype=bool)
    forced = np.zeros_like(entries)
    for count in range(1, rows + 2):
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=entries & ~forced,
            exits=forced,
            init_cash=10000.0,
            size=1.0,
            size_type="percent",
            fees=0.0,
            slippage=0.0,
            freq="1D",
            engine="rust",
        )
        updated = np.zeros_like(forced)
        for trade in portfolio.trades.records.itertuples():
            start, end = int(trade.entry_idx) + 1, int(trade.exit_idx)
            if start <= end:
                updated[start : end + 1, int(trade.col)] = True
        if np.array_equal(updated, forced):
            break
        forced = updated
    else:
        pytest.fail("synthetic reruns did not converge")
    assert count == rows
    assert portfolio.orders.records.idx.tolist() == list(range(rows))
    assert portfolio.orders.records.side.tolist() == [0, 1] * (rows // 2)


def _synthetic_initial_guess(entries, exits, limit):
    forced = np.zeros_like(entries)
    for col in range(entries.shape[1]):
        opened = None
        for row in range(entries.shape[0]):
            if opened is None:
                if entries[row, col] and not exits[row, col]:
                    opened = row
            elif row - opened >= limit:
                forced[row, col] = True
                opened = None
            elif exits[row, col] and not entries[row, col]:
                opened = None
    return forced


def test_synthetic_warm_start_matches_position_callback_with_shared_cash():
    """Check an initial guess against actual fills, including rejected entries."""
    for seed in range(50):
        rng = np.random.default_rng(seed)
        rows, cols, limit = 80, 4, 4
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.04, (rows, cols)), axis=0)))
        entries = rng.random((rows, cols)) < 0.25
        exits = rng.random((rows, cols)) < 0.1
        common = dict(
            init_cash=1200.0,
            cash_sharing=True,
            group_by=True,
            size=10.0,
            size_type="amount",
            fees=0.001,
            slippage=0.001,
            sl_stop=0.05,
            freq="1D",
        )
        reference = vbt.Portfolio.from_signals(
            close,
            signal_func_nb=_position_time_signal,
            signal_args=(entries, exits, np.full(cols, -1), np.full(cols, -1), limit),
            engine="numba",
            **common,
        )
        forced = _synthetic_initial_guess(entries, exits, limit)
        seen = set()
        for _ in range(rows * cols + 1):
            portfolio = vbt.Portfolio.from_signals(
                close, entries=entries & ~forced, exits=exits | forced, engine="rust", **common
            )
            updated = np.zeros_like(forced)
            for trade in portfolio.trades.records.itertuples():
                start, end = int(trade.entry_idx) + limit, int(trade.exit_idx)
                if start <= end:
                    updated[start : end + 1, int(trade.col)] = True
            if np.array_equal(updated, forced):
                break
            key = updated.tobytes()
            if key in seen:
                pytest.fail(f"synthetic warm-start reruns cycled on seed {seed}")
            seen.add(key)
            forced = updated
        else:
            pytest.fail(f"synthetic warm-start reruns exceeded the cap on seed {seed}")
        columns = ["idx", "col", "side", "size", "price", "fees"]
        pd.testing.assert_frame_equal(
            portfolio.orders.records[columns].reset_index(drop=True),
            reference.orders.records[columns].reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )


def test_synthetic_dense_entry_guess_is_stable_after_one_rust_run():
    rows = 1200
    close = pd.DataFrame({"stock": np.full(rows, 100.0)})
    entries = np.ones((rows, 1), dtype=bool)
    forced = np.zeros_like(entries)
    forced[1::2, 0] = True
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries & ~forced,
        exits=forced,
        init_cash=10000.0,
        size=1.0,
        size_type="percent",
        fees=0.0,
        slippage=0.0,
        freq="1D",
        engine="rust",
    )
    updated = np.zeros_like(forced)
    for trade in portfolio.trades.records.itertuples():
        start, end = int(trade.entry_idx) + 1, int(trade.exit_idx)
        if start <= end:
            updated[start : end + 1, int(trade.col)] = True
    assert np.array_equal(updated, forced)
    assert portfolio.orders.records.idx.tolist() == list(range(rows))
    assert portfolio.orders.records.side.tolist() == [0, 1] * (rows // 2)


@pytest.mark.parametrize(("batch_all", "warm"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("call_seq", ["default", "auto"])
def test_synthetic_rust_reruns_match_callback_with_price_stops(batch_all, warm, call_seq):
    """Compare retries and intraday stops; this is not production validation."""
    partial_exit_seen = False
    rejection_seen = False
    reordered = False
    for seed in range(20):
        rng = np.random.default_rng(seed)
        rows, cols, limit = 65, 3, 4
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.04, (rows, cols)), axis=0)))
        opened = close.shift(1).fillna(close.iloc[0])
        high = pd.DataFrame(np.maximum(opened, close) * 1.01)
        low = pd.DataFrame(np.minimum(opened, close) * 0.99)
        entries = rng.random((rows, cols)) < 0.2
        exits = rng.random((rows, cols)) < 0.08
        max_size = np.full((rows, cols), np.inf)
        max_size[rng.random((rows, cols)) < 0.2] = 4.0
        min_size = np.zeros((rows, cols))
        min_size[rng.random((rows, cols)) < 0.1] = 15.0
        common = dict(
            init_cash=1200.0,
            cash_sharing=True,
            group_by=True,
            size=10.0,
            size_type="amount",
            max_size=max_size,
            min_size=min_size,
            fees=0.001,
            slippage=0.001,
            sl_stop=0.05,
            open=opened,
            high=high,
            low=low,
            log=True,
            call_seq=call_seq,
            attach_call_seq=True,
            freq="1D",
        )
        reference = vbt.Portfolio.from_signals(
            close,
            signal_func_nb=_position_time_signal,
            signal_args=(entries, exits, np.full(cols, -1), np.full(cols, -1), limit),
            engine="numba",
            **common,
        )
        forced = _synthetic_initial_guess(entries, exits, limit) if warm else np.zeros_like(entries)
        seen = set()
        for _ in range(entries.size + 1):
            portfolio = vbt.Portfolio.from_signals(
                close, entries=entries & ~forced, exits=exits | forced, engine="rust", **common
            )
            if batch_all:
                updated = np.zeros_like(entries)
                for trade in portfolio.trades.records.itertuples():
                    start = int(trade.entry_idx) + limit
                    stop = int(trade.exit_idx)
                    if start < rows and start <= stop:
                        updated[start : stop + 1, int(trade.col)] = True
                if np.array_equal(updated, forced):
                    break
                key = updated.tobytes()
                if key in seen:
                    pytest.fail(f"Timed-exit reruns cycled on seed {seed}")
                seen.add(key)
                forced = updated
                continue
            due = []
            for trade in portfolio.trades.records.itertuples():
                start = int(trade.entry_idx) + limit
                stop = int(trade.exit_idx) + (1 if trade.status == 0 else 0)
                due.extend(
                    (row, int(trade.col))
                    for row in range(start, stop)
                    if not forced[row, int(trade.col)]
                )
            if not due:
                break
            row, col = min(due)
            forced[row, col] = True
        else:
            pytest.fail("Timed-exit reruns exceeded the iteration cap")
        columns = ["idx", "col", "side", "size", "price", "fees"]
        actual = portfolio.orders.records[columns].reset_index(drop=True)
        expected = reference.orders.records[columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-10)
        rejection_seen |= (reference.logs.records_readable["Result Status"] == "Rejected").any()
        reordered |= (reference.call_seq.to_numpy() != np.arange(cols)).any()
        positions = np.zeros(cols)
        opened_at = [None] * cols
        orders = {(int(o.idx), int(o.col)): o for o in actual.itertuples()}
        if batch_all:
            assert len(orders) == len(actual)
            for row in range(rows):
                for col in range(cols):
                    if opened_at[col] is not None and row - opened_at[col] >= limit:
                        assert forced[row, col], (seed, row, col)
                    if forced[row, col]:
                        assert opened_at[col] is not None and row - opened_at[col] >= limit, (
                            seed,
                            row,
                            col,
                        )
                    order = orders.get((row, col))
                    if order is None:
                        continue
                    if order.side == 0:
                        assert opened_at[col] is None
                        opened_at[col] = row
                        positions[col] = order.size
                    else:
                        positions[col] -= order.size
                        if positions[col] < 1e-9:
                            positions[col] = 0
                            opened_at[col] = None
            positions.fill(0)
        for order in expected.itertuples():
            if order.side == 0:
                positions[order.col] += order.size
            else:
                partial_exit_seen |= order.size < positions[order.col] - 1e-9
                positions[order.col] -= order.size
    assert partial_exit_seen
    assert rejection_seen
    if call_seq == "auto":
        assert reordered


def test_static_timed_exit_does_not_retry_a_partial_fill():
    """A single forced exit leaves the remaining shares open after a partial fill."""
    close = pd.DataFrame({"stock": [100.0] * 7})
    entries = np.zeros((7, 1), dtype=np.bool_)
    entries[0, 0] = True
    exits = np.zeros_like(entries)
    forced = np.zeros_like(entries)
    forced[3, 0] = True
    max_size = np.full((7, 1), np.inf)
    max_size[3, 0] = 4.0
    common = dict(
        init_cash=10000.0,
        size=10.0,
        size_type="amount",
        max_size=max_size,
        fees=0.0,
        slippage=0.0,
        freq="1D",
    )
    rust = vbt.Portfolio.from_signals(
        close, entries=entries & ~forced, exits=forced, engine="rust", **common
    )
    callback = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(entries, exits, np.full(1, -1), np.full(1, -1), 3),
        engine="numba",
        **common,
    )
    assert rust.orders.records["size"].tolist() == [10.0, 4.0]
    assert (rust.trades.records.status == 0).any()
    assert callback.orders.records["size"].tolist() == [10.0, 4.0, 6.0]
    assert callback.orders.records.idx.tolist() == [0, 3, 4]


@pytest.mark.parametrize("failure", ["partial", "rejected"])
def test_synthetic_rust_rerun_retries_unfilled_timed_exit(failure):
    """Probe retries in two toy cases; no production implementation is implied."""
    close = pd.DataFrame({"stock": [100.0] * 7})
    entries = np.zeros((7, 1), dtype=np.bool_)
    entries[0, 0] = True
    exits = np.zeros_like(entries)
    max_size = np.full((7, 1), np.inf)
    min_size = np.zeros((7, 1))
    if failure == "partial":
        max_size[3, 0] = 4.0
    else:
        min_size[3, 0] = 20.0
    common = dict(
        init_cash=10000.0,
        size=10.0,
        size_type="amount",
        max_size=max_size,
        min_size=min_size,
        fees=0.0,
        slippage=0.0,
        freq="1D",
    )
    reference = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=_position_time_signal,
        signal_args=(entries, exits, np.full(1, -1), np.full(1, -1), 3),
        engine="numba",
        **common,
    )
    forced = np.zeros_like(entries)
    for _ in range(entries.size + 1):
        portfolio = vbt.Portfolio.from_signals(
            close, entries=entries & ~forced, exits=exits | forced, engine="rust", **common
        )
        due = []
        for trade in portfolio.trades.records.itertuples():
            start = int(trade.entry_idx) + 3
            stop = int(trade.exit_idx) + (1 if trade.status == 0 else 0)
            due.extend(
                (row, int(trade.col))
                for row in range(start, stop)
                if not forced[row, int(trade.col)]
            )
        if not due:
            break
        row, col = min(due)
        forced[row, col] = True
    else:
        pytest.fail("Timed-exit reruns exceeded the iteration cap")
    columns = ["idx", "col", "side", "size", "price", "fees"]
    pd.testing.assert_frame_equal(
        portfolio.orders.records[columns],
        reference.orders.records[columns],
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    assert portfolio.orders.records.idx.iloc[-1] == 4


def test_synthetic_rust_reruns_batch_independent_cash_groups():
    """Probe one pending event per cash group; this is not production validation."""
    batched = False
    for seed in range(8):
        rng = np.random.default_rng(seed)
        rows, groups, per_group, limit = 65, 3, 3, 4
        cols = groups * per_group
        labels = np.repeat(np.arange(groups), per_group)
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.04, (rows, cols)), axis=0)))
        opened = close.shift(1).fillna(close.iloc[0])
        high = pd.DataFrame(np.maximum(opened, close) * 1.01)
        low = pd.DataFrame(np.minimum(opened, close) * 0.99)
        entries = rng.random((rows, cols)) < 0.2
        exits = rng.random((rows, cols)) < 0.08
        max_size = np.full((rows, cols), np.inf)
        max_size[rng.random((rows, cols)) < 0.2] = 4.0
        min_size = np.zeros((rows, cols))
        min_size[rng.random((rows, cols)) < 0.1] = 15.0
        common = dict(
            init_cash=10000.0,
            cash_sharing=True,
            group_by=labels,
            size=10.0,
            size_type="amount",
            max_size=max_size,
            min_size=min_size,
            fees=0.001,
            slippage=0.001,
            sl_stop=0.05,
            open=opened,
            high=high,
            low=low,
            freq="1D",
        )
        reference = vbt.Portfolio.from_signals(
            close,
            signal_func_nb=_position_time_signal,
            signal_args=(entries, exits, np.full(cols, -1), np.full(cols, -1), limit),
            engine="numba",
            **common,
        )
        forced = np.zeros_like(entries)
        for _ in range(entries.size + 1):
            portfolio = vbt.Portfolio.from_signals(
                close, entries=entries & ~forced, exits=exits | forced, engine="rust", **common
            )
            earliest = {}
            for trade in portfolio.trades.records.itertuples():
                start = int(trade.entry_idx) + limit
                stop = int(trade.exit_idx) + (1 if trade.status == 0 else 0)
                for row in range(start, stop):
                    col = int(trade.col)
                    if not forced[row, col]:
                        group = int(labels[col])
                        earliest[group] = min(earliest.get(group, (row, col)), (row, col))
                        break
            if not earliest:
                break
            batched |= len(earliest) > 1
            for row, col in earliest.values():
                forced[row, col] = True
        else:
            pytest.fail("Timed-exit reruns exceeded the iteration cap")
        columns = ["idx", "col", "side", "size", "price", "fees"]
        actual = portfolio.orders.records[columns].reset_index(drop=True)
        expected = reference.orders.records[columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-10)
    assert batched
