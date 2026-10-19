"""Synthetic portfolio tests for the production-rule replica."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.replica import REASON_DELISTED, ReplicaConfig, run_replica
from ggTrader.paper.cash_sweep import SWEEP_TRADE_REASON
from ggTrader.paper.catastrophe_stop import CATASTROPHE_STOP_REASON
from ggTrader.paper.risk import RiskConfig

C4 = {s: [10.0] * 5 for s in ["DDD", "AAA", "CCC", "BBB"]}


def _panel(
    prices: dict[str, list[float]],
    spy: list[float],
    entries: dict[str, list[int]] | None = None,
    exits: dict[str, list[int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build aligned close and signal fixtures with entries at listed bars."""
    index = pd.bdate_range("2024-01-01", periods=len(spy))
    close = pd.DataFrame(prices, index=index)
    ent = pd.DataFrame(False, index=index, columns=close.columns)
    ext = ent.copy()
    for frame, signals in [(ent, entries), (ext, exits)]:
        for symbol, bars in (signals or {}).items():
            frame.loc[index[bars], symbol] = True
    return close, ent, ext, pd.Series(spy, index=index)


def test_buys_fill_alphabetically_up_to_three_slots():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy)
    buys = res.trades.query("reason == 'signal' and side == 'buy'")
    assert list(buys.symbol) == ["AAA", "BBB", "CCC"]
    assert (buys.date == close.index[2]).all()  # signal bar 1, filled bar 2
    assert res.truncated_days == 1


def test_lag_zero_fills_on_signal_bar():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy, ReplicaConfig(signal_lag_bars=0))
    assert (res.trades.query("reason == 'signal'").date == close.index[1]).all()


def test_position_is_3_3pct_of_nav():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy)
    first = res.trades.query("reason == 'signal'").iloc[0]
    assert first.notional == pytest.approx(0.033 * res.equity.iloc[1], rel=1e-5)  # cents rounding


def test_sweep_buys_down_to_reserve():
    close, ent, ext, spy = _panel({"AAA": [10.0] * 3}, [100.0] * 3)
    res = run_replica(close, ent, ext, spy)
    assert res.exposure.loc[close.index[1], "spy"] == pytest.approx(0.95, abs=1e-3)
    assert res.exposure.loc[close.index[1], "cash"] == pytest.approx(0.05, abs=1e-3)


def test_funding_sell_precedes_buys_and_cash_never_negative():
    close, ent, ext, spy = _panel(C4, [100.0] * 5, entries={s: [1] for s in C4})
    res = run_replica(close, ent, ext, spy)
    day2 = res.trades[res.trades.date == close.index[2]]
    assert (day2.iloc[0].reason, day2.iloc[0].side) == (SWEEP_TRADE_REASON, "sell")
    assert (res.exposure.cash >= -1e-9).all()


def test_exit_signal_sells_on_next_bar():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 6}, [100.0] * 6, entries={"AAA": [1]}, exits={"AAA": [3]}
    )
    res = run_replica(close, ent, ext, spy)
    sells = res.trades.query("reason == 'signal' and side == 'sell'")
    assert list(sells.date) == [close.index[4]]


def test_exits_before_fill_are_ignored_but_fill_bar_exit_counts():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 5}, [100.0] * 5, entries={"AAA": [1]}, exits={"AAA": [0, 1]}
    )
    assert (
        run_replica(close, ent, ext, spy).trades.query("side == 'sell' and symbol == 'AAA'").empty
    )
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 5}, [100.0] * 5, entries={"AAA": [1]}, exits={"AAA": [2]}
    )
    sells = run_replica(close, ent, ext, spy).trades.query("side == 'sell' and symbol == 'AAA'")
    assert list(sells.date) == [close.index[3]]


def test_catastrophe_stop_sells_at_minus_25pct():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0, 10.0, 10.0, 7.4, 7.4]}, [100.0] * 5, entries={"AAA": [1]}
    )
    res = run_replica(close, ent, ext, spy)
    stop = res.trades[res.trades.reason == CATASTROPHE_STOP_REASON]
    assert list(stop.symbol) == ["AAA"] and list(stop.date) == [close.index[3]]


def test_delisted_position_sold_at_last_close():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0, 10.0, 10.0, 11.0, np.nan, np.nan]}, [100.0] * 6, entries={"AAA": [1]}
    )
    res = run_replica(close, ent, ext, spy)
    d = res.trades[res.trades.reason == REASON_DELISTED]
    assert list(d.date) == [close.index[4]]
    assert d.iloc[0].price == pytest.approx(11.0 * (1 - 5e-4))
    assert res.equity.notna().all()


def test_missing_spy_bar_skips_sweep():
    close, ent, ext, spy = _panel({"AAA": [10.0] * 3}, [100.0, np.nan, 100.0])
    res = run_replica(close, ent, ext, spy)
    assert res.trades.query("symbol == 'SPY'").date.min() == close.index[2]
    assert res.equity.notna().all()


def test_drawdown_halt_stops_all_trading():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 7}, [100, 100, 100, 84, 84, 84, 100], entries={"AAA": [3, 4]}
    )
    res = run_replica(close, ent, ext, spy)
    assert res.trades[res.trades.date.isin(close.index[3:6])].empty
    assert res.halted_days == 3


def test_unaffordable_buy_is_skipped_without_partial_fill():
    panel = _panel(
        {"AAA": [10.0] * 5, "BBB": [10.0] * 5},
        [100.0] * 5,
        entries={"AAA": [1], "BBB": [1]},
    )
    cfg = ReplicaConfig(
        start_cash=1000.0,
        min_clip_usd=100_000.0,
        risk=RiskConfig(max_positions=3, position_pct=0.6, max_concentration_pct=1.0),
    )
    res = run_replica(*panel, cfg)
    assert list(res.trades.symbol) == ["AAA"]
    assert res.trades.iloc[0].notional == 600.0
    assert (res.exposure.cash >= 0).all()


def test_daily_halt_recovers_and_catches_up_missed_exit():
    close, ent, ext, spy = _panel(
        {"AAA": [10.0] * 6},
        [100, 100, 100, 100, 96, 96],
        entries={"AAA": [1]},
        exits={"AAA": [3]},
    )
    res = run_replica(close, ent, ext, spy)
    assert res.halted_days == 1
    assert res.trades[res.trades.date == close.index[4]].empty
    sells = res.trades.query("symbol == 'AAA' and side == 'sell'")
    assert list(sells.date) == [close.index[5]]


def test_temporary_missing_stock_close_marks_forward_without_delisting():
    panel = _panel(
        {"AAA": [10.0, 10.0, 10.0, np.nan, 11.0]},
        [100.0] * 5,
        entries={"AAA": [1]},
    )
    res = run_replica(*panel)
    assert res.trades.query("reason == 'delisted'").empty
    assert res.equity.iloc[3] == pytest.approx(res.equity.iloc[2])
    assert res.equity.iloc[4] > res.equity.iloc[3]


def test_cash_interest_starts_on_first_decision_bar():
    panel = _panel({"AAA": [10.0] * 5}, [100.0] * 5)
    rates = pd.Series(0.001, index=panel[0].index)
    cfg = ReplicaConfig(start_cash=1000.0, min_clip_usd=100_000.0)
    res = run_replica(*panel, cfg, cash_rate_daily=rates)
    np.testing.assert_allclose(res.equity, 1000.0 * 1.001 ** np.arange(5))
