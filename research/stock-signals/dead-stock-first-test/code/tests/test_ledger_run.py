"""Tests for the tax-ledger execution of the monthly replay and its SPY twin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ledger_run import BookResult, TaxParams, run_book, run_spy
from replay import TargetStep, dead_exit_dates, exit_close, plan_trades, span_end_dates

from ggTrader.data.bars import Bars
from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.lab.dead_stocks import SIM_PRICE_FLOOR
from ggTrader.lab.strategies.contract import ParamGrid
from ggTrader.lab.strategies.xs_momentum import XsMomentum

pytestmark = pytest.mark.study

TAX = TaxParams(st_rate=0.24, lt_rate=0.15)
NO_TAX = TaxParams(0.0, 0.0)
CASH = 100_000.0


def frame(days: pd.DatetimeIndex, **paths: object) -> pd.DataFrame:
    """Closes from per-symbol arrays or scalars (a scalar is a flat price)."""
    cols = {
        name: (np.full(len(days), float(p)) if np.isscalar(p) else np.asarray(p, dtype=float))
        for name, p in paths.items()
    }
    return pd.DataFrame(cols, index=days)


def step(date: str, targets: dict[str, float]) -> TargetStep:
    stamp = pd.Timestamp(date)
    return TargetStep(decision_date=stamp, date=stamp, targets=targets)


def empty_exits() -> pd.Series:
    return pd.Series([], index=pd.Index([], dtype=object), dtype="datetime64[ns]")


def run(close, steps, *, exits=None, ex=None, params=TAX, cost=0.0, **kwargs) -> BookResult:
    return run_book(
        close,
        steps,
        empty_exits() if exits is None else exits,
        close if ex is None else ex,
        params=params,
        one_way_cost=cost,
        start=str(close.index[0].date()),
        end=str(close.index[-1].date()),
        start_cash=kwargs.pop("start_cash", CASH),
        **kwargs,
    )


def test_single_winner_taxed_at_long_term_rate() -> None:
    days = pd.bdate_range("2019-01-02", "2020-02-10")  # last day is 404 days after the buy
    price = np.where(days >= "2019-06-03", 20.0, 10.0)
    result = run(frame(days, A=price), [step("2019-01-02", {"A": 1.0})])
    gain = CASH  # the stock doubled on all of the money
    assert result.tax_paid == pytest.approx(0.15 * gain, abs=1e-6)
    assert result.after_tax_final == pytest.approx(CASH + gain - 0.15 * gain, abs=1e-6)
    assert result.unpaid_tax_left == 0.0


def test_short_term_gain_taxed_at_short_rate() -> None:
    days = pd.bdate_range("2019-01-02", "2019-06-28")
    price = np.where(days >= "2019-03-01", 20.0, 10.0)
    result = run(frame(days, A=price), [step("2019-01-02", {"A": 1.0})])
    assert result.tax_paid == pytest.approx(0.24 * CASH, abs=1e-6)
    assert result.after_tax_final == pytest.approx(CASH + CASH * (1 - 0.24), abs=1e-6)


def test_costs_reduce_pretax_value_by_exact_amount() -> None:
    days = pd.bdate_range("2019-01-02", "2019-03-29")
    cost = 0.0015
    result = run(frame(days, A=10.0), [step("2019-01-02", {"A": 1.0})], params=NO_TAX, cost=cost)
    traded = CASH / (1.0 + cost)  # the buy spends all cash, so the traded value is this
    assert result.cost_paid == pytest.approx(2 * cost * traded, abs=1e-6)
    assert result.after_tax_final == pytest.approx(CASH - 0.003 * traded, abs=1e-6)
    assert result.pretax_equity.iloc[-1] == pytest.approx(CASH - 0.003 * traded, abs=1e-6)


def dying_fixture(loss: float, successor: bool = False) -> tuple:
    """X doubles-and-more and is sold in 2012; D dies mid-2012. Both held from January."""
    days = pd.bdate_range("2011-12-01", "2013-03-29")
    x = np.where(days >= "2012-06-01", 300.0, 100.0)
    d = np.where(days <= "2012-06-15", 10.0, np.nan)
    close = frame(days, X=x, D=d)
    spans = pd.DataFrame(
        {
            "symbol": ["D"],
            "last_date": [pd.Timestamp("2012-06-15")],
            "has_successor": [successor],
        }
    )
    ex = exit_close(close, spans, loss)
    steps = [step("2012-01-03", {"X": 0.5, "D": 0.5}), step("2012-09-04", {})]
    return close, spans, ex, steps


def test_dead_exit_loss_reaches_ledger_in_death_year() -> None:
    close, spans, ex1, steps = dying_fixture(1.0)
    ex0 = exit_close(close, spans, 0.0)
    exits = dead_exit_dates(close, spans)
    dead = run(close, steps, exits=exits, ex=ex1)
    flat = run(close, steps, exits=exits, ex=ex0)
    sold = [t for t in dead.trades if t.reason == "dead_exit"]
    assert [t.symbol for t in sold] == ["D"]
    written_off = -sold[0].qty * (10.0 - sold[0].price)  # dollars lost: qty is negative
    assert dead.n_dead_exits == 1
    assert dead.dead_exit_loss_realised == pytest.approx(written_off, abs=1e-6)
    # X gain 100,000 and D loss 50,000 land in 2012: the tax falls by rate x loss.
    assert flat.tax_paid == pytest.approx(0.24 * 100_000.0, abs=1e-6)
    assert dead.tax_paid == pytest.approx(0.24 * (100_000.0 - written_off), abs=1e-6)
    assert flat.tax_paid - dead.tax_paid == pytest.approx(0.24 * written_off, abs=1e-6)


def test_dead_exit_loss_zero_realises_nothing_extra() -> None:
    days = pd.bdate_range("2019-01-02", "2019-12-31")
    d = np.where(days <= "2019-06-14", 10.0, np.nan)
    close = frame(days, D=d)
    spans = pd.DataFrame({"symbol": ["D"], "last_date": [pd.Timestamp("2019-06-14")]})
    result = run(
        close,
        [step("2019-01-02", {"D": 1.0})],
        exits=dead_exit_dates(close, spans),
        ex=exit_close(close, spans, 0.0),
    )
    assert result.n_dead_exits == 1
    assert result.dead_exit_loss_realised == pytest.approx(0.0, abs=1e-9)
    assert result.tax_paid == 0.0
    assert result.after_tax_final == pytest.approx(CASH, abs=1e-6)


def test_successor_stock_sold_on_last_bar_at_unchanged_price() -> None:
    days = pd.bdate_range("2019-01-02", "2019-12-31")
    s = np.where(days <= "2019-06-14", 50.0, np.nan)
    close = frame(days, S=s)
    spans = pd.DataFrame(
        {"symbol": ["S"], "last_date": [pd.Timestamp("2019-06-14")], "has_successor": [True]}
    )
    result = run(
        close,
        [step("2019-01-02", {"S": 1.0})],
        exits=span_end_dates(close, spans),
        ex=exit_close(close, spans, 1.0),
        successors=frozenset({"S"}),
    )
    sold = [t for t in result.trades if t.reason == "dead_exit"]
    assert len(sold) == 1
    assert sold[0].date == pd.Timestamp("2019-06-14")
    assert sold[0].price == 50.0
    assert result.n_dead_exits == 0
    assert result.dead_exit_loss_realised == 0.0
    assert result.after_tax_final == pytest.approx(CASH, abs=1e-6)


def test_spy_twin_uses_same_costs_and_rates() -> None:
    days = pd.bdate_range("2019-01-02", "2020-06-30")
    spy = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.4, len(days))), index=days, name="SPY")
    kwargs = dict(start=str(days[0].date()), end=str(days[-1].date()), start_cash=CASH)
    plain = run_spy(spy, params=NO_TAX, one_way_cost=0.0, **kwargs)
    assert plain.after_tax_final == pytest.approx(CASH * spy.iloc[-1] / spy.iloc[0], rel=1e-12)
    assert plain.tax_paid == 0.0
    real = run_spy(spy, params=TAX, one_way_cost=0.0015, **kwargs)
    twin = run_book(
        spy.to_frame(),
        [step(str(days[0].date()), {"SPY": 1.0})],
        empty_exits(),
        spy.to_frame(),
        params=TAX,
        one_way_cost=0.0015,
        **kwargs,
    )
    assert real.after_tax_final == pytest.approx(twin.after_tax_final, rel=1e-12)
    assert real.tax_paid > 0.0 and real.cost_paid > 0.0
    assert real.after_tax_final < plain.after_tax_final


def test_ledger_chain_verifies() -> None:
    close, spans, ex, steps = dying_fixture(1.0)
    result = run(close, steps, exits=dead_exit_dates(close, spans), ex=ex, cost=0.001)
    assert result.chain_verified is True


def test_taxes_never_exceed_gains_when_all_losers() -> None:
    days = pd.bdate_range("2019-01-02", "2020-06-30")
    down = 100.0 * np.exp(-np.linspace(0.0, 0.6, len(days)))
    close = frame(days, A=down, B=down * 0.5)
    steps = [
        step("2019-01-02", {"A": 1.0}),
        step("2019-07-01", {"B": 1.0}),
        step("2020-01-02", {"A": 0.5, "B": 0.5}),
    ]
    result = run(close, steps, cost=0.001)
    assert result.tax_paid == 0.0
    assert result.after_tax_final < CASH


def test_year_start_tax_sells_pro_rata_when_cash_is_short() -> None:
    days = pd.bdate_range("2019-01-02", "2020-03-31")
    a = np.where(days >= "2019-03-01", 20.0, 10.0)
    close = frame(days, A=a, B=10.0)
    steps = [step("2019-01-02", {"A": 1.0}), step("2019-06-03", {"B": 1.0})]
    result = run(close, steps)
    assert result.tax_paid == pytest.approx(0.24 * CASH, abs=1e-6)
    assert result.unpaid_tax_left == 0.0
    reasons = [t.reason for t in result.trades]
    assert "tax_sale" in reasons
    assert result.after_tax_final == pytest.approx(CASH + CASH - 0.24 * CASH, abs=1e-6)


def test_unfillable_buy_is_skipped_and_cash_kept() -> None:
    days = pd.bdate_range("2019-01-02", "2019-04-30")
    b = np.where(days == "2019-02-01", np.nan, 20.0)
    close = frame(days, A=10.0, B=b)
    result = run(close, [step("2019-02-01", {"A": 0.5, "B": 0.5})], params=NO_TAX)
    assert [t.symbol for t in result.trades if t.reason == "rebalance"] == ["A"]
    assert result.skipped_buys == 1
    held_value = 0.5 * CASH
    assert result.pretax_equity.loc["2019-02-04"] == pytest.approx(CASH, abs=1e-6)
    assert [t.qty for t in result.trades if t.symbol == "A"][0] == pytest.approx(held_value / 10.0)


def test_buy_of_a_stock_whose_series_already_ended_is_skipped() -> None:
    days = pd.bdate_range("2019-01-02", "2019-04-30")
    d = np.where(days <= "2019-02-01", 10.0, np.nan)
    close = frame(days, A=10.0, D=d)
    spans = pd.DataFrame({"symbol": ["D"], "last_date": [pd.Timestamp("2019-02-01")]})
    result = run(
        close,
        [step("2019-02-01", {"A": 0.5, "D": 0.5})],
        exits=dead_exit_dates(close, spans),
        ex=exit_close(close, spans, 1.0),
        params=NO_TAX,
    )
    assert "D" not in {t.symbol for t in result.trades}
    assert result.skipped_buys == 1


def test_halted_held_stock_valued_at_last_print() -> None:
    days = pd.bdate_range("2019-01-02", "2019-05-31")
    h = np.where((days >= "2019-03-01") & (days <= "2019-03-29"), np.nan, 10.0)
    close = frame(days, H=h, A=20.0)
    targets = {"H": 0.5, "A": 0.5}
    steps = [step("2019-01-02", targets), step("2019-03-15", targets)]
    result = run(close, steps, params=NO_TAX)
    traded = [t for t in result.trades if t.reason != "liquidation"]
    assert {t.date for t in traded} == {pd.Timestamp("2019-01-02")}
    halted = result.pretax_equity.loc["2019-03-01":"2019-03-29"]
    assert len(halted) > 10
    np.testing.assert_allclose(halted.to_numpy(), CASH, atol=1e-6)


def random_close(n: int = 30, seed: int = 7) -> pd.DataFrame:
    days = pd.bdate_range("2018-01-01", "2019-12-31")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, size=(len(days), n))
    prices = 50.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=days, columns=[f"S{i:02d}" for i in range(n)])


def no_spans() -> pd.DataFrame:
    return pd.DataFrame({"symbol": [], "last_date": pd.to_datetime([])})


def test_cash_never_negative() -> None:
    close = random_close()
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)
    steps = plan_trades(
        close,
        eligible,
        no_spans(),
        top_n=5,
        lookback=40,
        skip=5,
        dead_loss=1.0,
        start="2019-01-02",
        end="2019-12-31",
    )
    result = run_book(
        close,
        steps,
        empty_exits(),
        close,
        params=TAX,
        one_way_cost=0.02,
        start="2019-01-02",
        end="2019-12-31",
    )
    assert len(steps) >= 10
    assert result.min_cash >= -1e-9
    assert result.chain_verified


def test_matches_engine_without_dead_stocks() -> None:
    """Replay with no cost and no tax equals the engine's weight simulation."""
    close = random_close()
    start, end = "2019-01-02", "2019-12-31"
    lookback, skip, top_n = 40, 5, 5
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)
    steps = plan_trades(
        close,
        eligible,
        no_spans(),
        top_n=top_n,
        lookback=lookback,
        skip=skip,
        dead_loss=1.0,
        start=start,
        end=end,
    )
    replay = run_book(
        close,
        steps,
        empty_exits(),
        close,
        params=NO_TAX,
        one_way_cost=0.0,
        start=start,
        end=end,
        start_cash=CASH,
    )
    utc = close.copy()
    utc.index = utc.index.tz_localize("UTC")
    bars = Bars(
        close=utc,
        open=utc,
        high=utc,
        low=utc,
        volume=utc * 0.0 + 1000.0,
        requested=tuple(utc.columns),
        missing_symbols=(),
        last_valid_bar=utc.apply(lambda s: s.last_valid_index()),
    )
    rebal = pd.DatetimeIndex([s.decision_date for s in steps]).tz_localize("UTC")
    grid = ParamGrid.single({"top_n": top_n, "lookback": lookback, "skip": skip})
    weights = XsMomentum().build(bars, grid, rebal=rebal)
    weights.columns = weights.columns.get_level_values(-1)
    sim = engine_portfolio.run_weights(
        weights,
        utc[weights.columns],
        {"START_CASH": CASH, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D"},
    )
    engine = sim.value.iloc[:, 0]
    engine.index = engine.index.tz_convert(None)
    first_exec = steps[0].date
    month_ends = [s.decision_date for s in steps if s.decision_date >= first_exec]
    month_ends += [steps[-1].date]
    assert len(month_ends) >= 10
    mine = replay.pretax_equity
    for stamp in month_ends:
        assert mine.loc[stamp] == pytest.approx(engine.loc[stamp], rel=1e-6), stamp
    after = mine.index >= first_exec
    np.testing.assert_allclose(
        mine[after].to_numpy(), engine.reindex(mine.index)[after].to_numpy(), rtol=1e-6
    )


def test_floor_constant_is_the_lab_floor() -> None:
    close, spans, ex, _ = dying_fixture(1.0)
    last = pd.Timestamp("2012-06-15")
    assert ex.loc[last, "D"] == pytest.approx(10.0 * SIM_PRICE_FLOOR)
