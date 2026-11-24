"""Tests for the tax-ledger execution of the monthly replay and its SPY twin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ledger_run import BookResult, TaxParams, _Execution, run_book, run_spy
from replay import TargetStep, exit_close, plan_trades, span_end_dates

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
    kwargs.setdefault("successors", frozenset())
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
    exits = span_end_dates(close, spans)
    dead = run(close, steps, exits=exits, ex=ex1)
    flat = run(close, steps, exits=exits, ex=ex0)
    sold = [t for t in dead.trades if t.reason == "dead_exit"]
    assert [t.symbol for t in sold] == ["D"]
    written_off = -sold[0].qty * (10.0 - sold[0].price)  # dollars lost: qty is negative
    assert dead.n_dead_exits == 1
    assert dead.dead_exit_writeoff == pytest.approx(written_off, abs=1e-6)
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
        exits=span_end_dates(close, spans),
        ex=exit_close(close, spans, 0.0),
    )
    assert result.n_dead_exits == 1
    assert result.dead_exit_writeoff == pytest.approx(0.0, abs=1e-9)
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
    assert result.dead_exit_writeoff == 0.0
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
        successors=frozenset(),
        **kwargs,
    )
    assert real.after_tax_final == pytest.approx(twin.after_tax_final, rel=1e-12)
    assert real.tax_paid > 0.0 and real.cost_paid > 0.0
    assert real.after_tax_final < plain.after_tax_final


def test_ledger_chain_verifies() -> None:
    close, spans, ex, steps = dying_fixture(1.0)
    result = run(close, steps, exits=span_end_dates(close, spans), ex=ex, cost=0.001)
    assert result.chain_verified is True


def test_all_loser_book_pays_no_tax_and_loses_money() -> None:
    """Only shows zero tax on losers (the wash and carry paths have their own tests)."""
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
    assert result.unpaid_tax_left == 0.0
    assert result.after_tax_final < CASH
    # With no tax paid, the after-tax result is the pre-tax one.
    assert result.after_tax_final == pytest.approx(result.pretax_equity.iloc[-1], abs=1e-6)


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
        exits=span_end_dates(close, spans),
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
        successors=frozenset(),
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
        successors=frozenset(),
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


def test_last_day_target_is_not_bought_then_sold() -> None:
    days = pd.bdate_range("2019-01-02", "2019-06-28")
    close = frame(days, A=10.0, B=20.0)
    base = [step("2019-01-02", {"A": 1.0})]
    last = str(days[-1].date())
    with_last = run(close, [*base, step(last, {"B": 1.0})], cost=0.001)
    without = run(close, base, cost=0.001)
    assert len(with_last.trades) == len(without.trades)
    assert with_last.cost_paid == pytest.approx(without.cost_paid, abs=1e-9)
    assert with_last.after_tax_final == pytest.approx(without.after_tax_final, abs=1e-9)
    assert "B" not in {t.symbol for t in with_last.trades}
    np.testing.assert_allclose(
        with_last.pretax_equity.to_numpy(), without.pretax_equity.to_numpy(), atol=1e-9
    )


def test_window_of_fewer_than_two_trading_days_is_refused() -> None:
    days = pd.bdate_range("2019-01-02", "2019-01-31")
    close = frame(days, A=10.0)
    for end in ("2019-01-02", "2018-12-31"):
        with pytest.raises(ValueError, match="two trading days"):
            run_book(
                close,
                [],
                empty_exits(),
                close,
                params=TAX,
                one_way_cost=0.0,
                start="2019-01-02",
                end=end,
                successors=frozenset(),
            )
        with pytest.raises(ValueError, match="two trading days"):
            run_spy(close["A"], params=TAX, one_way_cost=0.0, start="2019-01-02", end=end)


def two_endings(drop_cell: bool = False) -> tuple:
    """Dead stock D and bought-out stock S, both held, both ending in mid-June."""
    days = pd.bdate_range("2019-01-02", "2019-12-31")
    gone = days > "2019-06-14"
    close = frame(days, D=np.where(gone, np.nan, 10.0), S=np.where(gone, np.nan, 50.0), A=20.0)
    spans = pd.DataFrame(
        {
            "symbol": ["D", "S"],
            "last_date": [pd.Timestamp("2019-06-14")] * 2,
            "has_successor": [False, True],
        }
    )
    ex = exit_close(close, spans, 1.0)
    if drop_cell:
        ex.loc["2019-06-14", "D"] = np.nan
    steps = [step("2019-01-02", {"D": 0.3, "S": 0.3, "A": 0.4})]
    return close, spans, ex, steps


def test_successors_is_required_and_must_be_known_and_only_dead_are_counted() -> None:
    close, spans, ex, steps = two_endings()
    ends = span_end_dates(close, spans)
    kwargs = dict(params=TAX, one_way_cost=0.0, start="2019-01-02", end="2019-12-31")
    with pytest.raises(TypeError):
        run_book(close, steps, ends, ex, **kwargs)
    with pytest.raises(ValueError, match="X1"):
        run_book(close, steps, ends, ex, successors=frozenset({"S", "X1"}), **kwargs)
    ok = run_book(close, steps, ends, ex, successors=frozenset({"S"}), **kwargs)
    assert ok.n_dead_exits == 1  # D only
    assert ok.dead_exit_writeoff == pytest.approx(0.3 * CASH * (1 - SIM_PRICE_FLOOR), abs=1e-6)
    sold = {t.symbol: t.price for t in ok.trades if t.reason == "dead_exit"}
    assert sold["S"] == 50.0 and sold["D"] == pytest.approx(10.0 * SIM_PRICE_FLOOR)
    assert run_book(close, steps, ends, ex, successors=frozenset(), **kwargs).n_dead_exits == 2


def test_missing_exit_price_of_a_held_stock_raises_with_symbol_and_date() -> None:
    close, spans, ex, steps = two_endings(drop_cell=True)
    with pytest.raises(ValueError, match=r"D on 2019-06-14"):
        run(close, steps, exits=span_end_dates(close, spans), ex=ex)
    # A stock that is not held on its exit date needs no exit price.
    run(close, [step("2019-01-02", {"A": 1.0})], exits=span_end_dates(close, spans), ex=ex)


def assessed_total(result: BookResult) -> float:
    """Sum of every tax assessment and later correction the book recorded."""
    return sum(
        float(e.payload["amount"])
        for e in result.events
        if e.kind in ("tax_assessment", "tax_adjustment")
    )


def test_funding_sale_is_pro_rata_across_positions_and_taxed() -> None:
    days = pd.bdate_range("2019-01-02", "2020-03-31")
    a = np.where(days >= "2019-09-03", 30.0, np.where(days >= "2019-03-01", 20.0, 10.0))
    b = np.where(days >= "2019-09-03", 20.0, np.where(days >= "2019-03-01", 15.0, 10.0))
    c = np.where(days >= "2019-03-01", 20.0, 10.0)
    close = frame(days, A=a, B=b, C=c)
    steps = [
        step("2019-01-02", {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}),
        step("2019-06-03", {"A": 0.5, "B": 0.5}),  # C's gain is realised and reinvested
    ]
    result = run(close, steps)
    qty = {}  # shares bought per (symbol, date)
    for t in result.trades:
        if t.reason == "rebalance" and t.qty > 0:
            qty[(t.symbol, t.date.strftime("%Y-%m-%d"))] = t.qty
    funding = [t for t in result.trades if t.reason == "tax_sale"]
    assert {t.symbol for t in funding} == {"A", "B"}
    sold = {t.symbol: -t.qty for t in funding}
    held_a = qty[("A", "2019-01-02")] + qty[("A", "2019-06-03")]
    held_b = qty[("B", "2019-01-02")] + qty[("B", "2019-06-03")]
    assert sold["A"] / held_a == pytest.approx(sold["B"] / held_b, rel=1e-6)
    # HIFO takes the June lots first; the sale is smaller than each June lot.
    assert sold["A"] < qty[("A", "2019-06-03")] and sold["B"] < qty[("B", "2019-06-03")]
    # The funding sales went through the book and realised a gain in 2020.
    realised = sum(
        float(e.payload["gain"])
        for e in result.events
        if e.kind == "sell" and e.date == funding[0].date
    )
    assert realised == pytest.approx(sold["A"] * (30.0 - 20.0) + sold["B"] * (20.0 - 15.0))
    # Hand figure: 2019 tax on C's gain, 2020 tax on the June lots (short term) and the January
    # lots (long term at the final sale, after more than a year).
    c_gain = qty[("C", "2019-01-02")] * 10.0
    short = qty[("A", "2019-06-03")] * 10.0 + qty[("B", "2019-06-03")] * 5.0
    long_ = qty[("A", "2019-01-02")] * 20.0 + qty[("B", "2019-01-02")] * 10.0
    expected = 0.24 * c_gain + 0.24 * short + 0.15 * long_
    assert assessed_total(result) == pytest.approx(expected, rel=1e-9)
    assert result.tax_paid + result.unpaid_tax_left == pytest.approx(assessed_total(result))
    assert result.unpaid_tax_left == 0.0
    assert result.min_cash >= -1e-9


def test_cash_short_after_three_rounds_leaves_unpaid_tax_and_cash_floor() -> None:
    days = pd.bdate_range("2019-01-02", "2020-03-31")
    c = np.where(days >= "2019-03-01", 20.0, 10.0)
    d = np.where(days >= "2019-12-02", 0.01, 10.0)  # the reinvested money collapses
    close = frame(days, C=c, D=d)
    steps = [step("2019-01-02", {"C": 1.0}), step("2019-06-03", {"D": 1.0})]
    result = run(close, steps)
    tax_2019 = 0.24 * CASH  # C doubled: 100,000 gain, realised in June
    assert assessed_total(result) == pytest.approx(tax_2019, rel=1e-9)
    assert result.unpaid_tax_left > 0.9 * tax_2019
    assert 0.0 < result.tax_paid < 0.1 * tax_2019
    assert result.tax_paid + result.unpaid_tax_left == pytest.approx(tax_2019, rel=1e-9)
    assert result.min_cash >= -1e-9
    sold = [t for t in result.trades if t.reason == "tax_sale"]
    assert [t.symbol for t in sold] == ["D"]  # everything that could be sold was sold
    assert result.chain_verified


def test_wash_sale_cross_year_disallows_loss() -> None:
    days = pd.bdate_range("2012-01-02", "2013-03-28")
    x = np.where(days >= "2012-06-01", 200.0, 100.0)
    w = np.where(days >= "2012-06-01", 8.0, 10.0)
    close = frame(days, X=x, W=w)
    steps = [
        step("2012-01-03", {"X": 0.5, "W": 0.5}),
        step("2012-12-20", {}),  # sell everything: X gains 50,000, W loses 10,000
        step("2013-01-10", {"W": 1.0}),  # re-buy W within 30 days, in the next year
    ]
    result = run(close, steps)
    # The 2012 tax was assessed on 2013-01-02 with the loss allowed (9,600), then corrected up.
    assert result.tax_paid == pytest.approx(0.24 * 50_000.0, abs=1e-6)  # loss disallowed
    assert result.tax_paid > 0.24 * 40_000.0 + 1.0  # not the 9,600 the loss would give
    assert result.unpaid_tax_left == 0.0
    adjust = [e for e in result.events if e.kind == "wash_adjustment"]
    assert len(adjust) == 1
    assert float(adjust[0].payload["disallowed"]) == pytest.approx(10_000.0, abs=1e-6)
    assert float(adjust[0].payload["basis_per_share"]) == pytest.approx(8.0 + 2.0, abs=1e-9)
    amended = [e for e in result.events if e.kind == "tax_adjustment"]
    assert [int(e.payload["year"]) for e in amended] == [2012]
    assert float(amended[0].payload["amount"]) == pytest.approx(0.24 * 10_000.0, abs=1e-6)


# Seeds 86 and 117 overspent cash by about 1e-11 before the clamp existed.
@pytest.mark.parametrize("seed", [*range(10), 86, 117])
def test_scaled_buys_never_exceed_cash_by_rounding(seed: int) -> None:
    """Many names, odd prices, cost 0.00064: cash binds and rounding must not overspend."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-02", "2019-12-31")
    n = 60
    walk = np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(len(days), n)), axis=0))
    close = pd.DataFrame(
        rng.uniform(3.0, 400.0, size=n) * walk,
        index=days,
        columns=[f"S{i:02d}" for i in range(n)],
    )
    steps = []
    for when in ("2019-01-02", "2019-03-01", "2019-05-01", "2019-07-01", "2019-09-03"):
        names = rng.choice(close.columns, size=50, replace=False)
        weights = rng.dirichlet(np.ones(50))
        steps.append(step(when, {str(s): float(w) for s, w in zip(names, weights, strict=True)}))
    result = run(close, steps, cost=0.00064)
    assert result.min_cash >= 0.0
    assert result.chain_verified
    assert len([t for t in result.trades if t.reason == "rebalance" and t.qty > 0]) >= 50


def make_execution(cash: float) -> _Execution:
    columns = pd.Index(["A"])
    return _Execution(columns, TAX, 0.001, cash, frozenset(), {})


def test_buy_clamps_a_rounding_sized_excess_to_cash() -> None:
    run_ = make_execution(1_000.0)
    fill = 47.73431396484375 * 1.001
    qty = 1_000.0 / fill * (1.0 + 3e-12)  # asks for a hair more than the cash allows
    run_.buy("A", qty, 47.73431396484375, pd.Timestamp("2019-01-02"))
    assert run_.book.cash >= 0.0
    assert run_.book.cash < 1e-9
    assert run_.trades[0].qty <= 1_000.0 / fill


def test_buy_far_over_cash_is_a_logic_error_not_clamped() -> None:
    run_ = make_execution(1_000.0)
    with pytest.raises(ValueError, match=r"A.*2019-01-02.*cash"):
        run_.buy("A", 100.0, 10.0, pd.Timestamp("2019-01-02"))
    assert run_.trades == [] and run_.book.cash == 1_000.0


def test_revised_tax_is_paid_the_day_a_january_rebuy_raises_it() -> None:
    days = pd.bdate_range("2012-01-02", "2013-03-28")
    x = np.where(days >= "2012-06-01", 200.0, 100.0)
    w = np.where(days >= "2012-06-01", 8.0, 10.0)
    close = frame(days, X=x, W=w)
    steps = [
        step("2012-01-03", {"X": 0.5, "W": 0.5}),
        step("2012-12-20", {}),  # X gains 50,000, W loses 10,000
        step("2013-01-10", {"W": 1.0}),  # re-buy W within 30 days: the loss is disallowed
    ]
    result = run(close, steps)
    rebuy_day = pd.Timestamp("2013-01-10")
    raised = [e for e in result.events if e.kind == "tax_adjustment" and e.date == rebuy_day]
    assert len(raised) == 1
    assert int(raised[0].payload["year"]) == 2012
    assert float(raised[0].payload["amount"]) == pytest.approx(0.24 * 10_000.0, abs=1e-6)
    assert float(raised[0].payload["revised_tax"]) > 0.24 * 40_000.0  # was 9,600 when assessed
    # The extra tax is paid that same day: by the end of the 10th, payments equal assessments.
    kinds = ("tax_assessment", "tax_adjustment")
    assessed = sum(
        float(e.payload["amount"]) for e in result.events if e.kind in kinds and e.date <= rebuy_day
    )
    paid = sum(
        float(e.payload["amount"])
        for e in result.events
        if e.kind == "tax_payment" and e.date <= rebuy_day
    )
    assert paid == pytest.approx(assessed, abs=1e-6)
    assert paid == pytest.approx(0.24 * 50_000.0, abs=1e-6)
    assert any(e.kind == "tax_payment" and e.date == rebuy_day for e in result.events)
    assert result.min_cash >= -1e-9
    assert result.unpaid_tax_left == 0.0 and result.unpaid_tax_flag is False


def test_account_value_curve_ends_at_after_tax_final_and_matches_pretax_without_tax() -> None:
    close, spans, ex, steps = dying_fixture(1.0)
    ends = span_end_dates(close, spans)
    taxed = run(close, steps, exits=ends, ex=ex, cost=0.001)
    assert taxed.account_value.iloc[-1] == pytest.approx(taxed.after_tax_final, abs=1e-6)
    gap = taxed.pretax_equity - taxed.account_value
    assert gap.iloc[-1] == pytest.approx(taxed.tax_paid, abs=1e-6)
    assert (gap >= -1e-9).all()
    free = run(close, steps, exits=ends, ex=ex, cost=0.001, params=NO_TAX)
    np.testing.assert_allclose(
        free.account_value.to_numpy(), free.pretax_equity.to_numpy(), atol=1e-9
    )
    assert free.account_value.index.equals(free.pretax_equity.index)


def test_unpaid_tax_at_the_end_is_netted_and_flagged() -> None:
    days = pd.bdate_range("2019-01-02", "2020-03-31")
    c = np.where(days >= "2019-03-01", 20.0, 10.0)
    d = np.where(days >= "2019-12-02", 0.01, 10.0)  # the reinvested money collapses
    close = frame(days, C=c, D=d)
    steps = [step("2019-01-02", {"C": 1.0}), step("2019-06-03", {"D": 1.0})]
    result = run(close, steps)
    assert result.unpaid_tax_left > 0.9 * 0.24 * CASH
    assert result.unpaid_tax_flag is True
    # Nothing is left but the bill: the final value is the cash (zero) minus the unpaid tax.
    assert result.after_tax_final == pytest.approx(-result.unpaid_tax_left, abs=1e-6)
    assert result.after_tax_final < 0.0
    assert result.account_value.iloc[-1] == pytest.approx(result.after_tax_final, abs=1e-6)
    assert result.cagr_after_tax == -1.0
    paid = run(close, [step("2019-01-02", {"C": 1.0})])
    assert paid.unpaid_tax_flag is False
