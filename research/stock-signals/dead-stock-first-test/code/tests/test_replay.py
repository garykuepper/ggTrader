"""Tests for the monthly replay planner and its dead-stock exits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import replay
from replay import Trade, dead_exit_dates, exit_close, plan_trades, span_end_dates

from ggTrader.data.bars import Bars
from ggTrader.lab.dead_stocks import SIM_PRICE_FLOOR
from ggTrader.lab.strategies.contract import ParamGrid
from ggTrader.lab.strategies.xs_momentum import XsMomentum

pytestmark = pytest.mark.study

DAYS = pd.bdate_range("2018-01-01", "2019-12-31")
START = "2019-01-02"
END = "2019-12-31"
LOOKBACK = 40
SKIP = 5


def make_close(n_symbols: int = 12, seed: int = 0, days: pd.DatetimeIndex = DAYS) -> pd.DataFrame:
    """Random-walk closes, one column per symbol S00, S01, ..."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, size=(len(days), n_symbols))
    prices = 50.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=days, columns=[f"S{i:02d}" for i in range(n_symbols)])


def runaway(close: pd.DataFrame) -> np.ndarray:
    """A path that grows 2% a day: it out-scores every random walk on any date."""
    return 10.0 * np.exp(0.02 * np.arange(len(close)))


def all_eligible(close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=close.index, columns=close.columns)


def make_spans(rows: list[tuple]) -> pd.DataFrame:
    """Spans rows as (symbol, last_date) with the optional has_successor flag."""
    frame = pd.DataFrame(
        rows,
        columns=["symbol", "last_date"] + (["has_successor"] if rows and len(rows[0]) == 3 else []),
    )
    frame["last_date"] = pd.to_datetime(frame["last_date"])
    return frame


def empty_spans() -> pd.DataFrame:
    return pd.DataFrame({"symbol": [], "last_date": pd.to_datetime([])})


def plan(close, eligible=None, spans=None, **kwargs):
    kwargs.setdefault("top_n", 3)
    kwargs.setdefault("lookback", LOOKBACK)
    kwargs.setdefault("skip", SKIP)
    kwargs.setdefault("dead_loss", 1.0)
    kwargs.setdefault("start", START)
    kwargs.setdefault("end", END)
    return plan_trades(
        close,
        all_eligible(close) if eligible is None else eligible,
        empty_spans() if spans is None else spans,
        **kwargs,
    )


def month_ends(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    inside = index[(index >= start) & (index <= end)]
    return list(pd.Series(inside, index=inside).groupby(inside.to_period("M")).max())


def test_trade_dataclass_fields() -> None:
    trade = Trade(pd.Timestamp("2019-01-03"), "S00", 10.0, 5.0, "rebalance")
    assert (trade.qty, trade.price, trade.reason) == (10.0, 5.0, "rebalance")
    with pytest.raises(AttributeError):
        trade.qty = 1.0  # type: ignore[misc]


def test_trade_is_next_day() -> None:
    close = make_close()
    steps = plan(close)
    assert steps
    decisions = month_ends(close.index, START, END)
    for step in steps:
        assert step.decision_date in decisions
        pos = close.index.get_loc(step.decision_date)
        assert step.date == close.index[pos + 1]
    # Dec 31 2019 is the last bar of the data: no next trading day, so no step for it.
    assert pd.Timestamp("2019-12-31") not in [s.decision_date for s in steps]
    assert len(steps) == 11


def test_decision_day_prices_only() -> None:
    close = make_close()
    steps = plan(close)
    altered = close.copy()
    rng = np.random.default_rng(99)
    for step in steps:
        later = altered.index > step.decision_date
        altered.loc[later] = rng.uniform(1.0, 500.0, size=(later.sum(), close.shape[1]))
        # Only the prices after this day are rewritten, so earlier days keep their values.
        again = [s for s in plan(altered) if s.decision_date == step.decision_date]
        assert again[0].targets == step.targets
        altered = close.copy()


def test_top_n_equal_weight_and_only_eligible() -> None:
    close = make_close()
    # S00 is the runaway winner everywhere but is never eligible.
    close["S00"] = runaway(close)
    eligible = all_eligible(close)
    eligible["S00"] = False
    steps = plan(close, eligible, top_n=4)
    for step in steps:
        assert "S00" not in step.targets
        assert len(step.targets) == 4
        assert all(w == pytest.approx(0.25) for w in step.targets.values())
    # With it eligible the winner is always chosen.
    for step in plan(close, top_n=4):
        assert "S00" in step.targets


def test_fewer_eligible_than_top_n_splits_over_those_eligible() -> None:
    close = make_close()
    eligible = all_eligible(close)
    eligible.loc[:, close.columns[2:]] = False
    for step in plan(close, eligible, top_n=5):
        assert set(step.targets) == {"S00", "S01"}
        assert all(w == pytest.approx(0.5) for w in step.targets.values())


def test_no_eligible_stock_gives_empty_targets() -> None:
    close = make_close()
    eligible = all_eligible(close) & False
    for step in plan(close, eligible):
        assert step.targets == {}


def test_eligibility_frame_with_a_utc_index_is_aligned() -> None:
    close = make_close()
    utc_close = close.copy()
    utc_close.index = utc_close.index.tz_localize("UTC")
    naive = plan(close)
    aware = plan(utc_close, all_eligible(close))  # naive eligible frame, UTC close
    assert [s.targets for s in naive] == [s.targets for s in aware]


def test_dead_stock_exit_on_last_bar_at_haircut() -> None:
    close = make_close()
    last = pd.Timestamp("2019-06-14")
    close.loc[close.index > last, "S03"] = np.nan
    spans = make_spans([("S03", last)])
    exits = dead_exit_dates(close, spans)
    assert list(exits.index) == ["S03"]
    assert exits["S03"] == last
    half = exit_close(close, spans, 0.5)
    assert half.loc[last, "S03"] == pytest.approx(0.5 * close.loc[last, "S03"])
    gone = exit_close(close, spans, 1.0)
    assert gone.loc[last, "S03"] == pytest.approx(close.loc[last, "S03"] * SIM_PRICE_FLOOR)
    kept = exit_close(close, spans, 0.0)
    assert kept.loc[last, "S03"] == close.loc[last, "S03"]
    # Nothing else moves.
    other = half.drop(columns="S03")
    pd.testing.assert_frame_equal(other, close.drop(columns="S03"))
    assert half.drop(index=last).equals(close.drop(index=last))


def test_successor_stock_not_written_down() -> None:
    close = make_close()
    last = pd.Timestamp("2019-06-14")
    close.loc[close.index > last, "S03"] = np.nan
    spans = make_spans([("S03", last, True)])
    assert dead_exit_dates(close, spans).empty
    for loss in (0.0, 0.5, 1.0):
        out = exit_close(close, spans, loss)
        assert out.loc[last, "S03"] == close.loc[last, "S03"]
    # The end date is still reported for the ledger, flagged as having a successor.
    ends = span_end_dates(close, spans)
    assert ends["S03"] == last


def test_halted_stock_valued_at_last_print() -> None:
    close = make_close()
    # S01 is the top scorer; it stops printing for a stretch that covers a decision day.
    close["S01"] = runaway(close)
    decision = month_ends(close.index, START, END)[3]
    halted = (close.index > decision - pd.Timedelta(days=6)) & (close.index <= decision)
    close.loc[halted, "S01"] = np.nan
    assert np.isnan(close.loc[decision, "S01"])
    step = next(s for s in plan(close, top_n=2) if s.decision_date == decision)
    assert "S01" in step.targets


def test_no_entry_after_death() -> None:
    close = make_close()
    # The strongest stock dies on 2019-06-14 but its stale, forward-filled score would keep it
    # on top; the eligibility frame here allows it on every day, so the planner itself must
    # drop it once its last bar is before the decision day.
    close["S02"] = runaway(close)
    last = pd.Timestamp("2019-06-14")
    close.loc[close.index > last, "S02"] = np.nan
    spans = make_spans([("S02", last)])
    steps = plan(close, spans=spans, top_n=3)
    alive = [s for s in steps if s.decision_date <= last]
    dead = [s for s in steps if s.decision_date > last]
    assert alive and dead
    assert all("S02" in s.targets for s in alive)
    assert all("S02" not in s.targets for s in dead)
    assert all(len(s.targets) == 3 for s in dead)


def test_stock_whose_last_bar_is_tomorrow_is_still_chosen_today() -> None:
    close = make_close()
    close["S02"] = runaway(close)
    decision = month_ends(close.index, START, END)[5]
    next_day = close.index[close.index.get_loc(decision) + 1]
    close.loc[close.index > decision, "S02"] = np.nan  # no bar on the execution day
    steps = plan(close, spans=make_spans([("S02", decision)]), top_n=3)
    step = next(s for s in steps if s.decision_date == decision)
    assert step.date == next_day
    assert "S02" in step.targets  # unfillable buys are the ledger's problem, not a substitute


def test_decisions_identical_when_only_future_span_endings_differ() -> None:
    base = make_close()
    base["S02"] = runaway(base)
    decisions = month_ends(base.index, START, END)
    decision = decisions[4]
    soon = base.index[base.index.get_loc(decision) + 1]
    later = base.index[base.index.get_loc(decision) + 30]

    def with_ending(last: pd.Timestamp | None):
        close = base.copy()
        spans = empty_spans()
        if last is not None:
            close.loc[close.index > last, "S02"] = np.nan
            spans = make_spans([("S02", last)])
        return plan(close, spans=spans, top_n=3), close

    first, close_first = with_ending(soon)
    second, close_second = with_ending(later)
    third, _ = with_ending(None)
    # Closes agree through the decision day, so everything up to it must agree.
    pd.testing.assert_frame_equal(close_first.loc[:decision], close_second.loc[:decision])
    for steps in (second, third):
        for ours, theirs in zip(first, steps, strict=True):
            if ours.decision_date <= soon:
                assert ours.targets == theirs.targets
    chosen = next(s for s in first if s.decision_date == decision)
    assert "S02" in chosen.targets


def test_dead_exit_dates_only_inside_the_window() -> None:
    close = make_close()
    spans = make_spans(
        [
            ("S00", "2017-06-01"),  # ended before the data starts
            ("S01", "2019-03-15"),
            ("S02", "2030-01-01"),  # ends after the data
            ("NOT_THERE", "2019-03-15"),  # not in close
        ]
    )
    exits = dead_exit_dates(close, spans)
    assert dict(exits) == {"S01": pd.Timestamp("2019-03-15")}


def test_dead_exit_date_snaps_back_to_the_last_bar() -> None:
    close = make_close()
    saturday = pd.Timestamp("2019-03-16")
    exits = dead_exit_dates(close, make_spans([("S01", saturday)]))
    assert exits["S01"] == pd.Timestamp("2019-03-15")


def test_spans_without_optional_columns_are_accepted() -> None:
    close = make_close()
    spans = pd.DataFrame({"symbol": ["S01"], "last_date": [pd.Timestamp("2019-03-15")]})
    assert not dead_exit_dates(close, spans).empty
    assert not exit_close(close, spans, 0.5).equals(close)
    assert plan(close, spans=spans)


def test_bad_dead_loss_is_refused() -> None:
    close = make_close()
    with pytest.raises(ValueError, match="dead_loss"):
        plan(close, dead_loss=1.5)


def test_bad_window_is_refused() -> None:
    close = make_close()
    with pytest.raises(ValueError, match="start"):
        plan(close, start="2019-06-01", end="2019-01-01")


def test_weights_follow_xs_momentum_convention() -> None:
    """Same names and weights as ``XsMomentum.build`` on the same decision days."""
    close = make_close(n_symbols=14, seed=3)
    # Ties: two stocks with identical paths must be broken by column order, as in build().
    close["S13"] = close["S12"]
    eligible = all_eligible(close)
    eligible.loc["2019-01-01":"2019-06-30", "S05"] = False
    eligible.loc["2019-04-01":, "S07"] = False
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
    top_n = 4
    steps = plan(close, eligible, top_n=top_n)
    rebal = pd.DatetimeIndex([s.decision_date for s in steps]).tz_localize("UTC")
    assert len(steps) >= 3
    grid = ParamGrid.single({"top_n": top_n, "lookback": LOOKBACK, "skip": SKIP})
    eligible_utc = eligible.copy()
    eligible_utc.index = eligible_utc.index.tz_localize("UTC")
    weights = XsMomentum().build(bars, grid, rebal=rebal, eligible=eligible_utc)
    weights.columns = weights.columns.get_level_values(-1)
    for step in steps:
        row = weights.loc[step.date.tz_localize("UTC")]
        expected = {sym: w for sym, w in row.items() if w > 0}
        assert set(step.targets) == set(expected)
        for sym, weight in expected.items():
            assert step.targets[sym] == pytest.approx(weight)
        assert sum(step.targets.values()) == pytest.approx(1.0)


def test_module_exposes_planner_constants() -> None:
    assert replay.REBALANCE_REASON == "rebalance"
    assert replay.DEAD_EXIT_REASON == "dead_exit"


def test_span_with_both_successor_and_total_loss_is_refused() -> None:
    close = make_close()
    spans = pd.DataFrame(
        {
            "symbol": ["S01"],
            "last_date": [pd.Timestamp("2019-03-15")],
            "has_successor": [True],
            "total_loss": [True],
        }
    )
    with pytest.raises(ValueError, match="S01"):
        exit_close(close, spans, 0.5)
    with pytest.raises(ValueError, match="S01"):
        dead_exit_dates(close, spans)
    with pytest.raises(ValueError, match="S01"):
        span_end_dates(close, spans)
    with pytest.raises(ValueError, match="S01"):
        plan(close, spans=spans)


def test_successor_alone_keeps_last_price_at_full_loss() -> None:
    close = make_close()
    last = pd.Timestamp("2019-03-15")
    spans = pd.DataFrame(
        {
            "symbol": ["S01"],
            "last_date": [last],
            "has_successor": [True],
            "total_loss": [False],
        }
    )
    out = exit_close(close, spans, 1.0)
    assert out.loc[last, "S01"] == close.loc[last, "S01"]
    assert dead_exit_dates(close, spans).empty


def test_total_loss_alone_goes_to_the_floor() -> None:
    close = make_close()
    last = pd.Timestamp("2019-03-15")
    spans = pd.DataFrame(
        {"symbol": ["S01"], "last_date": [last], "has_successor": [False], "total_loss": [True]}
    )
    out = exit_close(close, spans, 0.0)
    assert out.loc[last, "S01"] == pytest.approx(close.loc[last, "S01"] * SIM_PRICE_FLOOR)
