"""Exercise the arm helpers on synthetic data."""

import pandas as pd
import pytest
from ed_arms import arm_b_schedule

pytestmark = pytest.mark.study


def test_yearly_schedule() -> None:
    dates = list(pd.to_datetime(["2020-01-31", "2020-02-28", "2021-01-29"]))
    assert arm_b_schedule(dates) == [dates[0]]


def test_planner_excludes_prior_jump_but_not_future_jump() -> None:
    from ed_arms import plan_arm_trades

    index = pd.bdate_range("2019-01-01", "2021-03-31")
    close = pd.DataFrame({"A": 10.0, "B": 10.0}, index=index)
    close.loc["2019-12-02":, "A"] = 20.0
    eligible = pd.DataFrame(True, index=index, columns=close.columns)
    spans = pd.DataFrame(columns=["symbol", "last_date", "has_successor"])
    kwargs = dict(top_n=1, lookback=60, skip=5, start="2020-01-01", end="2021-03-31")
    a = plan_arm_trades("A", close, eligible, spans, **kwargs)
    b = plan_arm_trades("B", close, eligible, spans, **kwargs)
    c = plan_arm_trades("C", close, eligible, spans, **kwargs)
    assert len(a) > len(b) == 2
    assert a[0].targets == {"A": 1.0}
    assert c[0].targets == {"B": 1.0}
    later = close.copy()
    later.loc["2020-02-03":, "B"] = 30.0
    changed = plan_arm_trades("C", later, eligible, spans, **kwargs)
    assert changed[0] == c[0]


@pytest.mark.parametrize(("days", "expected"), [(300, 24.0), (400, 15.0)])
def test_tax_book_applies_rate_for_actual_holding(days: int, expected: float) -> None:
    from ed_ledger_run import TaxLotBook, TaxParams

    bought = pd.Timestamp("2018-01-01")
    sold = bought + pd.Timedelta(days=days)
    book = TaxLotBook(TaxParams(0.24, 0.15), cash=1000)
    book.buy("A", 1, 100, bought)
    book.sell("A", 1, 200, sold)
    assert book.year_end(sold.year, sold) == pytest.approx(expected)


def test_arm_c_aligns_naive_eligibility_to_utc_close() -> None:
    from ed_arms import plan_arm_trades

    index = pd.bdate_range("2019-01-01", "2020-03-31", tz="UTC")
    close = pd.DataFrame({"A": 10.0}, index=index)
    eligible = pd.DataFrame(True, index=index.tz_localize(None), columns=close.columns)
    spans = pd.DataFrame(columns=["symbol", "last_date", "has_successor"])
    steps = plan_arm_trades(
        "C",
        close,
        eligible,
        spans,
        top_n=1,
        lookback=60,
        skip=5,
        start="2020-01-01",
        end="2020-03-31",
    )
    assert steps and steps[0].targets == {"A": 1.0}


def test_yearly_schedule_strictly_exceeds_anniversary_on_historical_calendar() -> None:
    import pandas_market_calendars as calendars

    sessions = calendars.get_calendar("NYSE").valid_days("1969-02-03", "1998-07-31")
    decisions = (
        pd.Series(sessions, index=sessions).groupby(sessions.tz_localize(None).to_period("M")).max()
    )
    scheduled = arm_b_schedule(list(decisions))
    assert all(b > a + pd.DateOffset(years=1) for a, b in zip(scheduled, scheduled[1:]))


def test_reopening_jump_after_missing_days_is_excluded() -> None:
    from ed_arms import plan_arm_trades

    index = pd.bdate_range("2019-01-01", "2020-03-31")
    close = pd.DataFrame({"A": 10.0, "B": 10.0}, index=index)
    close.loc["2019-12-16":"2019-12-20", "A"] = float("nan")
    close.loc["2019-12-23":, "A"] = 15.0
    eligible = pd.DataFrame(True, index=index, columns=close.columns)
    spans = pd.DataFrame(columns=["symbol", "last_date", "has_successor"])
    steps = plan_arm_trades(
        "C",
        close,
        eligible,
        spans,
        top_n=1,
        lookback=60,
        skip=5,
        start="2020-01-01",
        end="2020-03-31",
    )
    assert steps[0].targets == {"B": 1.0}


def test_arm_b_historical_execution_intervals_and_first_lot(monkeypatch) -> None:
    import ed_replay
    import pandas_market_calendars as calendars
    from ed_arms import plan_arm_trades
    from ed_ledger_run import TaxLotBook, TaxParams

    sessions = calendars.get_calendar("NYSE").valid_days("1969-02-03", "1998-07-31")
    decisions = (
        pd.Series(sessions, index=sessions).groupby(sessions.tz_localize(None).to_period("M")).max()
    )
    candidates = []
    for decision in decisions:
        position = sessions.get_loc(decision) + 1
        if position < len(sessions):
            candidates.append(ed_replay.TargetStep(decision, sessions[position], {"A": 1.0}))
    monkeypatch.setattr(ed_replay, "plan_trades", lambda *args, **kwargs: candidates)
    steps = plan_arm_trades(
        "B",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        top_n=1,
        lookback=1,
        skip=0,
        start="1969-02-03",
        end="1998-07-31",
    )
    assert len(steps) > 2
    expected = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate.date > expected[-1].date + pd.DateOffset(years=1):
            expected.append(candidate)
    assert steps == expected
    assert all(b.date > a.date + pd.DateOffset(years=1) for a, b in zip(steps, steps[1:]))
    book = TaxLotBook(TaxParams(0.24, 0.15), cash=1000)
    book.buy("A", 1, 100, steps[0].date)
    realized = book.sell("A", 1, 200, steps[1].date)
    assert realized[0].term == "LT"
