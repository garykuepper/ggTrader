"""Exercise the arm helpers on synthetic data."""

import pandas as pd
import pytest
from ed_arms import arm_b_schedule, jump_excluded

pytestmark = pytest.mark.study


def test_yearly_schedule() -> None:
    dates = list(pd.to_datetime(["2020-01-31", "2020-02-28", "2021-01-29"]))
    assert arm_b_schedule(dates) == [dates[0], dates[2]]


def test_excludes_only_prior_jumps_and_handles_gaps() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"])
    returns = pd.DataFrame({1: [0, 0.2, 0, 0], 2: [0, None, 0, 0.5], 3: [0, 0, 0, 0]}, index=dates)
    assert jump_excluded(returns, dates[3]) == {1}
    assert jump_excluded(returns, dates[2], lookback=1) == {1}


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
