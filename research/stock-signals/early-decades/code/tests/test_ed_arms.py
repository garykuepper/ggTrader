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
