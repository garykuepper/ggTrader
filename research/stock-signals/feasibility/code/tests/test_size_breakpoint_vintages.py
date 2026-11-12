"""Verify descriptive breakpoint-vintage comparisons on invented numbers."""

import pandas as pd
import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "size_breakpoint_vintages.py")
pytestmark = pytest.mark.study


def test_comparison_uses_older_vintage_as_denominator():
    index = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
    old = pd.DataFrame({"median_usd_millions": [100.0, 200.0, 300.0]}, index=index)
    new = pd.DataFrame({"median_usd_millions": [110.0, 200.0, 270.0]}, index=index)
    result = module.compare(old, new)
    assert result["months"] == 3
    assert result["different_medians"] == 2
    assert result["max_absolute_relative_difference"] == pytest.approx(0.1)


def test_no_common_month_fails():
    old = pd.DataFrame({"median_usd_millions": [1.0]}, index=pd.to_datetime(["2023-01-31"]))
    new = pd.DataFrame({"median_usd_millions": [1.0]}, index=pd.to_datetime(["2024-01-31"]))
    with pytest.raises(ValueError, match="overlap"):
        module.compare(old, new)
