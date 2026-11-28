"""Check pilot comparisons with small, offline examples."""

import pandas as pd
import pytest
from fc_pilot import coverage, filter_listing, price_agreement

pytestmark = pytest.mark.study


def test_coverage_counts_missing_symbols() -> None:
    listed = pd.DataFrame({"symbol": ["A", "B", "C"]})
    bars = pd.DataFrame({"symbol": ["A", "B", "B"]})
    assert coverage(listed, bars) == {"listed": 3, "returned": 2, "missing": 1, "share": 2 / 3}


def test_price_agreement_uses_tolerance() -> None:
    bars = pd.DataFrame({"symbol": ["A", "B"], "close": [101.0, 104.0]})
    reference = pd.DataFrame({"symbol": ["A", "B"], "close": [100.0, 100.0]})
    assert price_agreement(bars, reference, 0.02) == 0.5


def test_listing_filter_drops_etf_and_test_issues() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "exchange": ["Q"] * 3,
            "is_etf": [False, True, False],
            "is_test": [False, False, True],
        }
    )
    assert filter_listing(frame)["symbol"].tolist() == ["A"]
