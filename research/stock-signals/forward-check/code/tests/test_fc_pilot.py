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


def test_fetch_listing_parses_both_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    import fc_pilot

    nasdaq = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares\nAAA|A|Q|N|N|100|N|N\n"
        "File Creation Time: now|||||||\n"
    )
    other = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
        "Test Issue|NASDAQ Symbol\nBBB|B|N|BBB|Y|100|N|BBB\n"
        "File Creation Time: now|||||||\n"
    )
    monkeypatch.setattr(
        fc_pilot,
        "read_directory",
        lambda url: io.StringIO(nasdaq if "nasdaqlisted" in url else other),
    )
    result = fc_pilot.fetch_listing()
    assert result["symbol"].tolist() == ["AAA"]


def test_yahoo_bars_normalizes_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime

    import fc_pilot
    import yfinance

    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["AAA"]])
    frame = pd.DataFrame([[1, 2, 1, 2, 100]], columns=columns)
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: frame)
    result = fc_pilot.fetch_bars(["AAA"], datetime.date(2026, 11, 4), "yahoo")
    assert result[["symbol", "close"]].to_dict("records") == [{"symbol": "AAA", "close": 2}]


def test_alpaca_bars_normalize_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime

    import fc_pilot
    import requests

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "bars": {
                    "AAA": [{"t": "2026-11-04T04:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 100}]
                }
            }

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    result = fc_pilot.fetch_bars(["AAA"], datetime.date(2026, 11, 4), "alpaca")
    assert result[["symbol", "close"]].to_dict("records") == [{"symbol": "AAA", "close": 2}]
