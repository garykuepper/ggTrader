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
        status_code = 200

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


def test_return_agreement_compares_daily_changes() -> None:
    from fc_pilot import return_agreement

    old = pd.DataFrame({"symbol": ["A", "B"], "close": [100.0, 100.0]})
    new = pd.DataFrame({"symbol": ["A", "B"], "close": [102.0, 110.0]})
    ref_old = pd.DataFrame({"symbol": ["A", "B"], "close": [50.0, 50.0]})
    ref_new = pd.DataFrame({"symbol": ["A", "B"], "close": [51.0, 50.0]})
    assert return_agreement(old, new, ref_old, ref_new, 0.005) == (2, 0.5)


def test_classify_security_uses_listing_name() -> None:
    from fc_pilot import classify_security

    assert classify_security("ABC", "Example Corp - Common Stock") == "common"
    assert classify_security("ABCW", "Example Corp Warrant") == "warrants"
    assert classify_security("ABC", "Example Inc Units") == "units"
    assert classify_security("ABC", "Example Inc Preferred Stock") == "preferred"
    assert classify_security("ABC", "Example Inc Rights") == "rights"
    assert classify_security("ABC", "Unclear security") == "other"


def test_alignment_reports_returns_and_largest_disagreements() -> None:
    from fc_alignment import compare

    a = pd.DataFrame({"symbol": ["A", "B"], "previous": [100.0, 100.0], "close": [102.0, 120.0]})
    y = pd.DataFrame({"symbol": ["A", "B"], "previous": [100.0, 100.0], "close": [102.0, 100.0]})
    result = compare(a, y)
    assert result["matched"] == 2
    assert result["within_half_point"] == 0.5
    assert result["largest"][0]["symbol"] == "B"
    assert result["largest"][0]["difference"] == pytest.approx(0.2)


def test_alpaca_can_request_sip_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime

    import fc_pilot
    import requests

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"bars": {}}

    seen = []

    def fake_get(url: str, **kwargs: object) -> Response:
        seen.append(kwargs["params"].copy())
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _: None)
    fc_pilot.fetch_bars(["AAA"], datetime.date(2026, 11, 4), "alpaca", feed="sip")
    assert seen[0]["feed"] == "sip"
    assert seen[0]["adjustment"] == "raw"
