"""Tests for the FOMC meeting-calendar loader -- free, scraped from the
Federal Reserve's own public calendar pages (no API key)."""

from __future__ import annotations

import pandas as pd
import pytest
from ggTrader.lab.fomc_calendar import (
    parse_historical_page,
    parse_main_calendar_page,
)


class TestParseHistoricalPage:
    def test_same_month_two_day_meeting(self):
        html = "<p>January 27-28 Meeting - 2015 Beige Book</p>"
        dates = parse_historical_page(html, year=2015)
        assert dates == [pd.Timestamp("2015-01-28")]

    def test_cross_month_two_day_meeting_full_names(self):
        html = "<p>June 30-July 1 Meeting - 1998</p>"
        dates = parse_historical_page(html, year=1998)
        assert dates == [pd.Timestamp("1998-07-01")]

    def test_cross_month_two_day_meeting_abbreviated_names(self):
        html = "<p>Jan/Feb 31-1 Meeting - 2017</p>"
        dates = parse_historical_page(html, year=2017)
        assert dates == [pd.Timestamp("2017-02-01")]

    def test_single_day_meeting(self):
        html = "<p>March 31 Meeting - 1998</p>"
        dates = parse_historical_page(html, year=1998)
        assert dates == [pd.Timestamp("1998-03-31")]

    def test_multiple_meetings_sorted(self):
        html = "<p>March 17-18 Meeting</p><p>January 27-28 Meeting</p>"
        dates = parse_historical_page(html, year=2015)
        assert dates == [pd.Timestamp("2015-01-28"), pd.Timestamp("2015-03-18")]

    def test_no_meetings_returns_empty_list(self):
        assert parse_historical_page("<p>nothing here</p>", year=2015) == []


class TestParseMainCalendarPage:
    def test_extracts_dates_from_presconf_ids(self):
        html = "fomcpresconf20260318 fomcpresconf20260429"
        dates = parse_main_calendar_page(html)
        assert dates == [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-04-29")]

    def test_dedupes_and_sorts(self):
        html = "fomcpresconf20260429 fomcpresconf20260318 fomcpresconf20260318"
        dates = parse_main_calendar_page(html)
        assert dates == [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-04-29")]

    def test_no_ids_returns_empty_list(self):
        assert parse_main_calendar_page("<p>nothing here</p>") == []


@pytest.mark.integration
def test_historical_fomc_dates_end_to_end_live_fetch():
    """Live smoke test against the real Fed site -- confirms the scraper
    still works against the current page structure. Spot-checks two known
    dates rather than the full list (formats vary too much year to year
    for a single hardcoded 'exact list' assertion to be worth maintaining)."""
    from ggTrader.lab.fomc_calendar import historical_fomc_announcement_dates

    dates = historical_fomc_announcement_dates(start_year=2015, end_year=2016)
    date_strs = {d.strftime("%Y-%m-%d") for d in dates}
    assert "2015-12-16" in date_strs  # "liftoff" meeting
    assert "2016-12-14" in date_strs
    assert len(dates) >= 14  # 8/year x 2 years, minus any known gaps
