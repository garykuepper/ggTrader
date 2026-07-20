"""FOMC scheduled-meeting announcement dates: free, scraped from the
Federal Reserve's own public calendar pages (no API key). Backs the
fomc_drift strategy's event list.

Two source pages, stitched together because the Fed changed its site
structure partway through:
- fomchistorical{year}.htm (through 2020): lists each meeting as
  "{Month} {D1}-{D2} Meeting" (same-month), "{Mon1}/{Mon2} {D1}-{D2}
  Meeting" (cross-month, often abbreviated month names), or "{Month} {D}
  Meeting" (single-day, common pre-2007). The announcement date is the
  LAST day of the range.
- fomccalendars.htm (2012+, including future scheduled meetings): embeds
  "fomcpresconfYYYYMMDD" anchor IDs directly encoding each meeting's
  announcement date -- reliable for every meeting since 2019 (all 8/year
  got press conferences that year) and quarterly meetings 2012-2018.

Deliberately does NOT include unscheduled/emergency actions (e.g. the
March 2020 inter-meeting cuts) -- the pre-FOMC-drift mechanism requires
the meeting to have been publicly scheduled well in advance, which is
what distinguishes "day before a known event" from "day before a
surprise," and emergency actions don't appear in either source's regular
meeting-list format.
"""

from __future__ import annotations

import re
from typing import Callable, List

import pandas as pd
import requests

HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
MAIN_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

_FULL_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_ABBR_MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_MONTH_INDEX = {m: i + 1 for i, m in enumerate(_FULL_MONTHS)}
_MONTH_INDEX.update({m: i + 1 for i, m in enumerate(_ABBR_MONTHS)})
_MONTHS_PATTERN = "|".join(_FULL_MONTHS + _ABBR_MONTHS)

_MEETING_PATTERN = re.compile(
    rf"(?<![A-Za-z])({_MONTHS_PATTERN})(?:/({_MONTHS_PATTERN}))?\s+(\d{{1,2}})(?:[-–](\d{{1,2}}))?\s+Meeting"
)
_PRESCONF_ID_PATTERN = re.compile(r"fomcpresconf(\d{4})(\d{2})(\d{2})")

HttpFetch = Callable[[str], str]


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text)


def parse_historical_page(html: str, year: int) -> List[pd.Timestamp]:
    """Parse one fomchistorical{year}.htm page into announcement dates."""
    text = _strip_html(html)
    out: List[pd.Timestamp] = []
    for m in _MEETING_PATTERN.finditer(text):
        month1, month2, d1, d2 = m.groups()
        if month2:
            month, day = month2, d2
        elif d2:
            month, day = month1, d2
        else:
            month, day = month1, d1
        out.append(pd.Timestamp(year=year, month=_MONTH_INDEX[month], day=int(day)))
    return sorted(out)


def parse_main_calendar_page(html: str) -> List[pd.Timestamp]:
    """Parse the main fomccalendars.htm page's fomcpresconfYYYYMMDD IDs."""
    dates = {
        pd.Timestamp(year=int(y), month=int(mo), day=int(d))
        for y, mo, d in _PRESCONF_ID_PATTERN.findall(html)
    }
    return sorted(dates)


def _default_http_fetch(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return resp.text


def historical_fomc_announcement_dates(
    start_year: int = 2011,
    end_year: int | None = None,
    http_fetch: HttpFetch | None = None,
) -> List[pd.Timestamp]:
    """All scheduled FOMC announcement dates in [start_year, end_year].

    Years <= 2020 come from the per-year historical pages; 2021+ comes
    from the main calendar page's fomcpresconf IDs (the historical-page
    format wasn't continued past 2020). end_year defaults to the current
    year.
    """
    fetch = http_fetch or _default_http_fetch
    end_year = end_year or pd.Timestamp.now().year

    dates: List[pd.Timestamp] = []
    for year in range(start_year, min(end_year, 2020) + 1):
        html = fetch(HISTORICAL_URL.format(year=year))
        dates.extend(parse_historical_page(html, year))

    if end_year >= 2021:
        html = fetch(MAIN_CALENDAR_URL)
        recent = parse_main_calendar_page(html)
        dates.extend(d for d in recent if 2021 <= d.year <= end_year)

    return sorted(set(dates))
