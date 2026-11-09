"""The S&P Dow Jones Indices press releases for the S&P 500: a second, official source of changes.

The weekly job's other change sources are a Wikipedia table and a GitHub CSV, both volunteer work.
S&P DJI itself announces every S&P 500 change in a press release, distributed on PR Newswire
(free, no login). Each release holds a table ``Effective Date | Index Name | Action | Company
Name | Ticker | GICS Sector`` that lists the S&P 500 additions and deletions with the date S&P
says they take effect. This module fetches the S&P DJI news page, picks the releases about the
S&P 500, fetches each recent one and parses that table.

Used two ways by the weekly job (``membership_weekly``): a warning when an announced change is
not reflected in our list, and a third date in a proposed events file, where an official date
that agrees with another source clears the ``needs_confirmation`` flag. Nothing here touches the
database, and a failure of this source never closes the live gate (it only raises a warning).
"""

from __future__ import annotations

import functools
import html
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser

import requests

from ggTrader.data.core.index_constituents import normalize_yf_ticker
from ggTrader.data.membership_sources import (
    GetFn,
    SourceFetch,
    SourceStructureError,
    _failure,
    _request,
)

PRESS_BASE = "https://www.prnewswire.com"
#: The newest S&P DJI releases (about 25, a few months back): one request.
PRESS_LISTING_URL = PRESS_BASE + "/news/s%26p-dow-jones-indices/"
#: Source names as stored in ``universe.source_snapshots``.
PRESS_LISTING = "sp_press_listing"
PRESS_PREFIX = "sp_press_"
#: Only releases published this many days back (or newer) are fetched, at most ``MAX_RELEASES``.
LOOKBACK_DAYS = 75
MAX_RELEASES = 8
#: A release is about the S&P 500 when this word sequence is in its address slug (title).
_SLUG_SP500 = re.compile(r"(?:^|-)sp-500(?:-|$)")
#: The opening tag of a release card. The card's text is then taken with plain ``str.find`` calls,
#: so the cost stays linear in the page size on any input (a lazy ``.*?`` pattern over the whole
#: card was quadratic on a page of unclosed cards).
_CARD_OPEN = re.compile(
    r'<a\s+class="newsreleaseconsolidatelink[^"]*"\s+href="(?P<href>/news-releases/[^"]+\.html)"'
    r"[^>]*>"
)
_CARD_CLASS = "newsreleaseconsolidatelink"
_STAMP = re.compile(r"(?<![A-Za-z])([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s+(\d{4})(?!\d)")
_RELEASE_ID = re.compile(r"-(\d{6,})\.html$")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTH_WORDS = frozenset(
    [
        *_MONTHS,
        "sept",
        *"january february march april june july august september october".split(),
        "november",
        "december",
    ]
)
_BODY_MARK = 'class="release-body'
_HEADER = ("effective date", "index name", "action", "company name", "ticker", "gics sector")
_ACTIONS = {"addition": "add", "deletion": "remove"}


@dataclass(frozen=True)
class PressRelease:
    """One entry of the news page."""

    url: str
    title: str
    published: date
    release_id: str

    @property
    def source_name(self) -> str:
        """The name of this release's row in ``universe.source_snapshots``."""
        return PRESS_PREFIX + self.release_id


@dataclass(frozen=True)
class PressChange:
    """One S&P 500 addition or deletion from a release's table."""

    effective: date
    action: str  # 'add' | 'remove'
    ticker: str
    company: str
    published: date
    url: str
    release_id: str = "0"

    @property
    def order(self) -> tuple[date, int, date]:
        """Newest release last: publication day, then release id (it grows with time), then date."""
        return (self.published, int(self.release_id), self.effective)


@dataclass
class PressFetch:
    """The news page and the releases fetched from it; never raises."""

    listing: SourceFetch
    releases: list[tuple[PressRelease, SourceFetch]] = field(default_factory=list)


def _month(text: str) -> int | None:
    key = text.strip().lower()[:3]
    return _MONTHS.index(key) + 1 if key in _MONTHS else None


def parse_press_date(text: str) -> date | None:
    """``Nov 4, 2026``, ``Sept 21, 2026`` or ``Nov 06, 2026, 18:09 ET`` -> a date, else None."""
    found = _STAMP.search(html.unescape(text))
    if not found:
        return None
    if found.group(1).lower() not in _MONTH_WORDS:
        return None
    month = _month(found.group(1))
    if month is None:
        return None
    try:
        return date(int(found.group(3)), month, int(found.group(2)))
    except ValueError:
        return None


def _structure_guard(parse):
    """Turn any unexpected exception of a parser into a ``SourceStructureError`` (code internal).

    A hostile or oddly broken page (for example a huge numeric character reference that makes
    ``html.unescape`` raise) must read as a changed page, never escape into the weekly job.
    """

    @functools.wraps(parse)
    def guarded(*args, **kwargs):
        try:
            return parse(*args, **kwargs)
        except SourceStructureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SourceStructureError("internal", f"{type(exc).__name__}: {exc}"[:200]) from exc

    return guarded


@_structure_guard
def parse_press_listing(raw: bytes) -> list[PressRelease]:
    """The S&P 500 releases on the news page, newest first.

    Raises ``SourceStructureError`` when the page has no release cards at all (the markup
    changed); a page with cards but none about the S&P 500 returns an empty list.
    """
    text = raw.decode("utf-8", errors="replace")
    opens = list(_CARD_OPEN.finditer(text))
    if not opens:
        raise SourceStructureError("cards_not_found", "no release cards on the S&P DJI news page")
    if text.count(_CARD_CLASS) > len(opens):
        raise SourceStructureError(
            "bad_card",
            f"{text.count(_CARD_CLASS)} release cards on the page but {len(opens)} readable",
        )
    out: list[PressRelease] = []
    seen: set[str] = set()
    for i, card in enumerate(opens):
        href = card.group("href")
        slug = href.rsplit("/", 1)[-1]
        ident = _RELEASE_ID.search(href)
        limit = opens[i + 1].start() if i + 1 < len(opens) else len(text)  # windows never overlap
        end = text.find("</a>", card.end(), limit)
        window = text[card.end() : end if end >= 0 else limit]
        small_at = window.find("<small>")
        small_end = window.find("</small>", small_at) if small_at >= 0 else -1
        published = parse_press_date(window[small_at + 7 : small_end]) if small_end >= 0 else None
        if ident is None or published is None:
            raise SourceStructureError(
                "bad_card", f"cannot read the id or date of release card {href!r}"
            )
        if not _SLUG_SP500.search(slug) or ident.group(1) in seen:
            continue
        seen.add(ident.group(1))
        title_end = window.find("</h3>", small_end)
        raw_title = window[small_end + 8 : title_end if title_end >= 0 else len(window)]
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw_title)))
        out.append(PressRelease(PRESS_BASE + href, title.strip(), published, ident.group(1)))
    return sorted(out, key=lambda r: (r.published, r.release_id), reverse=True)


class _Rows(HTMLParser):
    """Collects the text of every table row's cells, in the release body only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


@_structure_guard
def parse_press_release(raw: bytes, release: PressRelease) -> list[PressChange]:
    """The S&P 500 additions and deletions in a release's summary table.

    Raises ``SourceStructureError`` when the release body or the table is missing or its header
    changed, or a row cannot be read. A table with no S&P 500 row (a release that only touches
    other indices) returns an empty list.
    """
    text = raw.decode("utf-8", errors="replace")
    start = text.find(_BODY_MARK)
    if start < 0:
        raise SourceStructureError("body_not_found", f"release {release.release_id}: no body")
    parser = _Rows()
    parser.feed(text[start:])
    rows = parser.rows
    header_at = next(
        (i for i, r in enumerate(rows) if tuple(c.lower() for c in r[: len(_HEADER)]) == _HEADER),
        None,
    )
    if header_at is None:
        raise SourceStructureError(
            "table_not_found",
            f"release {release.release_id}: no 'Effective Date | Index Name | Action | ...' table",
        )
    changes: list[PressChange] = []
    problems: list[str] = []
    for row in rows[header_at + 1 :]:
        index = row[1].strip().lower() if len(row) > 1 else ""
        if index != "s&p 500":
            continue
        if len(row) < 5:
            problems.append(f"truncated S&P 500 row {row!r}")
            continue
        action = _ACTIONS.get(row[2].strip().lower())
        day = parse_press_date(row[0])
        ticker = row[4].strip()
        if action is None or day is None or not ticker:
            problems.append(f"unreadable S&P 500 row {row!r}")
            continue
        changes.append(
            PressChange(
                day,
                action,
                ticker,
                row[3].strip(),
                release.published,
                release.url,
                release.release_id,
            )
        )
    if problems:
        raise SourceStructureError("bad_row", "; ".join(problems[:3]), problems)
    return changes


def fetch_press(
    *,
    get: GetFn | None = None,
    now: Callable[[], datetime] | None = None,
    lookback_days: int = LOOKBACK_DAYS,
    max_releases: int = MAX_RELEASES,
) -> PressFetch:
    """Fetch the news page, then each S&P 500 release published within ``lookback_days``.

    Never raises. At most ``1 + max_releases`` requests. A release that cannot be fetched comes
    back with ``ok=False``; an unparseable news page yields no releases (the caller parses the
    listing again to report the structure error).
    """
    clock = now or (lambda: datetime.now(timezone.utc))
    get = get or requests.get
    try:
        listing = _request(get, PRESS_LISTING, PRESS_LISTING_URL, clock, accept="text/html")
    except Exception as exc:  # noqa: BLE001 - one source must never abort the job
        listing = _failure(
            PRESS_LISTING, PRESS_LISTING_URL, clock(), "internal", f"{type(exc).__name__}: {exc}"
        )
    result = PressFetch(listing)
    if not listing.ok or listing.raw is None:
        return result
    try:
        wanted = parse_press_listing(listing.raw)
    except SourceStructureError:
        return result
    cutoff = clock().astimezone(timezone.utc).date() - timedelta(days=lookback_days)
    for release in [r for r in wanted if r.published >= cutoff][:max_releases]:
        try:
            fetched = _request(get, release.source_name, release.url, clock, accept="text/html")
        except Exception as exc:  # noqa: BLE001
            fetched = _failure(
                release.source_name,
                release.url,
                clock(),
                "internal",
                f"{type(exc).__name__}: {exc}",
            )
        fetched.last_change = release.published
        result.releases.append((release, fetched))
    return result


def latest_changes(changes: Sequence[PressChange]) -> list[PressChange]:
    """One row per ``(action, ticker)``: the one from the newest release (a correction wins)."""
    best: dict[tuple[str, str], PressChange] = {}
    for change in sorted(changes, key=lambda c: c.order):
        best[(change.action, normalize_yf_ticker(change.ticker))] = change
    return sorted(best.values(), key=lambda c: (c.effective, c.action, c.ticker))
