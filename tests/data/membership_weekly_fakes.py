"""Builders for the weekly-job tests: wiki pages, chinobing CSVs, a fake network and a fake store.

Nothing here opens a socket or a database connection. The wikitext follows the structure of the real
pages (checked against the live ledger on 2026-11-08): two header rows, rows in the three editors'
styles (line-initial ``|``, line-initial ``||`` and two cells on one line), refs reused by name, a
row with no refs cell, and a stray pipe after a ticker.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ggTrader.data.membership_holdings import HoldingsResult
from ggTrader.data.membership_press import PRESS_LISTING_URL
from ggTrader.data.membership_sources import (
    CHINO_COMMITS_URL,
    WIKI_API,
)
from ggTrader.data.membership_weekly import SnapshotRow
from ggTrader.data.universe_db import VintageData
from tests.data.test_membership_holdings import CLASSES, history
from tests.data.test_membership_upstream import FakeResponse, vintage_data
from tests.data.xlsx_builder import holdings_xlsx

NOW = datetime(2026, 11, 9, 14, 0, tzinfo=timezone.utc)
TODAY = NOW.date()
LEDGER_TITLE = "Historical components of the S&P 500"
CURRENT_TITLE = "List of S&P 500 companies"

REAL_HEADER = """{| class="wikitable sortable" id="changes"
|-
! data-sort-type="date" rowspan="2" | Effective Date
! colspan="2" | Added
! colspan="2" | Removed
! rowspan="2" | Reason
! rowspan="2" | Refs
|-
! Ticker || Security || Ticker || Security"""

#: ``(date text, added, added name, removed, removed name, reason, refs markup)``
LedgerTuple = tuple[str, str, str, str, str, str, str]


def press_ref(announced: str, name: str = "") -> str:
    """A ref in the shape the page uses: a cite web template with a ``date=`` parameter."""
    tag = f' name="{name}"' if name else ""
    stamp = datetime.strptime(announced, "%B %d, %Y").strftime("%Y%m%d")  # the URL embeds it too
    return (
        f"<ref{tag}>{{{{Cite web |title=Release |url=https://www.spglobal.com/spdji/en/documents/"
        f"indexnews/announcements/{stamp}-1/x.pdf |date={announced} |access-date=2026-11-08 "
        f"|website=S&P Dow Jones Indices}}}}</ref>"
    )


def ledger_row(index: int, row: LedgerTuple) -> str:
    """One data row in one of three styles, chosen by position (as the real page mixes them)."""
    day, add, add_name, rem, rem_name, reason, refs = row
    style = index % 3
    if style == 0:
        cells = [day, add, f"[[{add_name}]]" if add_name else "", rem, f"[[{rem_name}]]", reason]
        return "|-\n" + "\n".join(f"|{c}" for c in cells) + (f"\n|{refs}" if refs else "")
    if style == 1:
        cells = [f" {day} ", f" {add} ", f" [[{add_name}]]", f" {rem} ", f" [[{rem_name}]]", reason]
        return "|-\n" + "\n".join(f"|| {c}" for c in cells) + (f"\n|| {refs}" if refs else "")
    return (  # the removed ticker and name share one line, a stray pipe follows the added ticker
        f"|-\n|| {day}\n|| {add} |\n|| [[{add_name}]]\n|| {rem}|| [[{rem_name}]]\n|| {reason}"
        + (f"\n|| {refs}" if refs else "")
    )


def filler_rows(count: int = 110) -> list[LedgerTuple]:
    """Old, plain rows so the table is big enough to look like the ledger."""
    out = []
    for i in range(count):
        day = date(2008, 1, 1) + timedelta(days=15 * i)
        out.append(
            (
                f"{day:%B} {day.day}, {day.year}",
                f"ZA{i:03d}",
                f"Added {i}",
                f"ZB{i:03d}",
                f"Removed {i}",
                "Market capitalization changes.",
                "",
            )
        )
    return out


def ledger_wikitext(rows: Sequence[LedgerTuple], *, header: str = REAL_HEADER) -> str:
    """The page's wikitext: prose, the table (newest row first) and the trailer."""
    body = "\n".join(ledger_row(i, row) for i, row in enumerate([*rows, *filler_rows()]))
    return f"Intro text.\n{header}\n{body}\n|}}\n\n==See also==\n* [[List of S&P 500 companies]]\n"


def parse_json(wikitext: str, title: str = LEDGER_TITLE, revid: int = 111) -> bytes:
    """The ``action=parse`` reply (``formatversion=2``) for a wikitext."""
    payload = {"parse": {"title": title, "pageid": 1, "revid": revid, "wikitext": wikitext}}
    return json.dumps(payload).encode()


def ledger_json(rows: Sequence[LedgerTuple], **kw: Any) -> bytes:
    """The ledger page reply."""
    return parse_json(ledger_wikitext(rows, **kw))


CURRENT_HEADER = """{| class="wikitable sortable mw-collapsible sticky-header" id="constituents"
|-
![[Ticker symbol|Symbol]]
! Security !! [[Global Industry Classification Standard|GICS]] Sector !! GICS Sub-Industry !! Headquarters Location !! Date added !! [[Central Index Key|CIK]] !! Founded"""  # noqa: E501


def current_wikitext(tickers: Sequence[str], *, header: str = CURRENT_HEADER) -> str:
    """The members table: symbols in the page's three template spellings."""
    templates = ["NyseSymbol", "NasdaqSymbol", "BZX link"]
    rows = []
    for i, ticker in enumerate(tickers):
        symbol = ticker.replace("-", ".")
        rows.append(
            f"|-\n|| {{{{{templates[i % 3]}|{symbol}}}}}\n|| [[Company {i}]]\n|| Industrials \n"
            f"|| Machinery \n|| [[Town, State]] \n|| 2001-01-01 \n|| 0000000001\n|| 1900"
        )
    return f"Lead.\n== S&P 500 component stocks ==\n{header}\n" + "\n".join(rows) + "\n|}\n"


def current_json(tickers: Sequence[str], **kw: Any) -> bytes:
    """The current-members page reply."""
    return parse_json(current_wikitext(tickers, **kw), CURRENT_TITLE, 222)


def components_csv(last: Sequence[str], last_day: date, *, rows: int = 120) -> bytes:
    """Daily snapshots ending at ``last_day`` (tickers in chinobing's ``BRK-B`` spelling)."""
    lines = ["date,tickers"]
    older = ",".join(sorted([*last, "OLDONE"][1:] + ["OLDONE"]))
    final = ",".join(sorted(t.replace(".", "-") for t in last))
    for i in range(rows - 1, 0, -1):
        lines.append(f'{last_day - timedelta(days=i)},"{older}"')
    lines.append(f'{last_day},"{final}"')
    lines.insert(3, lines[3])  # the real file repeats a few dates
    return ("\n".join(lines) + "\n").encode()


def changes_csv(rows: Sequence[tuple[date, list[str], list[str]]]) -> bytes:
    """``date,added_tickers,removed_tickers`` with Python-list cells, plus old filler rows."""
    lines = ["date,added_tickers,removed_tickers"]
    for i in range(110):
        day = date(2010, 1, 1) + timedelta(days=20 * i)
        lines.append(f"{day},\"['YA{i:03d}']\",\"['YB{i:03d}']\"")
    for day, added, removed in rows:
        lines.append(f'{day},"{added!r}","{removed!r}"')
    return ("\n".join(lines) + "\n").encode()


def revisions_json(ledger_edit: str, current_edit: str) -> bytes:
    """The ``action=query&prop=revisions`` reply for the two pages."""
    pages = [
        {"title": LEDGER_TITLE, "revisions": [{"revid": 1, "timestamp": ledger_edit}]},
        {"title": CURRENT_TITLE, "revisions": [{"revid": 2, "timestamp": current_edit}]},
    ]
    return json.dumps({"query": {"pages": pages}}).encode()


#: ``(published text, slug, title)`` of a listing card; the slug carries the release id.
PressCard = tuple[str, str, str]
#: ``(effective text, index, action, company, ticker)`` of a release table row.
PressRow = tuple[str, str, str, str, str]
MIDCAP_CARD: PressCard = (
    "Nov 06, 2026, 18:09 ET",
    "james-hardie-industries-set-to-join-sp-midcap-400-302932466",
    "James Hardie Industries Set to Join S&amp;P MidCap 400",
)


def press_listing_html(cards: Sequence[PressCard] = (MIDCAP_CARD,)) -> bytes:
    """The S&P DJI news page in the shape PR Newswire serves it (checked 2026-11-09)."""
    items = "".join(
        f"""<div class="row newsCards" lang="en-US"><div class="col-sm-12 card" role="group">
<a class="newsreleaseconsolidatelink display-outline w-100" href="/news-releases/{slug}.html">
<h3 class="no-top-margin remove-outline"><small>{stamp}</small>
  {title}
</h3><p class="remove-outline">S&amp;P Dow Jones Indices will make the following...</p>
</a></div></div>"""
        for stamp, slug, title in cards
    )
    return f"<!DOCTYPE html><html><body>{items}</body></html>".encode()


def press_release_html(rows: Sequence[PressRow], *, header: bool = True) -> bytes:
    """A release page: boilerplate, then the body with the summary table."""
    head = (
        "".join(
            f"<td>{c}</td>"
            for c in (
                "Effective Date",
                "Index Name",
                "Action",
                "Company Name",
                "Ticker",
                "GICS Sector",
            )
        )
        if header
        else "<td>Date</td><td>Index</td>"
    )
    body = "".join(
        "<tr>" + "".join(f"<td> {c} </td>" for c in (*row, "Industrials")) + "</tr>" for row in rows
    )
    return (
        "<html><head><style>.x{color:red}</style></head><body><div>nav</div>"
        '<div class="release-body container"><p>NEW YORK /PRNewswire/ -- S&amp;P Dow Jones '
        "Indices will make the following changes:</p>"
        f"<table><tr>{head}</tr>{body}</table></div></body></html>"
    ).encode()


def reply(body: bytes, status: int = 200) -> FakeResponse:
    """A complete 200 reply (``Content-Length`` matches the body)."""
    return FakeResponse(status, body, {"content-length": str(len(body))})


class World:
    """A consistent set of sources (all agreeing with ``ours``) that tests then perturb.

    ``ours`` is our latest list in raw spelling. Attributes are the bytes each source serves;
    replace one (or set an ``Exception`` in ``errors``) to break that source.
    """

    def __init__(self, ours: list[str]) -> None:
        self.ours = ours
        self.spy = holdings_xlsx(ours)
        self.ledger = ledger_json(
            [("September 21, 2026", "BE", "Bloom", "TAP", "Molson", "Cap changes.", "")]
        )
        self.current = current_json([t.replace(".", "-") for t in ours])
        self.components = components_csv(ours, date(2026, 11, 8))
        self.changes = changes_csv([(date(2026, 9, 21), ["BE"], ["TAP"])])
        self.ledger_edit = "2026-11-03T10:00:00Z"
        self.current_edit = "2026-11-08T10:00:00Z"
        self.commit_date = "2026-11-08T01:00:00Z"
        self.press_listing = press_listing_html()
        self.press_releases: dict[str, bytes] = {}  # release id -> page
        self.errors: dict[str, Exception | FakeResponse] = {}
        self.calls: list[tuple[str, dict[str, str], dict[str, str]]] = []

    def __call__(self, url: str, params: dict[str, str] | None = None, **kw: Any) -> FakeResponse:
        params = params or {}
        self.calls.append((url, params, kw.get("headers", {})))
        key = self._key(url, params)
        if key in self.errors:
            err = self.errors[key]
            if isinstance(err, Exception):
                raise err
            return err
        if key == "spy":
            return reply(self.spy)
        if key == "ledger":
            return reply(self.ledger)
        if key == "current":
            return reply(self.current)
        if key == "revisions":
            return reply(revisions_json(self.ledger_edit, self.current_edit))
        if key == "commits":
            return reply(
                json.dumps([{"commit": {"committer": {"date": self.commit_date}}}]).encode()
            )
        if key == "components":
            return reply(self.components)
        if key == "press_listing":
            return reply(self.press_listing)
        if key.startswith("press_"):
            page = self.press_releases.get(key.removeprefix("press_"))
            return reply(page) if page is not None else reply(b"not found", 404)
        return reply(self.changes)

    @staticmethod
    def _key(url: str, params: dict[str, str]) -> str:
        if "ssga.com" in url:
            return "spy"
        if url == PRESS_LISTING_URL:
            return "press_listing"
        if "prnewswire.com/news-releases/" in url:
            return "press_" + url.rsplit("-", 1)[-1].removesuffix(".html")
        if url == WIKI_API:
            if params.get("action") == "query":
                return "revisions"
            return "ledger" if "Historical" in params["page"] else "current"
        if url == CHINO_COMMITS_URL:
            return "commits"
        return "components" if "components" in url else "changes"


class FakeWeeklyStore:
    """An in-memory ``WeeklyStore``; it can only read a vintage and append run records."""

    def __init__(
        self,
        ours: list[str],
        *,
        known_through: date | None = None,
        fail_record: bool = False,
        fail_latest: bool = False,
    ) -> None:
        self._vintage: VintageData | None = vintage_data(history(ours))
        self._known_through = known_through
        self.fail_record = fail_record
        self.fail_latest = fail_latest
        self.runs: list[tuple[HoldingsResult, list[SnapshotRow]]] = []

    def latest(self, index_name: str) -> VintageData | None:
        if self.fail_latest:
            raise RuntimeError("postgres://user:secret@host/db refused the connection")
        return self._vintage

    def known_through(self, load_id: int, index_name: str) -> date | None:
        return self._known_through

    def record_run(self, index_name: str, result: HoldingsResult, rows: list[SnapshotRow]) -> int:
        if self.fail_record:
            raise RuntimeError("postgres://user:secret@host/db is down")
        self.runs.append((result, rows))
        return len(self.runs)


class RecordingNotifier:
    """Stands in for a notifier: remembers every message, sends nothing anywhere."""

    def __init__(self, delivers: bool = True) -> None:
        self.messages: list[str] = []
        self.delivers = delivers

    def send(self, message: str) -> bool:
        self.messages.append(message)
        return self.delivers


def base_ours(n: int = 495) -> list[str]:
    """Our list: ``n`` plain names plus the four share-class names, raw spelling."""
    return [f"X{i:04d}" for i in range(n)] + CLASSES


class NoNetwork(Exception):
    """Raised by the guards that make a real network call impossible in these tests."""


def no_real_network(*_: Any, **__: Any) -> Any:
    """Installed over ``requests`` and ``socket`` entry points by the tests' autouse fixture."""
    raise NoNetwork("a test tried to reach the network")
