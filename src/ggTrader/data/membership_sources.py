"""The public sources the weekly membership job reads: fetchers and parsers, no database.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, Appendix C1. The upstream
feed is dead, so the S&P 500 list is written from evidence and cross-checked against four public
sources besides the SPDR SPY holdings (``membership_holdings``):

* ``wikipedia_ledger``: the change ledger "Historical components of the S&P 500", read through the
  MediaWiki API (``action=parse``, ``prop=wikitext``);
* ``wikipedia_current``: "List of S&P 500 companies" (the current members), same API;
* ``chinobing_components`` and ``chinobing_changes``: the two raw CSVs of
  chinobing/historical_sp500_constituents on GitHub.

Each fetch is independent: a failure is returned as a :class:`SourceFetch` with ``ok=False``, never
raised, so one dead source cannot hide the others. The job is polite: one request per file, a
descriptive User-Agent, a timeout, no retry, ``maxlag`` on the wiki API, and two small metadata
calls (the wiki pages' last-edit times in one MediaWiki query, the GitHub repository's last commit)
that give each source its own *last-change date*.

The parsers are strict about structure on purpose. The ledger table is read by header: a missing,
renamed or extra column, a row with the wrong number of cells, or a table that vanished raises
:class:`SourceStructureError` so the job reports "the page changed" instead of reading garbage.
Every network call takes an injectable ``get`` (``requests.get``-compatible), so tests run offline.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import requests

from ggTrader.data.core.index_constituents import normalize_yf_ticker
from ggTrader.data.universe_algebra import MAX_MEMBERS, MIN_MEMBERS

#: Identifies the job to the servers (Wikimedia asks for a descriptive agent with a contact).
USER_AGENT = (
    "ggTrader-membership-check/1.0 (https://github.com/garykuepper/ggTrader; "
    "weekly personal research job, about 8 requests per run)"
)
#: Seconds to wait for each request.
TIMEOUT_SECONDS = 60
#: Largest body any one source may send before it is refused (the biggest real file is 8 MB).
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
#: Wikimedia's polite-client hint: the API refuses while the database lags more than this.
MAX_LAG_SECONDS = 5

WIKI_API = "https://en.wikipedia.org/w/api.php"
LEDGER_PAGE = "Historical_components_of_the_S&P_500"
CURRENT_PAGE = "List_of_S&P_500_companies"
CHINO_REPO = "chinobing/historical_sp500_constituents"
CHINO_RAW = f"https://raw.githubusercontent.com/{CHINO_REPO}/main/"
CHINO_COMPONENTS_URL = CHINO_RAW + "sp_500_historical_components.csv"
CHINO_CHANGES_URL = CHINO_RAW + "sp500_changes_since_1996.csv"
CHINO_COMMITS_URL = f"https://api.github.com/repos/{CHINO_REPO}/commits"

#: Source names, as stored in ``universe.source_snapshots.source``.
LEDGER = "wikipedia_ledger"
CURRENT = "wikipedia_current"
COMPONENTS = "chinobing_components"
CHANGES = "chinobing_changes"
SPY = "spy_holdings"
FETCHED_SOURCES = (LEDGER, CURRENT, COMPONENTS, CHANGES)

#: Header the change ledger must have, top row then second row (compared case-insensitively).
LEDGER_TOP_HEADER = ("effective date", "added", "removed", "reason", "refs")
LEDGER_SUB_HEADER = ("ticker", "security", "ticker", "security")
#: Cells in a ledger data row: date, added (ticker, security), removed (ticker, security), reason,
#: refs.
LEDGER_CELLS = 7
#: A ledger with fewer data rows than this is a truncated or replaced page, not the ledger.
MIN_LEDGER_ROWS = 100
#: Rows this many from the top of the page (the newest) must parse; an older bad row is a warning.
STRICT_NEWEST_ROWS = 60
#: Columns the current-members table must have, in this order at the front.
CURRENT_REQUIRED_HEADER = ("symbol", "security")

_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_LONG_DATE = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z0-9]{1,2})?$")
_PRESS_URL_DATE = re.compile(r"press\.spglobal\.com/(\d{4})-(\d{2})-(\d{2})-")
_PDF_URL_DATE = re.compile(r"/announcements/(\d{4})(\d{2})(\d{2})-")
_REF_PARAM_DATE = re.compile(r"(?<![-\w])date\s*=\s*([^|}\n<]+)", re.I)
_REF_DEFINITION = re.compile(r"<ref\s+name\s*=\s*\"?([^\">/\s]+)\"?[^>/]*>(.*?)</ref>", re.S)
_REF_USE = re.compile(r"<ref\s+name\s*=\s*\"?([^\">/\s]+)\"?\s*/>")

GetFn = Callable[..., Any]


class SourceStructureError(ValueError):
    """A source parsed, but not into the structure it is expected to have.

    ``code`` is a short machine word (``columns_changed``, ``table_not_found``, ``bad_row``,
    ``too_few_rows``, ``api_error``, ``bad_csv``); ``problems`` lists every defect found.
    """

    def __init__(self, code: str, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.problems = problems or [message]


# --------------------------------------------------------------------------- fetch records


@dataclass
class SourceFetch:
    """What one fetch brought back: the bytes and their provenance, or why there are none.

    ``last_change`` is the source's own last-change date when known (the wiki page's last edit,
    the repository's last commit); it is filled in by :func:`fetch_sources` after the metadata
    calls and stays ``None`` when they failed (the job then says it cannot tell).
    """

    name: str
    url: str
    retrieved_at: datetime
    ok: bool
    raw: bytes | None = None
    reason: str | None = None
    error: str | None = None
    http_status: int | None = None
    etag: str | None = None
    last_change: date | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _failure(
    name: str, url: str, when: datetime, reason: str, error: str, **kw: Any
) -> SourceFetch:
    return SourceFetch(name, url, when, False, None, reason, error, **kw)


def _request(
    get: GetFn,
    name: str,
    url: str,
    clock: Callable[[], datetime],
    *,
    params: dict[str, str] | None = None,
    accept: str = "*/*",
    expect_json: bool = False,
) -> SourceFetch:
    """One GET: HTTP 200, complete body, size cap; failures come back as ``ok=False``."""
    try:
        response = get(
            url,
            params=params,
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": accept,
                "Accept-Encoding": "identity",  # keeps Content-Length equal to the body we count
            },
        )
    except requests.RequestException as exc:
        return _failure(name, url, clock(), "unreachable", f"{type(exc).__name__}: {exc}")
    when = clock()
    status = int(response.status_code)
    if status != 200:
        return _failure(name, url, when, "http_status", f"HTTP {status}", http_status=status)
    raw = bytes(response.content)
    if len(raw) > MAX_DOWNLOAD_BYTES:
        return _failure(name, url, when, "too_large", f"{len(raw)} bytes", http_status=status)
    if not raw:
        return _failure(name, url, when, "empty", "the body is empty", http_status=status)
    declared = response.headers.get("content-length")
    if declared is not None and not response.headers.get("content-encoding"):
        if declared.isdigit() and int(declared) != len(raw):
            return _failure(
                name,
                url,
                when,
                "truncated",
                f"{len(raw)} bytes but content-length says {declared}",
                http_status=status,
            )
    if expect_json:
        try:
            json.loads(raw)
        except ValueError:
            return _failure(name, url, when, "not_json", "the body is not JSON", http_status=status)
    return SourceFetch(
        name,
        url,
        when,
        True,
        raw,
        http_status=status,
        etag=response.headers.get("etag"),
        detail={"final_url": str(getattr(response, "url", None) or url)},
    )


def wiki_params(page: str) -> dict[str, str]:
    """The MediaWiki ``action=parse`` query for one page's wikitext (``maxlag`` for politeness)."""
    return {
        "action": "parse",
        "page": page,
        "prop": "wikitext|revid",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "maxlag": str(MAX_LAG_SECONDS),
    }


def fetch_wiki_page(
    name: str, page: str, *, get: GetFn, clock: Callable[[], datetime]
) -> SourceFetch:
    """Fetch one wiki page's parse JSON; an API-level error object is a failure too."""
    result = _request(
        get,
        name,
        WIKI_API,
        clock,
        params=wiki_params(page),
        accept="application/json",
        expect_json=True,
    )
    result.detail["page"] = page
    if not result.ok:
        return result
    body = json.loads(result.raw or b"")
    if isinstance(body, dict) and "error" in body:
        code = (body.get("error") or {}).get("code", "unknown")
        result.ok, result.reason, result.error = False, "api_error", f"MediaWiki error: {code}"
        return result
    revid = (body.get("parse") or {}).get("revid") if isinstance(body, dict) else None
    result.detail["revid"] = revid
    return result


def fetch_wiki_edit_times(
    pages: Iterable[str], *, get: GetFn, clock: Callable[[], datetime]
) -> dict[str, datetime] | str:
    """Last-edit instants of the pages in one MediaWiki query, or an error text.

    The key is the page title with spaces as in the API (``List of S&P 500 companies``).
    """
    result = _request(
        get,
        "wikipedia_revisions",
        WIKI_API,
        clock,
        params={
            "action": "query",
            "prop": "revisions",
            "rvprop": "ids|timestamp",
            "titles": "|".join(pages),
            "format": "json",
            "formatversion": "2",
            "maxlag": str(MAX_LAG_SECONDS),
        },
        accept="application/json",
        expect_json=True,
    )
    if not result.ok:
        return f"{result.reason}: {result.error}"
    try:
        found = json.loads(result.raw or b"")["query"]["pages"]
        return {
            p["title"]: datetime.fromisoformat(
                p["revisions"][0]["timestamp"].replace("Z", "+00:00")
            )
            for p in found
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return "invalid: the revisions reply has no timestamps"


def fetch_chino_last_commit(*, get: GetFn, clock: Callable[[], datetime]) -> datetime | str:
    """The instant of the GitHub repository's newest commit, or an error text."""
    result = _request(
        get,
        "chinobing_commits",
        CHINO_COMMITS_URL,
        clock,
        params={"per_page": "1"},
        accept="application/vnd.github+json",
        expect_json=True,
    )
    if not result.ok:
        return f"{result.reason}: {result.error}"
    try:
        stamp = json.loads(result.raw or b"")[0]["commit"]["committer"]["date"]
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (KeyError, IndexError, TypeError, ValueError):
        return "invalid: the commits reply has no commit date"


def fetch_sources(
    *, get: GetFn | None = None, now: Callable[[], datetime] | None = None
) -> dict[str, SourceFetch]:
    """Fetch the four public sources and their last-change dates; never raises.

    Seven requests: the two wiki pages, one query for both pages' last-edit times, the two
    chinobing CSVs and the repository's newest commit (the SPY workbook is fetched by
    ``membership_holdings``).
    """
    clock = now or (lambda: datetime.now(timezone.utc))
    get = get or requests.get  # looked up at call time so tests can replace it
    out: dict[str, SourceFetch] = {}

    def guarded(name: str, url: str, make: Callable[[], SourceFetch]) -> SourceFetch:
        try:
            return make()
        except Exception as exc:  # noqa: BLE001 - one source must never abort the others
            return _failure(name, url, clock(), "internal", f"{type(exc).__name__}: {exc}")

    out[LEDGER] = guarded(
        LEDGER, WIKI_API, lambda: fetch_wiki_page(LEDGER, LEDGER_PAGE, get=get, clock=clock)
    )
    out[CURRENT] = guarded(
        CURRENT, WIKI_API, lambda: fetch_wiki_page(CURRENT, CURRENT_PAGE, get=get, clock=clock)
    )
    for name, url in ((COMPONENTS, CHINO_COMPONENTS_URL), (CHANGES, CHINO_CHANGES_URL)):
        out[name] = guarded(
            name, url, lambda name=name, url=url: _request(get, name, url, clock, accept="text/csv")
        )
    edits = guarded_value(
        lambda: fetch_wiki_edit_times(
            [LEDGER_PAGE.replace("_", " "), CURRENT_PAGE.replace("_", " ")], get=get, clock=clock
        )
    )
    commit = guarded_value(lambda: fetch_chino_last_commit(get=get, clock=clock))
    for name, page in ((LEDGER, LEDGER_PAGE), (CURRENT, CURRENT_PAGE)):
        source = out[name]
        if isinstance(edits, dict) and page.replace("_", " ") in edits:
            instant = edits[page.replace("_", " ")]
            source.last_change = instant.astimezone(timezone.utc).date()
            source.detail["last_edit"] = instant.astimezone(timezone.utc).isoformat()
        else:
            source.detail["last_change_error"] = (
                edits if isinstance(edits, str) else "the page is not in the revisions reply"
            )
    for name in (COMPONENTS, CHANGES):
        source = out[name]
        if isinstance(commit, datetime):
            source.last_change = commit.astimezone(timezone.utc).date()
            source.detail["last_commit"] = commit.astimezone(timezone.utc).isoformat()
        else:
            source.detail["last_change_error"] = commit
    return out


def guarded_value(make: Callable[[], Any]) -> Any:
    """Run a metadata call; any exception becomes an error text (the date is then 'unknown')."""
    try:
        return make()
    except Exception as exc:  # noqa: BLE001 - a missing date must not abort the run
        return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- wikitext tokens


def _opaque_end(text: str, i: int) -> int | None:
    """End index of a ``<ref ...>...</ref>``, ``<ref .../>`` or ``<!-- -->`` starting at ``i``."""
    if text.startswith("<!--", i):
        end = text.find("-->", i)
        return len(text) if end < 0 else end + 3
    if text.startswith("<ref", i):
        head = text.find(">", i)
        if head < 0:
            return len(text)
        if text[head - 1] == "/":
            return head + 1
        end = text.find("</ref>", head)
        return len(text) if end < 0 else end + 6
    return None


def split_cells(text: str, *, header: bool = False) -> list[str]:
    """Split one table row's wikitext into raw cell texts, markup left in place.

    A cell starts at a line-initial ``|`` (``!`` in a header row) or at ``||`` (``!!`` in a header
    row) outside ``{{ }}``, ``[[ ]]``, ``<ref>`` and comments; a line-initial ``||`` is one marker
    (the editors' habit), not an empty cell. Text before the first marker is dropped.
    """
    lead, sep = ("!", "!!") if header else ("|", "||")
    cells: list[str] = []
    current: list[str] = []
    started = False
    depth_t = depth_l = 0
    i, n = 0, len(text)
    while i < n:
        opaque = _opaque_end(text, i)
        if opaque is not None:
            current.append(text[i:opaque])
            i = opaque
            continue
        line_start = i == 0 or text[i - 1] == "\n"
        if depth_t == 0 and depth_l == 0:
            if line_start and text[i] == lead:
                if started:
                    cells.append("".join(current))
                current, started = [], True
                i += 1
                if text.startswith(lead, i) and not header:  # "|| x" at line start
                    i += 1
                continue
            if started and text.startswith(sep, i):
                cells.append("".join(current))
                current = []
                i += len(sep)
                continue
            if header and started and text.startswith("||", i):
                cells.append("".join(current))
                current = []
                i += 2
                continue
        if text.startswith("{{", i):
            depth_t += 1
            current.append("{{")
            i += 2
        elif text.startswith("}}", i) and depth_t:
            depth_t -= 1
            current.append("}}")
            i += 2
        elif text.startswith("[[", i):
            depth_l += 1
            current.append("[[")
            i += 2
        elif text.startswith("]]", i) and depth_l:
            depth_l -= 1
            current.append("]]")
            i += 2
        else:
            current.append(text[i])
            i += 1
    if started:
        cells.append("".join(current))
    return cells


_ATTRIBUTES = re.compile(r"^\s*(?:[A-Za-z-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s|]+)\s*)+\|(?!\|)")


def clean_cell(cell: str) -> str:
    """Plain text of a cell: attributes, refs, comments, templates and link markup removed."""
    text = _ATTRIBUTES.sub("", cell, count=1)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    return re.sub(r"\s+", " ", text.replace("'''", "").replace("''", "")).strip()


def _table_text(wikitext: str, marker: str) -> str:
    """The text of the wikitable whose opening line contains ``marker``."""
    match = re.search(r"^\{\|[^\n]*" + re.escape(marker) + r"[^\n]*$", wikitext, re.M)
    if match is None:
        raise SourceStructureError("table_not_found", f"no table with {marker} on the page")
    end = wikitext.find("\n|}", match.start())
    return wikitext[match.start() : len(wikitext) if end < 0 else end]


def _split_table(table: str) -> tuple[list[str], list[str]]:
    """``(header segments, data segments)``: rows split at ``|-``; header rows start with ``!``."""
    parts = re.split(r"\n\|-+[^\n]*", table)[1:]  # part 0 is the table's opening line
    header: list[str] = []
    for part in parts:
        first = next((ln for ln in part.split("\n") if ln.strip()), "")
        if first.startswith("!"):
            header.append(part)
        else:
            break
    return header, parts[len(header) :]


def _wikitext(raw: bytes) -> tuple[str, int | None]:
    try:
        body = json.loads(raw)
        parsed = body["parse"]
        return str(parsed["wikitext"]), parsed.get("revid")
    except (ValueError, KeyError, TypeError):
        error = None
        try:
            error = json.loads(raw).get("error", {}).get("code")
        except (ValueError, AttributeError):
            pass
        raise SourceStructureError(
            "api_error", f"the reply has no parse.wikitext (error: {error})"
        ) from None


def parse_wiki_date(text: str) -> date | None:
    """``October 30, 2026`` or ``2026-10-30`` (also inside longer text) -> a date, else None."""
    match = _LONG_DATE.search(text)
    if match and match.group(1).lower() in _MONTHS:
        try:
            return date(
                int(match.group(3)), _MONTHS.index(match.group(1).lower()) + 1, int(match.group(2))
            )
        except ValueError:
            return None
    iso = _ISO_DATE.search(text)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- the change ledger


@dataclass(frozen=True)
class LedgerRow:
    """One change in the ledger: a date, the name added and the name removed (either may be '')."""

    position: int
    effective: date
    added: str
    added_name: str
    removed: str
    removed_name: str
    reason: str
    announcement: date | None
    ref_dates: tuple[date, ...]


@dataclass
class Ledger:
    """The parsed ledger, newest page position first as on the page."""

    rows: list[LedgerRow]
    revid: int | None
    top_header: tuple[str, ...]
    sub_header: tuple[str, ...]
    warnings: list[str]

    @property
    def newest_effective(self) -> date:
        """The latest effective date on the page."""
        return max(row.effective for row in self.rows)


def _labels(segment: str) -> list[str]:
    return [clean_cell(c).lower() for c in split_cells(segment, header=True)]


def _ref_dates(cell: str, definitions: dict[str, str]) -> tuple[date, ...]:
    """Every press-release date a ref cell mentions: ``date=`` parameters and URL-embedded dates."""
    text = cell
    for name in _REF_USE.findall(cell):
        text += " " + definitions.get(name, "")
    found: set[date] = set()
    for value in _REF_PARAM_DATE.findall(text):
        parsed = parse_wiki_date(value)
        if parsed:
            found.add(parsed)
    for pattern in (_PRESS_URL_DATE, _PDF_URL_DATE):
        for y, m, d in pattern.findall(text):
            try:
                found.add(date(int(y), int(m), int(d)))
            except ValueError:
                continue
    return tuple(sorted(found))


def _ticker(cell: str) -> str:
    text = clean_cell(cell).rstrip("|").strip()
    return text


def parse_ledger(raw: bytes) -> Ledger:
    """Parse the ledger page; raise :class:`SourceStructureError` if its shape is not the expected.

    Checked: the ``id="changes"`` table exists; its two header rows are exactly
    ``Effective Date / Added / Removed / Reason / Refs`` and ``Ticker / Security / Ticker /
    Security``; at least :data:`MIN_LEDGER_ROWS` data rows; every data row has seven cells. A row
    whose date or ticker will not parse is a defect (an error among the newest
    :data:`STRICT_NEWEST_ROWS` rows, otherwise a warning).
    """
    wikitext, revid = _wikitext(raw)
    table = _table_text(wikitext, 'id="changes"')
    header, data = _split_table(table)
    if len(header) != 2:
        raise SourceStructureError(
            "columns_changed", f"the table has {len(header)} header rows, expected 2"
        )
    top, sub = tuple(_labels(header[0])), tuple(_labels(header[1]))
    problems: list[str] = []
    for found, wanted, what in (
        (top, LEDGER_TOP_HEADER, "header"),
        (sub, LEDGER_SUB_HEADER, "sub-header"),
    ):
        if found != wanted:
            missing = [w for w in wanted if w not in found]
            extra = [f for f in found if f not in wanted]
            problems.append(
                f"{what} is {list(found)}, expected {list(wanted)} "
                f"(missing {missing}, unexpected {extra})"
            )
    if problems:
        raise SourceStructureError(
            "columns_changed", "the ledger's columns changed: " + "; ".join(problems), problems
        )
    definitions = {m[0]: m[1] for m in _REF_DEFINITION.findall(table)}
    rows: list[LedgerRow] = []
    errors: list[str] = []
    warnings: list[str] = []
    for position, segment in enumerate(data):
        if not segment.strip():
            continue
        cells = split_cells(segment)
        problem: str | None = None
        if len(cells) == LEDGER_CELLS - 1:
            cells.append("")  # an older row with no refs cell at all: the last cell is empty
        if len(cells) != LEDGER_CELLS:
            problem = f"row {position} has {len(cells)} cells, expected {LEDGER_CELLS}"
        else:
            effective = parse_wiki_date(clean_cell(cells[0]))
            added, removed = _ticker(cells[1]), _ticker(cells[3])
            if effective is None:
                problem = f"row {position}: date {clean_cell(cells[0])!r} does not parse"
            elif any(t and not _TICKER.match(t) for t in (added, removed)):
                problem = f"row {position}: bad ticker {added!r}/{removed!r}"
            elif not (added or removed):
                problem = f"row {position}: neither an added nor a removed ticker"
        if problem is not None:
            (errors if position < STRICT_NEWEST_ROWS else warnings).append(problem)
            continue
        refs = _ref_dates(cells[6], definitions)
        rows.append(
            LedgerRow(
                position,
                effective,
                added,
                clean_cell(cells[2]),
                removed,
                clean_cell(cells[4]),
                clean_cell(cells[5]),
                min(refs) if refs else None,
                refs,
            )
        )
    if errors:
        raise SourceStructureError("bad_row", "; ".join(errors[:5]), errors)
    if len(rows) < MIN_LEDGER_ROWS:
        raise SourceStructureError(
            "too_few_rows", f"{len(rows)} ledger rows, fewer than the {MIN_LEDGER_ROWS} expected"
        )
    return Ledger(rows, revid, top, sub, warnings)


# --------------------------------------------------------------------------- the current list


@dataclass(frozen=True)
class CurrentList:
    """The current members as the page states them (normalised to ``BRK-B`` form)."""

    tickers: tuple[str, ...]
    revid: int | None


_SYMBOL_TEMPLATE = re.compile(
    r"\{\{[^|{}]+\|\s*([^|{}=]+?)\s*(?:\|[^{}]*)?\}\}"
)  # {{NyseSymbol|MMM}}


def parse_current_list(raw: bytes) -> CurrentList:
    """Parse the current-members table; raise :class:`SourceStructureError` on a changed shape."""
    wikitext, revid = _wikitext(raw)
    table = _table_text(wikitext, 'id="constituents"')
    header, data = _split_table(table)
    labels = [clean_cell(c).lower() for part in header for c in split_cells(part, header=True)]
    if tuple(labels[: len(CURRENT_REQUIRED_HEADER)]) != CURRENT_REQUIRED_HEADER:
        raise SourceStructureError(
            "columns_changed",
            f"the members table starts with {labels[:3]}, expected {list(CURRENT_REQUIRED_HEADER)}",
        )
    tickers: list[str] = []
    problems: list[str] = []
    for position, segment in enumerate(data):
        cells = split_cells(segment)
        if not cells:
            continue
        if len(cells) != len(labels):
            problems.append(f"row {position} has {len(cells)} cells, expected {len(labels)}")
            continue
        match = _SYMBOL_TEMPLATE.search(cells[0])
        text = match.group(1) if match else clean_cell(cells[0])
        text = clean_cell(text)
        if not _TICKER.match(text.upper()):
            problems.append(f"row {position}: bad symbol {text!r}")
            continue
        tickers.append(normalize_yf_ticker(text))
    if problems:
        raise SourceStructureError("bad_row", "; ".join(problems[:5]), problems)
    if len(tickers) != len(set(tickers)):
        raise SourceStructureError("bad_row", "duplicate symbols in the members table")
    if not MIN_MEMBERS <= len(tickers) <= MAX_MEMBERS:
        raise SourceStructureError(
            "too_few_rows", f"{len(tickers)} symbols is outside {MIN_MEMBERS}..{MAX_MEMBERS}"
        )
    return CurrentList(tuple(sorted(tickers)), revid)


# --------------------------------------------------------------------------- chinobing CSVs


@dataclass(frozen=True)
class ChinoSnapshot:
    """The last row of the components file: the date and the normalised tickers."""

    day: date
    tickers: tuple[str, ...]
    n_rows: int
    first_day: date


@dataclass(frozen=True)
class ChinoChange:
    """One row of the changes file (``day`` is the date chinobing *observed* the change)."""

    day: date
    added: tuple[str, ...]
    removed: tuple[str, ...]


def _csv_rows(raw: bytes, header: list[str]) -> list[list[str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SourceStructureError("bad_csv", "the file is not UTF-8") from None
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or [c.strip().lower() for c in rows[0]] != header:
        raise SourceStructureError(
            "bad_csv", f"the header is {rows[:1]}, expected {header}", [f"header {rows[:1]}"]
        )
    return [r for r in rows[1:] if any(c.strip() for c in r)]


def parse_chino_components(raw: bytes) -> ChinoSnapshot:
    """The newest snapshot of ``sp_500_historical_components.csv``.

    Dates must not go backwards (the file repeats a few days, so they are not strictly increasing).
    """
    rows = _csv_rows(raw, ["date", "tickers"])
    if len(rows) < 100:
        raise SourceStructureError("too_few_rows", f"{len(rows)} snapshot rows, expected 100+")
    try:
        days = [date.fromisoformat(r[0].strip()) for r in rows]
    except (ValueError, IndexError):
        raise SourceStructureError("bad_csv", "a snapshot date is not ISO") from None
    if any(b < a for a, b in zip(days, days[1:], strict=False)):
        raise SourceStructureError("bad_csv", "snapshot dates go backwards")
    last = sorted({normalize_yf_ticker(t) for t in rows[-1][1].split(",") if t.strip()})
    if not MIN_MEMBERS <= len(last) <= MAX_MEMBERS:
        raise SourceStructureError(
            "too_few_rows", f"the newest snapshot has {len(last)} tickers, outside range"
        )
    return ChinoSnapshot(days[-1], tuple(last), len(rows), days[0])


def _ticker_list(text: str) -> tuple[str, ...]:
    value = ast.literal_eval(text) if text.strip() else []
    if not isinstance(value, (list, tuple)) or not all(isinstance(t, str) for t in value):
        raise ValueError("not a list of strings")
    return tuple(value)


def parse_chino_changes(raw: bytes) -> list[ChinoChange]:
    """All rows of ``sp500_changes_since_1996.csv`` (``date, added_tickers, removed_tickers``)."""
    rows = _csv_rows(raw, ["date", "added_tickers", "removed_tickers"])
    if len(rows) < 100:
        raise SourceStructureError("too_few_rows", f"{len(rows)} change rows, expected 100+")
    out: list[ChinoChange] = []
    for i, row in enumerate(rows):
        try:
            out.append(
                ChinoChange(
                    date.fromisoformat(row[0].strip()), _ticker_list(row[1]), _ticker_list(row[2])
                )
            )
        except (ValueError, SyntaxError, IndexError):
            raise SourceStructureError("bad_csv", f"change row {i + 2} cannot be parsed") from None
    return out
