"""Tests for reading dead stocks' Form 8-K exit events and classifying item evidence."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from sec_client import SecFetchError, SecNotFound
from sec_filings import (
    EVENTS_STATUS_NO_OLDER_PAGES_NEEDED,
    EVENTS_STATUS_NOT_FOUND,
    EVENTS_STATUS_OK,
    EVENTS_STATUS_OLDER_PAGE_NOT_FOUND,
    EVENTS_STATUSES,
    EVIDENCE_ITEM_RULES,
    count_items_missing,
    evidence_class,
    extract_8k_events,
    fetch_events,
    fetch_events_with_status,
    older_page_urls,
    submissions_url,
)

pytestmark = pytest.mark.study


def page_fixture(rows: list[tuple[Any, ...]]) -> dict[str, list[Any]]:
    """Build one SEC submissions block from (form, filingDate, items[, accession, doc]) tuples."""
    forms: list[str] = []
    dates: list[str] = []
    items: list[Any] = []
    accessions: list[str] = []
    docs: list[str] = []
    for idx, row in enumerate(rows, start=1):
        form, filed, raw_items = row[0], row[1], row[2]
        accession = row[3] if len(row) > 3 else f"0000000001-10-{idx:06d}"
        doc = row[4] if len(row) > 4 else f"doc{idx}.htm"
        forms.append(form)
        dates.append(filed)
        items.append(raw_items)
        accessions.append(accession)
        docs.append(doc)
    return {
        "form": forms,
        "filingDate": dates,
        "items": items,
        "accessionNumber": accessions,
        "primaryDocument": docs,
    }


def recent_fixture(
    rows: list[tuple[Any, ...]],
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a top-level SEC submissions payload holding ``recent`` and optional ``files``."""
    return {
        "cik": "0000000001",
        "filings": {
            "recent": page_fixture(rows),
            "files": list(files) if files is not None else [],
        },
    }


def json_bytes(payload: Any) -> bytes:
    """Encode a JSON fixture as UTF-8 bytes for the fake client."""
    return json.dumps(payload).encode("utf-8")


class FakeClient:
    """Serves canned bytes by URL; raises ``SecNotFound`` when absent or any stored exception."""

    def __init__(self, bodies: dict[str, object]) -> None:
        self.bodies = bodies
        self.requested: list[str] = []

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        body = self.bodies.get(url)
        if body is None:
            raise SecNotFound(f"not found: {url}")
        if isinstance(body, Exception):
            raise body
        assert isinstance(body, bytes)
        return body


# --- Six verbatim tests from the Task 7 brief ------------------------------------------------


def test_extract_events_filters_form_and_window():
    subs = recent_fixture(
        [
            ("8-K", "2010-07-01", "2.01,3.03"),
            ("10-Q", "2010-07-02", ""),
            ("8-K", "2010-12-01", "5.02"),
            ("8-K/A", "2010-07-10", "2.01"),
        ]
    )
    out = extract_8k_events(subs, pages=[], around=date(2010, 7, 5), days=90)
    assert [e["filing_date"] for e in out] == ["2010-07-01", "2010-07-10"]
    assert out[0]["items"] == ["2.01", "3.03"]


def test_older_pages_are_read():
    subs = recent_fixture([])
    page = page_fixture([("8-K", "2010-07-01", "1.03")])
    assert extract_8k_events(subs, pages=[page], around=date(2010, 7, 5), days=30)[0]["items"] == [
        "1.03"
    ]


def test_evidence_class_order():
    assert evidence_class([{"items": ["1.03", "2.01"]}]) == "bankruptcy_filed"
    assert evidence_class([{"items": ["3.03"]}]) == "acquired_or_merged"
    assert evidence_class([{"items": ["3.01"]}]) == "delisted_other"
    assert evidence_class([{"items": ["8.01"]}]) == "no_8k_found"
    assert evidence_class([]) == "no_8k_found"


def test_older_page_urls():
    subs = {"filings": {"files": [{"name": "CIK0000000001-submissions-001.json"}]}}
    assert older_page_urls(subs) == [
        "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json"
    ]


def test_submissions_url_pads_cik():
    assert submissions_url("0000000042") == "https://data.sec.gov/submissions/CIK0000000042.json"


def test_empty_items_are_kept_and_flagged():
    subs = recent_fixture([("8-K", "2010-07-01", "")])
    out = extract_8k_events(subs, pages=[], around=date(2010, 7, 5), days=30)
    assert out[0]["items"] == []


# --- Extra tests covering design decisions (a)-(f) and edge cases ----------------------------


def test_submissions_url_pads_unpadded_cik_and_rejects_out_of_range():
    assert submissions_url("42") == "https://data.sec.gov/submissions/CIK0000000042.json"
    for bad in ("0", "-5", "10000000000", "abc"):
        with pytest.raises(ValueError):
            submissions_url(bad)


def test_older_page_urls_handles_empty_and_rejects_traversal():
    assert older_page_urls({}) == []
    assert older_page_urls({"filings": {"files": []}}) == []
    with pytest.raises(ValueError):
        older_page_urls({"filings": {"files": [{"name": "../other.json"}]}})
    with pytest.raises(ValueError):
        older_page_urls({"filings": {"files": [{"name": "sub/page.json"}]}})


def test_extract_events_shape_window_edges_and_items_formats():
    # Inclusive edges: 2010-07-05 +/- 30 days is 2010-06-05 .. 2010-08-04.
    subs = recent_fixture(
        [
            ("8-K", "2010-06-04", "1.03", "0000000001-10-000001", "too_early.htm"),
            ("8-K", "2010-06-05", " 2.01 , 3.03 , ", "0000000001-10-000002", "start.htm"),
            ("8-K/A", "2010-08-04", ["5.01", "9.01"], "0000000001-10-000003", "end.htm"),
            ("8-K", "2010-08-05", "3.01", "0000000001-10-000004", "too_late.htm"),
        ]
    )
    out = extract_8k_events(subs, pages=[], around=date(2010, 7, 5), days=30)
    assert out == [
        {
            "accession": "0000000001-10-000002",
            "filing_date": "2010-06-05",
            "items": ["2.01", "3.03"],
            "primary_document": "start.htm",
        },
        {
            "accession": "0000000001-10-000003",
            "filing_date": "2010-08-04",
            "items": ["5.01", "9.01"],
            "primary_document": "end.htm",
        },
    ]


def test_extract_events_sorts_by_filing_date_then_accession_and_deduplicates():
    subs = recent_fixture(
        [
            ("8-K", "2010-07-10", "5.01", "0000000001-10-000005", "c.htm"),
            ("8-K", "2010-07-01", "2.01", "0000000001-10-000002", "b.htm"),
            ("8-K", "2010-07-01", "3.01", "0000000001-10-000001", "a.htm"),
        ]
    )
    # Duplicate accession across recent and an older page is kept once.
    page = page_fixture(
        [
            ("8-K", "2010-07-01", "3.01", "0000000001-10-000001", "a.htm"),
            ("8-K", "2010-06-20", "1.03", "0000000001-10-000000", "old.htm"),
        ]
    )
    out = extract_8k_events(subs, pages=[page], around=date(2010, 7, 5), days=30)
    assert [(e["filing_date"], e["accession"]) for e in out] == [
        ("2010-06-20", "0000000001-10-000000"),
        ("2010-07-01", "0000000001-10-000001"),
        ("2010-07-01", "0000000001-10-000002"),
        ("2010-07-10", "0000000001-10-000005"),
    ]


def test_extract_events_rejects_mismatched_parallel_arrays():
    bad = {
        "filings": {
            "recent": {
                "form": ["8-K"],
                "filingDate": ["2010-07-01", "2010-07-02"],
                "items": ["1.03"],
                "accessionNumber": ["0000000001-10-000001"],
                "primaryDocument": ["a.htm"],
            }
        }
    }
    with pytest.raises(ValueError):
        extract_8k_events(bad, pages=[], around=date(2010, 7, 5), days=30)


def test_fetch_events_skips_older_pages_when_recent_reaches_before_window():
    # Window is 2010-06-05 .. 2010-08-04; recent reaches back to 2010-06-04 (strictly before).
    older_name = "CIK0000000042-submissions-001.json"
    subs = recent_fixture(
        [
            ("8-K", "2010-07-10", "2.01", "0000000042-10-000002", "b.htm"),
            ("10-Q", "2010-06-04", "", "0000000042-10-000001", "a.htm"),
        ],
        files=[{"name": older_name}],
    )
    main_url = submissions_url("42")
    client = FakeClient({main_url: json_bytes(subs)})
    events, status = fetch_events_with_status(client, "42", around=date(2010, 7, 5), days=30)
    assert status == EVENTS_STATUS_NO_OLDER_PAGES_NEEDED
    assert status in EVENTS_STATUSES
    assert client.requested == [main_url]
    assert [e["accession"] for e in events] == ["0000000042-10-000002"]


def test_fetch_events_reads_older_page_when_recent_stops_on_or_after_window_start():
    # Earliest recent date equals window start (2010-06-05): boundary filings may sit on page 1.
    older_name = "CIK0000000042-submissions-001.json"
    subs = recent_fixture(
        [
            ("8-K", "2010-07-10", "2.01", "0000000042-10-000003", "c.htm"),
            ("10-Q", "2010-06-05", "", "0000000042-10-000002", "b.htm"),
        ],
        files=[{"name": older_name}],
    )
    page = page_fixture([("8-K", "2010-06-05", "1.03", "0000000042-10-000001", "a.htm")])
    main_url = submissions_url("42")
    page_url = f"https://data.sec.gov/submissions/{older_name}"
    client = FakeClient({main_url: json_bytes(subs), page_url: json_bytes(page)})

    events, status = fetch_events_with_status(client, "42", around=date(2010, 7, 5), days=30)
    assert status == EVENTS_STATUS_OK
    assert client.requested == [main_url, page_url]
    assert [(e["filing_date"], e["accession"]) for e in events] == [
        ("2010-06-05", "0000000042-10-000001"),
        ("2010-07-10", "0000000042-10-000003"),
    ]
    # Plain fetch_events delegates to fetch_events_with_status and returns the sorted list.
    assert fetch_events(client, "42", around=date(2010, 7, 5), days=30) == events


def test_fetch_events_uses_page_date_ranges_to_skip_irrelevant_older_pages():
    # Recent only covers 2015+; of three older pages, only page 2 overlaps 2010-06-05..2010-08-04.
    subs = recent_fixture(
        [("10-K", "2015-03-01", "", "0000000042-15-000001", "k.htm")],
        files=[
            {
                "name": "CIK0000000042-submissions-001.json",
                "filingFrom": "2011-01-01",
                "filingTo": "2014-12-31",
            },
            {
                "name": "CIK0000000042-submissions-002.json",
                "filingFrom": "2009-01-01",
                "filingTo": "2010-12-31",
            },
            {
                "name": "CIK0000000042-submissions-003.json",
                "filingFrom": "2000-01-01",
                "filingTo": "2008-12-31",
            },
        ],
    )
    page2 = page_fixture([("8-K", "2010-07-02", "5.01", "0000000042-10-000001", "m.htm")])
    main_url = submissions_url("42")
    page2_url = "https://data.sec.gov/submissions/CIK0000000042-submissions-002.json"
    client = FakeClient({main_url: json_bytes(subs), page2_url: json_bytes(page2)})

    events, status = fetch_events_with_status(client, "42", around=date(2010, 7, 5), days=30)
    assert status == EVENTS_STATUS_OK
    assert client.requested == [main_url, page2_url]
    assert [e["items"] for e in events] == [["5.01"]]


def test_fetch_events_with_no_older_pages_listed_returns_ok():
    subs = recent_fixture([("8-K", "2010-07-01", "2.01", "0000000042-10-000001", "a.htm")])
    main_url = submissions_url("42")
    client = FakeClient({main_url: json_bytes(subs)})
    events, status = fetch_events_with_status(client, "42", around=date(2010, 7, 5), days=30)
    assert status == EVENTS_STATUS_OK
    assert client.requested == [main_url]
    assert [e["items"] for e in events] == [["2.01"]]


def test_fetch_events_returns_empty_and_not_found_on_404():
    client = FakeClient({})
    events, status = fetch_events_with_status(client, "42", around=date(2010, 7, 5), days=90)
    assert events == []
    assert status == EVENTS_STATUS_NOT_FOUND
    assert status in EVENTS_STATUSES
    assert fetch_events(client, "42", around=date(2010, 7, 5), days=90) == []


def test_fetch_events_records_missing_older_page_without_crashing():
    older_name = "CIK0000000042-submissions-001.json"
    subs = recent_fixture(
        [("8-K", "2010-07-10", "3.01", "0000000042-10-000002", "b.htm")],
        files=[{"name": older_name}],
    )
    main_url = submissions_url("42")
    client = FakeClient({main_url: json_bytes(subs)})  # older page URL is absent -> 404

    events, status = fetch_events_with_status(client, "42", around=date(2010, 7, 5), days=30)
    assert status == EVENTS_STATUS_OLDER_PAGE_NOT_FOUND
    assert status in EVENTS_STATUSES
    assert [e["accession"] for e in events] == ["0000000042-10-000002"]


def test_fetch_events_does_not_swallow_non_404_fetch_errors():
    main_url = submissions_url("42")
    client = FakeClient({main_url: SecFetchError("HTTP 500")})
    with pytest.raises(SecFetchError, match="500"):
        fetch_events_with_status(client, "42", around=date(2010, 7, 5), days=90)


def test_evidence_item_rules_and_priority_across_multiple_events():
    assert [label for label, _ in EVIDENCE_ITEM_RULES] == [
        "bankruptcy_filed",
        "acquired_or_merged",
        "delisted_other",
    ]
    assert evidence_class([{"items": ["3.01"]}, {"items": ["1.03"]}]) == "bankruptcy_filed"
    assert evidence_class([{"items": ["3.01"]}, {"items": ["5.01"]}]) == "acquired_or_merged"
    assert evidence_class([{"items": ["2.01"]}]) == "acquired_or_merged"
    assert evidence_class([{"items": []}]) == "no_8k_found"


def test_count_items_missing_counts_empty_item_events():
    assert count_items_missing([]) == 0
    events = [
        {"accession": "1", "filing_date": "2010-07-01", "items": [], "primary_document": "a.htm"},
        {
            "accession": "2",
            "filing_date": "2010-07-02",
            "items": ["2.01"],
            "primary_document": "b.htm",
        },
        {"accession": "3", "filing_date": "2010-07-03", "items": [], "primary_document": "c.htm"},
    ]
    assert count_items_missing(events) == 2
