"""Test metadata selection over invented SEC submissions pages."""

import pytest
import sec_submissions

pytestmark = pytest.mark.study


def _page(dates, forms, accessions, names, report_dates=None):
    return {
        "filingDate": dates,
        "form": forms,
        "accessionNumber": accessions,
        "primaryDocument": names,
        "reportDate": report_dates if report_dates is not None else ["2020-01-01"] * len(dates),
    }


def _submissions():
    return {
        "cik": "0000000123",
        "filings": {
            "recent": _page(["2025-02-01"], ["10-K"], ["0000000123-25-000001"], ["new.htm"]),
            "files": [{"name": "old.json", "filingFrom": "2023-01-01", "filingTo": "2024-12-31"}],
        },
    }


def test_selects_periodic_filing_from_historical_page():
    pages = {
        "old.json": _page(
            ["2024-11-06", "2024-10-01", "2024-01-01"],
            ["10-Q", "8-K", "10-K"],
            ["0000000123-24-000003", "0000000123-24-000002", "0000000123-24-000001"],
            ["q.htm", "event.htm", "k.htm"],
            ["2024-09-30", "", "2023-12-31"],
        )
    }
    item = sec_submissions.latest_periodic_filing(
        _submissions(), pages, "2024-12-31", expected_cik=123
    )
    assert item["filing"] == {
        "filing_date": "2024-11-06",
        "report_date": "2024-09-30",
        "form": "10-Q",
        "accession": "0000000123-24-000003",
        "document": "q.htm",
    }
    assert item["future_range_discrepancies"] == []


def test_missing_historical_page_stops_instead_of_guessing():
    with pytest.raises(ValueError, match="Missing historical"):
        sec_submissions.latest_periodic_filing(_submissions(), {}, "2024-12-31", expected_cik=123)


def test_wrong_cik_and_misaligned_columns_are_rejected():
    doc = _submissions()
    with pytest.raises(ValueError, match="CIK"):
        sec_submissions.latest_periodic_filing(doc, {}, "2024-12-31", expected_cik=999)
    page = _page(["2024-01-01"], ["10-Q"], [], ["q.htm"])
    with pytest.raises(ValueError, match="columns"):
        sec_submissions.latest_periodic_filing(
            doc, {"old.json": page}, "2024-12-31", expected_cik=123
        )


def test_same_day_periodic_filings_are_not_arbitrarily_ordered():
    page = _page(
        ["2024-11-06"] * 2,
        ["10-K", "10-Q"],
        ["0000000123-24-000001", "0000000123-24-000002"],
        ["a.htm", "b.htm"],
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        sec_submissions.latest_periodic_filing(
            _submissions(), {"old.json": page}, "2024-12-31", expected_cik=123
        )


@pytest.mark.parametrize(
    "page",
    [
        _page(["2022-01-01"], ["10-K"], ["0000000123-22-000001"], ["k.htm"]),
        _page(["2024-11-06"], ["10-Q "], ["0000000123-24-000001"], ["q.htm"]),
        _page(["2024-11-06"], ["10-Q"], ["invalid"], ["q.htm"]),
        _page(["2024-11-06"], ["10-Q"], ["0000000123-24-000001"], ["../q.htm"]),
        _page(
            ["2024-11-06", "2024-08-01"],
            ["10-Q", "10-Q"],
            ["0000000123-24-000001"] * 2,
            ["q.htm"] * 2,
        ),
    ],
)
def test_rejects_misfiled_or_invalid_page_rows(page):
    doc = _submissions()
    doc["filings"]["recent"] = _page([], [], [], [])
    with pytest.raises(ValueError):
        sec_submissions.latest_periodic_filing(
            doc, {"old.json": page}, "2024-12-31", expected_cik=123
        )


def test_cutoff_before_observed_history_is_unresolved():
    with pytest.raises(ValueError, match="history"):
        sec_submissions.latest_periodic_filing(
            _submissions(),
            {"old.json": _page(["2024-01-01"], ["10-K"], ["0000000123-24-000001"], ["k.htm"])},
            "2022-12-31",
            expected_cik=123,
        )


def test_also_requires_pages_advertised_after_cutoff():
    doc = _submissions()
    doc["filings"]["files"][0]["filingFrom"] = "2025-01-01"
    doc["filings"]["files"][0]["filingTo"] = "2025-12-31"
    with pytest.raises(ValueError, match="Missing historical"):
        sec_submissions.latest_periodic_filing(doc, {}, "2024-12-31", expected_cik=123)


def test_reports_future_range_error_without_using_future_filing():
    page = _page(
        ["2024-11-06", "2025-01-02"],
        ["10-Q", "8-K"],
        ["0000000123-24-000001", "0000000123-25-000002"],
        ["q.htm", "e.htm"],
    )
    result = sec_submissions.latest_periodic_filing(
        _submissions(), {"old.json": page}, "2024-12-31", expected_cik=123
    )
    assert result["filing"]["accession"] == "0000000123-24-000001"
    assert result["future_range_discrepancies"] == [
        {"page": "old.json", "filing_date": "2025-01-02", "accession": "0000000123-25-000002"}
    ]


def test_cutoff_is_filing_date_not_publication_time():
    doc = _submissions()
    doc["filings"]["files"] = []
    result = sec_submissions.latest_periodic_filing(
        doc, {}, filing_date_cutoff="2025-02-01", expected_cik=123
    )
    assert result["filing"]["filing_date"] == "2025-02-01"


@pytest.mark.parametrize("report_date", ["", "2025-01-01", "20240930"])
def test_invalid_report_date_is_not_used(report_date):
    doc = _submissions()
    doc["filings"]["files"] = []
    doc["filings"]["recent"] = _page(
        ["2024-11-01"], ["10-K"], ["0000000123-24-000001"], ["k.htm"], [report_date]
    )
    with pytest.raises(ValueError, match="report"):
        sec_submissions.latest_periodic_filing(doc, {}, "2024-12-31", expected_cik=123)


def test_old_filing_without_document_or_report_date_does_not_block_newer_filing():
    doc = _submissions()
    doc["filings"]["files"] = []
    doc["filings"]["recent"] = _page(
        ["2020-05-01", "2024-11-01"],
        ["10-Q", "10-K"],
        ["0000000123-20-000001", "0000000123-24-000001"],
        ["", "k.htm"],
        ["", "2024-09-30"],
    )
    selected = sec_submissions.latest_periodic_filing(doc, {}, "2024-12-31", expected_cik=123)
    assert selected["filing"]["report_date"] == "2024-09-30"


def test_can_select_latest_annual_even_after_a_quarterly_filing():
    doc = _submissions()
    doc["filings"]["files"] = []
    doc["filings"]["recent"] = _page(
        ["2024-02-01", "2024-11-01"],
        ["10-K", "10-Q"],
        ["0000000123-24-000001", "0000000123-24-000002"],
        ["k.htm", "q.htm"],
        ["2023-12-31", "2024-09-30"],
    )
    result = sec_submissions.latest_periodic_filing(
        doc, {}, "2024-12-31", expected_cik=123, form="10-K"
    )
    assert result["filing"]["form"] == "10-K"
    assert result["filing"]["report_date"] == "2023-12-31"


def test_range_mismatch_is_visible_and_requires_explicit_opt_in():
    doc = _submissions()
    page = _page(["2025-01-01"], ["10-Q"], ["0000000123-25-000002"], ["q.htm"])
    with pytest.raises(ValueError, match="range"):
        sec_submissions.latest_periodic_filing(
            doc, {"old.json": page}, "2025-01-15", expected_cik=123
        )
    result = sec_submissions.latest_periodic_filing(
        doc,
        {"old.json": page},
        "2025-01-15",
        expected_cik=123,
        allow_range_discrepancies=True,
    )
    assert result["filing"]["accession"] == "0000000123-25-000002"
    assert result["prior_range_discrepancies"] == [
        {"page": "old.json", "filing_date": "2025-01-01", "accession": "0000000123-25-000002"}
    ]


def test_recent_listing_candidates_retain_paths_and_exclude_future():
    doc = _submissions()
    doc["filings"]["recent"] = _page(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        ["8-A12B", "25-NSE", "25/A", "10-K"],
        [f"0000000123-24-{i:06d}" for i in range(1, 5)],
        ["registration.htm", "xslF25X02/primary_doc.xml", "amendment.htm", "k.htm"],
    )
    result = sec_submissions.recent_listing_filings(doc, "2024-01-03", expected_cik=123)
    assert [row["form"] for row in result["filings"]] == ["8-A12B", "25-NSE"]
    assert result["filings"][1]["document"] == "xslF25X02/primary_doc.xml"
    assert result["has_older_pages"] is True
    assert result["earliest_recent_filing_date"] == "2024-01-02"
    earlier = sec_submissions.recent_listing_filings(doc, "2024-01-01", expected_cik=123)
    assert earlier["filings"] == []
    assert earlier["earliest_recent_filing_date"] is None


@pytest.mark.parametrize("document", ["../bad.htm", "/bad.htm", "x/../bad.htm", "x%2fbad.htm"])
def test_recent_listing_candidates_refuse_unsafe_paths(document):
    doc = _submissions()
    doc["filings"]["recent"] = _page(["2024-01-02"], ["25"], ["0000000123-24-000001"], [document])
    with pytest.raises(ValueError, match="document"):
        sec_submissions.recent_listing_filings(doc, "2024-01-03", expected_cik=123)


def test_recent_listing_candidates_refuse_wrong_issuer_and_duplicate_accession():
    doc = _submissions()
    with pytest.raises(ValueError, match="CIK"):
        sec_submissions.recent_listing_filings(doc, "2025-02-02", expected_cik=999)
    doc["filings"]["recent"] = _page(
        ["2024-01-02", "2024-01-03"], ["25", "25/A"], ["0000000123-24-000001"] * 2, ["x.htm"] * 2
    )
    with pytest.raises(ValueError, match="Duplicate"):
        sec_submissions.recent_listing_filings(doc, "2024-01-02", expected_cik=123)


def test_listing_inventory_reads_all_pages_without_future_filings():
    doc = _submissions()
    doc["filings"]["recent"] = _page(
        ["2025-02-01"], ["25-NSE"], ["0000000123-25-000001"], ["xslF25X02/primary_doc.xml"]
    )
    pages = {
        "old.json": _page(
            ["2024-11-06", "2024-11-06", "2023-04-01"],
            ["25", "25/A", "8-A12B"],
            ["0000000123-24-000003", "0000000123-24-000004", "0000000123-23-000001"],
            ["a.htm", "b.htm", "c.htm"],
        )
    }
    result = sec_submissions.listing_filings(doc, pages, "2024-12-31", expected_cik=123)
    assert [row["form"] for row in result["filings"]] == ["8-A12B", "25", "25/A"]
    assert result["range_discrepancies"] == []
    assert result["earliest_observed_filing_date"] == "2023-04-01"
    earlier = sec_submissions.listing_filings(doc, pages, "2022-12-31", expected_cik=123)
    assert earlier["filings"] == []
    assert earlier["earliest_observed_filing_date"] is None


def test_listing_inventory_requires_every_referenced_page():
    with pytest.raises(ValueError, match="Missing historical"):
        sec_submissions.listing_filings(_submissions(), {}, "2024-12-31", expected_cik=123)


def test_listing_inventory_rejects_duplicate_across_pages():
    doc = _submissions()
    duplicate = "0000000123-25-000001"
    pages = {"old.json": _page(["2024-11-06"], ["25"], [duplicate], ["a.htm"])}
    with pytest.raises(ValueError, match="Duplicate"):
        sec_submissions.listing_filings(doc, pages, "2024-12-31", expected_cik=123)


def test_listing_inventory_reports_prior_range_discrepancies_only_when_allowed():
    doc = _submissions()
    pages = {"old.json": _page(["2025-01-01"], ["25"], ["0000000123-25-000002"], ["a.htm"])}
    with pytest.raises(ValueError, match="range"):
        sec_submissions.listing_filings(doc, pages, "2025-01-15", expected_cik=123)
    result = sec_submissions.listing_filings(
        doc, pages, "2025-01-15", expected_cik=123, allow_range_discrepancies=True
    )
    assert result["range_discrepancies"] == [
        {"page": "old.json", "filing_date": "2025-01-01", "accession": "0000000123-25-000002"}
    ]
