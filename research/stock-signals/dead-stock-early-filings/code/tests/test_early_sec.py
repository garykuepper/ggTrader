"""Tests for early-window SEC index search, filing-header parsing, and item classification."""

from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import pytest
from early_sec import (
    CONFIDENCE_CHECKED,
    CONFIDENCE_FORMER_NAME,
    CONFIDENCE_NAME_ONLY,
    CONFIDENCE_RANK,
    FORM8K_FORMS,
    FORM25_FORMS,
    Filing,
    FilingHeader,
    IndexRow,
    classify_company_events,
    clean_company_name,
    filing_url,
    find_filings,
    has_kept_filing,
    label_items,
    master_index_url,
    parse_8k_items,
    parse_filing_header,
    parse_master_index,
    quarters_for,
    resolve_confidence,
)

pytestmark = pytest.mark.study

INDEX_PREAMBLE = (
    "Description: Master Index of EDGAR Dissemination Feed\n"
    "Last Data Received: March 31, 2001\n"
    "CIK|Company Name|Form Type|Date Filed|Filename\n"
    "--------------------------------------------------------------------------------\n"
)

OLD_HEADER_ACQUISITION = """<SEC-HEADER>0000950134-01-002345.hdr.sgml : 20010316
ACCESSION NUMBER:\t\t0000950134-01-002345
CONFORMED SUBMISSION TYPE:\t8-K
PUBLIC DOCUMENT COUNT:\t\t2
CONFORMED PERIOD OF REPORT:\t20010315
ITEM INFORMATION:\t\tAcquisition or disposition of assets
ITEM INFORMATION:\t\tFinancial statements and exhibits
FILED AS OF DATE:\t\t20010315

FILER:

\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tNORTHSTAR WIDGET CORP
\t\tCENTRAL INDEX KEY:\t\t0000812345
\t\tSTATE OF INCORPORATION:\t\tDE

\tFORMER COMPANY:
\t\tFORMER CONFORMED NAME:\tOLDSTAR ELECTRONICS INC
\t\tDATE OF NAME CHANGE:\t19980501

\tFORMER COMPANY:
\t\tFORMER CONFORMED NAME:\tFIRSTSTAR LABS CORP
\t\tDATE OF NAME CHANGE:\t19950214
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<TEXT>
ITEM INFORMATION: Bankruptcy or receivership
</TEXT>
</DOCUMENT>
"""

OLD_HEADER_CONTROL = """<SEC-HEADER>0000950134-02-004100.hdr.sgml : 20020604
CONFORMED SUBMISSION TYPE:\t8-K/A
ITEM INFORMATION:\t\t  Changes in   control of registrant
FILED AS OF DATE:\t\t20020603

FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tHARBOR BRIDGE SYSTEMS INC
\t\tCENTRAL INDEX KEY:\t\t0000745120
</SEC-HEADER>
"""

OLD_HEADER_BANKRUPTCY = """<SEC-HEADER>0000950134-01-009876.hdr.sgml : 20010920
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\tBANKRUPTCY OR RECEIVERSHIP
ITEM INFORMATION:\t\tOther events
FILED AS OF DATE:\t\t20010919

FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tCEDAR VALLEY TELECOM INC
\t\tCENTRAL INDEX KEY:\t\t0000901234
</SEC-HEADER>
"""

NEW_HEADER_201 = """<SEC-HEADER>0001104659-04-028111.hdr.sgml : 20040915
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\t2.01
ITEM INFORMATION:\t\t9.01
FILED AS OF DATE:\t\t20040914

FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tSUMMIT PHOTONICS CORP
\t\tCENTRAL INDEX KEY:\t\t0001023456
</SEC-HEADER>
"""

NEW_HEADER_501 = """<SEC-HEADER>0001104659-04-031222.hdr.sgml : 20041022
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\tItem 5.01: Changes in Control of Registrant
FILED AS OF DATE:\t\t20041021

FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tREDWOOD INSTRUMENTS INC
\t\tCENTRAL INDEX KEY:\t\t0001034567
</SEC-HEADER>
"""

NEW_HEADER_103 = """<SEC-HEADER>0001104659-04-033333.hdr.sgml : 20041108
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\t1.03
FILED AS OF DATE:\t\t20041105

FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tRIVERBEND AIRLINES HOLDINGS INC
\t\tCENTRAL INDEX KEY:\t\t0001045678
</SEC-HEADER>
"""

NEW_HEADER_301 = """<SEC-HEADER>0001104659-04-035444.hdr.sgml : 20041202
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\t3.01
FILED AS OF DATE:\t\t20041201

FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tMAPLE SOFTWARE CORP
\t\tCENTRAL INDEX KEY:\t\t0001056789
</SEC-HEADER>
"""

UNKNOWN_ITEMS_HEADER = """<SEC-HEADER>0000950134-03-001111.hdr.sgml : 20030410
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\tOther events
ITEM INFORMATION:\t\tFinancial statements and exhibits
ITEM INFORMATION:\t\t8.01
FILED AS OF DATE:\t\t20030409

FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\tGRANITE BIO LABS INC
\t\tCENTRAL INDEX KEY:\t\t0001067890
</SEC-HEADER>
"""


def _make_master_zip(rows: list[tuple[str, str, str, str]]) -> bytes:
    lines = []
    for idx, (cik, name, form, filed) in enumerate(rows):
        cik_num = int(cik) if cik.isdigit() else 1
        accession = f"{cik_num:010d}-01-{idx:06d}"
        lines.append(f"{cik}|{name}|{form}|{filed}|edgar/data/{cik_num}/{accession}.txt\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("master.idx", INDEX_PREAMBLE + "".join(lines))
    return buf.getvalue()


# --- Required tests from task-4-brief.md (exact names) ---


def test_old_header_acquisition_item_is_acquisition_type() -> None:
    items = parse_8k_items(OLD_HEADER_ACQUISITION)
    assert "acquisition or disposition of assets" in items
    # Body text after <DOCUMENT> must never leak into header items.
    assert "bankruptcy or receivership" not in items
    assert label_items(items) == "acquisition_type"

    control_items = parse_8k_items(OLD_HEADER_CONTROL)
    assert "changes in control of registrant" in control_items
    assert label_items(control_items) == "acquisition_type"


def test_old_header_bankruptcy_item_is_bankruptcy() -> None:
    items = parse_8k_items(OLD_HEADER_BANKRUPTCY)
    assert "bankruptcy or receivership" in items
    assert label_items(items) == "bankruptcy"


def test_new_numbered_items_2_01_and_5_01_are_acquisition_type() -> None:
    items_201 = parse_8k_items(NEW_HEADER_201)
    assert "2.01" in items_201
    assert label_items(items_201) == "acquisition_type"

    items_501 = parse_8k_items(NEW_HEADER_501)
    assert "5.01" in items_501
    assert label_items(items_501) == "acquisition_type"


def test_1_03_is_bankruptcy() -> None:
    items = parse_8k_items(NEW_HEADER_103)
    assert items == frozenset({"1.03"})
    assert label_items(items) == "bankruptcy"


def test_unknown_items_are_other() -> None:
    items = parse_8k_items(UNKNOWN_ITEMS_HEADER)
    assert label_items(items) == "other"
    # Empty set, unmapped codes (including 3.03 which is not registered in this brief),
    # and bare positional numbers without label text must all map to "other".
    assert label_items(frozenset()) == "other"
    assert label_items(frozenset({"3.03", "8.01", "9.01"})) == "other"
    assert label_items(frozenset({"Item 2", "Item 3"})) == "other"


def test_quarters_for_a_december_last_bar_spans_the_next_quarters() -> None:
    # 2001-12-15: -30 days is 2001-11-15 (2001Q4), +180 days is 2002-06-13 (2002Q2).
    assert quarters_for(pd.Timestamp("2001-12-15")) == [(2001, 4), (2002, 1), (2002, 2)]
    # 2000-01-10: -30 days crosses back into 1999Q4, +180 days reaches 2000Q3.
    assert quarters_for(pd.Timestamp("2000-01-10")) == [
        (1999, 4),
        (2000, 1),
        (2000, 2),
        (2000, 3),
    ]


def test_name_cleaning_matches_inc_corp_and_punctuation_variants() -> None:
    assert clean_company_name("Northstar Widget, Inc.") == "NORTHSTAR WIDGET"
    assert clean_company_name("NORTHSTAR WIDGET CORP.") == "NORTHSTAR WIDGET"
    assert clean_company_name("The Northstar Widget Corporation") == "NORTHSTAR WIDGET"
    assert clean_company_name("Northstar-Widget Co., Ltd.") == "NORTHSTAR WIDGET"
    assert clean_company_name("Northstar Widget Holdings, LLC") == "NORTHSTAR WIDGET"
    assert clean_company_name("Northstar Widget Group PLC") == "NORTHSTAR WIDGET"
    assert clean_company_name("Northstar Widget LP") == "NORTHSTAR WIDGET"
    assert clean_company_name("Smith & Jones, Inc.") == clean_company_name(
        "SMITH AND JONES COMPANY"
    )
    assert clean_company_name("Smith&Jones Corp") == "SMITH AND JONES"
    # Trailing HOLDINGS/GROUP dropped only when another word remains; non-trailing stays.
    assert clean_company_name("Holdings Inc.") == "HOLDINGS"
    assert clean_company_name("The Group Corp.") == "GROUP"
    assert clean_company_name("Group Health Corp.") == "GROUP HEALTH"
    assert clean_company_name("Inc.") == "INC"
    assert clean_company_name("   ") == ""


def test_former_name_match_is_marked_lower_confidence() -> None:
    rows = [
        IndexRow(
            cik=812345,
            company_name="OLDSTAR ELECTRONICS, INC.",
            form="8-K",
            filed=date(2001, 3, 15),
            file_path="edgar/data/812345/0000950134-01-002345.txt",
        ),
        IndexRow(
            cik=999111,
            company_name="NORTHSTAR WIDGET CORP",
            form="8-K",
            filed=date(2001, 3, 20),
            file_path="edgar/data/999111/0000950134-01-002399.txt",
        ),
        IndexRow(
            cik=555555,
            company_name="UNRELATED BIO INC",
            form="8-K",
            filed=date(2001, 3, 25),
            file_path="edgar/data/555555/0000950134-01-002400.txt",
        ),
    ]

    found = find_filings(
        rows,
        company_name="Northstar Widget, Inc.",
        former_names=["Oldstar Electronics Corp."],
    )
    assert len(found) == 2
    by_cik = {f.cik: f for f in found}
    assert by_cik[999111].confidence == CONFIDENCE_NAME_ONLY
    assert by_cik[812345].confidence == CONFIDENCE_FORMER_NAME
    assert CONFIDENCE_RANK[by_cik[812345].confidence] < CONFIDENCE_RANK[by_cik[999111].confidence]

    # A former_name match is never promoted to checked, even when header CIK & form agree.
    header = parse_filing_header(OLD_HEADER_ACQUISITION)
    assert resolve_confidence(CONFIDENCE_FORMER_NAME, by_cik[812345], header) == (
        CONFIDENCE_FORMER_NAME
    )


def test_a_company_that_kept_filing_after_180_days_is_marked_kept_filing() -> None:
    last_bar = pd.Timestamp("2001-03-15")
    # Exactly 180 days after 2001-03-15 is 2001-09-11 (still inside the 180-day window).
    assert not has_kept_filing([date(2001, 4, 1), date(2001, 9, 11)], last_bar)
    # 181 days after 2001-03-15 is 2001-09-12 (strictly more than 180 days after -> kept_filing).
    assert has_kept_filing([date(2001, 4, 1), date(2001, 9, 12)], last_bar)
    assert not has_kept_filing([], last_bar)

    label = classify_company_events(
        items=frozenset({"other events"}),
        has_form25=False,
        filing_dates=[date(2001, 4, 1), date(2001, 9, 12)],
        last_bar=last_bar,
    )
    assert label == "kept_filing"


# --- Additional coverage: header fields, precedence, index parsing, confidence ---


def test_parse_filing_header_extracts_all_fields() -> None:
    header = parse_filing_header(OLD_HEADER_ACQUISITION)
    assert header == FilingHeader(
        form="8-K",
        filed=date(2001, 3, 15),
        company_name="NORTHSTAR WIDGET CORP",
        cik=812345,
        former_names=("OLDSTAR ELECTRONICS INC", "FIRSTSTAR LABS CORP"),
        items=frozenset(
            {"acquisition or disposition of assets", "financial statements and exhibits"}
        ),
    )


def test_delisting_item_3_01_and_label_priority_order() -> None:
    items_301 = parse_8k_items(NEW_HEADER_301)
    assert items_301 == frozenset({"3.01"})
    assert label_items(items_301) == "delisting_notice_only"

    # Bankruptcy beats acquisition_type and delisting_notice_only.
    assert label_items(frozenset({"2.01", "1.03", "3.01"})) == "bankruptcy"
    assert (
        label_items(
            frozenset({"Acquisition or disposition of assets", "Bankruptcy or receivership"})
        )
        == "bankruptcy"
    )
    # acquisition_type beats delisting_notice_only.
    assert label_items(frozenset({"5.01", "3.01"})) == "acquisition_type"


def test_classify_company_events_follows_registered_label_order() -> None:
    last_bar = pd.Timestamp("2001-03-15")
    later_filing = [date(2001, 10, 1)]  # > 180 days after last_bar

    assert (
        classify_company_events(
            {"1.03", "2.01"}, has_form25=True, filing_dates=later_filing, last_bar=last_bar
        )
        == "bankruptcy"
    )
    assert (
        classify_company_events(
            {"2.01"}, has_form25=True, filing_dates=later_filing, last_bar=last_bar
        )
        == "acquisition_type"
    )
    assert (
        classify_company_events(
            {"other events"}, has_form25=True, filing_dates=later_filing, last_bar=last_bar
        )
        == "delisting_notice_only"
    )
    assert (
        classify_company_events(
            {"other events"}, has_form25=False, filing_dates=later_filing, last_bar=last_bar
        )
        == "kept_filing"
    )
    assert (
        classify_company_events(
            {"other events"}, has_form25=False, filing_dates=[date(2001, 4, 1)], last_bar=last_bar
        )
        == "no_event_found"
    )


def test_resolve_confidence_promotes_name_only_when_header_matches() -> None:
    filing = Filing(
        form="8-K",
        company_name="NORTHSTAR WIDGET CORP",
        cik=812345,
        filed=date(2001, 3, 15),
        file_path="edgar/data/812345/0000950134-01-002345.txt",
        confidence=CONFIDENCE_NAME_ONLY,
    )
    header = parse_filing_header(OLD_HEADER_ACQUISITION)
    assert resolve_confidence(CONFIDENCE_NAME_ONLY, filing, header) == CONFIDENCE_CHECKED

    wrong_cik = FilingHeader(
        form="8-K",
        filed=date(2001, 3, 15),
        company_name="OTHER",
        cik=999999,
        former_names=(),
        items=frozenset(),
    )
    assert resolve_confidence(CONFIDENCE_NAME_ONLY, filing, wrong_cik) == CONFIDENCE_NAME_ONLY

    wrong_form = FilingHeader(
        form="10-K",
        filed=date(2001, 3, 15),
        company_name="NORTHSTAR WIDGET CORP",
        cik=812345,
        former_names=(),
        items=frozenset(),
    )
    assert resolve_confidence(CONFIDENCE_NAME_ONLY, filing, wrong_form) == CONFIDENCE_NAME_ONLY


def test_parse_master_index_preserves_file_path_and_find_filings_filters() -> None:
    archive = _make_master_zip(
        [
            ("0000812345", "NORTHSTAR WIDGET CORP", "8-K", "2001-03-15"),
            ("812345", "NORTHSTAR WIDGET CORP", "10-K", "2001-03-01"),
            ("812345", "NORTHSTAR WIDGET CORP", "25-NSE", "2001-03-20"),
            ("812345", "NORTHSTAR WIDGET CORP", "8-A12B", "2000-01-10"),
            ("900001", "", "485BPOS", "2001-02-01"),
        ]
    )
    rows = parse_master_index(archive)
    assert len(rows) == 5
    assert rows[0] == IndexRow(
        cik=812345,
        company_name="NORTHSTAR WIDGET CORP",
        form="8-K",
        filed=date(2001, 3, 15),
        file_path="edgar/data/812345/0000812345-01-000000.txt",
    )
    assert rows[4].company_name == ""

    all_hits = find_filings(rows, "Northstar Widget, Inc.")
    assert [f.form for f in all_hits] == ["8-A12B", "10-K", "8-K", "25-NSE"]

    event_hits = find_filings(rows, "Northstar Widget, Inc.", forms=FORM8K_FORMS | FORM25_FORMS)
    assert [f.form for f in event_hits] == ["8-K", "25-NSE"]
    assert find_filings(rows, "   ") == []


def test_urls_match_edgar_layout() -> None:
    assert master_index_url(2001, 1) == (
        "https://www.sec.gov/Archives/edgar/full-index/2001/QTR1/master.zip"
    )
    assert filing_url("edgar/data/812345/0000950134-01-002345.txt") == (
        "https://www.sec.gov/Archives/edgar/data/812345/0000950134-01-002345.txt"
    )
