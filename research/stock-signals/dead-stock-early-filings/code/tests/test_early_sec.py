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
    label_filing_header,
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
ITEM INFORMATION:\t\t5.01
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
    assert label_items(items_201, filed=date(2004, 9, 14)) == "acquisition_type"
    assert label_filing_header(parse_filing_header(NEW_HEADER_201)) == "acquisition_type"

    items_501 = parse_8k_items(NEW_HEADER_501)
    assert "5.01" in items_501
    assert label_items(items_501, filed=date(2004, 10, 21)) == "acquisition_type"
    assert label_filing_header(parse_filing_header(NEW_HEADER_501)) == "acquisition_type"


def test_1_03_is_bankruptcy() -> None:
    items = parse_8k_items(NEW_HEADER_103)
    assert items == frozenset({"1.03"})
    assert label_items(items, filed=date(2004, 11, 5)) == "bankruptcy"
    assert label_filing_header(parse_filing_header(NEW_HEADER_103)) == "bankruptcy"


def test_unknown_items_are_other() -> None:
    items = parse_8k_items(UNKNOWN_ITEMS_HEADER)
    assert label_items(items, filed=date(2003, 4, 9)) == "other"
    # Empty set, unmapped codes (including 3.03 which is not registered in this brief),
    # and bare positional numbers without label text must all map to "other".
    assert label_items(frozenset()) == "other"
    assert label_items(frozenset({"3.03", "8.01", "9.01"}), filed=date(2004, 9, 1)) == "other"
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
        [(date(2001, 4, 1), frozenset({"other events"}))],
        has_form25=False,
        filing_dates=[date(2001, 4, 1), date(2001, 9, 12)],
        last_bar=last_bar,
    )
    assert label == "kept_filing"


# --- Additional coverage: era cutoff, strict numeric declarations, headers, precedence ---


def _header_with(filed_raw: str | None, *item_lines: str) -> str:
    date_line = f"FILED AS OF DATE:\t\t{filed_raw}\n" if filed_raw is not None else ""
    items_block = "".join(f"ITEM INFORMATION:\t\t{item}\n" for item in item_lines)
    return (
        "<SEC-HEADER>0001104659-04-000001.hdr.sgml : 20040823\n"
        "CONFORMED SUBMISSION TYPE:\t8-K\n"
        f"{items_block}"
        f"{date_line}"
        "FILER:\n"
        "\tCOMPANY DATA:\n"
        "\t\tCOMPANY CONFORMED NAME:\t\tCASCADE TEST CORP\n"
        "\t\tCENTRAL INDEX KEY:\t\t0001099999\n"
        "</SEC-HEADER>\n"
    )


def test_era_boundary_2004_08_22_vs_2004_08_23_for_2_01_and_1_03() -> None:
    # Strictly before 2004-08-23: item numbers are 'other'.
    pre_201 = _header_with("20040822", "2.01")
    assert parse_8k_items(pre_201) == frozenset()
    assert label_filing_header(parse_filing_header(pre_201)) == "other"
    assert label_items(frozenset({"2.01"}), filed=date(2004, 8, 22)) == "other"

    pre_103 = _header_with("20040822", "1.03")
    assert parse_8k_items(pre_103) == frozenset()
    assert label_filing_header(parse_filing_header(pre_103)) == "other"
    assert label_items(frozenset({"1.03"}), filed=date(2004, 8, 22)) == "other"

    assert label_items(frozenset({"3.01"}), filed=date(2004, 8, 22)) == "other"

    # On 2004-08-23: numbered item mapping takes effect.
    on_201 = _header_with("20040823", "2.01")
    assert parse_8k_items(on_201) == frozenset({"2.01"})
    assert label_filing_header(parse_filing_header(on_201)) == "acquisition_type"
    assert label_items(frozenset({"2.01"}), filed=date(2004, 8, 23)) == "acquisition_type"

    on_103 = _header_with("20040823", "1.03")
    assert parse_8k_items(on_103) == frozenset({"1.03"})
    assert label_filing_header(parse_filing_header(on_103)) == "bankruptcy"
    assert label_items(frozenset({"1.03"}), filed=date(2004, 8, 23)) == "bankruptcy"

    on_301 = _header_with("20040823", "3.01")
    assert parse_8k_items(on_301) == frozenset({"3.01"})
    assert label_filing_header(parse_filing_header(on_301)) == "delisting_notice_only"


def test_text_labels_count_in_every_era_including_later_wordings() -> None:
    # Old acquisition text label on 2004-08-23 still counts as acquisition_type.
    old_label_post_cutoff = _header_with("20040823", "Acquisition or disposition of assets")
    assert label_filing_header(parse_filing_header(old_label_post_cutoff)) == "acquisition_type"

    # Later-era acquisition wording counts in both eras and when undated.
    later_acq = "  Completion of   Acquisition or Disposition of Assets "
    assert label_filing_header(parse_filing_header(_header_with("20040822", later_acq))) == (
        "acquisition_type"
    )
    assert label_filing_header(parse_filing_header(_header_with("20040823", later_acq))) == (
        "acquisition_type"
    )
    assert label_items(frozenset({later_acq})) == "acquisition_type"

    # Later-era delisting notice wording counts in both eras and when undated.
    later_delist = (
        "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; "
        "Transfer of Listing"
    )
    assert label_filing_header(parse_filing_header(_header_with("20040822", later_delist))) == (
        "delisting_notice_only"
    )
    assert label_filing_header(parse_filing_header(_header_with("20040823", later_delist))) == (
        "delisting_notice_only"
    )
    assert label_items(frozenset({later_delist})) == "delisting_notice_only"

    # Bankruptcy text label on 2004-08-23 still counts as bankruptcy.
    bankruptcy_post = _header_with("20040823", "Bankruptcy or receivership")
    assert label_filing_header(parse_filing_header(bankruptcy_post)) == "bankruptcy"


def test_numbers_inside_descriptions_are_other() -> None:
    desc_201 = "Other events (agreement section 2.01)"
    desc_103 = "Other events (agreement section 1.03)"
    post_date = date(2004, 9, 1)

    for desc in (desc_201, desc_103):
        header = _header_with("20040901", desc)
        items = parse_8k_items(header)
        assert "2.01" not in items
        assert "1.03" not in items
        assert label_items(items, filed=post_date) == "other"
        assert label_filing_header(parse_filing_header(header)) == "other"
        # Direct call to label_items must not re-extract numbers from descriptions either.
        assert label_items(frozenset({desc}), filed=post_date) == "other"

    # Whole normalised text only: a registered label or code wrapped in extra prose is 'other'.
    assert (
        label_items(frozenset({"Item 5.01: Changes in Control of Registrant"}), filed=post_date)
        == "other"
    )
    assert (
        label_items(
            frozenset({"Other events (Acquisition or disposition of assets)"}), filed=post_date
        )
        == "other"
    )


def test_declaration_list_after_cutoff_is_acquisition_type() -> None:
    for declaration in ("2.01, 9.01", "2.01 and 9.01", "2.01, 5.01, and 9.01", "2.01 9.01"):
        header = _header_with("20040901", declaration)
        items = parse_8k_items(header)
        assert "2.01" in items
        assert "9.01" in items
        assert label_filing_header(parse_filing_header(header)) == "acquisition_type"
        assert label_items(frozenset({declaration}), filed=date(2004, 9, 1)) == "acquisition_type"

    # The same declaration list before the cutoff or without a date is 'other'.
    pre_header = _header_with("20040822", "2.01, 9.01")
    assert parse_8k_items(pre_header) == frozenset()
    assert label_filing_header(parse_filing_header(pre_header)) == "other"
    assert label_items(frozenset({"2.01, 9.01"})) == "other"


def test_number_with_no_date_is_other() -> None:
    # Missing FILED AS OF DATE -> numeric codes excluded and labelled 'other'.
    undated_201 = _header_with(None, "2.01")
    assert parse_8k_items(undated_201) == frozenset()
    parsed = parse_filing_header(undated_201)
    assert parsed.filed is None
    assert label_filing_header(parsed) == "other"

    # Unreadable FILED AS OF DATE -> treated as missing date (text labels only).
    bad_date_103 = _header_with("NOTADATE", "1.03")
    parsed_bad = parse_filing_header(bad_date_103)
    assert parsed_bad.filed is None
    assert parsed_bad.items == frozenset()
    assert label_filing_header(parsed_bad) == "other"

    # Explicit filed=None ignores header date and rejects numeric codes.
    assert parse_8k_items(NEW_HEADER_201, filed=None) == frozenset()

    # Undated label_items calls treat bare numbers as 'other', but still match text labels.
    assert label_items(frozenset({"2.01"})) == "other"
    assert label_items(frozenset({"1.03"})) == "other"
    assert label_items(frozenset({"3.01"})) == "other"
    assert label_items(frozenset({"Acquisition or disposition of assets"})) == "acquisition_type"


def test_unreadable_explicit_date_treats_numbers_as_other_and_keeps_text_labels() -> None:
    # Unreadable explicit date strings and NaT never raise; numbers become 'other'.
    assert label_items(frozenset({"2.01"}), filed="2004-02-30") == "other"
    assert label_items(frozenset({"1.03"}), filed="NOTADATE") == "other"
    assert label_items(frozenset({"2.01"}), filed=pd.NaT) == "other"

    # A bad explicit date never prevents a registered text label from classifying.
    assert (
        label_items(frozenset({"Acquisition or disposition of assets"}), filed="2004-02-30")
        == "acquisition_type"
    )
    assert (
        label_items(frozenset({"2.01", "Acquisition or disposition of assets"}), filed="NOTADATE")
        == "acquisition_type"
    )

    # parse_8k_items with an unreadable explicit date drops numbers and keeps text labels.
    mixed_header = _header_with("20040901", "2.01", "Acquisition or disposition of assets")
    items = parse_8k_items(mixed_header, filed="NOTADATE")
    assert items == frozenset({"acquisition or disposition of assets"})
    assert label_items(items, filed="NOTADATE") == "acquisition_type"

    # Dated-pair company classification does not abort on an unreadable date string.
    last_bar = pd.Timestamp("2004-02-15")
    assert (
        classify_company_events(
            [("2004-02-30", {"2.01"})],
            has_form25=False,
            filing_dates=[date(2004, 3, 1)],
            last_bar=last_bar,
        )
        == "no_event_found"
    )
    assert (
        classify_company_events(
            [("2004-02-30", {"2.01"}), ("NOTADATE", {"Acquisition or disposition of assets"})],
            has_form25=False,
            filing_dates=[date(2004, 3, 1)],
            last_bar=last_bar,
        )
        == "acquisition_type"
    )


def test_combining_mixed_era_filings_uses_each_filings_own_date() -> None:
    last_bar = pd.Timestamp("2004-08-20")

    # Pre-cutoff 2.01 + post-cutoff 8.01: post-cutoff date must not validate pre-cutoff 2.01.
    pre_201 = parse_filing_header(_header_with("20040822", "2.01"))
    post_801 = parse_filing_header(_header_with("20040910", "8.01"))
    assert (
        classify_company_events(
            [pre_201, post_801],
            has_form25=False,
            filing_dates=[date(2004, 8, 22), date(2004, 9, 10)],
            last_bar=last_bar,
        )
        == "no_event_found"
    )
    # Even if raw '2.01' is supplied directly with its pre-cutoff date, it stays 'other'.
    assert (
        classify_company_events(
            [(date(2004, 8, 22), {"2.01"}), (date(2004, 9, 10), {"8.01"})],
            has_form25=False,
            filing_dates=[date(2004, 8, 22), date(2004, 9, 10)],
            last_bar=last_bar,
        )
        == "no_event_found"
    )

    # Pre-cutoff 1.03 (invalid -> other) must not outrank valid post-cutoff 2.01.
    assert (
        classify_company_events(
            [(date(2004, 8, 22), {"1.03"}), (date(2004, 9, 10), {"2.01", "9.01"})],
            has_form25=False,
            filing_dates=[date(2004, 8, 22), date(2004, 9, 10)],
            last_bar=last_bar,
        )
        == "acquisition_type"
    )

    # Valid pre-cutoff acquisition text + valid post-cutoff 1.03 -> bankruptcy outranks acquisition.
    pre_acq = parse_filing_header(_header_with("20040822", "Acquisition or disposition of assets"))
    post_103 = parse_filing_header(_header_with("20040910", "1.03"))
    assert (
        classify_company_events(
            [pre_acq, post_103],
            has_form25=False,
            filing_dates=[date(2004, 8, 22), date(2004, 9, 10)],
            last_bar=last_bar,
        )
        == "bankruptcy"
    )


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
    post_date = date(2004, 12, 1)
    items_301 = parse_8k_items(NEW_HEADER_301)
    assert items_301 == frozenset({"3.01"})
    assert label_items(items_301, filed=post_date) == "delisting_notice_only"
    assert label_filing_header(parse_filing_header(NEW_HEADER_301)) == "delisting_notice_only"

    # Bankruptcy beats acquisition_type and delisting_notice_only.
    assert label_items(frozenset({"2.01", "1.03", "3.01"}), filed=post_date) == "bankruptcy"
    assert (
        label_items(
            frozenset({"Acquisition or disposition of assets", "Bankruptcy or receivership"})
        )
        == "bankruptcy"
    )
    # acquisition_type beats delisting_notice_only.
    assert label_items(frozenset({"5.01", "3.01"}), filed=post_date) == "acquisition_type"


def test_classify_company_events_follows_registered_label_order() -> None:
    last_bar = pd.Timestamp("2004-08-25")
    day_180 = [date(2005, 2, 21)]  # exactly 180 days after last_bar -> does not qualify
    day_181 = [date(2005, 2, 22)]  # 181 days after last_bar -> qualifies for kept_filing
    post_date = date(2004, 9, 1)

    # Order (amendment 2): ambiguous_match > bankruptcy > evidence_unresolved > kept_filing
    # > acquisition_type > delisting_notice_only > no_event_found.
    assert (
        classify_company_events(
            [(post_date, {"1.03"})],
            has_form25=True,
            filing_dates=day_181,
            last_bar=last_bar,
            ambiguous_match=True,
            evidence_unresolved=True,
        )
        == "ambiguous_match"
    )
    assert (
        classify_company_events(
            [(post_date, {"1.03", "2.01"})],
            has_form25=True,
            filing_dates=day_181,
            last_bar=last_bar,
            evidence_unresolved=True,
        )
        == "bankruptcy"
    )
    assert (
        classify_company_events(
            [(post_date, {"2.01"})],
            has_form25=True,
            filing_dates=day_181,
            last_bar=last_bar,
            evidence_unresolved=True,
        )
        == "evidence_unresolved"
    )
    # kept_filing (day 181) outranks acquisition_type and delisting_notice_only.
    assert (
        classify_company_events(
            [(post_date, {"2.01"})],
            has_form25=True,
            filing_dates=day_181,
            last_bar=last_bar,
        )
        == "kept_filing"
    )
    assert (
        classify_company_events(
            [(post_date, {"other events"})],
            has_form25=True,
            filing_dates=day_181,
            last_bar=last_bar,
        )
        == "kept_filing"
    )
    # Day 180 does not qualify as kept_filing, so acquisition_type and delisting_notice_only win.
    assert (
        classify_company_events(
            [(post_date, {"2.01"})],
            has_form25=True,
            filing_dates=day_180,
            last_bar=last_bar,
        )
        == "acquisition_type"
    )
    assert (
        classify_company_events(
            [(post_date, {"other events"})],
            has_form25=True,
            filing_dates=day_180,
            last_bar=last_bar,
        )
        == "delisting_notice_only"
    )
    assert (
        classify_company_events(
            [(post_date, {"other events"})],
            has_form25=False,
            filing_dates=[date(2004, 9, 1)],
            last_bar=last_bar,
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
