"""Unit tests for early_classify: pure classification, labels, binding, precedence, pilot."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import early_classify as ec
import early_sec
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.study


def _filing(
    *,
    form: str = "8-K",
    company_name: str = "ALPHA CORP",
    cik: int = 1001,
    filed: date = date(2001, 3, 15),
    file_path: str = "edgar/data/1001/0000001001-01-000001.txt",
    confidence: str = early_sec.CONFIDENCE_NAME_ONLY,
) -> early_sec.Filing:
    return early_sec.Filing(
        form=form,
        company_name=company_name,
        cik=cik,
        filed=filed,
        file_path=file_path,
        confidence=confidence,
    )


def _header(
    *,
    form: str | None = "8-K",
    filed: date | None = date(2001, 3, 15),
    company_name: str | None = "ALPHA CORP",
    cik: int | None = 1001,
    former_names: tuple[str, ...] = (),
    items: frozenset[str] = frozenset(),
) -> early_sec.FilingHeader:
    return early_sec.FilingHeader(
        form=form,
        filed=filed,
        company_name=company_name,
        cik=cik,
        former_names=former_names,
        items=items,
    )


def _hand_made_roster_and_names() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eleven-row roster covering all nine registered labels and both 8-K eras."""
    rows = [
        # 1. Pre-cutoff bankruptcy (checked)
        ("C01", 10001, "C01", "2001-03-10", "2001-02-28", "1-50", True, "ALPHA CORP", "mcgill", 1),
        # 2. Pre-cutoff acquisition_type (checked)
        ("C02", 10002, "C02", "2001-06-15", "2001-05-31", "1-50", True, "BETA INC", "mcgill", 3),
        # 3. Post-cutoff acquisition_type 2.01 (former_name tier -> never promoted to checked)
        ("C03", 10003, "C03", "2004-09-10", "2004-08-31", "51-150", False, "GAMMA CO", "mcgill", 1),
        # 4. Delisting notice only via Form 25-NSE with bound header (checked)
        (
            "C04",
            10004,
            "C04",
            "2002-04-20",
            "2002-03-29",
            "51-150",
            False,
            "DELTA LLC",
            "mcgill",
            3,
        ),
        # 5. Post-cutoff delisting notice only via 8-K item 3.01 (checked)
        ("C05", 10005, "C05", "2004-11-15", "2004-10-29", "151-500", False, "EPSILON", "mcgill", 2),
        # 6. Kept filing (>180 days after last_bar, only other events in window)
        (
            "C06",
            10006,
            "C06",
            "2002-08-01",
            "2002-07-31",
            "151-500",
            False,
            "ZETA CORP",
            "mcgill",
            1,
        ),
        # 7. No event found (company matched, only "other events" within window)
        ("C07", 10007, "C07", "2003-02-14", "2003-01-31", "501+", False, "ETA GROUP", "mcgill", 3),
        # 8. No company match (named, SEC lookup returns None)
        ("C08", 10008, "C08", "2003-07-22", "2003-06-30", "501+", False, "THETA INC", "mcgill", 1),
        # 9. Name unknown (no company name -> lookup must not be called)
        ("C09", 10009, "C09", "2000-01-18", "2000-01-03", "1-50", True, None, "", pd.NA),
        # 10. Header CIK mismatch -> evidence_unresolved (never attributes other company's event)
        (
            "C10",
            10010,
            "C10",
            "2004-10-05",
            "2004-09-30",
            "51-150",
            False,
            "IOTA CORP",
            "mcgill",
            3,
        ),
        # 11. Multiple CIKs share cleaned name -> ambiguous_match
        (
            "C11",
            10011,
            "C11",
            "2003-05-05",
            "2003-04-30",
            "151-500",
            False,
            "KAPPA INC",
            "mcgill",
            1,
        ),
    ]
    roster = pd.DataFrame(
        [
            {
                "column": r[0],
                "permno": r[1],
                "symbol": r[2],
                "last_bar": pd.Timestamp(r[3]),
                "decision_date": pd.Timestamp(r[4]),
                "rank_band": r[5],
                "held_by_rule": r[6],
            }
            for r in rows
        ]
    )
    roster["permno"] = roster["permno"].astype("Int64")
    names = roster.copy()
    names["company_name"] = pd.Series([r[7] for r in rows], dtype=object)
    names["name_source"] = pd.Series([r[8] for r in rows], dtype=object)
    names["exchcd"] = pd.Series([r[9] for r in rows], dtype="Int64")
    return roster, names


def _fake_lookup_table() -> dict[str, ec.CompanyEvidence | None]:
    f1 = _filing(form="8-K", company_name="ALPHA CORP", cik=101, filed=date(2001, 3, 15))
    h1 = _header(
        form="8-K",
        filed=date(2001, 3, 15),
        company_name="ALPHA CORP",
        cik=101,
        items=frozenset({"bankruptcy or receivership"}),
    )

    f2 = _filing(form="8-K", company_name="BETA INC", cik=102, filed=date(2001, 6, 20))
    h2 = _header(
        form="8-K",
        filed=date(2001, 6, 20),
        company_name="BETA INC",
        cik=102,
        items=frozenset({"acquisition or disposition of assets"}),
    )

    f3 = _filing(
        form="8-K",
        company_name="GAMMA NEWCO",
        cik=103,
        filed=date(2004, 9, 20),
        confidence=early_sec.CONFIDENCE_FORMER_NAME,
    )
    h3 = _header(
        form="8-K",
        filed=date(2004, 9, 20),
        company_name="GAMMA NEWCO",
        cik=103,
        former_names=("GAMMA CO",),
        items=frozenset({"2.01", "9.01"}),
    )

    f4 = _filing(form="25-NSE", company_name="DELTA LLC", cik=104, filed=date(2002, 4, 25))
    h4 = _header(
        form="25-NSE",
        filed=date(2002, 4, 25),
        company_name="DELTA LLC",
        cik=104,
        items=frozenset(),
    )

    f5 = _filing(form="8-K", company_name="EPSILON", cik=105, filed=date(2004, 11, 20))
    h5 = _header(
        form="8-K",
        filed=date(2004, 11, 20),
        company_name="EPSILON",
        cik=105,
        items=frozenset({"3.01"}),
    )

    f6 = _filing(form="8-K", company_name="ZETA CORP", cik=106, filed=date(2002, 8, 5))
    h6 = _header(
        form="8-K",
        filed=date(2002, 8, 5),
        company_name="ZETA CORP",
        cik=106,
        items=frozenset({"other events"}),
    )

    f7 = _filing(form="8-K", company_name="ETA GROUP", cik=107, filed=date(2003, 2, 20))
    h7 = _header(
        form="8-K",
        filed=date(2003, 2, 20),
        company_name="ETA GROUP",
        cik=107,
        items=frozenset({"other events"}),
    )

    f10 = _filing(form="8-K", company_name="IOTA CORP", cik=110, filed=date(2004, 10, 12))
    h10 = _header(
        form="8-K",
        filed=date(2004, 10, 12),
        company_name="OTHER CORP",
        cik=999,  # Different CIK in header -> evidence_unresolved, never attributes bankruptcy
        items=frozenset({"1.03"}),
    )

    return {
        "ALPHA CORP": ec.CompanyEvidence(
            cik=101,
            company_name="ALPHA CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(f1, h1),),
            all_filing_dates=(date(2001, 3, 15), date(2002, 1, 1)),
            candidate_ciks=(101,),
        ),
        "BETA INC": ec.CompanyEvidence(
            cik=102,
            company_name="BETA INC",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(f2, h2),),
            all_filing_dates=(date(2001, 6, 20),),
            candidate_ciks=(102,),
        ),
        "GAMMA CO": ec.CompanyEvidence(
            cik=103,
            company_name="GAMMA NEWCO",
            match_confidence=early_sec.CONFIDENCE_FORMER_NAME,
            window_filings=(ec.FilingEvidence(f3, h3),),
            all_filing_dates=(date(2004, 9, 20),),
            candidate_ciks=(103,),
        ),
        "DELTA LLC": ec.CompanyEvidence(
            cik=104,
            company_name="DELTA LLC",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(f4, h4),),
            all_filing_dates=(date(2002, 4, 25),),
            candidate_ciks=(104,),
        ),
        "EPSILON": ec.CompanyEvidence(
            cik=105,
            company_name="EPSILON",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(f5, h5),),
            all_filing_dates=(date(2004, 11, 20),),
            candidate_ciks=(105,),
        ),
        "ZETA CORP": ec.CompanyEvidence(
            cik=106,
            company_name="ZETA CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(f6, h6),),
            all_filing_dates=(date(2002, 8, 5), date(2003, 5, 1)),  # > 180d after 2002-08-01
            candidate_ciks=(106,),
        ),
        "ETA GROUP": ec.CompanyEvidence(
            cik=107,
            company_name="ETA GROUP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(f7, h7),),
            all_filing_dates=(date(2003, 2, 20), date(2003, 6, 1)),  # <= 180d after 2003-02-14
            candidate_ciks=(107,),
        ),
        "THETA INC": None,
        "IOTA CORP": ec.CompanyEvidence(
            cik=110,
            company_name="IOTA CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(f10, h10),),
            all_filing_dates=(date(2004, 10, 12),),
            candidate_ciks=(110,),
        ),
        "KAPPA INC": ec.CompanyEvidence(
            cik=None,
            company_name="",
            match_confidence="",
            candidate_ciks=(111, 222),
        ),
    }


def test_classify_roster_covers_every_label_confidence_tier_and_output_columns() -> None:
    roster, names = _hand_made_roster_and_names()
    table = _fake_lookup_table()
    looked_up: list[str] = []

    def fake_lookup(company_name: str, _last_day: Any) -> ec.CompanyEvidence | None:
        looked_up.append(company_name)
        return table[company_name]

    out = ec.classify_roster(roster, names, fake_lookup)

    assert list(out.columns) == [
        "column",
        "permno",
        "symbol",
        "last_trading_day",
        "company_name",
        "name_source",
        "exchange",
        "rank_band",
        "held_by_rule",
        "label",
        "confidence",
        "evidence",
        "suggested_loss",
    ]
    assert len(out) == len(roster)
    # Unnamed row C09 must never invoke sec_lookup.
    assert len(looked_up) == 10

    by_col = out.set_index("column")

    assert by_col.loc["C01", "label"] == "bankruptcy"
    assert by_col.loc["C01", "confidence"] == "checked"
    assert "8-K" in by_col.loc["C01", "evidence"]
    assert "2001-03-15" in by_col.loc["C01", "evidence"]
    assert "bankruptcy or receivership" in by_col.loc["C01", "evidence"]

    assert by_col.loc["C02", "label"] == "acquisition_type"
    assert by_col.loc["C02", "confidence"] == "checked"
    assert by_col.loc["C02", "exchange"] == "Nasdaq"

    # former_name match is NEVER promoted to checked even when header CIK & form match.
    assert by_col.loc["C03", "label"] == "acquisition_type"
    assert by_col.loc["C03", "confidence"] == "former_name"
    assert "2.01" in by_col.loc["C03", "evidence"]

    assert by_col.loc["C04", "label"] == "delisting_notice_only"
    assert by_col.loc["C04", "confidence"] == "checked"
    assert "25-NSE" in by_col.loc["C04", "evidence"]

    assert by_col.loc["C05", "label"] == "delisting_notice_only"
    assert by_col.loc["C05", "confidence"] == "checked"
    assert by_col.loc["C05", "exchange"] == "AMEX"

    assert by_col.loc["C06", "label"] == "kept_filing"
    assert by_col.loc["C06", "confidence"] == "name_only"
    assert "2003-05-01" in by_col.loc["C06", "evidence"]

    assert by_col.loc["C07", "label"] == "no_event_found"
    assert by_col.loc["C07", "confidence"] == "name_only"

    assert by_col.loc["C08", "label"] == "no_company_match"
    assert by_col.loc["C08", "confidence"] == ""

    assert by_col.loc["C09", "label"] == "name_unknown"
    assert by_col.loc["C09", "confidence"] == ""
    assert by_col.loc["C09", "exchange"] == "unknown"

    # Header CIK mismatch MUST yield evidence_unresolved, never attributing CIK 999's bankruptcy.
    assert by_col.loc["C10", "label"] == "evidence_unresolved"
    assert by_col.loc["C10", "confidence"] == "name_only"
    assert "header_cik_mismatch" in by_col.loc["C10", "evidence"]

    # Two CIKs sharing one cleaned name -> ambiguous_match with count of CIKs, no suggested loss.
    assert by_col.loc["C11", "label"] == "ambiguous_match"
    assert by_col.loc["C11", "confidence"] == ""
    assert "2 CIKs" in by_col.loc["C11", "evidence"]

    # All 9 registered labels and all 3 defined confidence tiers were exercised.
    assert set(out["label"]) == set(ec.REGISTERED_LABELS)
    assert {c for c in out["confidence"] if c} == set(ec.CONFIDENCE_TIERS)

    # suggested_loss is 0.0 ONLY for acquisition_type, NaN (empty) for all other 8 labels.
    acq_mask = out["label"] == "acquisition_type"
    assert (out.loc[acq_mask, "suggested_loss"] == 0.0).all()
    assert out.loc[~acq_mask, "suggested_loss"].isna().all()


def test_label_counts_sum_to_the_roster_size() -> None:
    roster, names = _hand_made_roster_and_names()
    table = _fake_lookup_table()
    out = ec.classify_roster(roster, names, lambda name, _d: table[name])
    summary = ec.summarize_classifications(out)

    assert summary["total"] == len(roster)
    assert sum(summary["by_label"].values()) == len(roster)
    assert set(summary["by_label"]) == set(ec.REGISTERED_LABELS)
    assert sum(summary["by_confidence"].values()) == len(roster)
    assert sum(v["total"] for v in summary["by_year"].values()) == len(roster)
    assert sum(v["total"] for v in summary["by_exchange"].values()) == len(roster)
    assert sum(v["total"] for v in summary["by_rank_band"].values()) == len(roster)
    assert summary["by_held_by_rule"]["held"]["total"] + summary["by_held_by_rule"]["not_held"][
        "total"
    ] == len(roster)

    held_expected = int(roster["held_by_rule"].sum())
    assert summary["held_by_rule"]["total"] == held_expected
    assert sum(summary["held_by_rule"]["by_label"].values()) == held_expected


def test_header_binding_unreadability_and_amendment2_precedence() -> None:
    """Test every unresolved cause, day-180 vs day-181 kept_filing, and bankruptcy precedence."""
    last = date(2004, 9, 10)
    day_180 = last + timedelta(days=180)
    day_181 = last + timedelta(days=181)

    cases = [
        ("HDR_404", 30001, "HDR 404 CORP", last),
        ("HDR_UNREADABLE", 30002, "HDR UNREADABLE CORP", last),
        ("HDR_FORM_BAD", 30003, "HDR FORM BAD CORP", last),
        ("HDR_DATE_BAD", 30004, "HDR DATE BAD CORP", last),
        ("FILING_CIK_BAD", 30005, "FILING CIK BAD CORP", last),
        ("ACQ_DAY_181", 30006, "ACQ DAY 181 CORP", last),
        ("ACQ_DAY_180", 30007, "ACQ DAY 180 CORP", last),
        ("BK_OVER_ALL", 30008, "BK OVER ALL CORP", last),
        ("UNRES_OVER_KEPT", 30009, "UNRES OVER KEPT CORP", last),
        ("PRE_NUM", 30010, "PRE NUM INC", date(2004, 2, 10)),
        ("LISTING_ONLY", 30011, "LISTING CO", date(2002, 5, 10)),
    ]
    roster = pd.DataFrame(
        [
            {
                "column": col,
                "permno": permno,
                "symbol": col,
                "last_bar": pd.Timestamp(lb),
                "decision_date": pd.Timestamp(lb) - pd.Timedelta(days=10),
                "rank_band": "1-50",
                "held_by_rule": True,
            }
            for col, permno, _name, lb in cases
        ]
    )
    roster["permno"] = roster["permno"].astype("Int64")
    names = pd.DataFrame(
        {
            "column": [c[0] for c in cases],
            "company_name": [c[2] for c in cases],
            "name_source": ["mcgill"] * len(cases),
            "exchcd": pd.Series([1] * len(cases), dtype="Int64"),
        }
    )

    f_date = date(2004, 9, 15)
    acq_filing = _filing(form="8-K", company_name="X", cik=306, filed=f_date)
    acq_header = _header(form="8-K", filed=f_date, cik=306, items=frozenset({"2.01"}))

    acq_f180 = _filing(form="8-K", company_name="X", cik=307, filed=f_date)
    acq_h180 = _header(form="8-K", filed=f_date, cik=307, items=frozenset({"2.01"}))

    bk_f = _filing(
        form="8-K",
        company_name="X",
        cik=308,
        filed=f_date,
        file_path="edgar/data/308/0000000308-04-000001.txt",
    )
    bk_h = _header(form="8-K", filed=f_date, cik=308, items=frozenset({"1.03"}))
    bk_bad_f = _filing(
        form="8-K",
        company_name="X",
        cik=308,
        filed=date(2004, 9, 20),
        file_path="edgar/data/308/0000000308-04-000002.txt",
    )

    # Undated partial header: early_sec amendment-1 still treats undated numeric item as 'other',
    # AND classifier boundary separately flags header_unreadable (missing filed_date).
    undated_header = _header(form="8-K", filed=None, cik=302, items=frozenset({"2.01"}))
    assert early_sec.label_filing_header(undated_header) == "other"

    lookup_map: dict[str, ec.CompanyEvidence] = {
        "HDR 404 CORP": ec.CompanyEvidence(
            cik=301,
            company_name="HDR 404 CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(
                    _filing(form="8-K", cik=301, filed=f_date),
                    header=None,
                    issue="header_not_found",
                ),
            ),
            all_filing_dates=(f_date,),
            candidate_ciks=(301,),
        ),
        "HDR UNREADABLE CORP": ec.CompanyEvidence(
            cik=302,
            company_name="HDR UNREADABLE CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(
                    _filing(form="8-K", cik=302, filed=f_date),
                    header=undated_header,
                ),
            ),
            all_filing_dates=(f_date,),
            candidate_ciks=(302,),
        ),
        "HDR FORM BAD CORP": ec.CompanyEvidence(
            cik=303,
            company_name="HDR FORM BAD CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(
                    _filing(form="8-K", cik=303, filed=f_date),
                    header=_header(form="10-K", filed=f_date, cik=303, items=frozenset({"2.01"})),
                ),
            ),
            all_filing_dates=(f_date,),
            candidate_ciks=(303,),
        ),
        "HDR DATE BAD CORP": ec.CompanyEvidence(
            cik=304,
            company_name="HDR DATE BAD CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(
                    _filing(form="8-K", cik=304, filed=f_date),
                    header=_header(
                        form="8-K", filed=date(2004, 9, 18), cik=304, items=frozenset({"2.01"})
                    ),
                ),
            ),
            all_filing_dates=(f_date,),
            candidate_ciks=(304,),
        ),
        "FILING CIK BAD CORP": ec.CompanyEvidence(
            cik=305,
            company_name="FILING CIK BAD CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(
                    _filing(form="8-K", cik=999, filed=f_date),
                    header=_header(form="8-K", filed=f_date, cik=999, items=frozenset({"2.01"})),
                ),
            ),
            all_filing_dates=(f_date,),
            candidate_ciks=(305,),
        ),
        "ACQ DAY 181 CORP": ec.CompanyEvidence(
            cik=306,
            company_name="ACQ DAY 181 CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(acq_filing, acq_header),),
            all_filing_dates=(f_date, day_181),
            candidate_ciks=(306,),
        ),
        "ACQ DAY 180 CORP": ec.CompanyEvidence(
            cik=307,
            company_name="ACQ DAY 180 CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(ec.FilingEvidence(acq_f180, acq_h180),),
            all_filing_dates=(f_date, day_180),
            candidate_ciks=(307,),
        ),
        "BK OVER ALL CORP": ec.CompanyEvidence(
            cik=308,
            company_name="BK OVER ALL CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(bk_f, bk_h),
                ec.FilingEvidence(bk_bad_f, header=None, issue="header_not_found"),
            ),
            all_filing_dates=(f_date, day_181),
            candidate_ciks=(308,),
        ),
        "UNRES OVER KEPT CORP": ec.CompanyEvidence(
            cik=309,
            company_name="UNRES OVER KEPT CORP",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(
                    _filing(form="8-K", cik=309, filed=f_date),
                    header=None,
                    issue="header_not_found",
                ),
            ),
            all_filing_dates=(f_date, day_181),
            candidate_ciks=(309,),
        ),
        "PRE NUM INC": ec.CompanyEvidence(
            cik=310,
            company_name="PRE NUM INC",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(
                    _filing(form="8-K", cik=310, filed=date(2004, 2, 20)),
                    header=_header(
                        form="8-K",
                        filed=date(2004, 2, 20),
                        cik=310,
                        items=frozenset({"2.01"}),
                    ),
                ),
            ),
            all_filing_dates=(date(2004, 2, 20),),
            candidate_ciks=(310,),
        ),
        "LISTING CO": ec.CompanyEvidence(
            cik=311,
            company_name="LISTING CO",
            match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
            window_filings=(
                ec.FilingEvidence(_filing(form="8-A12B", cik=311, filed=date(2002, 5, 12)), None),
            ),
            all_filing_dates=(date(2002, 5, 12),),
            candidate_ciks=(311,),
        ),
    }

    out = ec.classify_roster(roster, names, lambda name, _d: lookup_map[name]).set_index("column")

    assert out.loc["HDR_404", "label"] == "evidence_unresolved"
    assert "header_not_found" in out.loc["HDR_404", "evidence"]
    assert pd.isna(out.loc["HDR_404", "suggested_loss"])

    assert out.loc["HDR_UNREADABLE", "label"] == "evidence_unresolved"
    assert "header_unreadable" in out.loc["HDR_UNREADABLE", "evidence"]
    assert "filed_date" in out.loc["HDR_UNREADABLE", "evidence"]

    assert out.loc["HDR_FORM_BAD", "label"] == "evidence_unresolved"
    assert "header_form_mismatch" in out.loc["HDR_FORM_BAD", "evidence"]

    assert out.loc["HDR_DATE_BAD", "label"] == "evidence_unresolved"
    assert "header_date_mismatch" in out.loc["HDR_DATE_BAD", "evidence"]

    assert out.loc["FILING_CIK_BAD", "label"] == "evidence_unresolved"
    assert "filing_cik_mismatch" in out.loc["FILING_CIK_BAD", "evidence"]

    # Day 181 -> kept_filing (no suggested loss), AND evidence still records acquisition notice.
    assert out.loc["ACQ_DAY_181", "label"] == "kept_filing"
    assert pd.isna(out.loc["ACQ_DAY_181", "suggested_loss"])
    assert "acquisition_type notice" in out.loc["ACQ_DAY_181", "evidence"]
    assert "8-K 2004-09-15 [2.01]" in out.loc["ACQ_DAY_181", "evidence"]

    # Day 180 -> does not qualify as kept_filing, so acquisition_type wins with 0.0 suggested loss.
    assert out.loc["ACQ_DAY_180", "label"] == "acquisition_type"
    assert out.loc["ACQ_DAY_180", "suggested_loss"] == 0.0
    assert out.loc["ACQ_DAY_180", "confidence"] == "checked"

    # Resolved bankruptcy outranks both evidence_unresolved and kept_filing.
    assert out.loc["BK_OVER_ALL", "label"] == "bankruptcy"
    assert out.loc["BK_OVER_ALL", "confidence"] == "checked"
    assert pd.isna(out.loc["BK_OVER_ALL", "suggested_loss"])

    # Unresolved header with no bankruptcy outranks kept_filing.
    assert out.loc["UNRES_OVER_KEPT", "label"] == "evidence_unresolved"
    assert pd.isna(out.loc["UNRES_OVER_KEPT", "suggested_loss"])

    # Pre-2004-08-23 bare 2.01 is not acquisition; 8-A12B listing form is ignored.
    assert out.loc["PRE_NUM", "label"] == "no_event_found"
    assert out.loc["LISTING_ONLY", "label"] == "no_event_found"


def test_select_pilot_rows_is_deterministic_and_uses_seed_20261125() -> None:
    permnos = list(range(50000, 50030))
    roster = pd.DataFrame(
        {
            "column": [f"P{p}" for p in permnos],
            "permno": pd.Series(permnos, dtype="Int64"),
            "symbol": [f"S{i}" for i in range(len(permnos))],
            "last_bar": pd.Timestamp("2001-06-15"),
            "decision_date": pd.Timestamp("2001-05-31"),
            "rank_band": "1-50",
            "held_by_rule": True,
        }
    )
    reversed_roster = roster.iloc[::-1].reset_index(drop=True)

    pilot_a = ec.select_pilot_rows(roster, pilot_size=10, seed=20261125)
    pilot_b = ec.select_pilot_rows(reversed_roster, pilot_size=10, seed=20261125)

    expected_idx = np.random.default_rng(20261125).permutation(len(roster))[:10]
    expected_permnos = [permnos[i] for i in expected_idx]

    assert list(pilot_a["permno"]) == expected_permnos
    assert list(pilot_b["permno"]) == expected_permnos
