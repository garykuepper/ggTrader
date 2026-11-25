"""Tests for Phase 3 open-case matching, evidence classification, merging, and coverage."""

from __future__ import annotations

from datetime import date
from typing import Any

import open_cases
import payout_table
import pytest
import sec_filings
from sec_name_index import IndexedFiling

pytestmark = pytest.mark.study


def _base_row(
    *,
    permno: str = "89260",
    symbol: str = "PERMNO-89260",
    window: str = "dev",
    universe: str = "A",
    exit_date: str = "2015-06-15",
    last_close: str = "50.0",
    stock_name: str = "ACME CORP",
    exchcd: str = "1",
    price_label: str = "calm_like_deal",
    evidence_class: str = "no_filing_match",
    match_confidence: str = "",
    match_rule_version: str = "",
    form25_accession: str = "",
    form25_provision: str = "",
    form25_security: str = "",
    event_accessions: str = "",
    events_status: str = "",
    suggested_loss: str = "",
    basis: str = "none",
    loss_note: str = (
        "no_filing_match: no Form 25 notice matched the company name; no loss can be set"
    ),
    cross_check: str = "not_comparable",
) -> dict[str, str]:
    """Build a 21-column text row matching the schema of ``_payouts.csv``."""
    return {
        "permno": permno,
        "symbol": symbol,
        "window": window,
        "universe": universe,
        "exit_date": exit_date,
        "last_close": last_close,
        "stock_name": stock_name,
        "exchcd": exchcd,
        "price_label": price_label,
        "evidence_class": evidence_class,
        "match_confidence": match_confidence,
        "match_rule_version": match_rule_version,
        "form25_accession": form25_accession,
        "form25_provision": form25_provision,
        "form25_security": form25_security,
        "event_accessions": event_accessions,
        "events_status": events_status,
        "suggested_loss": suggested_loss,
        "basis": basis,
        "loss_note": loss_note,
        "cross_check": cross_check,
    }


def _filing(
    cik: int,
    name: str,
    form: str,
    filed: date,
    accession: str = "0000100001-15-000001",
) -> IndexedFiling:
    """Build a hand-constructed ``IndexedFiling`` record."""
    return IndexedFiling(cik=cik, name=name, form=form, filed=filed, accession=accession)


def _event(
    accession: str,
    filing_date: str,
    items: list[str],
    primary_document: str = "doc.htm",
) -> dict[str, Any]:
    """Build a hand-constructed 8-K event dict matching ``extract_8k_events``."""
    return {
        "accession": accession,
        "filing_date": filing_date,
        "items": items,
        "primary_document": primary_document,
    }


def test_v1_candidate_stops_v1_1() -> None:
    # Roster name "WIDGET TECH" matches "WIDGET TECHNOLOGY INC" under v1 (TECH -> TECHNOLOGY),
    # and matches "WIDGET TECHNOLOGIES INC" only under v1.1 (plural folding).
    row = _base_row(stock_name="WIDGET TECH", exit_date="2015-06-15")
    filings = {
        1001: [_filing(1001, "WIDGET TECHNOLOGY INC", "8-K", date(2015, 5, 1), "acc-v1")],
        2002: [_filing(2002, "WIDGET TECHNOLOGIES INC", "8-K", date(2015, 5, 1), "acc-v11")],
    }

    match = open_cases.match_open_case(row, filings)

    assert match == open_cases.OpenMatch(
        status="matched",
        cik=1001,
        name_rule="v1",
        filer_name="WIDGET TECHNOLOGY INC",
        reason="",
        still_filing_check="done",
    )

    # Even when v1 produces two candidates (ambiguous), v1.1 must never run.
    ambiguous_filings = {
        1001: [_filing(1001, "WIDGET TECHNOLOGY INC", "8-K", date(2015, 5, 1), "acc-1")],
        1002: [_filing(1002, "WIDGET TECHNOLOGY CORP", "8-K", date(2015, 5, 2), "acc-2")],
        2002: [_filing(2002, "WIDGET TECHNOLOGIES INC", "8-K", date(2015, 5, 3), "acc-3")],
    }
    ambiguous = open_cases.match_open_case(row, ambiguous_filings)
    assert ambiguous.status == "ambiguous_match"
    assert ambiguous.cik is None
    assert "1001" in ambiguous.reason
    assert "1002" in ambiguous.reason
    assert "2002" not in ambiguous.reason


def test_v1_1_runs_only_when_v1_finds_none() -> None:
    # Trailing EDGAR slash tag /DE/ fails v1 but matches under v1.1.
    row = _base_row(stock_name="ALPHARMA INC-A", exit_date="2015-06-15")
    filings = {
        3003: [
            _filing(3003, "ALPHARMA INC /DE/", "10-Q", date(2015, 3, 10), "acc-later"),
            _filing(3003, "ALPHARMA INC /DE/", "8-K", date(2015, 1, 5), "acc-earliest"),
        ]
    }

    match = open_cases.match_open_case(row, filings)

    assert match == open_cases.OpenMatch(
        status="matched",
        cik=3003,
        name_rule="v1.1",
        filer_name="ALPHARMA INC /DE/",
        reason="",
        still_filing_check="done",
    )


def test_name_and_date_must_be_on_the_same_line() -> None:
    # The CIK carried the matching name in 2004, and filed inside the 2015 window only under a
    # different name; neither line satisfies both conditions, so there is no candidate.
    row = _base_row(stock_name="ACME CORP", exit_date="2015-06-15")
    filings = {
        1001: [
            _filing(1001, "ACME CORP", "10-K", date(2004, 3, 15), "acc-2004"),
            _filing(1001, "DIFFERENT NAME HOLDINGS", "8-K", date(2015, 6, 1), "acc-2015"),
        ]
    }

    match = open_cases.match_open_case(row, filings)

    assert match == open_cases.OpenMatch(
        status="no_filing_match",
        cik=None,
        name_rule="",
        filer_name="",
        reason="",
        still_filing_check="",
    )


def test_window_edges_candidate() -> None:
    # exit_date = 2015-06-15 (non-leap year):
    #   -365 days = 2014-06-15 (in), -366 days = 2014-06-14 (out)
    #   +180 days = 2015-12-12 (in), +181 days = 2015-12-13 (out)
    row = _base_row(stock_name="ACME CORP", exit_date="2015-06-15")

    in_start = open_cases.match_open_case(
        row, {1001: [_filing(1001, "ACME CORP", "8-K", date(2014, 6, 15))]}
    )
    out_before = open_cases.match_open_case(
        row, {1001: [_filing(1001, "ACME CORP", "8-K", date(2014, 6, 14))]}
    )
    in_end = open_cases.match_open_case(
        row, {1001: [_filing(1001, "ACME CORP", "8-K", date(2015, 12, 12))]}
    )
    out_after = open_cases.match_open_case(
        row, {1001: [_filing(1001, "ACME CORP", "8-K", date(2015, 12, 13))]}
    )

    assert in_start.status == "matched"
    assert in_start.cik == 1001
    assert out_before.status == "no_filing_match"
    assert out_before.cik is None
    assert in_end.status == "matched"
    assert in_end.cik == 1001
    assert out_after.status == "no_filing_match"
    assert out_after.cik is None


def test_two_live_filers_same_name_is_ambiguous() -> None:
    row = _base_row(stock_name="ACME CORP", exit_date="2015-06-15")
    filings = {
        4001: [_filing(4001, "ACME CORP", "8-K", date(2015, 5, 1), "acc-a")],
        4002: [_filing(4002, "ACME INC", "10-Q", date(2015, 5, 10), "acc-b")],
    }

    match = open_cases.match_open_case(row, filings)

    assert match.status == "ambiguous_match"
    assert match.cik is None
    assert match.name_rule == ""
    assert match.filer_name == ""
    assert match.still_filing_check == ""
    assert "4001" in match.reason
    assert "4002" in match.reason


def test_company_still_filing_rejected_from_day_181() -> None:
    # exit_date = 2015-06-15: day +180 is 2015-12-12, day +181 is 2015-12-13.
    row = _base_row(stock_name="ACME CORP", exit_date="2015-06-15")
    in_window_line = _filing(5001, "ACME CORP", "8-K", date(2015, 6, 1), "acc-base")

    # Day +180 periodic form does NOT reject.
    day_180 = open_cases.match_open_case(
        row,
        {5001: [in_window_line, _filing(5001, "ACME CORP", "10-K", date(2015, 12, 12), "acc-180")]},
    )
    assert day_180.status == "matched"
    assert day_180.cik == 5001
    assert day_180.still_filing_check == "done"

    # Day +181 periodic 10-K DOES reject.
    day_181 = open_cases.match_open_case(
        row,
        {5001: [in_window_line, _filing(5001, "ACME CORP", "10-K", date(2015, 12, 13), "acc-181")]},
    )
    assert day_181 == open_cases.OpenMatch(
        status="no_filing_match",
        cik=None,
        name_rule="",
        filer_name="",
        reason="company_still_filing: 5001",
        still_filing_check="",
    )

    # A 10-Q/A on or after day +181 DOES reject.
    amended_10q = open_cases.match_open_case(
        row,
        {
            5001: [
                in_window_line,
                _filing(5001, "ACME CORP", "10-Q/A", date(2016, 2, 1), "acc-10qa"),
            ]
        },
    )
    assert amended_10q.status == "no_filing_match"
    assert amended_10q.cik is None
    assert amended_10q.reason == "company_still_filing: 5001"
    assert amended_10q.still_filing_check == ""

    # An 8-K after day +181 NEVER rejects.
    late_8k = open_cases.match_open_case(
        row,
        {5001: [in_window_line, _filing(5001, "ACME CORP", "8-K", date(2016, 6, 1), "acc-8k")]},
    )
    assert late_8k.status == "matched"
    assert late_8k.cik == 5001
    assert late_8k.still_filing_check == "done"


def test_periodic_after_index_end_is_ignored_and_check_incomplete() -> None:
    # exit_date = 2024-10-02 -> exit + 181 days = 2025-04-01 > INDEX_LAST_DAY (2025-03-31).
    row = _base_row(stock_name="ACME CORP", exit_date="2024-10-02")
    filings = {
        6001: [
            _filing(6001, "ACME CORP", "8-K", date(2024, 9, 15), "acc-base"),
            _filing(6001, "ACME CORP", "10-K", date(2025, 4, 15), "acc-after-index"),
        ]
    }

    match = open_cases.match_open_case(row, filings)

    assert match == open_cases.OpenMatch(
        status="matched",
        cik=6001,
        name_rule="v1",
        filer_name="ACME CORP",
        reason="",
        still_filing_check="incomplete",
    )


def test_deal_form_window_edges() -> None:
    # exit_date = 2018-05-10 (non-leap year):
    #   -365 days = 2017-05-10 (in), -366 days = 2017-05-09 (out)
    #   +30 days  = 2018-06-09 (in), +31 days  = 2018-06-10 (out)
    exit_dt = date(2018, 5, 10)
    filings = [
        _filing(7001, "ACME CORP", "DEFM14A", date(2017, 5, 9), "acc-too-early"),
        _filing(7001, "ACME CORP", "SC TO-T/A", date(2018, 6, 9), "acc-plus-30"),
        _filing(7001, "ACME CORP", "DEFM14A", date(2017, 5, 10), "acc-minus-365"),
        _filing(7001, "ACME CORP", "10-K", date(2018, 3, 1), "acc-10k-ignored"),
        _filing(7001, "ACME CORP", "SC 14D9", date(2018, 6, 10), "acc-too-late"),
    ]

    found = open_cases.deal_form_filings(filings, exit_dt)

    assert [f.accession for f in found] == ["acc-minus-365", "acc-plus-30"]
    assert [f.form for f in found] == ["DEFM14A", "SC TO-T/A"]


def test_deal_forms_only_gets_no_loss() -> None:
    row = _base_row(stock_name="ACME CORP", exit_date="2018-05-10", exchcd="1")
    match = open_cases.OpenMatch(
        status="matched",
        cik=7001,
        name_rule="v1",
        filer_name="ACME CORP",
        reason="",
        still_filing_check="done",
    )
    filings = [
        _filing(7001, "ACME CORP", "DEFM14A", date(2018, 4, 1), "acc-defm"),
        _filing(7001, "ACME CORP", "SC 13E3/A", date(2018, 4, 20), "acc-13e3a"),
    ]

    out = open_cases.classify_open_case(
        row, match, events=[], events_status=sec_filings.EVENTS_STATUS_OK, filings=filings
    )

    assert tuple(out.keys()) == open_cases.OUTPUT_COLUMNS
    assert out["evidence_class"] == "deal_forms_only"
    assert out["suggested_loss"] == ""
    assert out["basis"] == "none"
    assert out["match_confidence"] == "name_only"
    assert out["match_rule_version"] == "v2"
    assert out["phase"] == "3"
    assert out["name_rule"] == "v1"
    assert out["reason"] == ""
    assert out["still_filing_check"] == "done"
    assert out["deal_form_accessions"] == "acc-defm;acc-13e3a"
    assert out["cross_check"] == "not_comparable"


def test_incomplete_events_keep_no_8k_found_and_reason() -> None:
    row = _base_row(stock_name="ACME CORP", exit_date="2018-05-10", exchcd="1")
    match = open_cases.OpenMatch(
        status="matched",
        cik=7001,
        name_rule="v1",
        filer_name="ACME CORP",
        reason="",
        still_filing_check="done",
    )
    filings = [_filing(7001, "ACME CORP", "DEFM14A", date(2018, 4, 1), "acc-defm")]
    partial_events = [_event("acc-8k-partial", "2018-05-01", ["2.01"])]

    out = open_cases.classify_open_case(
        row,
        match,
        events=partial_events,
        events_status=sec_filings.EVENTS_STATUS_OLDER_PAGE_NOT_FOUND,
        filings=filings,
    )

    assert out["evidence_class"] == "no_8k_found"
    assert out["reason"] == "events_incomplete"
    assert out["suggested_loss"] == ""
    assert out["basis"] == "none"
    assert out["events_status"] == "older_page_not_found"
    assert out["deal_form_accessions"] == "acc-defm"
    assert out["match_confidence"] == "name_only"
    assert out["match_rule_version"] == "v2"
    assert out["phase"] == "3"


def test_item_103_beats_deal_forms_bankruptcy_losses() -> None:
    match = open_cases.OpenMatch(
        status="matched",
        cik=8001,
        name_rule="v1",
        filer_name="ACME CORP",
        reason="",
        still_filing_check="done",
    )
    filings = [_filing(8001, "ACME CORP", "DEFM14A", date(2018, 4, 1), "acc-defm")]
    events = [
        _event("acc-8k-1", "2018-05-01", ["1.03", "2.01"]),
    ]

    for exchcd, expected_loss in (("1", "0.3"), ("2", "0.3"), ("3", "0.55")):
        row = _base_row(
            stock_name="ACME CORP",
            exit_date="2018-05-10",
            exchcd=exchcd,
            price_label="collapse_like",
        )
        out = open_cases.classify_open_case(
            row,
            match,
            events=events,
            events_status=sec_filings.EVENTS_STATUS_OK,
            filings=filings,
        )
        assert out["evidence_class"] == "bankruptcy_filed"
        assert out["suggested_loss"] == expected_loss
        assert out["basis"] == "literature_fill"
        assert out["match_confidence"] == "name_only"
        assert out["match_rule_version"] == "v2"
        assert out["cross_check"] == "agrees"
        assert out["deal_form_accessions"] == "acc-defm"


def test_acquisition_item_gives_zero_loss_name_only() -> None:
    row = _base_row(
        stock_name="ACME CORP",
        exit_date="2018-05-10",
        exchcd="3",
        price_label="calm_like_deal",
    )
    match = open_cases.OpenMatch(
        status="matched",
        cik=8002,
        name_rule="v1.1",
        filer_name="ACME CORP /DE/",
        reason="",
        still_filing_check="done",
    )
    events = [_event("acc-8k-acq", "2018-05-08", ["2.01", "9.01"])]

    out = open_cases.classify_open_case(
        row,
        match,
        events=events,
        events_status=sec_filings.EVENTS_STATUS_NO_OLDER_PAGES_NEEDED,
        filings=[],
    )

    assert out["evidence_class"] == "acquired_or_merged"
    assert out["suggested_loss"] == "0.0"
    assert out["basis"] == "filing_evidence"
    assert out["loss_note"] == payout_table.ACQUIRED_NOTE
    assert out["match_confidence"] == "name_only"
    assert out["match_rule_version"] == "v2"
    assert out["name_rule"] == "v1.1"
    assert out["event_accessions"] == "acc-8k-acq"
    assert out["events_status"] == "no_older_pages_needed"
    assert out["cross_check"] == "agrees"
    assert out["phase"] == "3"


def test_v2_match_is_never_confirmed() -> None:
    row = _base_row(stock_name="ACME CORP", exit_date="2018-05-10", exchcd="1")
    match = open_cases.OpenMatch(
        status="matched",
        cik=9001,
        name_rule="v1",
        filer_name="ACME CORP",
        reason="",
        still_filing_check="done",
    )
    cases = [
        ([_event("a1", "2018-05-09", ["2.01"])], sec_filings.EVENTS_STATUS_OK, []),
        ([_event("a2", "2018-05-09", ["1.03"])], sec_filings.EVENTS_STATUS_OK, []),
        ([_event("a3", "2018-05-09", ["3.01"])], sec_filings.EVENTS_STATUS_OK, []),
        (
            [],
            sec_filings.EVENTS_STATUS_OK,
            [_filing(9001, "ACME CORP", "DEFM14A", date(2018, 4, 1), "d1")],
        ),
        ([], sec_filings.EVENTS_STATUS_OK, []),
    ]

    for events, status, filings in cases:
        out = open_cases.classify_open_case(row, match, events, status, filings)
        assert out["match_confidence"] == "name_only"
        assert out["match_confidence"] != "confirmed"
        assert out["match_rule_version"] == "v2"


def test_phase2_no_8k_rows_keep_confidence_and_rule_version() -> None:
    confirmed_row = _base_row(
        permno="76598",
        symbol="PERMNO-76598",
        exit_date="2006-10-05",
        stock_name="MAVERICK TUBE CO",
        exchcd="1",
        evidence_class="no_8k_found",
        match_confidence="confirmed",
        match_rule_version="v1",
        form25_accession="0000876661-06-000789",
        form25_provision="17 CFR 240.12d2-2(a)(3)",
        form25_security="Common Stock",
        event_accessions="0000950137-06-008431",
        events_status="ok",
        suggested_loss="",
        basis="none",
        loss_note=(
            "no_8k_found: no acquisition, bankruptcy or delisting item found near the exit; "
            "no loss set"
        ),
    )
    deal_filings = [_filing(10001, "MAVERICK TUBE CORP", "DEFM14A", date(2006, 8, 15), "acc-defm")]

    upgraded = open_cases.classify_phase2_no_8k(confirmed_row, deal_filings)
    assert upgraded["evidence_class"] == "deal_forms_only"
    assert upgraded["match_confidence"] == "confirmed"
    assert upgraded["match_rule_version"] == "v1"
    assert upgraded["form25_accession"] == "0000876661-06-000789"
    assert upgraded["suggested_loss"] == ""
    assert upgraded["basis"] == "none"
    assert upgraded["phase"] == "3"
    assert upgraded["name_rule"] == ""
    assert upgraded["reason"] == ""
    assert upgraded["still_filing_check"] == ""
    assert upgraded["deal_form_accessions"] == "acc-defm"

    # Without deal forms, every original field stays unchanged and reason is no_deal_forms.
    name_only_row = dict(confirmed_row, match_confidence="name_only", match_rule_version="v1.1")
    unchanged = open_cases.classify_phase2_no_8k(name_only_row, [])
    for col in payout_table.TABLE_COLUMNS:
        assert unchanged[col] == name_only_row[col]
    assert unchanged["phase"] == "3"
    assert unchanged["reason"] == "no_deal_forms"
    assert unchanged["deal_form_accessions"] == ""

    # With incomplete events_status, class stays no_8k_found and reason is events_incomplete.
    incomplete_row = dict(confirmed_row, events_status="not_found")
    incomplete = open_cases.classify_phase2_no_8k(incomplete_row, deal_filings)
    for col in payout_table.TABLE_COLUMNS:
        assert incomplete[col] == incomplete_row[col]
    assert incomplete["phase"] == "3"
    assert incomplete["reason"] == "events_incomplete"
    assert incomplete["deal_form_accessions"] == "acc-defm"


def test_merge_keeps_evidenced_rows_unchanged() -> None:
    evidenced = _base_row(
        permno="48696",
        symbol="PERMNO-48696",
        exit_date="2005-07-29",
        stock_name="TITAN CP",
        evidence_class="acquired_or_merged",
        match_confidence="confirmed",
        match_rule_version="v1",
        form25_accession="9999999997-05-034197",
        event_accessions="0001104659-05-024772",
        events_status="ok",
        suggested_loss="0.0",
        basis="filing_evidence",
        loss_note=payout_table.ACQUIRED_NOTE,
        cross_check="agrees",
    )
    open_row = _base_row(permno="89260", symbol="PERMNO-89260", stock_name="UNITED DEFENSE")
    p3_row = open_cases.classify_open_case(
        open_row,
        open_cases.OpenMatch(
            status="no_filing_match",
            cik=None,
            name_rule="",
            filer_name="",
            reason="",
            still_filing_check="",
        ),
        events=[],
        events_status="",
        filings=[],
    )

    merged = open_cases.merge_open_cases([evidenced, open_row], {"PERMNO-89260": p3_row})

    assert [r["symbol"] for r in merged] == ["PERMNO-48696", "PERMNO-89260"]
    for col in payout_table.TABLE_COLUMNS:
        assert merged[0][col] == evidenced[col]
    assert merged[0]["phase"] == "1"
    assert merged[0]["name_rule"] == ""
    assert merged[0]["reason"] == ""
    assert merged[0]["still_filing_check"] == ""
    assert merged[0]["deal_form_accessions"] == ""
    assert merged[1]["phase"] == "3"


def test_merge_refuses_a_phase3_row_for_an_evidenced_symbol() -> None:
    evidenced = _base_row(
        permno="48696",
        symbol="PERMNO-48696",
        evidence_class="acquired_or_merged",
        match_confidence="confirmed",
        match_rule_version="v1",
        suggested_loss="0.0",
        basis="filing_evidence",
        loss_note=payout_table.ACQUIRED_NOTE,
        cross_check="agrees",
    )
    bogus_p3 = dict(
        evidenced,
        phase="3",
        name_rule="",
        reason="",
        still_filing_check="",
        deal_form_accessions="",
    )

    with pytest.raises(ValueError, match="PERMNO-48696"):
        open_cases.merge_open_cases([evidenced], {"PERMNO-48696": bogus_p3})


def test_merge_refuses_a_missing_open_row() -> None:
    evidenced = _base_row(
        permno="48696",
        symbol="PERMNO-48696",
        evidence_class="acquired_or_merged",
        suggested_loss="0.0",
        basis="filing_evidence",
    )
    open_row = _base_row(permno="89260", symbol="PERMNO-89260", evidence_class="no_filing_match")

    with pytest.raises(ValueError, match="PERMNO-89260"):
        open_cases.merge_open_cases([evidenced, open_row], {})


def test_make_matcher_prefix_for_16_character_names() -> None:
    row_16 = _base_row(symbol="SYM-16", stock_name="INTEGRATED DEVIC")
    row_15 = _base_row(symbol="SYM-15", stock_name="INTEGRATED DEVI")
    assert len(row_16["stock_name"]) == 16
    assert len(row_15["stock_name"]) == 15

    matcher = open_cases.make_matcher([row_16, row_15])

    # Longer filer name matches the 16-character cut-off by prefix, not the 15-character name.
    longer = matcher("INTEGRATED DEVICE TECHNOLOGY INC")
    assert longer == {"SYM-16": frozenset({"v1", "v1.1"})}

    # Exact cleaned equality matches the 15-character name (and not the longer 16-character prefix).
    exact_15 = matcher("INTEGRATED DEVI INC")
    assert exact_15 == {"SYM-15": frozenset({"v1", "v1.1"})}

    # Unrelated name returns an empty mapping (no empty-set entries).
    assert matcher("UNRELATED COMPANY LLC") == {}


def test_phase3_coverage_counts_moves() -> None:
    p1_evidenced = _base_row(
        permno="1",
        symbol="S1",
        exit_date="2010-01-10",
        stock_name="ALPHA CORP",
        evidence_class="acquired_or_merged",
        match_confidence="confirmed",
        match_rule_version="v1",
        suggested_loss="0.0",
        basis="filing_evidence",
        loss_note=payout_table.ACQUIRED_NOTE,
        cross_check="agrees",
    )
    open_acq = _base_row(permno="2", symbol="S2", exit_date="2012-03-01", stock_name="BETA INC")
    open_deal = _base_row(permno="3", symbol="S3", exit_date="2024-11-01", stock_name="GAMMA TECH")
    open_rejected = _base_row(
        permno="4", symbol="S4", exit_date="2014-04-01", stock_name="DELTA CORP"
    )
    p2_no8k = _base_row(
        permno="5",
        symbol="S5",
        exit_date="2008-03-07",
        stock_name="EPSILON CORP",
        evidence_class="no_8k_found",
        match_confidence="confirmed",
        match_rule_version="v1",
        events_status="ok",
    )
    unnamed = _base_row(
        permno="6",
        symbol="S6",
        exit_date="2011-09-08",
        stock_name="",
        evidence_class="name_unknown",
    )

    m_acq = open_cases.OpenMatch("matched", 2001, "v1", "BETA INC", "", "done")
    m_deal = open_cases.OpenMatch(
        "matched", 3001, "v1.1", "GAMMA TECHNOLOGIES INC", "", "incomplete"
    )
    m_rej = open_cases.OpenMatch("no_filing_match", None, "", "", "company_still_filing: 4001", "")
    m_unnamed = open_cases.OpenMatch("no_filing_match", None, "", "", "", "")

    p3_rows = {
        "S2": open_cases.classify_open_case(
            open_acq,
            m_acq,
            [_event("e2", "2012-02-20", ["2.01"])],
            sec_filings.EVENTS_STATUS_OK,
            [],
        ),
        "S3": open_cases.classify_open_case(
            open_deal,
            m_deal,
            [],
            sec_filings.EVENTS_STATUS_OK,
            [_filing(3001, "GAMMA TECHNOLOGIES INC", "DEFM14A", date(2024, 9, 1), "d3")],
        ),
        "S4": open_cases.classify_open_case(open_rejected, m_rej, [], "", []),
        "S5": open_cases.classify_phase2_no_8k(
            p2_no8k, [_filing(5001, "EPSILON CORP", "DEFM14A", date(2008, 1, 15), "d5")]
        ),
        "S6": open_cases.classify_open_case(unnamed, m_unnamed, [], "", []),
    }
    before = [p1_evidenced, open_acq, open_deal, open_rejected, p2_no8k, unnamed]
    after = open_cases.merge_open_cases(before, p3_rows)

    cov = open_cases.phase3_coverage(
        before,
        after,
        open_matches={"S2": m_acq, "S3": m_deal, "S4": m_rej, "S6": m_unnamed},
    )

    assert cov["before_by_class"]["acquired_or_merged"] == 1
    assert cov["before_by_class"]["no_filing_match"] == 3
    assert cov["before_by_class"]["no_8k_found"] == 1
    assert cov["before_by_class"]["name_unknown"] == 1
    assert cov["before_by_class"]["deal_forms_only"] == 0

    assert cov["after_by_class"]["acquired_or_merged"] == 2
    assert cov["after_by_class"]["deal_forms_only"] == 2
    assert cov["after_by_class"]["no_filing_match"] == 1
    assert cov["after_by_class"]["name_unknown"] == 1
    assert cov["deal_forms_only"] == 2

    assert cov["moves"] == {
        "no_8k_found -> deal_forms_only": 1,
        "no_filing_match -> acquired_or_merged": 1,
        "no_filing_match -> deal_forms_only": 1,
    }
    assert cov["matches"] == [
        {
            "roster_name": "BETA INC",
            "filer_name": "BETA INC",
            "cik": 2001,
            "name_rule": "v1",
            "exit_date": "2012-03-01",
        },
        {
            "roster_name": "GAMMA TECH",
            "filer_name": "GAMMA TECHNOLOGIES INC",
            "cik": 3001,
            "name_rule": "v1.1",
            "exit_date": "2024-11-01",
        },
    ]
    assert cov["v1_1_only_matches"] == [
        {
            "roster_name": "GAMMA TECH",
            "filer_name": "GAMMA TECHNOLOGIES INC",
            "cik": 3001,
            "name_rule": "v1.1",
            "exit_date": "2024-11-01",
        }
    ]
    assert cov["rejected_still_filing"] == [
        {
            "roster_name": "DELTA CORP",
            "reason": "company_still_filing: 4001",
            "exit_date": "2014-04-01",
        }
    ]
    assert cov["still_filing_incomplete"] == 1
    assert cov["best_case_unchanged"] is True

    # A Phase 3 bankruptcy literature fill keeps best_case_unchanged True.
    with_lit_bankruptcy = [
        after[0],
        dict(
            after[1],
            evidence_class="bankruptcy_filed",
            suggested_loss="0.3",
            basis="literature_fill",
        ),
        *after[2:],
    ]
    assert open_cases.phase3_coverage(before, with_lit_bankruptcy)["best_case_unchanged"] is True

    # Changing a Phase 1 loss flips best_case_unchanged to False.
    tampered_p1 = [dict(after[0], suggested_loss="0.5"), *after[1:]]
    assert open_cases.phase3_coverage(before, tampered_p1)["best_case_unchanged"] is False

    # Adding an unapproved non-zero loss to a Phase 3 row flips best_case_unchanged to False.
    tampered_p3 = [after[0], dict(after[1], suggested_loss="0.5"), *after[2:]]
    assert open_cases.phase3_coverage(before, tampered_p3)["best_case_unchanged"] is False


def test_unnamed_row_never_matches() -> None:
    blank_row = _base_row(symbol="UNNAMED-1", stock_name="", evidence_class="name_unknown")
    space_row = _base_row(symbol="UNNAMED-2", stock_name="   ", evidence_class="name_unknown")
    none_source_row = {
        **_base_row(symbol="UNNAMED-3", stock_name="IGNORED NAME", evidence_class="name_unknown"),
        "name_source": "none",
    }

    for row in (blank_row, space_row, none_source_row):
        assert open_cases.name_rules(row, "ACME CORP") == frozenset()
        assert open_cases.name_rules(row, "") == frozenset()
        assert open_cases.name_rules(row, "IGNORED NAME") == frozenset()

    matcher = open_cases.make_matcher([blank_row, space_row, none_source_row])
    assert matcher("ACME CORP") == {}
    assert matcher("") == {}
    assert matcher("IGNORED NAME") == {}

    filings = {1001: [_filing(1001, "IGNORED NAME", "8-K", date(2015, 6, 1))]}
    for row in (blank_row, space_row, none_source_row):
        match = open_cases.match_open_case(row, filings)
        assert match == open_cases.OpenMatch(
            status="no_filing_match",
            cik=None,
            name_rule="",
            filer_name="",
            reason="",
            still_filing_check="",
        )
