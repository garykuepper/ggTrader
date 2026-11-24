"""Tests for finding each dead stock's SEC Form 25 notice: quarters, names, windows, notices."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date

import pytest
import sec_filings
from sec_client import SecFetchError, SecNotFound
from sec_filings import (
    ABBREVIATIONS,
    form25_candidates,
    master_index_url,
    match_roster_row,
    normalize_name,
    notice_is_common_stock,
    quarters_covering,
    read_notice,
)

pytestmark = pytest.mark.study

HEADER = (
    "Description: Master Index of EDGAR Dissemination Feed\n"
    "CIK|Company Name|Form Type|Date Filed|Filename\n"
    "--------------------------------------------------------------------------------\n"
)


def cand(name: str, filed: str, cik: str = "0000000001", form: str = "25") -> dict:
    """One candidate filing with a single indexed party."""
    accession = f"{int(cik):010d}-10-{filed.replace('-', '')[-6:]}"
    return {
        "accession": accession,
        "form": form,
        "filing_date": filed,
        "parties": [{"cik": cik, "name": name, "filename": f"edgar/data/{int(cik)}/x.txt"}],
    }


def master_zip(rows: list[tuple[str, str, str, str]]) -> bytes:
    """A master.zip holding (cik, name, form, date) rows, with consistent file names."""
    lines = []
    for n, (cik, name, form, filed) in enumerate(rows):
        accession = f"{int(cik):010d}-10-{n:06d}"
        lines.append(f"{cik}|{name}|{form}|{filed}|edgar/data/{int(cik)}/{accession}.txt\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("master.idx", HEADER + "".join(lines))
    return buffer.getvalue()


class FakeClient:
    """Serves canned bodies by URL; a value that is an exception is raised instead."""

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


ROW = {"stock_name": "ALPHA CORP", "name_source": "mcgill", "exit_date": "2010-06-30"}


def test_normalize_name_drops_corporate_suffix():
    assert normalize_name("SiRF Technology Holdings, Inc.") == "SIRF TECHNOLOGY HOLDINGS"


def test_normalize_name_spells_out_short_forms_the_same_on_both_sides():
    assert normalize_name("LONGS DRG STRS") == normalize_name("Longs Drug Stores Corp.")
    assert normalize_name("TITAN CP") == normalize_name("Titan Corporation") == "TITAN"
    assert normalize_name("INTL COAL GROUP") == "INTERNATIONAL COAL GROUP"


def test_normalize_name_drops_several_trailing_suffix_words_but_never_everything():
    assert normalize_name("ACME CO INC") == "ACME"
    assert normalize_name("Inc.") == "INC"


def test_abbreviation_table_has_whole_word_keys_only():
    assert all(key == key.upper() and " " not in key for key in ABBREVIATIONS)


def test_match_requires_window_edges():
    on_edge_before = cand("ALPHA CORP", "2010-05-31")
    too_early = cand("ALPHA CORP", "2010-05-30")
    on_edge_after = cand("ALPHA CORP", "2010-12-27")
    too_late = cand("ALPHA CORP", "2010-12-28")
    assert match_roster_row(ROW, [on_edge_before])["status"] == "matched"
    assert match_roster_row(ROW, [too_early])["status"] == "none"
    assert match_roster_row(ROW, [on_edge_after])["status"] == "matched"
    assert match_roster_row(ROW, [too_late])["status"] == "none"


def test_two_filers_with_the_same_name_are_ambiguous():
    out = match_roster_row(
        ROW,
        [
            cand("ALPHA CORP", "2010-07-01", cik="0000000001"),
            cand("Alpha Corp.", "2010-07-02", cik="0000000002"),
        ],
    )
    assert out["status"] == "ambiguous" and len(out["ciks"]) == 2


def test_same_cik_filing_twice_is_one_match():
    out = match_roster_row(
        ROW,
        [
            cand("ALPHA CORP", "2010-07-09", cik="0000000001", form="25/A"),
            cand("ALPHA CORP", "2010-07-01", cik="0000000001"),
        ],
    )
    assert out["status"] == "matched" and out["ciks"] == ["0000000001"]
    # Filing order, earliest first, so the caller reads the first notice first.
    assert [a[-6:] for a in out["accessions"]] == ["100701", "100709"]


def test_unnamed_row_never_matches():
    row = {"stock_name": None, "name_source": "none", "exit_date": "2010-06-30"}
    assert match_roster_row(row, [cand("ALPHA CORP", "2010-07-01")])["status"] == "none"
    row = {"stock_name": "", "name_source": "none", "exit_date": "2010-06-30"}
    assert match_roster_row(row, [cand("", "2010-07-01")])["status"] == "none"


def test_a_name_shorter_than_16_characters_must_match_exactly():
    row = dict(ROW, stock_name="AMYLIN PHARMA")  # 13 characters: not cut off
    assert match_roster_row(row, [cand("Amylin Pharmaceuticals Inc", "2010-07-01")])["status"] == (
        "matched"
    )
    row = dict(ROW, stock_name="AMYLIN")
    assert match_roster_row(row, [cand("Amylin Pharmaceuticals Inc", "2010-07-01")])["status"] == (
        "none"
    )


def test_a_16_character_name_is_a_cut_off_name_and_matches_by_prefix():
    row = dict(ROW, stock_name="AMYLIN PHARMACEU")
    assert len(row["stock_name"]) == 16
    out = match_roster_row(row, [cand("Amylin Pharmaceuticals, Inc.", "2010-07-01")])
    assert out["status"] == "matched"
    other = match_roster_row(row, [cand("Zeta Pharmaceuticals", "2010-07-01")])
    assert other["status"] == "none"


def test_a_cut_off_name_that_starts_two_filers_is_ambiguous_never_a_pick():
    row = dict(ROW, stock_name="AMYLIN PHARMACEU")
    out = match_roster_row(
        row,
        [
            cand("Amylin Pharmaceuticals Inc", "2010-07-01", cik="0000000001"),
            cand("Amylin Pharmaceutical Holdings", "2010-07-02", cik="0000000002"),
        ],
    )
    assert out["status"] == "ambiguous" and out["ciks"] == ["0000000001", "0000000002"]


def test_a_cut_off_name_prefix_does_not_apply_to_a_short_name_or_to_a_longer_roster_name():
    # The filer's name must start with the roster name, not the other way round.
    row = dict(ROW, stock_name="AMYLIN PHARMACEU")
    assert match_roster_row(row, [cand("Amylin Pharma", "2010-07-01")])["status"] == "matched"
    assert match_roster_row(row, [cand("Amy", "2010-07-01")])["status"] == "none"


def test_any_party_may_match_and_each_party_counts_its_own_cik():
    both = cand("NEW YORK STOCK EXCHANGE LLC", "2010-07-01", cik="0000876661")
    both["parties"].append({"cik": "0000000001", "name": "Alpha Corp", "filename": "x"})
    out = match_roster_row(ROW, [both])
    assert out["status"] == "matched" and out["ciks"] == ["0000000001"]


def test_notice_is_common_stock():
    assert notice_is_common_stock("Common Stock, par value $0.01")
    assert notice_is_common_stock("Ordinary Shares")
    assert not notice_is_common_stock("Warrants to purchase Common Stock")
    assert not notice_is_common_stock("Series A Preferred Stock")
    assert not notice_is_common_stock("4.5% Convertible Notes due 2020")
    assert not notice_is_common_stock("Debentures")
    assert not notice_is_common_stock("Common Stock Purchase Rights")


def test_master_index_url():
    assert master_index_url(2010, 3) == (
        "https://www.sec.gov/Archives/edgar/full-index/2010/QTR3/master.zip"
    )


def test_quarters_covering_spans_year_boundary():
    assert quarters_covering(date(2009, 11, 15), date(2010, 2, 1)) == [(2009, 4), (2010, 1)]
    assert quarters_covering(date(2010, 3, 31), date(2010, 4, 1)) == [(2010, 1), (2010, 2)]
    assert quarters_covering(date(2010, 5, 1), date(2010, 5, 1)) == [(2010, 2)]


def test_form25_candidates_keeps_form_25_family_and_drops_registration_forms():
    rows = [
        ("1", "ALPHA CORP", "25", "2010-01-05"),
        ("1", "ALPHA CORP", "25/A", "2010-01-06"),
        ("2", "BETA CORP", "25-NSE", "2010-01-07"),
        ("3", "GAMMA CORP", "25-NSE/A", "2010-01-08"),
        ("4", "DELTA CORP", "8-A12B", "2010-01-09"),
    ]
    client = FakeClient({master_index_url(2010, 1): master_zip(rows)})
    candidates, skipped = form25_candidates(client, date(2010, 1, 1), date(2010, 3, 31))
    assert skipped == []
    assert sorted(c["form"] for c in candidates) == ["25", "25-NSE", "25-NSE/A", "25/A"]


def test_a_failing_quarter_is_skipped_not_fatal():
    good = master_zip([("1", "ALPHA CORP", "25", "2010-01-05")])
    client = FakeClient(
        {
            master_index_url(2010, 1): good,
            master_index_url(2010, 2): SecFetchError("HTTP 500"),
            master_index_url(2010, 3): b"not a zip",
        }
    )
    candidates, skipped = form25_candidates(client, date(2010, 1, 1), date(2010, 9, 30))
    assert [s["quarter"] for s in skipped] == [(2010, 2), (2010, 3)]
    assert all(s["reason"] for s in skipped)
    assert candidates


def test_a_missing_quarter_is_skipped_too():
    client = FakeClient({})
    candidates, skipped = form25_candidates(client, date(2010, 1, 1), date(2010, 3, 31))
    assert candidates == [] and [s["quarter"] for s in skipped] == [(2010, 1)]


CIK = "0000051143"
ACCESSION = "0000876661-17-000796"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/51143/000087666117000796/"
XML_URL = INDEX_URL + "primary_doc.xml"
INDEX_HTML = (
    f"<html><body><a href='/Archives/edgar/data/51143/000087666117000796/primary_doc.xml'>"
    f"x</a> {ACCESSION}</body></html>"
).encode()
NOTICE_XML = b"""<notificationOfRemoval>
<schemaVersion>X0203</schemaVersion>
<exchange><cik>0000876661</cik><entityName>NEW YORK STOCK EXCHANGE LLC</entityName></exchange>
<issuer><cik>0000051143</cik><entityName>EXAMPLE CORP</entityName></issuer>
<descriptionClassSecurity>Common Stock, par value $0.01</descriptionClassSecurity>
<ruleProvision>17 CFR 240.12d2-2(a)(3)</ruleProvision>
<signatureData><signatureDate>2017-05-01</signatureDate></signatureData>
</notificationOfRemoval>"""


def test_read_notice_reads_an_exchange_notice_and_records_both_hashes():
    client = FakeClient({INDEX_URL: INDEX_HTML, XML_URL: NOTICE_XML})
    out = read_notice(client, ACCESSION, CIK)
    assert out["status"] == "readable"
    assert out["accession"] == ACCESSION
    assert out["index_sha256"] == hashlib.sha256(INDEX_HTML).hexdigest()
    assert out["document_sha256"] == hashlib.sha256(NOTICE_XML).hexdigest()
    assert out["notice"]["security_description"] == "Common Stock, par value $0.01"
    assert out["notice"]["rule_provision"] == "17 CFR 240.12d2-2(a)(3)"
    assert client.requested == [INDEX_URL, XML_URL]


def test_read_notice_returns_unreadable_for_an_issuer_filed_html_notice():
    client = FakeClient({INDEX_URL: INDEX_HTML, XML_URL: b"<html>Form 25 filed by issuer</html>"})
    out = read_notice(client, ACCESSION, CIK)
    assert out["status"] == "unreadable" and out["reason"]
    assert out["accession"] == ACCESSION


def test_read_notice_returns_unreadable_when_the_document_or_index_is_missing_or_fails():
    only_index = FakeClient({INDEX_URL: INDEX_HTML})
    assert read_notice(only_index, ACCESSION, CIK)["status"] == "unreadable"
    broken = FakeClient({INDEX_URL: SecFetchError("HTTP 500")})
    out = read_notice(broken, ACCESSION, CIK)
    assert out["status"] == "unreadable" and "500" in out["reason"]


def test_read_notice_reports_a_valid_notice_about_a_different_issuer_as_a_mismatch():
    other = NOTICE_XML.replace(b"0000051143", b"0000000777")
    client = FakeClient({INDEX_URL: INDEX_HTML, XML_URL: other})
    out = read_notice(client, ACCESSION, CIK)
    assert out["status"] == "issuer_mismatch"
    assert out["found_issuer_cik"] == "0000000777" and out["reason"]
    assert client.requested == [INDEX_URL, XML_URL]  # both real URLs were read


def test_match_roster_row_lists_each_ciks_own_accessions():
    out = match_roster_row(
        ROW,
        [
            cand("ALPHA CORP", "2010-07-01", cik="0000000001"),
            cand("Alpha Corp.", "2010-07-02", cik="0000000002"),
            cand("ALPHA CORP", "2010-07-03", cik="0000000001", form="25/A"),
        ],
    )
    assert list(out["by_cik"]) == ["0000000001", "0000000002"]
    assert len(out["by_cik"]["0000000001"]) == 2 and len(out["by_cik"]["0000000002"]) == 1


def test_normalize_name_is_idempotent():
    for name in ["SiRF Technology Holdings, Inc.", "LONGS DRG STRS", "ACME CO INC", "Inc.", "GBT"]:
        once = normalize_name(name)
        assert normalize_name(once) == once
    for name in ABBREVIATIONS.values():
        assert normalize_name(name) == normalize_name(normalize_name(name))


def test_normalize_name_leaves_a_class_letter_after_inc_in_place():
    # "INC-A" is not a trailing suffix word, so the roster name does not equal "Alpharma Inc".
    assert normalize_name("ALPHARMA INC-A") == "ALPHARMA INC A"
    row = dict(ROW, stock_name="ALPHARMA INC-A")
    assert match_roster_row(row, [cand("Alpharma Inc", "2010-07-01")])["status"] == "none"


def test_normalize_name_keeps_group_when_inc_follows():
    assert normalize_name("GROUP INC") == "GROUP"
    assert normalize_name("Shaw Group Inc") == "SHAW GROUP"


def test_match_accepts_real_date_objects_as_well_as_iso_strings():
    row = {"stock_name": "ALPHA CORP", "name_source": "mcgill", "exit_date": date(2010, 6, 30)}
    inside = cand("ALPHA CORP", "2010-07-01")
    inside["filing_date"] = date(2010, 7, 1)
    assert match_roster_row(row, [inside])["status"] == "matched"


def test_a_complete_16_character_name_matches_by_prefix_and_two_filers_make_it_ambiguous():
    # Documented behaviour: a 16-character name is treated as cut off, even when complete.
    row = dict(ROW, stock_name="ACME HOLDINGS CO")
    assert len(row["stock_name"]) == 16
    out = match_roster_row(
        row,
        [
            cand("Acme Holdings Corp", "2010-07-01", cik="0000000001"),
            cand("Acme Holdings Group Inc", "2010-07-02", cik="0000000002"),
        ],
    )
    assert out["status"] == "ambiguous" and out["ciks"] == ["0000000001", "0000000002"]


def test_module_exposes_only_the_documented_forms():
    assert sec_filings.FORM25_FORMS == frozenset({"25", "25/A", "25-NSE", "25-NSE/A"})


# ---- matching rule v1.1 (second pass only for stocks v1 left unmatched) ------------------------


def test_v1_results_carry_rule_version_v1():
    assert match_roster_row(ROW, [cand("ALPHA CORP", "2010-07-01")])["rule_version"] == "v1"
    assert match_roster_row(ROW, [])["rule_version"] == "v1"
    unnamed = {"stock_name": "", "name_source": "none", "exit_date": "2010-06-30"}
    assert match_roster_row(unnamed, [cand("ALPHA CORP", "2010-07-01")])["rule_version"] == "v1"


def test_v11_matches_cut_off_inc_suffix_while_v1_misses():
    # "ATHLON ENERGY IN" is 16 characters; v1 strips INC from the filer before prefix check.
    row = dict(ROW, stock_name="ATHLON ENERGY IN")
    assert len(row["stock_name"]) == 16
    filings = [cand("Athlon Energy Inc.", "2010-07-01", cik="0001575189")]
    assert match_roster_row(row, filings)["status"] == "none"
    out = sec_filings.match_roster_row_v11(row, filings)
    assert out["status"] == "matched"
    assert out["ciks"] == ["0001575189"]
    assert out["rule_version"] == "v1.1"
    assert sec_filings.normalize_name_v11_untrimmed("Athlon Energy Inc.") == "ATHLON ENERGY INC"
    assert sec_filings.normalize_name_v11_untrimmed("ATHLON ENERGY IN") == "ATHLON ENERGY IN"


def test_v11_strips_share_class_markers_before_suffix_stripping():
    row = dict(ROW, stock_name="ALPHARMA INC-A")
    filings = [cand("ALPHARMA INC", "2010-07-01", cik="0000730469")]
    assert match_roster_row(row, filings)["status"] == "none"
    out = sec_filings.match_roster_row_v11(row, filings)
    assert out["status"] == "matched" and out["rule_version"] == "v1.1"
    for variant in (
        "ALPHARMA INC-A",
        "ALPHARMA INC-B",
        "ALPHARMA INC-C",
        "ALPHARMA INC CL A",
        "ALPHARMA INC CLASS A",
        "ALPHARMA INC CLASS B",
        "ALPHARMA INC - CL C",
        "ALPHARMA INC, CLASS A",
    ):
        assert sec_filings.normalize_name_v11(variant) == "ALPHARMA"
    # Bare trailing A (not hyphenated, not CL/CLASS) and unsupported class letters stay intact.
    assert sec_filings.normalize_name_v11("STUDIO A") == "STUDIO A"
    assert sec_filings.normalize_name_v11("ALPHARMA INC-D") == "ALPHARMA INC D"


def test_hcc_prefix_stays_matched_under_v1_and_also_matches_under_v11():
    row = dict(ROW, stock_name="HCC INS HLDGS IN")
    assert len(row["stock_name"]) == 16
    filings = [cand("HCC INSURANCE HOLDINGS INC/DE/", "2010-07-01", cik="0000888919")]
    v1_out = match_roster_row(row, filings)
    assert v1_out["status"] == "matched" and v1_out["rule_version"] == "v1"
    v11_out = sec_filings.match_roster_row_v11(row, filings)
    assert v11_out["status"] == "matched" and v11_out["rule_version"] == "v1.1"


def test_v11_strips_trailing_edgar_slash_tags_only():
    assert sec_filings.normalize_name_v11("CYTEC INDUSTRIES INC/DE/") == "CYTEC INDUSTRY"
    assert sec_filings.normalize_name_v11("ACME CORP/NEW") == "ACME"
    assert sec_filings.normalize_name_v11("ACME CORP/MD/") == "ACME"
    assert sec_filings.normalize_name_v11_untrimmed("ACME CORP/DE/") == "ACME CORP"
    # Interior slashes (and non-tag words after a slash) are not stripped.
    assert sec_filings.normalize_name_v11("AT&T/EDISON") == "AT T EDISON"
    assert sec_filings.normalize_name_v11("AT&T/ED CORP") == "AT T ED"
    row = dict(ROW, stock_name="CYTEC INDS INC")
    filings = [cand("CYTEC INDUSTRIES INC/DE/", "2010-07-01", cik="0000912513")]
    assert match_roster_row(row, filings)["status"] == "none"
    out = sec_filings.match_roster_row_v11(row, filings)
    assert out["status"] == "matched" and out["rule_version"] == "v1.1"


def test_v11_folds_plurals_to_singular_including_esterline_tech():
    row = dict(ROW, stock_name="ESTERLINE TECH")
    filings = [cand("Esterline Technologies Corp", "2010-07-01", cik="0000033619")]
    assert match_roster_row(row, filings)["status"] == "none"
    out = sec_filings.match_roster_row_v11(row, filings)
    assert out["status"] == "matched" and out["rule_version"] == "v1.1"
    assert (
        sec_filings.normalize_name_v11("ESTERLINE TECH")
        == sec_filings.normalize_name_v11("ESTERLINE TECHS")
        == sec_filings.normalize_name_v11("Esterline Technology Corp")
        == sec_filings.normalize_name_v11("Esterline Technologies Corp")
        == "ESTERLINE TECHNOLOGY"
    )
    # COMPANIES folds to COMPANY, which is then stripped by SUFFIX_WORDS.
    assert sec_filings.normalize_name_v11("ACME COMPANIES") == "ACME"
    assert sec_filings.normalize_name_v11_untrimmed("ACME COMPANIES") == "ACME COMPANY"
    # Whole words only, and every table entry maps cleanly.
    assert len(sec_filings.PLURAL_TO_SINGULAR) == 12
    for plural, singular in sec_filings.PLURAL_TO_SINGULAR.items():
        assert sec_filings.normalize_name_v11_untrimmed(f"ACME {plural}") == f"ACME {singular}"


def test_v11_normalize_is_idempotent():
    samples = [
        "ATHLON ENERGY IN",
        "ALPHARMA INC-A",
        "HCC INSURANCE HOLDINGS INC/DE/",
        "Esterline Technologies Corp",
        "ACME COMPANIES",
        "AT&T/EDISON",
        "Inc.",
    ]
    for name in samples:
        once = sec_filings.normalize_name_v11(name)
        assert sec_filings.normalize_name_v11(once) == once


def test_v11_keeps_ambiguity_window_edges_and_unnamed_rules():
    row = dict(ROW, stock_name="ESTERLINE TECH")
    both = [
        cand("Esterline Technologies Corp", "2010-07-01", cik="0000000001"),
        cand("Esterline Technology Inc/DE/", "2010-07-02", cik="0000000002"),
    ]
    amb = sec_filings.match_roster_row_v11(row, both)
    assert amb["status"] == "ambiguous" and amb["ciks"] == ["0000000001", "0000000002"]
    assert amb["rule_version"] == "v1.1"
    # Window edges: -30 / +180 inclusive, -31 / +181 excluded.
    assert (
        sec_filings.match_roster_row_v11(row, [cand("Esterline Technologies Corp", "2010-05-31")])[
            "status"
        ]
        == "matched"
    )
    assert (
        sec_filings.match_roster_row_v11(row, [cand("Esterline Technologies Corp", "2010-05-30")])[
            "status"
        ]
        == "none"
    )
    assert (
        sec_filings.match_roster_row_v11(row, [cand("Esterline Technologies Corp", "2010-12-27")])[
            "status"
        ]
        == "matched"
    )
    assert (
        sec_filings.match_roster_row_v11(row, [cand("Esterline Technologies Corp", "2010-12-28")])[
            "status"
        ]
        == "none"
    )
    unnamed = {"stock_name": None, "name_source": "none", "exit_date": "2010-06-30"}
    out = sec_filings.match_roster_row_v11(unnamed, [cand("ALPHA CORP", "2010-07-01")])
    assert out["status"] == "none" and out["rule_version"] == "v1.1"


def test_v11_known_limitation_bare_hyphen_class_suffix_strips_alpha_a():
    # Known false-positive class: v1.1 strips a trailing -A/-B/-C marker even when no corporate
    # suffix precedes it, so a company whose own name ends in "-A" collapses onto "ALPHA INC".
    # Because the Form 25 notice check only verifies the picked CIK against its own filing, v1.1
    # matches are always classified as name-only identity leads and never confirmed.
    assert sec_filings.normalize_name_v11("ALPHA-A") == "ALPHA"
    row = dict(ROW, stock_name="ALPHA-A")
    filings = [cand("ALPHA INC", "2010-07-01", cik="0000000001")]
    assert match_roster_row(row, filings)["status"] == "none"
    out = sec_filings.match_roster_row_v11(row, filings)
    assert out["status"] == "matched" and out["rule_version"] == "v1.1"


def test_v11_known_limitation_complete_16_char_name_prefix_collision():
    # Known limitation: "ACME SYSTEMS INC" is a complete 16-character name, not a truncated one,
    # yet v1.1 treats every 16-character roster name as a prefix over untrimmed words; after
    # folding SYSTEMS -> SYSTEM, "ACME SYSTEM INC" is a prefix of "ACME SYSTEM INCOME FUND".
    # The Form 25 notice check only proves the filing belongs to the picked CIK, not that the
    # picked CIK is the roster stock, which is why v1.1-only leads are never marked "confirmed".
    row = dict(ROW, stock_name="ACME SYSTEMS INC")
    assert len(row["stock_name"]) == 16
    filings = [cand("ACME SYSTEM INCOME FUND", "2010-07-01", cik="0000000001")]
    assert match_roster_row(row, filings)["status"] == "none"
    out = sec_filings.match_roster_row_v11(row, filings)
    assert out["status"] == "matched" and out["rule_version"] == "v1.1"
