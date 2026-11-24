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


def test_read_notice_rejects_a_notice_for_a_different_issuer():
    client = FakeClient({INDEX_URL: INDEX_HTML, XML_URL: NOTICE_XML})
    out = read_notice(client, ACCESSION, "0000000009")
    assert out["status"] == "unreadable"


def test_module_exposes_only_the_documented_forms():
    assert sec_filings.FORM25_FORMS == frozenset({"25", "25/A", "25-NSE", "25-NSE/A"})
