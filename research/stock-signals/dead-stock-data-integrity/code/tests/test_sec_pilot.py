"""Tests for the Form 25 pilot driver: quarter list, counts, refusals and the skipped limit."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import sec_pilot
from sec_client import SecFetchError, SecNotFound
from sec_filings import master_index_url
from sec_pilot import (
    OUTPUT_JSON,
    TooManySkippedQuarters,
    main,
    needed_quarters,
    run_form25,
    summarize,
)

pytestmark = pytest.mark.study

HEADER = (
    "CIK|Company Name|Form Type|Date Filed|Filename\n"
    "--------------------------------------------------------------------------------\n"
)
NOTICE = (
    "<notificationOfRemoval><schemaVersion>X0203</schemaVersion>"
    "<exchange><cik>0000876661</cik><entityName>NYSE LLC</entityName></exchange>"
    "<issuer><cik>{cik}</cik><entityName>X</entityName></issuer>"
    "<descriptionClassSecurity>{desc}</descriptionClassSecurity>"
    "<ruleProvision>17 CFR 240.12d2-2(a)(3)</ruleProvision>"
    "<signatureData><signatureDate>2010-07-01</signatureDate></signatureData>"
    "</notificationOfRemoval>"
)


def row(symbol: str, name: str, exit_date: str, source: str = "mcgill") -> dict[str, str]:
    """A roster row with only the columns the pilot reads."""
    return {
        "symbol": symbol,
        "permno": "1",
        "stock_name": name,
        "name_source": source,
        "exit_date": exit_date,
    }


def master_zip(rows: list[tuple[str, str, str, str, str]]) -> bytes:
    """A master.zip of (cik, name, form, date, accession) rows."""
    lines = [
        f"{cik}|{name}|{form}|{filed}|edgar/data/{int(cik)}/{acc}.txt\n"
        for cik, name, form, filed, acc in rows
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("master.idx", HEADER + "".join(lines))
    return buffer.getvalue()


def filing_files(cik: str, accession: str, desc: str) -> dict[str, bytes]:
    """The filing folder page and the notice document for one accession."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    page = (
        f"<html><a href='/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
        f"/primary_doc.xml'>x</a>{accession}</html>"
    ).encode()
    xml = NOTICE.format(cik=cik, desc=desc).encode()
    return {base: page, base + "primary_doc.xml": xml}


class FakeClient:
    """Canned bodies by URL; exception values are raised; no network, no cache."""

    network_fetches = 0
    cache_hits = 0
    corrupt_cache_refetches = 0

    def __init__(self, bodies: dict[str, object]) -> None:
        self.bodies = bodies

    def get(self, url: str) -> bytes:
        body = self.bodies.get(url)
        if body is None:
            raise SecNotFound(url)
        if isinstance(body, Exception):
            raise body
        assert isinstance(body, bytes)
        return body

    def manifest(self) -> dict[str, str]:
        return {url: "0" * 64 for url in sorted(self.bodies)}


ROWS = [
    row("A", "ALPHA CORP", "2010-02-15"),
    row("B", "BETA CORP", "2010-02-20"),
    row("C", "GAMMA CORP", "2010-02-25"),
    row("D", "ZETA CORP", "2010-02-26"),
    row("E", "", "2010-03-01", source="none"),
]
ACC_A = "0000000001-10-000001"
ACC_B1 = "0000000002-10-000001"
ACC_B2 = "0000000003-10-000001"
ACC_C = "0000000004-10-000001"


def good_client() -> FakeClient:
    """Alpha matches a readable common notice, Beta two filers, Gamma an unreadable notice."""
    bodies: dict[str, object] = {
        master_index_url(2010, 2): master_zip([]),
        master_index_url(2010, 3): master_zip([]),
        master_index_url(2010, 1): master_zip(
            [
                ("1", "Alpha Corp", "25", "2010-03-02", ACC_A),
                ("2", "Beta Corp", "25", "2010-03-03", ACC_B1),
                ("3", "BETA CORP.", "25", "2010-03-04", ACC_B2),
                ("4", "Gamma Corp", "25", "2010-03-05", ACC_C),
                ("5", "Alpha Corp", "8-A12B", "2010-03-06", "0000000005-10-000001"),
            ]
        ),
    }
    bodies.update(filing_files("0000000001", ACC_A, "Common Stock"))
    return FakeClient(bodies)


def test_needed_quarters_cover_exit_minus_30_to_plus_180_for_named_rows_only():
    rows = [
        row("A", "ALPHA", "2010-02-15"),  # 2010-01-16 .. 2010-08-14: Q1 to Q3
        row("B", "", "2015-02-15", source="none"),  # unnamed: needs nothing
        row("C", "GAMMA", "2009-12-31"),  # 2009-12-01 .. 2010-06-29: Q4 2009 to Q2 2010
    ]
    assert needed_quarters(rows) == [(2009, 4), (2010, 1), (2010, 2), (2010, 3)]


def test_run_form25_counts_every_kind_of_result():
    payload = run_form25(good_client(), ROWS, run_date="2026-11-24")
    summary = payload["summary"]
    assert (summary["matched"], summary["ambiguous"], summary["none"]) == (2, 1, 1)
    assert summary["name_unknown"] == 1 and summary["named_stocks"] == 4
    assert summary["exact_match_rate_of_named"] == 0.5
    assert (summary["notices_readable"], summary["notices_unreadable"]) == (1, 1)
    assert summary["notices_common_stock"] == 1
    by_symbol = {s["symbol"]: s for s in payload["stocks"]}
    assert by_symbol["A"]["notice"]["status"] == "readable"
    assert by_symbol["A"]["notice"]["is_common_stock"] is True
    assert by_symbol["B"]["status"] == "ambiguous" and by_symbol["B"]["notice"] is None
    assert by_symbol["C"]["notice"]["status"] == "unreadable" and by_symbol["C"]["notice"]["reason"]
    assert by_symbol["E"]["status"] == "name_unknown"
    assert payload["quarters_skipped"] == [] and len(payload["quarters_downloaded"]) == 3
    assert payload["manifest"] and payload["request_count"]["network_fetches"] == 0


def test_a_non_common_first_notice_is_passed_over_for_a_later_common_one():
    first, second = "0000000001-10-000001", "0000000001-10-000002"
    bodies = dict(good_client().bodies)
    bodies[master_index_url(2010, 1)] = master_zip(
        [
            ("1", "Alpha Corp", "25", "2010-03-02", first),
            ("1", "Alpha Corp", "25", "2010-03-09", second),
        ]
    )
    bodies.update(filing_files("0000000001", first, "Warrants to purchase Common Stock"))
    bodies.update(filing_files("0000000001", second, "Common Stock"))
    payload = run_form25(FakeClient(bodies), ROWS[:1])
    assert payload["stocks"][0]["notice"]["accession"] == second


def test_limit_quarters_downloads_only_the_first_quarters():
    payload = run_form25(good_client(), ROWS, limit_quarters=1)
    assert payload["limit_quarters"] == 1
    assert payload["quarters_downloaded"] == [[2010, 1]]
    assert payload["quarters_needed"] == 3


def test_too_many_skipped_quarters_stops_the_run():
    client = FakeClient({master_index_url(2010, 1): master_zip([])})  # Q2 and Q3 are missing
    with pytest.raises(TooManySkippedQuarters, match="2 quarterly indexes failed"):
        run_form25(client, ROWS, max_skipped=1)
    payload = run_form25(client, ROWS, max_skipped=2)  # exactly two missing: allowed
    assert len(payload["quarters_skipped"]) == 2


def test_a_quarter_that_will_not_parse_or_download_counts_as_skipped():
    bodies = {**good_client().bodies, master_index_url(2010, 2): b"not a zip"}
    bodies[master_index_url(2010, 3)] = SecFetchError("HTTP 500")
    payload = run_form25(FakeClient(bodies), ROWS[:1], max_skipped=2)
    assert [s["quarter"] for s in payload["quarters_skipped"]] == [[2010, 2], [2010, 3]]
    with pytest.raises(TooManySkippedQuarters):
        run_form25(FakeClient(bodies), ROWS[:1], max_skipped=1)


def test_summarize_with_no_named_stocks_has_no_rate():
    stocks = [{"status": "name_unknown", "notice": None}]
    assert summarize(stocks)["exact_match_rate_of_named"] is None


def write_roster(path: Path) -> Path:
    """A tiny roster csv with the columns the pilot reads."""
    path.write_text(
        "symbol,permno,stock_name,name_source,exit_date\nA,1,ALPHA CORP,mcgill,2010-02-15\n"
    )
    return path


def args_for(tmp_path: Path, *extra: str) -> list[str]:
    """A command line for ``main`` with temp folders and a tiny roster."""
    roster = write_roster(tmp_path / "roster.csv")
    return [
        "--phase",
        "form25",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out",
        str(tmp_path / "out"),
        "--roster",
        str(roster),
        *extra,
    ]


@pytest.fixture(autouse=True)
def user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A made-up fair-access name, never a real address."""
    monkeypatch.setenv("SEC_USER_AGENT", "ggTrader Study contact@example.test")


def test_main_refuses_to_overwrite_without_force(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / OUTPUT_JSON).write_text("{}")
    made: list[str] = []

    def factory(ua: str, cache: Path) -> FakeClient:
        made.append(ua)
        return good_client()

    with pytest.raises(SystemExit) as stop:
        main(args_for(tmp_path), client_factory=factory)
    assert stop.value.code == 2 and made == []
    assert (out / OUTPUT_JSON).read_text() == "{}"
    assert main(args_for(tmp_path, "--force"), client_factory=factory) == 0
    assert json.loads((out / OUTPUT_JSON).read_text())["summary"]["matched"] == 1


def test_main_needs_a_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SEC_USER_AGENT")
    with pytest.raises(SystemExit) as stop:
        main(args_for(tmp_path), client_factory=lambda ua, cache: good_client())
    assert stop.value.code == 2


def test_main_writes_the_result_and_the_cache_checksums(tmp_path: Path):
    assert main(args_for(tmp_path), client_factory=lambda ua, cache: good_client()) == 0
    assert (tmp_path / "out" / OUTPUT_JSON).is_file()
    assert (tmp_path / "cache" / "SHA256SUMS.txt").is_file()


def test_main_stops_with_an_error_when_too_many_quarters_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    def factory(ua: str, cache: Path) -> FakeClient:
        return FakeClient({})

    code = main(args_for(tmp_path, "--max-skipped", "1"), client_factory=factory)
    assert code == 2
    assert "quarterly indexes failed" in capsys.readouterr().err
    assert not (tmp_path / "out" / OUTPUT_JSON).exists()
    assert (tmp_path / "cache" / "SHA256SUMS.txt").is_file()


def test_the_default_roster_is_the_study_roster():
    assert sec_pilot.STUDY_DIR.name == "dead-stock-data-integrity"
    assert (sec_pilot.STUDY_DIR / sec_pilot.ROSTER_FILE).is_file()
