"""Tests for the Form 25 pilot driver: quarter list, counts, refusals and the skipped limit."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import sec_pilot
from sec_client import SecClient, SecFetchError, SecNotFound
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


def filing_files(
    cik: str, accession: str, desc: str, issuer: str | None = None
) -> dict[str, bytes]:
    """The filing folder page and the notice document; ``issuer`` defaults to ``cik``."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    page = (
        f"<html><a href='/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
        f"/primary_doc.xml'>x</a>{accession}</html>"
    ).encode()
    xml = NOTICE.format(cik=issuer or cik, desc=desc).encode()
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
    assert summary["confirmed_common"] == 1 and summary["unconfirmed_unreadable"] == 1
    assert summary["matched"] == 2  # headline: confirmed_common + unconfirmed_unreadable
    assert (summary["ambiguous"], summary["none"], summary["name_unknown"]) == (1, 1, 1)
    assert summary["name_matched"] == 3 and summary["named_stocks"] == 4
    assert summary["matched_rate_of_named"] == 0.5
    assert summary["non_common_only"] == 0 and summary["rejected_issuer_mismatch"] == 0
    by_symbol = {s["symbol"]: s for s in payload["stocks"]}
    assert by_symbol["A"]["notice"]["status"] == "readable"
    assert by_symbol["A"]["notice"]["is_common_stock"] is True
    assert by_symbol["A"]["confirmation"] == "confirmed_common"
    assert by_symbol["B"]["status"] == "ambiguous" and by_symbol["B"]["notice"] is None
    assert by_symbol["C"]["status"] == "matched"
    assert by_symbol["C"]["confirmation"] == "unconfirmed_unreadable"
    assert by_symbol["C"]["notice"]["status"] == "unreadable" and by_symbol["C"]["notice"]["reason"]
    assert by_symbol["D"]["status"] == "none" and by_symbol["D"]["confirmation"] == "not_checked"
    assert by_symbol["E"]["status"] == "name_unknown"
    assert {s["confirmation"] for s in payload["stocks"]} <= set(sec_pilot.CONFIRMATIONS)
    assert payload["quarters_skipped"] == [] and len(payload["quarters_downloaded"]) == 3
    assert payload["manifest"] and payload["request_count"]["network_fetches"] == 0


def count_parts(summary: dict) -> int:
    """The outcome counts that must add up to the roster size."""
    keys = (
        "confirmed_common",
        "unconfirmed_unreadable",
        "non_common_only",
        "rejected_issuer_mismatch",
        "ambiguous",
        "none",
        "name_unknown",
    )
    return sum(summary[k] for k in keys)


def test_summary_counts_add_up_to_the_roster_size():
    payload = run_form25(good_client(), ROWS)
    assert count_parts(payload["summary"]) == len(ROWS) == payload["summary"]["roster_rows"]


def test_an_exchange_named_like_the_stock_is_rejected_by_issuer_validation():
    # The exchange's own CIK shares the stock's name in the index; its notice names another issuer.
    acc = "0000876661-10-000001"
    bodies = dict(good_client().bodies)
    bodies[master_index_url(2010, 1)] = master_zip(
        [("876661", "Alpha Corp", "25", "2010-03-02", acc)]
    )
    bodies.update(filing_files("0000876661", acc, "Common Stock", issuer="0000000777"))
    payload = run_form25(FakeClient(bodies), ROWS[:1])
    stock = payload["stocks"][0]
    assert stock["status"] == "none" and stock["reason"] == "issuer_mismatch"
    assert stock["confirmation"] == "rejected_issuer_mismatch" and stock["ciks"] == []
    assert stock["rejected_ciks"][0]["cik"] == "0000876661"
    assert stock["rejected_ciks"][0]["reason"] == "issuer_mismatch"
    assert (
        payload["summary"]["rejected_issuer_mismatch"] == 1 and payload["summary"]["matched"] == 0
    )


def test_a_rejected_party_is_dropped_and_the_real_issuer_stays():
    real, exchange = "0000000001-10-000001", "0000876661-10-000001"
    bodies = dict(good_client().bodies)
    bodies[master_index_url(2010, 1)] = master_zip(
        [
            ("1", "Alpha Corp", "25", "2010-03-02", real),
            ("876661", "Alpha Corp", "25", "2010-03-02", exchange),
        ]
    )
    bodies.update(filing_files("0000000001", real, "Common Stock"))
    bodies.update(filing_files("0000876661", exchange, "Common Stock", issuer="0000000001"))
    stock = run_form25(FakeClient(bodies), ROWS[:1])["stocks"][0]
    assert stock["status"] == "matched" and stock["ciks"] == ["0000000001"]
    assert [r["cik"] for r in stock["rejected_ciks"]] == ["0000876661"]


def test_a_warrants_only_filing_is_not_a_match():
    acc = "0000000001-10-000001"
    bodies = dict(good_client().bodies)
    bodies[master_index_url(2010, 1)] = master_zip([("1", "Alpha Corp", "25", "2010-03-02", acc)])
    bodies.update(filing_files("0000000001", acc, "Warrants to purchase Common Stock"))
    payload = run_form25(FakeClient(bodies), ROWS[:1])
    stock = payload["stocks"][0]
    assert stock["status"] == "none" and stock["confirmation"] == "non_common_only"
    assert stock["reason"] == "non_common_only" and stock["ciks"] == []
    assert stock["rejected_ciks"][0]["reason"] == "non_common_only"
    assert payload["summary"]["non_common_only"] == 1 and payload["summary"]["matched"] == 0


def test_an_unreadable_html_notice_is_unconfirmed_but_kept():
    acc = "0000000001-10-000001"
    base = "https://www.sec.gov/Archives/edgar/data/1/000000000110000001/"
    bodies = dict(good_client().bodies)
    bodies[master_index_url(2010, 1)] = master_zip([("1", "Alpha Corp", "25", "2010-03-02", acc)])
    bodies[base] = f"<html><a href='/x/primary_doc.htm'>x</a>{acc}</html>".encode()
    stock = run_form25(FakeClient(bodies), ROWS[:1])["stocks"][0]  # no primary_doc.xml: 404
    assert stock["status"] == "matched" and stock["confirmation"] == "unconfirmed_unreadable"
    assert stock["ciks"] == ["0000000001"] and stock["notice"]["status"] == "unreadable"


def test_two_roster_stocks_matching_the_same_filing_are_both_processed():
    rows = [row("A", "ALPHA CORP", "2010-02-15"), row("A2", "Alpha Corp.", "2010-02-20")]
    payload = run_form25(good_client(), rows)
    assert [s["status"] for s in payload["stocks"]] == ["matched", "matched"]
    assert [s["confirmation"] for s in payload["stocks"]] == ["confirmed_common"] * 2
    assert payload["summary"]["confirmed_common"] == 2


def two_filer_bodies(a_files: dict[str, bytes], a_accessions: list[str]) -> dict[str, object]:
    """Q1 lists the given filings for CIK 1 and one common-stock filing for CIK 2."""
    b_acc = "0000000002-10-000001"
    bodies = dict(good_client().bodies)
    listed = [
        ("1", "Alpha Corp", "25", f"2010-03-0{n + 2}", acc) for n, acc in enumerate(a_accessions)
    ]
    listed.append(("2", "Alpha Corp", "25", "2010-03-09", b_acc))
    bodies[master_index_url(2010, 1)] = master_zip(listed)
    bodies.update(a_files)
    bodies.update(filing_files("0000000002", b_acc, "Common Stock"))
    return bodies


def test_a_cik_with_an_unreadable_filing_is_not_called_non_common_only():
    # A: a readable warrants notice and an unreadable filing (possibly the common shares).
    warrants, html = "0000000001-10-000001", "0000000001-10-000002"
    a_files = filing_files("0000000001", warrants, "Warrants to purchase Common Stock")
    base = "https://www.sec.gov/Archives/edgar/data/1/000000000110000002/"
    a_files[base] = f"<html><a href='/x/doc.htm'>x</a>{html}</html>".encode()  # no primary_doc.xml
    bodies = two_filer_bodies(a_files, [warrants, html])
    stock = run_form25(FakeClient(bodies), ROWS[:1])["stocks"][0]
    assert stock["status"] == "ambiguous" and stock["confirmation"] == "not_checked"
    assert stock["ciks"] == ["0000000001", "0000000002"]  # A is not silently dropped
    assert stock["rejected_ciks"] == []
    states = {c["cik"]: c["state"] for c in stock["cik_checks"]}
    assert states == {"0000000001": "unreadable", "0000000002": "confirmed_common"}


def test_a_cik_whose_every_filing_was_read_and_none_is_common_is_dropped():
    warrants = "0000000001-10-000001"
    bodies = two_filer_bodies(
        filing_files("0000000001", warrants, "Warrants to purchase Common Stock"), [warrants]
    )
    stock = run_form25(FakeClient(bodies), ROWS[:1])["stocks"][0]
    assert stock["status"] == "matched" and stock["ciks"] == ["0000000002"]
    assert stock["confirmation"] == "confirmed_common"
    assert [(r["cik"], r["reason"]) for r in stock["rejected_ciks"]] == [
        ("0000000001", "non_common_only")
    ]


def test_a_form_25_and_its_amendment_from_one_cik_are_one_match():
    first, amend = "0000000001-10-000001", "0000000001-10-000002"
    bodies = dict(good_client().bodies)
    bodies[master_index_url(2010, 1)] = master_zip(
        [
            ("1", "Alpha Corp", "25", "2010-03-02", first),
            ("1", "Alpha Corp", "25/A", "2010-03-09", amend),
        ]
    )
    bodies.update(filing_files("0000000001", first, "Common Stock"))
    bodies.update(filing_files("0000000001", amend, "Common Stock"))
    stock = run_form25(FakeClient(bodies), ROWS[:1])["stocks"][0]
    assert stock["status"] == "matched" and stock["ciks"] == ["0000000001"]
    assert stock["notice"]["accession"] == first


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
    stocks = [{"status": "name_unknown", "confirmation": "not_checked", "notice": None}]
    assert summarize(stocks)["matched_rate_of_named"] is None


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
    monkeypatch.setenv("SEC_USER_AGENT", "ggTrader test client")


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
    assert json.loads((out / OUTPUT_JSON).read_text())["summary"]["confirmed_common"] == 1


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


class OpenerResponse:
    """A minimal 200 response for a real ``SecClient`` driven by a fake opener."""

    status = 200

    def __init__(self, url: str, body: bytes) -> None:
        self.url, self.headers, self._body = url, {}, io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self.url

    def close(self) -> None:
        self._body.close()


def test_the_user_agent_is_sent_but_never_written_to_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    agent = "ggTrader test client unique-7f3a91"
    monkeypatch.setenv("SEC_USER_AGENT", agent)
    bodies = good_client().bodies
    seen: list[str | None] = []

    def opener(request: Any) -> OpenerResponse:
        seen.append(request.get_header("User-agent"))
        return OpenerResponse(request.full_url, bodies[request.full_url])

    def factory(ua: str, cache: Path) -> SecClient:
        return SecClient(ua, cache, min_interval=0, opener=opener)

    assert main(args_for(tmp_path), client_factory=factory) == 0
    assert seen and set(seen) == {agent}
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any(p.name == OUTPUT_JSON for p in written)
    assert any(p.name == "SHA256SUMS.txt" for p in written)
    assert all(agent.encode() not in p.read_bytes() for p in written)


def test_the_result_file_is_written_through_the_atomic_json_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ggTrader.utils import io as atomic_io

    assert sec_pilot.write_json is atomic_io.write_json
    calls: list[tuple[Path, bool]] = []

    def spy(path: Path, payload: Any, *, strict: bool = False, indent: int = 2) -> Path:
        calls.append((Path(path), strict))
        return atomic_io.write_json(path, payload, strict=strict, indent=indent)

    monkeypatch.setattr(sec_pilot, "write_json", spy)
    assert main(args_for(tmp_path), client_factory=lambda ua, cache: good_client()) == 0
    assert calls == [(tmp_path / "out" / OUTPUT_JSON, True)]
    assert not list((tmp_path / "out").glob("*.tmp"))


def run_form25_main(tmp_path: Path) -> dict[str, Any]:
    """Run the Form 25 phase on the tiny roster and return the written result."""
    assert main(args_for(tmp_path), good_client_factory()) == 0
    return json.loads((tmp_path / "out" / OUTPUT_JSON).read_text())


def good_client_factory() -> Any:
    return lambda ua, cache: good_client()


def test_the_form25_result_carries_a_provenance_block(tmp_path: Path):
    result = run_form25_main(tmp_path)
    prov = result["provenance"]
    assert prov["phase"] == "form25" and prov["run_date"] == result["run_date"]
    assert prov["roster_sha256"] == sec_pilot.sha256_file(tmp_path / "roster.csv")
    assert set(prov["code_sha256"]) == set(sec_pilot.CODE_FILES)
    for name, digest in prov["code_sha256"].items():
        assert digest == sec_pilot.sha256_file(sec_pilot.ROOT / name)
    assert prov["git_head"] and prov["git_dirty"] in (True, False, "unknown")
    assert prov["quarter_window"]["needed"] == result["quarters_needed"]
    assert prov["quarter_window"]["first"] and prov["quarter_window"]["last"]
    assert prov["quarter_window"]["start"] <= prov["quarter_window"]["end"]
    # What the writer produces is what the payouts phase's validator accepts.
    assert sec_pilot.form25_provenance_problems(prov, prov["roster_sha256"]) == []
    rules = prov["matching_rules"]
    assert rules["before_days"] == 30 and rules["after_days"] == 180
    assert rules["truncated_name_length"] == 16 and len(rules["abbreviations_sha256"]) == 64


@pytest.mark.parametrize(("status", "dirty"), [("", False), (" M code/x.py", True)])
def test_the_dirty_flag_follows_git_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, dirty: bool
):
    import dead_stock_data_integrity as study

    asked: list[tuple[str, ...]] = []

    def fake_git(root: Path, *args: str) -> str:
        asked.append(args)
        return "deadbeef" if args[0] == "rev-parse" else status

    monkeypatch.setattr(study, "_git", fake_git)
    prov = run_form25_main(tmp_path)["provenance"]
    assert prov["git_head"] == "deadbeef" and prov["git_dirty"] is dirty
    assert any(a[0] == "status" for a in asked)


def test_git_that_cannot_be_asked_is_recorded_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import dead_stock_data_integrity as study

    monkeypatch.setattr(study, "_git", lambda root, *args: None)
    prov = run_form25_main(tmp_path)["provenance"]
    assert prov["git_head"] == "unknown" and prov["git_dirty"] == "unknown"


def test_the_provenance_never_holds_the_user_agent_or_an_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    agent = "ggTrader test client unique-9c2e41 contact@example.test"
    monkeypatch.setenv("SEC_USER_AGENT", agent)
    text = json.dumps(run_form25_main(tmp_path)["provenance"])
    assert "unique-9c2e41" not in text and "@" not in text


def make_repo(tmp_path: Path) -> Path:
    """A real throw-away git repository with one tracked code file and an ``out`` folder."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "out").mkdir(parents=True)
    (repo / "code.py").write_text("x = 1\n")
    (repo / "roster.csv").write_text(
        "symbol,permno,stock_name,name_source,exit_date\nA,1,ALPHA CORP,mcgill,2010-02-15\n"
    )
    for cmd in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t", "commit", "-q", "-m", "x"],
    ):
        subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)
    return repo


def repo_provenance(repo: Path) -> dict[str, Any]:
    rows, digest = sec_pilot.load_roster(repo / "roster.csv")
    return sec_pilot.phase_provenance(
        "form25", digest, rows, out_dir=repo / "out", run_date="2026-11-24", root=repo
    )


def test_exclusions_are_exactly_the_three_output_names_and_hide_only_them(tmp_path: Path):
    repo = make_repo(tmp_path)
    for name in ("_form25_matches.json", "_payouts.csv", "_payouts.json"):
        (repo / "out" / name).write_text("{}")
    prov = repo_provenance(repo)
    assert prov["git_dirty_ignores"] == [
        "out/_form25_matches.json",
        "out/_payouts.csv",
        "out/_payouts.json",
    ]
    assert prov["git_dirty"] is False  # untracked results alone do not make the tree dirty
    (repo / "code.py").write_text("x = 2\n")
    assert repo_provenance(repo)["git_dirty"] is True  # a real code change still does


def test_a_symlinked_output_cannot_hide_a_changed_code_file(tmp_path: Path):
    repo = make_repo(tmp_path)
    (repo / "code.py").write_text("x = 2\n")
    (repo / "out" / "_payouts.csv").symlink_to(repo / "code.py")
    prov = repo_provenance(repo)
    # The excluded name is the link's own name; the file it points at is still counted.
    assert "out/_payouts.csv" in prov["git_dirty_ignores"]
    assert "code.py" not in prov["git_dirty_ignores"]
    assert prov["git_dirty"] is True


@pytest.mark.parametrize("name", ["_form25_matches.json", "_payouts.csv", "_payouts.json"])
@pytest.mark.parametrize("phase", ["form25", "payouts"])
def test_both_phases_refuse_an_output_that_is_a_symlink(tmp_path: Path, name: str, phase: str):
    argv = args_for(tmp_path)
    argv[1] = phase
    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim.py"
    victim.write_text("keep me")
    (out / name).symlink_to(victim)
    made: list[str] = []

    def factory(ua: str, cache: Path) -> Any:
        made.append(ua)
        return good_client()

    with pytest.raises(SystemExit) as stop:
        main([*argv, "--force"], client_factory=factory)
    assert stop.value.code == 2 and made == []
    assert victim.read_text() == "keep me"


def test_both_phases_refuse_an_output_folder_that_is_a_symlink(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "out").symlink_to(real)
    with pytest.raises(SystemExit) as stop:
        main(args_for(tmp_path), client_factory=lambda ua, cache: good_client())
    assert stop.value.code == 2


def test_the_form25_phase_reads_the_roster_once_and_records_that_hash(
    tmp_path: Path, open_counter: Any
):
    argv = args_for(tmp_path)
    roster = tmp_path / "roster.csv"
    original = roster.read_bytes()
    counter = open_counter(roster)
    assert main(argv, client_factory=lambda ua, cache: good_client()) == 0
    assert counter.reads == ["read_bytes"]
    prov = json.loads((tmp_path / "out" / OUTPUT_JSON).read_text())["provenance"]
    import hashlib

    assert prov["roster_sha256"] == hashlib.sha256(original).hexdigest()


def test_a_roster_changed_after_the_read_cannot_change_the_recorded_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    argv = args_for(tmp_path)
    roster = tmp_path / "roster.csv"
    used = sec_pilot.sha256_file(roster)

    def factory(ua: str, cache: Path) -> Any:
        roster.write_text(roster.read_text() + "B,2,BETA CORP,mcgill,2010-02-15\n")
        return good_client()

    assert main(argv, client_factory=factory) == 0
    prov = json.loads((tmp_path / "out" / OUTPUT_JSON).read_text())["provenance"]
    assert prov["roster_sha256"] == used  # the bytes parsed, not the file as it is later


# ---- matching rule v1.1 second-pass orchestration and provenance -------------------------------


def test_run_form25_runs_v11_only_on_v1_none_and_labels_results(monkeypatch: pytest.MonkeyPatch):
    import sec_filings

    v11_called_for: list[str] = []
    real_v11 = sec_filings.match_roster_row_v11

    def spy_v11(row_item: Any, candidates: list[dict], **kwargs: Any) -> dict:
        v11_called_for.append(str(row_item.get("stock_name")))
        return real_v11(row_item, candidates, **kwargs)

    monkeypatch.setattr(sec_filings, "match_roster_row_v11", spy_v11)

    rows = [
        row("V1_OK", "ALPHA CORP", "2010-02-15"),
        row("V1_AMB", "BETA CORP", "2010-02-20"),
        row("V11_OK", "ESTERLINE TECH", "2010-02-20"),
        row("V11_CUT", "ATHLON ENERGY IN", "2010-02-20"),
        row("V11_WARR", "ALPHARMA INC-A", "2010-02-20"),
        row("STILL_NONE", "ZETA CORP", "2010-02-20"),
        row("UNNAMED", "", "2010-03-01", source="none"),
    ]
    acc_est = "0000000010-10-000001"
    acc_ath = "0000000011-10-000001"
    acc_alp = "0000000012-10-000001"
    bodies: dict[str, object] = {
        master_index_url(2010, 2): master_zip([]),
        master_index_url(2010, 3): master_zip([]),
        master_index_url(2010, 1): master_zip(
            [
                ("1", "Alpha Corp", "25", "2010-03-02", ACC_A),
                ("2", "Beta Corp", "25", "2010-03-03", ACC_B1),
                ("3", "BETA CORP.", "25", "2010-03-04", ACC_B2),
                ("10", "Esterline Technologies Corp", "25", "2010-03-05", acc_est),
                ("11", "Athlon Energy Inc.", "25", "2010-03-06", acc_ath),
                ("12", "ALPHARMA INC", "25", "2010-03-07", acc_alp),
            ]
        ),
    }
    bodies.update(filing_files("0000000001", ACC_A, "Common Stock"))
    bodies.update(filing_files("0000000010", acc_est, "Common Stock"))
    # Athlon has no primary_doc.xml -> unconfirmed_unreadable under v1.1
    # Alpharma has warrants only -> rejected as non_common_only under v1.1
    bodies.update(filing_files("0000000012", acc_alp, "Warrants to purchase Common Stock"))

    payload = run_form25(FakeClient(bodies), rows, run_date="2026-11-24")
    # v1.1 must NEVER run for V1_OK (v1 matched), V1_AMB (v1 ambiguous), or UNNAMED.
    assert v11_called_for == [
        "ESTERLINE TECH",
        "ATHLON ENERGY IN",
        "ALPHARMA INC-A",
        "ZETA CORP",
    ]
    by_sym = {s["symbol"]: s for s in payload["stocks"]}
    assert by_sym["V1_OK"]["status"] == "matched" and by_sym["V1_OK"]["rule_version"] == "v1"
    assert by_sym["V1_AMB"]["status"] == "ambiguous" and by_sym["V1_AMB"]["rule_version"] == "v1"
    assert by_sym["V11_OK"]["status"] == "matched" and by_sym["V11_OK"]["rule_version"] == "v1.1"
    assert by_sym["V11_OK"]["confirmation"] == "confirmed_common"
    assert by_sym["V11_CUT"]["status"] == "matched" and by_sym["V11_CUT"]["rule_version"] == "v1.1"
    assert by_sym["V11_CUT"]["confirmation"] == "unconfirmed_unreadable"
    assert by_sym["V11_WARR"]["status"] == "none" and by_sym["V11_WARR"]["rule_version"] == "v1.1"
    assert by_sym["V11_WARR"]["confirmation"] == "non_common_only"
    assert by_sym["STILL_NONE"]["status"] == "none"
    assert by_sym["STILL_NONE"]["rule_version"] == "v1.1"
    assert (
        by_sym["UNNAMED"]["status"] == "name_unknown" and by_sym["UNNAMED"]["rule_version"] is None
    )

    summary = payload["summary"]
    assert summary["matched"] == 3
    assert summary["matched_v1"] == 1
    assert summary["matched_v1_1_only"] == 2
    assert summary["matched_v1"] + summary["matched_v1_1_only"] == summary["matched"]
    assert count_parts(summary) == len(rows) == summary["roster_rows"]


def test_matching_rules_provenance_records_v11_constants(tmp_path: Path):
    import sec_filings

    rules = run_form25_main(tmp_path)["provenance"]["matching_rules"]
    assert rules["rule_versions"] == ["v1", "v1.1"]
    assert len(rules["plural_to_singular_sha256"]) == 64
    assert rules["edgar_slash_tag_pattern"] == sec_filings.EDGAR_SLASH_TAG_PATTERN
    assert rules["share_class_pattern"] == sec_filings.SHARE_CLASS_PATTERN
