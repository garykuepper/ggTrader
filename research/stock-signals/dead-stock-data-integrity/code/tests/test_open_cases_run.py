"""Tests for the Phase 3 open-case driver: preconditions, two-pass flow, outputs, provenance."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import open_cases
import open_cases_run
import payout_table
import pytest
import sec_filings
import sec_pilot
from sec_client import SecClient, SecFetchError, SecNotFound

pytestmark = pytest.mark.study

INDEX_HEADER = (
    "Description: Master Index of EDGAR Dissemination Feed\n"
    "CIK|Company Name|Form Type|Date Filed|Filename\n"
    "--------------------------------------------------------------------------------\n"
)
CLEAN_GIT = {"git_head": "a" * 40, "git_dirty": False}
DEFAULT_UA = "ggTrader Phase3 Test contact@example.test"


def make_master_zip(rows: Sequence[tuple[str, str, str, str, str]]) -> bytes:
    """Build a small valid master.zip of ``(cik, name, form, filed, accession)`` rows."""
    lines = [
        f"{cik}|{name}|{form}|{filed}|edgar/data/{int(cik)}/{accession}.txt\n"
        for cik, name, form, filed, accession in rows
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("master.idx", INDEX_HEADER + "".join(lines))
    return buffer.getvalue()


def seed_cache(cache_dir: Path, url: str, body: bytes) -> None:
    """Write a verified ``SecClient`` cache entry directly on disk without network access."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    (cache_dir / f"{key}.body").write_bytes(body)
    meta = {
        "url": url,
        "sha256": hashlib.sha256(body).hexdigest(),
        "fetched_at": "2026-11-24T00:00:00+00:00",
        "status": 200,
    }
    (cache_dir / f"{key}.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def submissions_json(filings: Sequence[tuple[str, str, str, str, str]]) -> bytes:
    """Build an SEC submissions JSON payload from ``(form, date, items, acc, doc)`` tuples."""
    forms, dates, items, accs, docs = [], [], [], [], []
    for form, filed, item_codes, acc, doc in filings:
        forms.append(form)
        dates.append(filed)
        items.append(item_codes)
        accs.append(acc)
        docs.append(doc)
    doc_obj = {
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": dates,
                "items": items,
                "accessionNumber": accs,
                "primaryDocument": docs,
            },
            "files": [],
        }
    }
    return json.dumps(doc_obj).encode("utf-8")


def payout_row(
    *,
    permno: str,
    symbol: str,
    exit_date: str = "2015-05-15",
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
    """Build one 21-column row matching ``_payouts.csv``."""
    return {
        "permno": permno,
        "symbol": symbol,
        "window": "dev",
        "universe": "A",
        "exit_date": exit_date,
        "last_close": "50.0",
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


def write_payouts(path: Path, rows: Sequence[dict[str, str]]) -> Path:
    """Write a ``_payouts.csv`` file with the 21 Phase 2 columns."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(payout_table.TABLE_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path


def write_roster(path: Path, rows: Sequence[dict[str, str]]) -> Path:
    """Write a minimal ``_roster.csv`` matching the given payout rows."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["symbol", "permno", "stock_name", "name_source", "exit_date"],
        lineterminator="\n",
    )
    writer.writeheader()
    for r in rows:
        name = r["stock_name"]
        writer.writerow(
            {
                "symbol": r["symbol"],
                "permno": r["permno"],
                "stock_name": name,
                "name_source": "mcgill" if name else "none",
                "exit_date": r["exit_date"],
            }
        )
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path


def write_form25(path: Path, stocks: Sequence[dict[str, Any]]) -> Path:
    """Write a minimal ``_form25_matches.json`` payload."""
    path.write_text(json.dumps({"phase": "form25", "stocks": list(stocks)}, indent=2) + "\n")
    return path


class FakeClient:
    """Fake SEC client with canned submissions bodies, cache-only archives, and request counters."""

    def __init__(
        self,
        bodies: dict[str, Any],
        *,
        archive_bodies: dict[str, bytes] | None = None,
    ) -> None:
        self.bodies = dict(bodies)
        self.archive_bodies = dict(archive_bodies or {})
        self._cached_bodies: dict[str, bytes] = {}
        self.requested: list[str] = []
        self.network_fetches = 0
        self.cache_hits = 0
        self.corrupt_cache_refetches = 0

    def _read_verified_cache(self, url: str) -> bytes | None:
        if url in self._cached_bodies:
            return self._cached_bodies[url]
        return self.archive_bodies.get(url)

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        if url in self._cached_bodies:
            self.cache_hits += 1
            return self._cached_bodies[url]
        self.network_fetches += 1
        body = self.bodies.get(url)
        if body is None:
            raise SecNotFound(url)
        if isinstance(body, Exception):
            raise body
        assert isinstance(body, bytes)
        self._cached_bodies[url] = body
        return body

    def manifest(self) -> dict[str, str]:
        all_urls = sorted(set(self.bodies) | set(self.archive_bodies))
        return {url: "0" * 64 for url in all_urls}


class FakeOpenerResponse:
    """Minimal 200 response object for driving a real ``SecClient`` without network access."""

    status = 200

    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.headers: dict[str, str] = {}
        self._buf = io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)

    def geturl(self) -> str:
        return self.url

    def close(self) -> None:
        self._buf.close()


def _setup_study_dir(
    tmp_path: Path,
    rows: Sequence[dict[str, str]],
    form25_stocks: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    """Create temp ``out`` and ``cache`` directories populated with Phase 1/2 inputs."""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    payouts = write_payouts(out / "_payouts.csv", rows)
    (out / "_payouts.json").write_text('{"phase": "payouts"}\n', encoding="utf-8")
    form25 = write_form25(out / "_form25_matches.json", form25_stocks)
    roster = write_roster(out / "_roster.csv", rows)
    return {
        "out": out,
        "cache": cache,
        "payouts": payouts,
        "form25": form25,
        "roster": roster,
    }


def _cli_args(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--cache-dir",
        str(paths["cache"]),
        "--out",
        str(paths["out"]),
        "--payouts",
        str(paths["payouts"]),
        "--form25",
        str(paths["form25"]),
        "--roster",
        str(paths["roster"]),
        *extra,
    ]


def _seed_quarters(
    cache_dir: Path,
    quarters: Sequence[tuple[int, int]],
    by_quarter: dict[tuple[int, int], Sequence[tuple[str, str, str, str, str]]],
) -> dict[str, bytes]:
    """Seed cached master.zip files for ``quarters`` and return ``{url: bytes}``."""
    archives: dict[str, bytes] = {}
    for year, quarter in quarters:
        url = sec_filings.master_index_url(year, quarter)
        body = make_master_zip(by_quarter.get((year, quarter), ()))
        seed_cache(cache_dir, url, body)
        archives[url] = body
    return archives


@pytest.fixture(autouse=True)
def _default_env_and_clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a test SEC_USER_AGENT and a clean git provenance stub by default."""
    monkeypatch.setenv("SEC_USER_AGENT", DEFAULT_UA)
    monkeypatch.setattr(open_cases_run, "git_provenance", lambda root, ignore=(): dict(CLEAN_GIT))


def test_driver_refuses_missing_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    # Seed all needed quarters except the second one (2015Q3).
    missing_q = quarters[1]
    archives = _seed_quarters(
        paths["cache"],
        [q for q in quarters if q != missing_q],
        {},
    )
    client = FakeClient({}, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    err = capsys.readouterr().err
    assert rc == 2
    assert f"{missing_q[0]}Q{missing_q[1]}" in err
    assert DEFAULT_UA not in err
    assert not (paths["out"] / "_payouts_v2.csv").exists()
    assert not (paths["out"] / "_payouts_v2.json").exists()
    assert (paths["cache"] / "SHA256SUMS.txt").is_file()


def test_driver_refuses_unset_user_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    paths = _setup_study_dir(tmp_path, rows, [])
    called: list[str] = []

    def factory(ua: str, cache: Path) -> FakeClient:
        called.append(ua)
        return FakeClient({})

    rc = open_cases_run.main(_cli_args(paths), client_factory=factory)

    err = capsys.readouterr().err
    assert rc == 2
    assert called == []
    assert "SEC_USER_AGENT" in err
    assert "contact@example.test" not in err


def test_driver_refuses_existing_output_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(paths["cache"], quarters, {})

    csv_path = paths["out"] / "_payouts_v2.csv"
    csv_path.write_text("existing\n", encoding="utf-8")
    made: list[str] = []

    def factory(ua: str, cache: Path) -> FakeClient:
        made.append(ua)
        return FakeClient({}, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=factory)
    assert rc == 2
    assert made == []
    assert "already exists" in capsys.readouterr().err
    assert csv_path.read_text(encoding="utf-8") == "existing\n"

    # Also refuse when only _payouts_v2.json exists without --force.
    csv_path.unlink()
    json_path = paths["out"] / "_payouts_v2.json"
    json_path.write_text("{}\n", encoding="utf-8")
    rc_json = open_cases_run.main(_cli_args(paths), client_factory=factory)
    assert rc_json == 2
    assert made == []
    assert json_path.read_text(encoding="utf-8") == "{}\n"

    # With --force, the run succeeds and replaces outputs.
    rc_force = open_cases_run.main(_cli_args(paths, "--force"), client_factory=factory)
    assert rc_force == 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["phase"] == "open_cases"


def test_driver_refuses_dirty_tree_unless_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(paths["cache"], quarters, {})

    monkeypatch.setattr(
        open_cases_run,
        "git_provenance",
        lambda root, ignore=(): {"git_head": "b" * 40, "git_dirty": True},
    )
    made: list[str] = []

    def factory(ua: str, cache: Path) -> FakeClient:
        made.append(ua)
        return FakeClient({}, archive_bodies=archives)

    rc_dirty = open_cases_run.main(_cli_args(paths), client_factory=factory)
    assert rc_dirty == 2
    assert made == []
    assert "not clean" in capsys.readouterr().err

    # git_dirty == "unknown" is also not False and must be refused without --allow-dirty.
    monkeypatch.setattr(
        open_cases_run,
        "git_provenance",
        lambda root, ignore=(): {"git_head": "unknown", "git_dirty": "unknown"},
    )
    rc_unknown = open_cases_run.main(_cli_args(paths), client_factory=factory)
    assert rc_unknown == 2
    assert made == []

    # With --allow-dirty, the run proceeds and records allow_dirty=True in provenance.
    monkeypatch.setattr(
        open_cases_run,
        "git_provenance",
        lambda root, ignore=(): {"git_head": "b" * 40, "git_dirty": True},
    )
    rc_allowed = open_cases_run.main(_cli_args(paths, "--allow-dirty"), client_factory=factory)
    assert rc_allowed == 0
    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    assert payload["provenance"]["allow_dirty"] is True
    assert payload["provenance"]["git_dirty"] is True
    assert payload["provenance"]["git_head"] == "b" * 40


def test_driver_writes_all_rows_and_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        payout_row(
            permno="10001",
            symbol="PERMNO-10001",
            exit_date="2015-05-15",
            stock_name="DONE DEAL CORP",
            evidence_class="acquired_or_merged",
            match_confidence="confirmed",
            match_rule_version="v1",
            form25_accession="0000876661-15-000001",
            form25_provision="17 CFR 240.12d2-2(a)(3)",
            form25_security="Common Stock",
            event_accessions="0000000101-15-000001",
            events_status="ok",
            suggested_loss="0.0",
            basis="filing_evidence",
            loss_note=(
                "filings report an acquisition or merger event; payment amount and form not "
                "verified"
            ),
            cross_check="agrees",
        ),
        payout_row(
            permno="10002",
            symbol="PERMNO-10002",
            exit_date="2015-05-15",
            stock_name="ACME CORP",
            evidence_class="no_filing_match",
        ),
        payout_row(
            permno="10003",
            symbol="PERMNO-10003",
            exit_date="2015-05-15",
            stock_name="WIDGET TECH",
            evidence_class="no_filing_match",
        ),
        payout_row(
            permno="10004",
            symbol="PERMNO-10004",
            exit_date="2015-05-15",
            stock_name="ALIVE CORP",
            evidence_class="no_filing_match",
        ),
        payout_row(
            permno="10005",
            symbol="PERMNO-10005",
            exit_date="2024-12-01",
            stock_name="LATE EXIT CORP",
            evidence_class="no_filing_match",
        ),
        payout_row(
            permno="10006",
            symbol="PERMNO-10006",
            exit_date="2015-05-15",
            stock_name="",
            evidence_class="name_unknown",
            loss_note="name_unknown: no company name in our data to match; no loss can be set",
        ),
    ]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(
        paths["cache"],
        quarters,
        {
            (2015, 2): [
                ("2001", "ACME CORP", "8-K", "2015-05-01", "0000002001-15-000001"),
                (
                    "2002",
                    "WIDGET TECHNOLOGIES INC",
                    "DEFM14A",
                    "2015-04-20",
                    "0000002002-15-000001",
                ),
                ("2003", "ALIVE CORP", "8-K", "2015-05-01", "0000002003-15-000001"),
            ],
            (2015, 4): [
                ("2003", "ALIVE CORP", "10-K", "2015-11-20", "0000002003-15-000002"),
            ],
            (2024, 4): [
                ("2004", "LATE EXIT CORP", "8-K", "2024-11-20", "0000002004-24-000001"),
            ],
        },
    )
    bodies = {
        sec_filings.submissions_url("2001"): submissions_json(
            [("8-K", "2015-05-10", "2.01", "0000002001-15-000002", "d.htm")]
        ),
        sec_filings.submissions_url("2002"): submissions_json(
            [("8-K", "2015-05-10", "8.01", "0000002002-15-000002", "d.htm")]
        ),
        sec_filings.submissions_url("2004"): submissions_json(
            [("8-K", "2024-11-25", "5.01", "0000002004-24-000002", "d.htm")]
        ),
    }
    client = FakeClient(bodies, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    assert rc == 0
    out_text = capsys.readouterr().out
    assert "v1.1-only matches 1" in out_text
    assert "deal_forms_only 1" in out_text
    assert "rejected still filing 1" in out_text
    assert "incomplete still-filing checks 1" in out_text

    csv_bytes = (paths["out"] / "_payouts_v2.csv").read_bytes()
    assert b"\r\n" not in csv_bytes
    parsed_csv = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"), newline="")))
    assert list(parsed_csv[0].keys()) == list(open_cases.OUTPUT_COLUMNS)
    assert len(parsed_csv) == 6
    assert [r["symbol"] for r in parsed_csv] == [
        "PERMNO-10001",
        "PERMNO-10002",
        "PERMNO-10003",
        "PERMNO-10004",
        "PERMNO-10005",
        "PERMNO-10006",
    ]

    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    assert payload["phase"] == "open_cases"
    assert len(payload["rows"]) == 6
    assert payload["rows"] == parsed_csv
    assert payload["unnamed_rows"] == ["PERMNO-10006"]
    assert payload["events_status_counts"] == {"ok": 3}
    assert payload["events_fetch_errors"] == []
    assert payload["request_count"]["archive_downloads"] == 0
    assert payload["request_count"]["network_fetches"] == 3
    assert payload["manifest"]

    cov = payload["coverage"]
    assert cov["roster_rows"] == 6
    assert cov["after_by_class"]["acquired_or_merged"] == 3
    assert cov["after_by_class"]["deal_forms_only"] == 1
    assert cov["after_by_class"]["no_filing_match"] == 1
    assert cov["after_by_class"]["name_unknown"] == 1
    assert cov["deal_forms_only"] == 1
    assert cov["still_filing_incomplete"] == 1
    assert len(cov["v1_1_only_matches"]) == 1
    assert cov["v1_1_only_matches"][0]["cik"] == 2002
    assert cov["v1_1_only_matches"][0]["filer_name"] == "WIDGET TECHNOLOGIES INC"
    assert len(cov["rejected_still_filing"]) == 1
    assert cov["rejected_still_filing"][0]["reason"] == "company_still_filing: 2003"
    assert cov["best_case_unchanged"] is True

    prov = payload["provenance"]
    assert prov["git_head"] == "a" * 40
    assert prov["git_dirty"] is False
    assert prov["allow_dirty"] is False
    assert prov["payouts_sha256"] == sec_pilot.sha256_file(paths["payouts"])
    assert prov["roster_sha256"] == sec_pilot.sha256_file(paths["roster"])
    assert prov["form25_sha256"] == sec_pilot.sha256_file(paths["form25"])
    assert prov["brief_sha256"] == sec_pilot.sha256_file(
        open_cases_run.ROOT / open_cases_run.BRIEF_REL
    )
    expected_code_files = set(sec_pilot.CODE_FILES) | {
        "research/stock-signals/dead-stock-data-integrity/code/sec_name_index.py",
        "research/stock-signals/dead-stock-data-integrity/code/open_cases.py",
        "research/stock-signals/dead-stock-data-integrity/code/open_cases_run.py",
    }
    assert set(prov["code_sha256"]) == expected_code_files
    for rel_path, digest in prov["code_sha256"].items():
        assert digest == sec_pilot.sha256_file(open_cases_run.ROOT / rel_path)
    assert prov["quarter_window"]["needed"] == len(quarters)
    assert prov["quarter_window"]["first"] == list(quarters[0])
    assert prov["quarter_window"]["last"] == list(quarters[-1])
    assert prov["quarter_window"]["start"] <= prov["quarter_window"]["end"]
    rc_block = prov["rule_constants"]
    assert rc_block["deal_forms"] == [
        "DEFM14A",
        "SC 13E3",
        "SC 13E3/A",
        "SC 14D9",
        "SC 14D9/A",
        "SC TO-T",
        "SC TO-T/A",
    ]
    assert rc_block["periodic_forms"] == ["10-K", "10-K/A", "10-Q", "10-Q/A"]
    assert rc_block["deal_days_before"] == 365
    assert rc_block["deal_days_after"] == 30
    assert rc_block["candidate_days_before"] == 365
    assert rc_block["candidate_days_after"] == 180
    assert rc_block["still_filing_first_day"] == 181
    assert rc_block["index_last_day"] == "2025-03-31"
    assert (paths["cache"] / "SHA256SUMS.txt").is_file()


def test_driver_never_writes_phase1_files(tmp_path: Path) -> None:
    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(paths["cache"], quarters, {})

    protected = [
        paths["out"] / "_payouts.csv",
        paths["out"] / "_payouts.json",
        paths["out"] / "_form25_matches.json",
        paths["out"] / "_roster.csv",
    ]
    before_hashes = {p.name: sec_pilot.sha256_file(p) for p in protected}

    rc = open_cases_run.main(
        _cli_args(paths),
        client_factory=lambda ua, cache: FakeClient({}, archive_bodies=archives),
    )

    assert rc == 0
    after_hashes = {p.name: sec_pilot.sha256_file(p) for p in protected}
    assert after_hashes == before_hashes


def test_evidence_rows_pass_through_unchanged(tmp_path: Path) -> None:
    rows = [
        payout_row(
            permno="10001",
            symbol="PERMNO-10001",
            exit_date="2015-05-15",
            stock_name="ACQ CORP",
            evidence_class="acquired_or_merged",
            match_confidence="confirmed",
            match_rule_version="v1",
            form25_accession="0000876661-15-000001",
            form25_provision="17 CFR 240.12d2-2(a)(3)",
            form25_security="Common Stock",
            event_accessions="0000000101-15-000001",
            events_status="ok",
            suggested_loss="0.0",
            basis="filing_evidence",
            loss_note=(
                "filings report an acquisition or merger event; payment amount and form not "
                "verified"
            ),
            cross_check="agrees",
        ),
        payout_row(
            permno="10002",
            symbol="PERMNO-10002",
            exit_date="2015-05-20",
            stock_name="BANKRUPT CORP",
            exchcd="3",
            price_label="collapse_like",
            evidence_class="bankruptcy_filed",
            match_confidence="name_only",
            match_rule_version="v1.1",
            form25_accession="0001354457-15-000002",
            event_accessions="0000000102-15-000001",
            events_status="ok",
            suggested_loss="0.55",
            basis="literature_fill",
            loss_note="bankruptcy literature fill",
            cross_check="agrees",
        ),
        payout_row(
            permno="10003",
            symbol="PERMNO-10003",
            exit_date="2015-05-25",
            stock_name="DELIST CORP",
            evidence_class="delisted_other",
            match_confidence="confirmed",
            match_rule_version="v1",
            form25_accession="0000876661-15-000003",
            event_accessions="0000000103-15-000001",
            events_status="ok",
            suggested_loss="",
            basis="none",
            loss_note=(
                "delisted_other: filings report a delisting notice only; no loss can be set from "
                "them"
            ),
            cross_check="not_comparable",
        ),
    ]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(paths["cache"], quarters, {})
    client = FakeClient({}, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    assert rc == 0
    assert client.requested == []
    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    for original, after in zip(rows, payload["rows"], strict=True):
        for col in payout_table.TABLE_COLUMNS:
            assert after[col] == original[col]
        assert after["phase"] == "1"
        assert after["name_rule"] == ""
        assert after["reason"] == ""
        assert after["still_filing_check"] == ""
        assert after["deal_form_accessions"] == ""


def test_five_phase2_rows_keep_cik_and_confidence(tmp_path: Path) -> None:
    specs = [
        ("20001", "PERMNO-20001", "ONE CORP", "confirmed", "v1", "0000005001"),
        ("20002", "PERMNO-20002", "TWO CORP", "confirmed", "v1", "0000005002"),
        ("20003", "PERMNO-20003", "THREE CORP", "name_only", "v1", "0000005003"),
        ("20004", "PERMNO-20004", "FOUR CORP", "name_only", "v1.1", "0000005004"),
        ("20005", "PERMNO-20005", "FIVE CORP", "name_only", "v1", "0000005005"),
    ]
    rows = [
        payout_row(
            permno=permno,
            symbol=symbol,
            exit_date="2015-05-15",
            stock_name=name,
            evidence_class="no_8k_found",
            match_confidence=conf,
            match_rule_version=rv,
            form25_accession=f"0000876661-15-00000{idx}",
            events_status="ok",
            loss_note=(
                "no_8k_found: no acquisition, bankruptcy or delisting item found near the exit; "
                "no loss set"
            ),
        )
        for idx, (permno, symbol, name, conf, rv, _cik) in enumerate(specs, start=1)
    ]
    form25_stocks = [
        {"permno": permno, "symbol": symbol, "status": "matched", "ciks": [cik]}
        for permno, symbol, _name, _conf, _rv, cik in specs
    ]
    paths = _setup_study_dir(tmp_path, rows, form25_stocks)
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    # CIKs 5001 and 5004 have deal forms in window; 5002, 5003, 5005 do not.
    archives = _seed_quarters(
        paths["cache"],
        quarters,
        {
            (2015, 2): [
                ("5001", "ONE CORP", "DEFM14A", "2015-04-20", "0000005001-15-000001"),
                ("5004", "FOUR CORP", "SC 14D9", "2015-05-01", "0000005004-15-000001"),
                ("5005", "FIVE CORP", "10-Q", "2015-05-01", "0000005005-15-000001"),
            ],
        },
    )
    client = FakeClient({}, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    assert rc == 0
    # Phase 2 no_8k_found rows never refetch submissions / 8-K events.
    assert client.requested == []

    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    by_sym = {r["symbol"]: r for r in payload["rows"]}
    assert by_sym["PERMNO-20001"]["evidence_class"] == "deal_forms_only"
    assert by_sym["PERMNO-20001"]["match_confidence"] == "confirmed"
    assert by_sym["PERMNO-20001"]["match_rule_version"] == "v1"
    assert by_sym["PERMNO-20001"]["suggested_loss"] == ""
    assert by_sym["PERMNO-20001"]["basis"] == "none"
    assert by_sym["PERMNO-20001"]["deal_form_accessions"] == "0000005001-15-000001"
    assert by_sym["PERMNO-20001"]["phase"] == "3"

    assert by_sym["PERMNO-20002"]["evidence_class"] == "no_8k_found"
    assert by_sym["PERMNO-20002"]["match_confidence"] == "confirmed"
    assert by_sym["PERMNO-20002"]["match_rule_version"] == "v1"
    assert by_sym["PERMNO-20002"]["reason"] == "no_deal_forms"

    assert by_sym["PERMNO-20003"]["evidence_class"] == "no_8k_found"
    assert by_sym["PERMNO-20003"]["match_confidence"] == "name_only"
    assert by_sym["PERMNO-20003"]["reason"] == "no_deal_forms"

    assert by_sym["PERMNO-20004"]["evidence_class"] == "deal_forms_only"
    assert by_sym["PERMNO-20004"]["match_confidence"] == "name_only"
    assert by_sym["PERMNO-20004"]["match_rule_version"] == "v1.1"
    assert by_sym["PERMNO-20004"]["deal_form_accessions"] == "0000005004-15-000001"

    assert by_sym["PERMNO-20005"]["evidence_class"] == "no_8k_found"
    assert by_sym["PERMNO-20005"]["match_confidence"] == "name_only"
    assert by_sym["PERMNO-20005"]["reason"] == "no_deal_forms"
    assert payload["coverage"]["deal_forms_only"] == 2


def test_archive_downloads_reported_zero_and_growth_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(
        paths["cache"],
        quarters,
        {(2015, 2): [("3001", "ACME CORP", "8-K", "2015-05-01", "0000003001-15-000001")]},
    )
    bodies = {
        sec_filings.submissions_url("3001"): submissions_json(
            [("8-K", "2015-05-10", "2.01", "0000003001-15-000002", "d.htm")]
        )
    }

    # Clean cache-only archives report archive_downloads == 0 even when submissions fetch runs.
    good_client = FakeClient(bodies, archive_bodies=archives)
    rc_ok = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: good_client)
    assert rc_ok == 0
    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    assert payload["request_count"]["archive_downloads"] == 0
    assert payload["request_count"]["network_fetches"] == 1

    # A misbehaving archives_factory that increments client.network_fetches is refused with exit 2.
    bad_out = tmp_path / "out_bad"
    bad_out.mkdir()
    bad_paths = {**paths, "out": bad_out}

    def downloading_archives(
        client: Any, req_quarters: Sequence[tuple[int, int]]
    ) -> Iterable[tuple[str, bytes]]:
        client.network_fetches += 1
        for year, quarter in req_quarters:
            yield f"{year}Q{quarter}", make_master_zip(())

    rc_bad = open_cases_run.main(
        _cli_args(bad_paths),
        client_factory=lambda ua, cache: FakeClient(bodies, archive_bodies=archives),
        archives_factory=downloading_archives,
    )
    assert rc_bad == 2
    assert "network fetch" in capsys.readouterr().err
    assert not (bad_out / "_payouts_v2.csv").exists()
    assert not (bad_out / "_payouts_v2.json").exists()


def test_events_fetch_error_is_recorded_and_run_continues(tmp_path: Path) -> None:
    rows = [
        payout_row(
            permno="10001",
            symbol="PERMNO-10001",
            exit_date="2015-05-15",
            stock_name="FAIL FETCH CORP",
            evidence_class="no_filing_match",
        ),
        payout_row(
            permno="10002",
            symbol="PERMNO-10002",
            exit_date="2015-05-15",
            stock_name="GOOD FETCH CORP",
            evidence_class="no_filing_match",
        ),
    ]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(
        paths["cache"],
        quarters,
        {
            (2015, 2): [
                ("4001", "FAIL FETCH CORP", "DEFM14A", "2015-05-01", "0000004001-15-000001"),
                ("4002", "GOOD FETCH CORP", "8-K", "2015-05-01", "0000004002-15-000001"),
            ],
        },
    )
    bodies = {
        sec_filings.submissions_url("4001"): SecFetchError("HTTP 500 server error"),
        sec_filings.submissions_url("4002"): submissions_json(
            [("8-K", "2015-05-10", "2.01", "0000004002-15-000002", "d.htm")]
        ),
    }
    client = FakeClient(bodies, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    assert rc == 0
    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    by_sym = {r["symbol"]: r for r in payload["rows"]}
    # Failed 8-K fetch keeps no_8k_found with events_incomplete, still listing deal accessions.
    assert by_sym["PERMNO-10001"]["evidence_class"] == "no_8k_found"
    assert by_sym["PERMNO-10001"]["events_status"] == "fetch_error"
    assert by_sym["PERMNO-10001"]["reason"] == "events_incomplete"
    assert by_sym["PERMNO-10001"]["deal_form_accessions"] == "0000004001-15-000001"
    # Second stock still completes normally.
    assert by_sym["PERMNO-10002"]["evidence_class"] == "acquired_or_merged"
    assert by_sym["PERMNO-10002"]["events_status"] == "ok"
    assert by_sym["PERMNO-10002"]["suggested_loss"] == "0.0"

    assert payload["events_status_counts"] == {"fetch_error": 1, "ok": 1}
    assert len(payload["events_fetch_errors"]) == 1
    err_entry = payload["events_fetch_errors"][0]
    assert err_entry == {
        "symbol": "PERMNO-10001",
        "permno": 10001,
        "cik": 4001,
        "category": "fetch_error",
        "error_type": "SecFetchError",
    }


def test_unnamed_rows_pass_through(tmp_path: Path) -> None:
    rows = [
        payout_row(
            permno="87283",
            symbol="PERMNO-87283",
            exit_date="2015-05-15",
            stock_name="",
            exchcd="",
            evidence_class="name_unknown",
            loss_note="name_unknown: no company name in our data to match; no loss can be set",
        ),
        payout_row(
            permno="75900",
            symbol="PERMNO-75900",
            exit_date="2015-05-20",
            stock_name="",
            exchcd="",
            evidence_class="name_unknown",
            loss_note="name_unknown: no company name in our data to match; no loss can be set",
        ),
    ]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(paths["cache"], quarters, {})

    rc = open_cases_run.main(
        _cli_args(paths),
        client_factory=lambda ua, cache: FakeClient({}, archive_bodies=archives),
    )

    assert rc == 0
    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    assert payload["unnamed_rows"] == ["PERMNO-87283", "PERMNO-75900"]
    for original, after in zip(rows, payload["rows"], strict=True):
        for col in payout_table.TABLE_COLUMNS:
            assert after[col] == original[col]
        assert after["phase"] == "3"
        assert after["name_rule"] == ""
        assert after["reason"] == "no_name_after_probe"
        assert after["still_filing_check"] == ""
        assert after["deal_form_accessions"] == ""


def test_contact_line_never_in_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_ua = "ggTrader Phase3 SecretToken-9f4c21 contact@example.test"
    monkeypatch.setenv("SEC_USER_AGENT", secret_ua)

    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    _seed_quarters(
        paths["cache"],
        quarters,
        {(2015, 2): [("6001", "ACME CORP", "8-K", "2015-05-01", "0000006001-15-000001")]},
    )
    sub_url = sec_filings.submissions_url("6001")
    sub_body = submissions_json([("8-K", "2015-05-10", "2.01", "0000006001-15-000002", "d.htm")])
    seen_headers: list[str | None] = []

    def fake_opener(request: Any) -> FakeOpenerResponse:
        assert request.full_url == sub_url
        seen_headers.append(request.get_header("User-agent"))
        return FakeOpenerResponse(request.full_url, sub_body)

    def factory(ua: str, cache: Path) -> SecClient:
        return SecClient(ua, cache, min_interval=0, opener=fake_opener)

    rc = open_cases_run.main(_cli_args(paths), client_factory=factory)

    captured = capsys.readouterr()
    assert rc == 0
    assert seen_headers == [secret_ua]
    assert "SecretToken-9f4c21" not in captured.out
    assert "SecretToken-9f4c21" not in captured.err
    assert "contact@example.test" not in captured.out
    assert "contact@example.test" not in captured.err

    for output_file in (
        paths["out"] / "_payouts_v2.csv",
        paths["out"] / "_payouts_v2.json",
        paths["cache"] / "SHA256SUMS.txt",
    ):
        text = output_file.read_text(encoding="utf-8")
        assert "SecretToken-9f4c21" not in text
        assert "contact@example.test" not in text
        assert secret_ua not in text


def test_driver_refuses_symlinks_bad_class_duplicate_and_missing_form25(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]

    # 1. Symlinked output file is refused even with --force.
    paths_sym = _setup_study_dir(tmp_path / "sym", rows, [])
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    (paths_sym["out"] / "_payouts_v2.csv").symlink_to(victim)
    assert open_cases_run.main(_cli_args(paths_sym, "--force")) == 2
    assert "symbolic link" in capsys.readouterr().err
    assert victim.read_text(encoding="utf-8") == "keep"

    # 2. Unknown evidence_class is refused.
    bad_class_rows = [
        payout_row(
            permno="10001",
            symbol="PERMNO-10001",
            exit_date="2015-05-15",
            evidence_class="not_a_real_class",
        )
    ]
    paths_cls = _setup_study_dir(tmp_path / "cls", bad_class_rows, [])
    assert open_cases_run.main(_cli_args(paths_cls)) == 2
    assert "unknown evidence_class" in capsys.readouterr().err

    # 3. Duplicate symbol is refused.
    dup_rows = [
        payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15"),
        payout_row(permno="10002", symbol="PERMNO-10001", exit_date="2015-05-15"),
    ]
    paths_dup = _setup_study_dir(tmp_path / "dup", dup_rows, [])
    assert open_cases_run.main(_cli_args(paths_dup)) == 2
    assert "duplicate symbol" in capsys.readouterr().err

    # 4. Missing Form 25 stock for a no_8k_found row is refused.
    no_8k_rows = [
        payout_row(
            permno="10001",
            symbol="PERMNO-10001",
            exit_date="2015-05-15",
            evidence_class="no_8k_found",
            events_status="ok",
        )
    ]
    paths_f25 = _setup_study_dir(tmp_path / "f25", no_8k_rows, [])
    assert open_cases_run.main(_cli_args(paths_f25)) == 2
    assert "Form 25 file lacks a stock entry" in capsys.readouterr().err


def test_malformed_submissions_never_count_as_complete_search(tmp_path: Path) -> None:
    malformed_cases = [
        ("70001", "PERMNO-70001", "EMPTY OBJ CORP", "7001", b"{}"),
        ("70002", "PERMNO-70002", "LIST FILINGS CORP", "7002", b'{"filings": []}'),
        ("70003", "PERMNO-70003", "EMPTY FILINGS CORP", "7003", b'{"filings": {}}'),
        (
            "70004",
            "PERMNO-70004",
            "ERROR OBJ CORP",
            "7004",
            b'{"error": "rate limit exceeded", "status": 429}',
        ),
        ("70005", "PERMNO-70005", "NON JSON CORP", "7005", b"not json at all"),
    ]
    rows = [
        payout_row(
            permno=permno,
            symbol=symbol,
            exit_date="2015-05-15",
            stock_name=name,
            evidence_class="no_filing_match",
        )
        for permno, symbol, name, _cik, _raw in malformed_cases
    ]
    # Add two well-formed controls: one deal_forms_only and one 8-K acquired_or_merged.
    rows.extend(
        [
            payout_row(
                permno="70006",
                symbol="PERMNO-70006",
                exit_date="2015-05-15",
                stock_name="GOOD DEAL FORM CORP",
                evidence_class="no_filing_match",
            ),
            payout_row(
                permno="70007",
                symbol="PERMNO-70007",
                exit_date="2015-05-15",
                stock_name="GOOD EIGHT K CORP",
                evidence_class="no_filing_match",
            ),
        ]
    )

    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    # Give EVERY stock a cached DEFM14A deal form in window so only complete 8-K searches promote.
    q2_rows = [
        (cik, name, "DEFM14A", "2015-05-01", f"000000{cik}-15-000001")
        for _permno, _sym, name, cik, _raw in malformed_cases
    ]
    q2_rows.extend(
        [
            ("7006", "GOOD DEAL FORM CORP", "DEFM14A", "2015-05-01", "0000007006-15-000001"),
            ("7007", "GOOD EIGHT K CORP", "DEFM14A", "2015-05-01", "0000007007-15-000001"),
        ]
    )
    archives = _seed_quarters(paths["cache"], quarters, {(2015, 2): q2_rows})
    bodies: dict[str, Any] = {
        sec_filings.submissions_url(cik): raw for _p, _s, _n, cik, raw in malformed_cases
    }
    bodies[sec_filings.submissions_url("7006")] = submissions_json(
        [("8-K", "2015-05-10", "8.01", "0000007006-15-000002", "d.htm")]
    )
    bodies[sec_filings.submissions_url("7007")] = submissions_json(
        [("8-K", "2015-05-10", "2.01", "0000007007-15-000002", "d.htm")]
    )
    client = FakeClient(bodies, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    assert rc == 0
    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    by_sym = {r["symbol"]: r for r in payload["rows"]}
    errors_by_sym = {entry["symbol"]: entry for entry in payload["events_fetch_errors"]}

    for permno, symbol, _name, cik, _raw in malformed_cases:
        row_out = by_sym[symbol]
        assert row_out["evidence_class"] == "no_8k_found"
        assert row_out["reason"] == "events_incomplete"
        assert row_out["events_status"] == "malformed_submissions"
        assert row_out["deal_form_accessions"] == f"000000{cik}-15-000001"
        err = errors_by_sym[symbol]
        assert err["permno"] == int(permno)
        assert err["cik"] == int(cik)
        assert err["category"] == "malformed_submissions"
        assert err["error_type"] in {"ValueError", "JSONDecodeError"}
        assert set(err) == {"symbol", "permno", "cik", "category", "error_type"}

    # Well-formed responses still classify normally.
    assert by_sym["PERMNO-70006"]["evidence_class"] == "deal_forms_only"
    assert by_sym["PERMNO-70006"]["events_status"] == "ok"
    assert by_sym["PERMNO-70006"]["reason"] == ""
    assert by_sym["PERMNO-70007"]["evidence_class"] == "acquired_or_merged"
    assert by_sym["PERMNO-70007"]["events_status"] == "ok"
    assert by_sym["PERMNO-70007"]["suggested_loss"] == "0.0"
    assert payload["events_status_counts"] == {"malformed_submissions": 5, "ok": 2}
    assert payload["coverage"]["deal_forms_only"] == 1


def test_contact_line_never_leaks_on_fetch_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_ua = "ggTrader Phase3 SecretErrToken-7b31 contact@example.test"
    monkeypatch.setenv("SEC_USER_AGENT", secret_ua)

    rows = [
        payout_row(
            permno="80001",
            symbol="PERMNO-80001",
            exit_date="2015-05-15",
            stock_name="LEAKY ERROR CORP",
            evidence_class="no_filing_match",
        )
    ]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    archives = _seed_quarters(
        paths["cache"],
        quarters,
        {
            (2015, 2): [
                ("8001", "LEAKY ERROR CORP", "DEFM14A", "2015-05-01", "0000008001-15-000001")
            ]
        },
    )
    bodies = {
        sec_filings.submissions_url("8001"): SecFetchError(
            f"HTTP 403 forbidden for User-Agent={secret_ua}"
        )
    }
    client = FakeClient(bodies, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    captured = capsys.readouterr()
    assert rc == 0
    for token in (secret_ua, "SecretErrToken-7b31", "contact@example.test"):
        assert token not in captured.out
        assert token not in captured.err

    for output_file in (
        paths["out"] / "_payouts_v2.csv",
        paths["out"] / "_payouts_v2.json",
        paths["cache"] / "SHA256SUMS.txt",
    ):
        text = output_file.read_text(encoding="utf-8")
        for token in (secret_ua, "SecretErrToken-7b31", "contact@example.test"):
            assert token not in text

    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    assert payload["events_fetch_errors"] == [
        {
            "symbol": "PERMNO-80001",
            "permno": 80001,
            "cik": 8001,
            "category": "fetch_error",
            "error_type": "SecFetchError",
        }
    ]


def test_unreadable_or_malformed_inputs_refuse_with_exit_2_and_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-05-15")]
    real_read_bytes = Path.read_bytes

    def _assert_refused(paths: dict[str, Path], rc: int, *expected_tokens: str) -> str:
        err = capsys.readouterr().err
        assert rc == 2
        for token in expected_tokens:
            assert token in err
        assert not (paths["out"] / "_payouts_v2.csv").exists()
        assert not (paths["out"] / "_payouts_v2.json").exists()
        assert (paths["cache"] / "SHA256SUMS.txt").is_file()
        return err

    # 1. Unreadable _payouts.csv (PermissionError on Path.read_bytes).
    paths_p = _setup_study_dir(tmp_path / "unreadable_payouts", base_rows, [])

    def deny_payouts(self: Path) -> bytes:
        if self == paths_p["payouts"]:
            raise PermissionError("raw-secret-payouts-msg")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", deny_payouts)
    err_p = _assert_refused(
        paths_p,
        open_cases_run.main(_cli_args(paths_p)),
        str(paths_p["payouts"]),
        "PermissionError",
    )
    assert "raw-secret-payouts-msg" not in err_p

    # 2. Unreadable _form25_matches.json.
    paths_f = _setup_study_dir(tmp_path / "unreadable_form25", base_rows, [])

    def deny_form25(self: Path) -> bytes:
        if self == paths_f["form25"]:
            raise PermissionError("raw-secret-form25-msg")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", deny_form25)
    err_f = _assert_refused(
        paths_f,
        open_cases_run.main(_cli_args(paths_f)),
        str(paths_f["form25"]),
        "PermissionError",
    )
    assert "raw-secret-form25-msg" not in err_f

    # 3. Unreadable frozen Phase 3 brief.
    paths_b = _setup_study_dir(tmp_path / "unreadable_brief", base_rows, [])
    brief_path = open_cases_run.ROOT / open_cases_run.BRIEF_REL

    def deny_brief(self: Path) -> bytes:
        if self == brief_path:
            raise PermissionError("raw-secret-brief-msg")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", deny_brief)
    err_b = _assert_refused(
        paths_b,
        open_cases_run.main(_cli_args(paths_b)),
        str(brief_path),
        "PermissionError",
    )
    assert "raw-secret-brief-msg" not in err_b
    monkeypatch.setattr(Path, "read_bytes", real_read_bytes)

    # 4. Roster row with an invalid exit date.
    paths_r = _setup_study_dir(tmp_path / "bad_roster_date", base_rows, [])
    bad_roster_rows = [payout_row(permno="10001", symbol="PERMNO-10001", exit_date="2015-99-99")]
    write_roster(paths_r["roster"], bad_roster_rows)
    _assert_refused(
        paths_r,
        open_cases_run.main(_cli_args(paths_r)),
        str(paths_r["roster"]),
        "ValueError",
    )

    # 5. _payouts.csv containing non-UTF-8 bytes.
    paths_u = _setup_study_dir(tmp_path / "non_utf8_payouts", base_rows, [])
    paths_u["payouts"].write_bytes(b"\xff\xfe\x00bad-utf8")
    _assert_refused(
        paths_u,
        open_cases_run.main(_cli_args(paths_u)),
        str(paths_u["payouts"]),
        "UnicodeDecodeError",
    )


def test_malformed_main_submissions_beats_older_page_errors_and_preserves_404(
    tmp_path: Path,
) -> None:
    specs = [
        ("71001", "PERMNO-71001", "MISSING RECENT 404 CORP", "7101"),
        ("71002", "PERMNO-71002", "MISSING RECENT ERR CORP", "7102"),
        ("71003", "PERMNO-71003", "MAIN 404 CORP", "7103"),
        ("71004", "PERMNO-71004", "VALID OLDER 404 CORP", "7104"),
        ("71005", "PERMNO-71005", "VALID OLDER ERR CORP", "7105"),
    ]
    rows = [
        payout_row(
            permno=permno,
            symbol=symbol,
            exit_date="2015-05-15",
            stock_name=name,
            evidence_class="no_filing_match",
        )
        for permno, symbol, name, _cik in specs
    ]
    paths = _setup_study_dir(tmp_path, rows, [])
    quarters = sec_pilot.needed_quarters(sec_pilot.read_roster(paths["roster"]))
    q2_rows = [
        (cik, name, "DEFM14A", "2015-05-01", f"000000{cik}-15-000001")
        for _permno, _sym, name, cik in specs
    ]
    archives = _seed_quarters(paths["cache"], quarters, {(2015, 2): q2_rows})

    def _page_descriptor(name: str) -> dict[str, str]:
        return {"name": name, "filingFrom": "2010-01-01", "filingTo": "2016-12-31"}

    def _valid_main_needing_older(page_name: str, cik: str) -> bytes:
        # Earliest recent filing sits inside the 90-day window so fetch_events_with_status must
        # consult the listed older page.
        doc = {
            "filings": {
                "recent": {
                    "form": ["10-Q"],
                    "filingDate": ["2015-05-01"],
                    "items": [""],
                    "accessionNumber": [f"000000{cik}-15-000002"],
                    "primaryDocument": ["q.htm"],
                },
                "files": [_page_descriptor(page_name)],
            }
        }
        return json.dumps(doc).encode("utf-8")

    page_7101 = "CIK0000007101-submissions-001.json"
    page_7102 = "CIK0000007102-submissions-001.json"
    page_7104 = "CIK0000007104-submissions-001.json"
    page_7105 = "CIK0000007105-submissions-001.json"
    page_url_7101 = sec_filings.SUBMISSIONS_PAGE_URL.format(name=page_7101)
    page_url_7102 = sec_filings.SUBMISSIONS_PAGE_URL.format(name=page_7102)
    page_url_7104 = sec_filings.SUBMISSIONS_PAGE_URL.format(name=page_7104)
    page_url_7105 = sec_filings.SUBMISSIONS_PAGE_URL.format(name=page_7105)

    bodies: dict[str, Any] = {
        # Missing `recent` key, valid older page descriptor returning 404.
        sec_filings.submissions_url("7101"): json.dumps(
            {"filings": {"files": [_page_descriptor(page_7101)]}}
        ).encode("utf-8"),
        page_url_7101: SecNotFound(page_url_7101),
        # Missing `recent` key, valid older page descriptor raising SecFetchError.
        sec_filings.submissions_url("7102"): json.dumps(
            {"filings": {"files": [_page_descriptor(page_7102)]}}
        ).encode("utf-8"),
        page_url_7102: SecFetchError("HTTP 500 older page failure"),
        # Main response 404 must remain ordinary not_found with no error entry.
        sec_filings.submissions_url("7103"): SecNotFound(sec_filings.submissions_url("7103")),
        # Valid main response with missing older page keeps older_page_not_found.
        sec_filings.submissions_url("7104"): _valid_main_needing_older(page_7104, "7104"),
        page_url_7104: SecNotFound(page_url_7104),
        # Valid main response with failing older page keeps fetch_error.
        sec_filings.submissions_url("7105"): _valid_main_needing_older(page_7105, "7105"),
        page_url_7105: SecFetchError("HTTP 503 older page unavailable"),
    }
    client = FakeClient(bodies, archive_bodies=archives)

    rc = open_cases_run.main(_cli_args(paths), client_factory=lambda ua, cache: client)

    assert rc == 0
    payload = json.loads((paths["out"] / "_payouts_v2.json").read_text(encoding="utf-8"))
    by_sym = {r["symbol"]: r for r in payload["rows"]}
    errors_by_sym = {entry["symbol"]: entry for entry in payload["events_fetch_errors"]}

    for permno, symbol, cik in (
        (71001, "PERMNO-71001", 7101),
        (71002, "PERMNO-71002", 7102),
    ):
        row_out = by_sym[symbol]
        assert row_out["evidence_class"] == "no_8k_found"
        assert row_out["events_status"] == "malformed_submissions"
        assert row_out["reason"] == "events_incomplete"
        assert row_out["deal_form_accessions"] == f"000000{cik}-15-000001"
        assert errors_by_sym[symbol] == {
            "symbol": symbol,
            "permno": permno,
            "cik": cik,
            "category": "malformed_submissions",
            "error_type": "ValueError",
        }

    # Malformed main responses stop before fetching listed older pages.
    assert page_url_7101 not in client.requested
    assert page_url_7102 not in client.requested

    # Main 404 stays not_found without adding an error record.
    assert by_sym["PERMNO-71003"]["evidence_class"] == "no_8k_found"
    assert by_sym["PERMNO-71003"]["events_status"] == "not_found"
    assert by_sym["PERMNO-71003"]["reason"] == "events_incomplete"
    assert "PERMNO-71003" not in errors_by_sym

    # Valid main responses still report older-page outcomes accurately.
    assert page_url_7104 in client.requested
    assert by_sym["PERMNO-71004"]["evidence_class"] == "no_8k_found"
    assert by_sym["PERMNO-71004"]["events_status"] == "older_page_not_found"
    assert by_sym["PERMNO-71004"]["reason"] == "events_incomplete"
    assert "PERMNO-71004" not in errors_by_sym

    assert page_url_7105 in client.requested
    assert by_sym["PERMNO-71005"]["evidence_class"] == "no_8k_found"
    assert by_sym["PERMNO-71005"]["events_status"] == "fetch_error"
    assert by_sym["PERMNO-71005"]["reason"] == "events_incomplete"
    assert errors_by_sym["PERMNO-71005"] == {
        "symbol": "PERMNO-71005",
        "permno": 71005,
        "cik": 7105,
        "category": "fetch_error",
        "error_type": "SecFetchError",
    }

    assert len(payload["events_fetch_errors"]) == 3
    assert payload["events_status_counts"] == {
        "fetch_error": 1,
        "malformed_submissions": 2,
        "not_found": 1,
        "older_page_not_found": 1,
    }
