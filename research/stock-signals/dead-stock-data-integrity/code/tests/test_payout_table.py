"""Tests for the payout table: evidence classes, suggested losses, coverage and the driver phase."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import sec_pilot
from payout_table import (
    EVIDENCE_CLASSES,
    LITERATURE_FILLS,
    TABLE_COLUMNS,
    build_payout_table,
    clean_exchange,
    coverage_summary,
    suggested_loss,
)
from sec_client import SecNotFound
from sec_pilot import main, run_payouts

pytestmark = pytest.mark.study

CIK = "0000000001"


def roster_row(
    permno: int,
    *,
    exchcd: float | None = 1.0,
    label: str = "unclear",
    name: str = "ALPHA CORP",
    source: str = "mcgill",
    exit_date: str = "2010-02-15",
) -> dict[str, Any]:
    """One roster row with the columns the table reads."""
    return {
        "symbol": f"PERMNO-{permno}",
        "permno": permno,
        "window": "dev",
        "universe": "A",
        "exit_date": exit_date,
        "last_close": 10.0,
        "stock_name": name,
        "exchcd": exchcd,
        "name_source": source,
        "price_label": label,
    }


def roster_fixture(n: int = 3, unnamed: bool = False) -> pd.DataFrame:
    """A roster of ``n`` rows; ``unnamed`` makes the first row a stock with no name."""
    rows = [roster_row(i + 1) for i in range(n)]
    if unnamed:
        rows[0].update(stock_name=None, name_source="none", exchcd=None)
    return pd.DataFrame(rows)


def match(
    status: str = "matched",
    confirmation: str = "confirmed_common",
    *,
    accession: str = "0000000001-10-000001",
    notice: bool = True,
) -> dict[str, Any]:
    """A per-stock record as the Form 25 phase writes it."""
    deciding = None
    if notice:
        deciding = {
            "status": "readable",
            "accession": accession,
            "notice": {
                "security_description": "Common Stock",
                "rule_provision": "17 CFR 240.12d2-2(a)(3)",
            },
        }
    ciks = [CIK] if status == "matched" else []
    return {
        "status": status,
        "confirmation": confirmation,
        "ciks": ciks,
        "accessions": [accession] if status == "matched" else [],
        "notice": deciding if status == "matched" else None,
    }


def event(items: list[str], accession: str = "0000000009-10-000001") -> dict[str, Any]:
    """One 8-K event."""
    return {"filing_date": "2010-02-10", "accession": accession, "items": items}


def conflict_fixture() -> tuple[pd.DataFrame, dict, dict, dict]:
    """Stock 1 bankrupt but calm; 2 acquired but collapsed; 3 acquired and calm (agrees)."""
    roster = pd.DataFrame(
        [
            roster_row(1, label="calm_like_deal"),
            roster_row(2, label="collapse_like"),
            roster_row(3, label="calm_like_deal"),
        ]
    )
    matches = {i: match() for i in (1, 2, 3)}
    events = {1: [event(["1.03"])], 2: [event(["2.01"])], 3: [event(["2.01"])]}
    return roster, matches, {}, events


def table_fixture() -> pd.DataFrame:
    """Three stocks: a conflict with a loss, an agreeing deal with a loss, one with no loss."""
    roster = pd.DataFrame(
        [
            roster_row(1, label="calm_like_deal"),
            roster_row(2, label="calm_like_deal"),
            roster_row(3, label="unclear"),
        ]
    )
    matches = {i: match() for i in (1, 2, 3)}
    events = {1: [event(["1.03"])], 2: [event(["2.01"])], 3: [event(["3.01"])]}
    return build_payout_table(roster, matches, {}, events)


def test_suggested_loss_table():
    assert suggested_loss("acquired_or_merged", 1) == (0.0, "filing_evidence")
    assert suggested_loss("bankruptcy_filed", 1) == (0.30, "literature_fill")
    assert suggested_loss("bankruptcy_filed", 2) == (0.30, "literature_fill")
    assert suggested_loss("bankruptcy_filed", 3) == (0.55, "literature_fill")
    assert suggested_loss("delisted_other", 1) == (None, "none")
    assert suggested_loss("no_filing_match", 3) == (None, "none")


def test_suggested_loss_with_unknown_exchange_is_none():
    assert suggested_loss("bankruptcy_filed", None) == (None, "none")
    assert suggested_loss("bankruptcy_filed", 4) == (None, "none")
    assert suggested_loss("bankruptcy_filed", float("nan")) == (None, "none")


def test_literature_fills_name_both_papers():
    assert LITERATURE_FILLS[(1, 2)]["loss"] == 0.30 and LITERATURE_FILLS[(3,)]["loss"] == 0.55
    assert "Shumway (1997)" in LITERATURE_FILLS[(1, 2)]["source"]
    assert "Shumway and Warther (1999)" in LITERATURE_FILLS[(3,)]["source"]


def test_literature_fills_carry_their_qualification():
    for fill in LITERATURE_FILLS.values():
        assert fill["source_status"] == "unverified library copy"
        assert fill["meaning"] == (
            "missing-delisting-return fill used in the literature, "
            "not a measurement of these stocks"
        )
        assert fill["library_id"] and fill["locators"] and fill["doi"]


def test_the_seven_evidence_classes():
    assert EVIDENCE_CLASSES == (
        "bankruptcy_filed",
        "acquired_or_merged",
        "delisted_other",
        "no_8k_found",
        "no_filing_match",
        "name_unknown",
        "ambiguous_match",
    )


def test_cross_check_flags_conflicts_both_ways():
    table = build_payout_table(*conflict_fixture())
    assert table.set_index("permno")["cross_check"].to_dict() == {
        1: "conflict",
        2: "conflict",
        3: "agrees",
    }


def test_cross_check_not_comparable_when_the_label_is_unclear_or_no_evidence():
    roster = pd.DataFrame([roster_row(1, label="unclear"), roster_row(2, label="collapse_like")])
    table = build_payout_table(roster, {}, {}, {})  # no matches: nothing to compare
    assert set(table["cross_check"]) == {"not_comparable"}


def test_every_roster_row_survives_into_the_table():
    roster = roster_fixture(n=5)
    table = build_payout_table(roster, matches={}, notices={}, events={})
    assert len(table) == 5 and set(table["evidence_class"]) <= set(EVIDENCE_CLASSES)
    assert list(table.columns) == list(TABLE_COLUMNS)
    assert set(table["evidence_class"]) == {"no_filing_match"}  # named, but nothing was matched


def test_unnamed_stock_is_name_unknown():
    table = build_payout_table(roster_fixture(unnamed=True), {}, {}, {})
    assert table["evidence_class"].iloc[0] == "name_unknown"
    assert table["suggested_loss"].isna().iloc[0] and table["basis"].iloc[0] == "none"


def test_ambiguous_status_maps_to_ambiguous_match():
    roster = roster_fixture(n=1)
    table = build_payout_table(roster, {1: match("ambiguous", "not_checked")}, {}, {})
    assert table["evidence_class"].iloc[0] == "ambiguous_match"
    assert table["match_confidence"].iloc[0] == ""


@pytest.mark.parametrize("confirmation", ["non_common_only", "rejected_issuer_mismatch"])
def test_a_rejected_match_is_no_filing_match(confirmation: str):
    roster = roster_fixture(n=1)
    table = build_payout_table(roster, {1: match("none", confirmation)}, {}, {})
    assert table["evidence_class"].iloc[0] == "no_filing_match"
    assert table["match_confidence"].iloc[0] == ""
    assert table["basis"].iloc[0] == "none"


def test_match_confidence_and_basis_for_confirmed_and_name_only_stocks():
    roster = pd.DataFrame([roster_row(1), roster_row(2), roster_row(3)])
    matches = {
        1: match(),
        2: match(confirmation="unconfirmed_unreadable", notice=False),
        3: match(confirmation="unconfirmed_unreadable", notice=False),
    }
    events = {1: [event(["2.01"])], 2: [event(["2.01"])], 3: [event(["3.01"])]}
    table = build_payout_table(roster, matches, {}, events).set_index("permno")
    assert table["match_confidence"].to_dict() == {1: "confirmed", 2: "name_only", 3: "name_only"}
    assert table["suggested_loss"].iloc[0] == 0.0 and table["suggested_loss"].iloc[1] == 0.0
    assert table["basis"].to_dict() == {
        1: "filing_evidence",
        2: "filing_evidence",
        3: "none",
    }
    # An unreadable notice has no accession or provision of its own.
    assert table.loc[2, "form25_provision"] == "" and table.loc[2, "form25_security"] == ""
    assert table.loc[1, "form25_accession"] == "0000000001-10-000001"
    assert table.loc[1, "form25_security"] == "Common Stock"


def test_name_only_bankruptcy_keeps_the_literature_basis_and_counts_apart():
    roster = pd.DataFrame([roster_row(1, exchcd=3.0)])
    table = build_payout_table(
        roster,
        {1: match(confirmation="unconfirmed_unreadable", notice=False)},
        {},
        {1: [event(["1.03"])]},
    )
    assert table["suggested_loss"].iloc[0] == 0.55
    assert table["basis"].iloc[0] == "literature_fill"  # the basis names the number's source
    assert table["match_confidence"].iloc[0] == "name_only"
    cov = coverage_summary(table)
    assert cov["by_basis_and_confidence"]["literature_fill"] == {
        "confirmed": 0,
        "name_only": 1,
        "none": 0,
    }
    assert cov["by_basis"] == {"filing_evidence": 0, "literature_fill": 1, "none": 0}
    assert "filing_evidence_name_only" not in json.dumps(cov)


@pytest.mark.parametrize(
    ("code", "expected"),
    [(3.0, 3), (3, 3), (1.0, 1), (None, None), (float("nan"), None), (float("inf"), None)]
    + [(float("-inf"), None), ("3", None), ("NYSE", None), (3.5, None), (True, None)],
)
def test_clean_exchange_only_trusts_finite_whole_numbers(code: Any, expected: int | None):
    assert clean_exchange(code) == expected
    assert suggested_loss("bankruptcy_filed", code)[0] == (
        {1: 0.30, 2: 0.30, 3: 0.55}.get(expected) if expected else None
    )


def test_events_unavailable_or_incomplete_counts_every_non_ok_status():
    roster = pd.DataFrame([roster_row(i) for i in range(1, 7)])
    matches = {i: match() for i in range(1, 7)}
    status = {
        1: "ok",
        2: "no_older_pages_needed",
        3: "fetch_error",
        4: "not_found",
        5: "older_page_not_found",
        6: "some_new_status",
    }
    table = build_payout_table(roster, matches, {}, {}, events_status=status)
    gap = coverage_summary(table)["events_unavailable_or_incomplete"]
    assert gap["total"] == 4
    assert gap["by_status"] == {
        "fetch_error": 1,
        "not_found": 1,
        "older_page_not_found": 1,
        "some_new_status": 1,
    }
    assert "no 8-K was filed" in gap["note"] and "could not be read" in gap["note"]
    unmatched = build_payout_table(roster, {}, {}, {})  # nothing fetched: no gap to report
    assert coverage_summary(unmatched)["events_unavailable_or_incomplete"]["total"] == 0


def test_evidence_class_comes_from_the_eight_k_items_and_event_accessions_are_listed():
    roster = pd.DataFrame([roster_row(1), roster_row(2)])
    events = {1: [event(["3.01"], "A-1"), event(["8.01"], "A-2")], 2: []}
    table = build_payout_table(roster, {1: match(), 2: match()}, {}, events).set_index("permno")
    assert table.loc[1, "evidence_class"] == "delisted_other"
    assert table.loc[1, "event_accessions"] == "A-1;A-2"
    assert (
        table.loc[2, "evidence_class"] == "no_8k_found" and table.loc[2, "event_accessions"] == ""
    )


def test_events_status_is_carried_into_the_table():
    roster = pd.DataFrame([roster_row(1)])
    table = build_payout_table(roster, {1: match()}, {}, {}, events_status={1: "not_found"})
    assert table["events_status"].iloc[0] == "not_found"


def test_coverage_summary_counts():
    cov = coverage_summary(table_fixture())
    assert cov["with_suggested_loss"] == 2 and cov["conflicts"] == 1


def test_coverage_summary_splits_confirmed_from_name_only_and_counts_every_class():
    roster = pd.DataFrame([roster_row(1), roster_row(2), roster_row(3, source="none", name=None)])
    matches = {1: match(), 2: match(confirmation="unconfirmed_unreadable", notice=False)}
    events = {1: [event(["2.01"])], 2: [event(["2.01"])]}
    table = build_payout_table(roster, matches, {}, events)
    cov = coverage_summary(table, items_missing=4, quarters_skipped=2)
    assert set(cov["by_class"]) == set(EVIDENCE_CLASSES)
    assert cov["by_class"]["acquired_or_merged"] == 2 and cov["by_class"]["name_unknown"] == 1
    assert cov["by_match_confidence"] == {"confirmed": 1, "name_only": 1, "none": 1}
    assert cov["by_class_and_confidence"]["acquired_or_merged"] == {"confirmed": 1, "name_only": 1}
    assert cov["with_suggested_loss_confirmed"] == 1 and cov["with_suggested_loss_name_only"] == 1
    assert cov["items_missing"] == 4 and cov["quarters_skipped"] == 2
    assert cov["roster_rows"] == 3
    assert sum(cov["by_class"].values()) == 3


# ---- the payouts phase of the SEC driver -------------------------------------------------------

SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{int(CIK):010d}.json"


class FakeClient:
    """Canned bodies by URL; no network, no cache."""

    network_fetches = 0
    cache_hits = 0
    corrupt_cache_refetches = 0

    def __init__(self, bodies: dict[str, object]) -> None:
        self.bodies = bodies
        self.asked: list[str] = []

    def get(self, url: str) -> bytes:
        self.asked.append(url)
        body = self.bodies.get(url)
        if body is None:
            raise SecNotFound(url)
        if isinstance(body, Exception):
            raise body
        assert isinstance(body, bytes)
        return body

    def manifest(self) -> dict[str, str]:
        return {url: "0" * 64 for url in sorted(self.bodies)}


def submissions_body(items: str = "2.01") -> bytes:
    """A submissions file with one 8-K filed five days before the exit date."""
    recent = {
        "form": ["8-K"],
        "filingDate": ["2010-02-10"],
        "items": [items],
        "accessionNumber": ["0000000009-10-000001"],
        "primaryDocument": ["x.htm"],
    }
    return json.dumps({"filings": {"recent": recent, "files": []}}).encode()


FORM25_PAYLOAD = {
    "phase": "form25",
    "stocks": [
        {"symbol": "PERMNO-1", "permno": "1", "exit_date": "2010-02-15", **match()},
        {
            "symbol": "PERMNO-2",
            "permno": "2",
            "exit_date": "2010-02-15",
            **match("none", "rejected_issuer_mismatch"),
        },
    ],
}


def write_inputs(tmp_path: Path) -> list[str]:
    """A two-stock roster and its Form 25 result; returns the command line for ``main``."""
    roster = tmp_path / "roster.csv"
    header = "symbol,permno,window,universe,exit_date,last_close,stock_name,exchcd,name_source"
    roster.write_text(
        header
        + ",price_label\n"
        + "PERMNO-1,1,dev,A,2010-02-15,10.0,ALPHA CORP,1,mcgill,calm_like_deal\n"
        + "PERMNO-2,2,dev,A,2010-02-15,10.0,BETA CORP,3,mcgill,unclear\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "_form25_matches.json").write_text(json.dumps(FORM25_PAYLOAD))
    return [
        "--phase",
        "payouts",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out",
        str(out),
        "--roster",
        str(roster),
    ]


@pytest.fixture(autouse=True)
def user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A made-up fair-access name, never a real address."""
    monkeypatch.setenv("SEC_USER_AGENT", "ggTrader test client")


def sec_client() -> FakeClient:
    return FakeClient({SUBMISSIONS_URL: submissions_body()})


def test_run_payouts_fetches_events_only_for_matched_stocks():
    client = sec_client()
    rows = [
        {**roster_row(1, label="calm_like_deal"), "permno": "1", "exchcd": "1"},
        {**roster_row(2), "permno": "2", "exchcd": "3"},
    ]
    table, payload = run_payouts(client, rows, FORM25_PAYLOAD, run_date="2026-11-24")
    assert client.asked == [SUBMISSIONS_URL]  # stock 2 was rejected: no events fetched
    table = table.set_index("permno")
    assert table.loc[1, "evidence_class"] == "acquired_or_merged"
    assert table.loc[2, "evidence_class"] == "no_filing_match"
    assert payload["events_status_counts"] == {"ok": 1}
    assert payload["coverage_summary"]["roster_rows"] == 2


def test_run_payouts_reads_the_exchange_code_from_csv_text():
    client = FakeClient({SUBMISSIONS_URL: submissions_body("1.03")})
    rows = [{**roster_row(1), "permno": "1", "exchcd": "3.0"}]
    table, payload = run_payouts(client, rows, {"stocks": FORM25_PAYLOAD["stocks"][:1]})
    assert table["suggested_loss"].iloc[0] == 0.55 and table["basis"].iloc[0] == "literature_fill"
    assert payload["literature_fills"]["3"]["source_status"] == "unverified library copy"
    assert payload["coverage_summary"]["events_unavailable_or_incomplete"]["total"] == 0


def test_run_payouts_records_a_failed_events_fetch_and_carries_on():
    client = FakeClient({SUBMISSIONS_URL: sec_pilot.SecFetchError("HTTP 500")})
    rows = [{**roster_row(1), "permno": "1", "exchcd": "1"}]
    table, payload = run_payouts(client, rows, {"stocks": FORM25_PAYLOAD["stocks"][:1]})
    assert payload["events_status_counts"] == {"fetch_error": 1}
    assert table["events_status"].iloc[0] == "fetch_error"
    gap = payload["coverage_summary"]["events_unavailable_or_incomplete"]
    assert gap["by_status"] == {"fetch_error": 1}


def test_main_payouts_writes_csv_json_and_checksums(tmp_path: Path):
    argv = write_inputs(tmp_path)
    assert main(argv, client_factory=lambda ua, cache: sec_client()) == 0
    out = tmp_path / "out"
    with (out / "_payouts.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[0]["evidence_class"] == "acquired_or_merged"
    assert rows[0]["match_confidence"] == "confirmed" and rows[0]["suggested_loss"] == "0.0"
    assert rows[1]["suggested_loss"] == "" and rows[1]["basis"] == "none"
    result = json.loads((out / "_payouts.json").read_text())
    assert result["coverage_summary"]["with_suggested_loss"] == 1
    assert result["provenance"]["roster_sha256"] and result["provenance"]["form25_sha256"]
    assert "git_head" in result["provenance"] and "git_dirty" in result["provenance"]
    assert result["manifest"] and result["request_count"]["network_fetches"] == 0
    assert "ggTrader test client" not in (out / "_payouts.json").read_text()


def test_main_payouts_refuses_to_overwrite_without_force(tmp_path: Path):
    argv = write_inputs(tmp_path)
    (tmp_path / "out" / "_payouts.csv").write_text("old")
    made: list[str] = []

    def factory(ua: str, cache: Path) -> FakeClient:
        made.append(ua)
        return sec_client()

    with pytest.raises(SystemExit) as stop:
        main(argv, client_factory=factory)
    assert stop.value.code == 2 and made == []
    assert (tmp_path / "out" / "_payouts.csv").read_text() == "old"
    assert main([*argv, "--force"], client_factory=factory) == 0


def test_main_payouts_refuses_when_the_form25_file_is_missing(tmp_path: Path):
    argv = write_inputs(tmp_path)
    (tmp_path / "out" / "_form25_matches.json").unlink()
    made: list[str] = []

    def factory(ua: str, cache: Path) -> FakeClient:
        made.append(ua)
        return sec_client()

    with pytest.raises(SystemExit) as stop:
        main(argv, client_factory=factory)
    assert stop.value.code == 2 and made == []


def test_main_payouts_refuses_a_form25_file_for_another_roster(tmp_path: Path):
    argv = write_inputs(tmp_path)
    payload = {"stocks": FORM25_PAYLOAD["stocks"][:1]}
    (tmp_path / "out" / "_form25_matches.json").write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as stop:
        main(argv, client_factory=lambda ua, cache: sec_client())
    assert stop.value.code == 2
