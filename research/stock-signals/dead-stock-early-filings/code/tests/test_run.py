"""End-to-end tests for the dead-stock-early-filings driver and SecLookup (no network, no DB)."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import early_classify as ec
import early_sec
import numpy as np
import pandas as pd
import pytest

from ggTrader.utils.study import load_driver, study_dir

pytestmark = pytest.mark.study

run = load_driver(study_dir(__file__) / "code" / "run.py")

DUMMY_CONTACT = "Test Agent test@example.invalid"
DUMMY_EMAIL = "test@example.invalid"


def _make_synthetic_inputs(
    *,
    n_stocks: int = 30,
    n_deaths: int = 12,
    n_held_deaths: int | None = None,
) -> dict[str, Any]:
    """Synthetic early-window panel (warmup + 2000-2001) with deterministic deaths and names."""
    days = pd.bdate_range("1998-06-01", "2001-12-31", tz="UTC")
    t = np.arange(len(days), dtype=float)[:, None]
    drifts = (n_stocks - np.arange(n_stocks, dtype=float))[None, :] * 1e-4
    prices = 50.0 * np.exp(t * drifts)
    cols = [f"PERMNO-{10000 + i}" for i in range(n_stocks)]
    close = pd.DataFrame(prices, index=days, columns=cols)
    spy = pd.Series(100.0 * np.exp(t[:, 0] * 2e-4), index=days)
    eligible = pd.DataFrame(True, index=days, columns=cols)

    month_dates = pd.bdate_range("2000-02-15", "2001-11-15", freq="BMS") + pd.Timedelta(days=10)
    span_records: list[dict[str, Any]] = []
    name_records: list[dict[str, Any]] = []
    if n_held_deaths is None:
        dying_plan = [(i, month_dates[i % len(month_dates)]) for i in range(n_deaths)]
    else:
        held_plan = [(i, month_dates[0]) for i in range(n_held_deaths)]
        unheld_start = n_held_deaths + 50
        unheld_count = n_deaths - n_held_deaths
        assert unheld_start + unheld_count <= n_stocks
        unheld_plan = [
            (unheld_start + j, month_dates[(j % (len(month_dates) - 1)) + 1])
            for j in range(unheld_count)
        ]
        dying_plan = held_plan + unheld_plan

    for k, (idx, target_date) in enumerate(dying_plan):
        col = cols[idx]
        death_day = close.index[close.index.searchsorted(pd.Timestamp(target_date, tz="UTC"))]
        close.loc[death_day + pd.Timedelta(days=1) :, col] = np.nan
        span_records.append(
            {
                "symbol": col,
                "permno": 10000 + idx,
                "last_date": death_day.tz_localize(None),
                "has_successor": False,
                "total_loss": False,
            }
        )
        name_records.append(
            {
                "column": col,
                "company_name": f"SYNTHETIC CO {idx:03d}" if k % 5 != 4 else None,
                "name_source": "mcgill" if k % 5 != 4 else "",
                "exchcd": (1 if idx % 2 == 0 else 3) if k % 5 != 4 else pd.NA,
            }
        )

    span_rows = pd.DataFrame(span_records)
    names = pd.DataFrame(name_records)
    names["exchcd"] = names["exchcd"].astype("Int64")

    return {
        "close": close,
        "spy_close": spy,
        "eligible": eligible,
        "span_rows": span_rows,
        "release_spans": span_rows,
        "names": names,
        "data_provenance": {
            "candidates": {
                "data_release": {"release_id": 1, "release_sha256": "relhash001"},
                "bars_sha256": ["barhash001"],
                "n_batches": 1,
                "drift": False,
                "drift_symbols": [],
            },
            "spy": {
                "data_release": {"release_id": 1, "release_sha256": "relhash001"},
                "bars_sha256": ["spyhash001"],
                "n_batches": 1,
                "drift": False,
                "drift_symbols": [],
            },
            "n_candidate_columns": n_stocks,
        },
    }


def _fake_lookup_factory(
    _named_roster: pd.DataFrame,
    _client: early_sec.EarlySecClient | None,
) -> Any:
    """Deterministic fake SEC lookup cycling through labels based on company name."""

    def lookup(company_name: str, last_day: date) -> ec.CompanyEvidence | None:
        num = int(company_name.split()[-1])
        kind = num % 5
        if kind == 0:
            filing = early_sec.Filing(
                form="8-K",
                company_name=company_name,
                cik=1000 + num,
                filed=last_day,
                file_path=f"edgar/data/{1000 + num}/0001-00-000001.txt",
            )
            header = early_sec.FilingHeader(
                form="8-K",
                filed=last_day,
                company_name=company_name,
                cik=1000 + num,
                former_names=(),
                items=frozenset({"acquisition or disposition of assets"}),
                complete=True,
            )
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(ec.FilingEvidence(filing, header),),
                all_filing_dates=(last_day,),
                candidate_ciks=(1000 + num,),
            )
        if kind == 1:
            filing = early_sec.Filing(
                form="8-K",
                company_name=company_name,
                cik=1000 + num,
                filed=last_day,
                file_path=f"edgar/data/{1000 + num}/0001-00-000002.txt",
            )
            header = early_sec.FilingHeader(
                form="8-K",
                filed=last_day,
                company_name=company_name,
                cik=1000 + num,
                former_names=(),
                items=frozenset({"bankruptcy or receivership"}),
                complete=True,
            )
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(ec.FilingEvidence(filing, header),),
                all_filing_dates=(last_day,),
                candidate_ciks=(1000 + num,),
            )
        if kind == 2:
            filing = early_sec.Filing(
                form="25-NSE",
                company_name=company_name,
                cik=1000 + num,
                filed=last_day,
                file_path=f"edgar/data/{1000 + num}/0001-00-000003.txt",
            )
            header = early_sec.FilingHeader(
                form="25-NSE",
                filed=last_day,
                company_name=company_name,
                cik=1000 + num,
                former_names=(),
                items=frozenset(),
                complete=True,
            )
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(ec.FilingEvidence(filing, header),),
                all_filing_dates=(last_day,),
                candidate_ciks=(1000 + num,),
            )
        if kind == 3:
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(),
                all_filing_dates=(date(last_day.year + 1, last_day.month, 15),),
                candidate_ciks=(1000 + num,),
            )
        return None

    return lookup


def test_no_contact_line_in_the_output_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _make_synthetic_inputs(n_stocks=25, n_deaths=10)
    out_json = tmp_path / "_pilot.json"
    out_csv = tmp_path / "_pilot_labels.csv"
    seen_clients: list[early_sec.EarlySecClient | None] = []

    def factory(named_roster: pd.DataFrame, client: early_sec.EarlySecClient | None) -> Any:
        seen_clients.append(client)
        return _fake_lookup_factory(named_roster, client)

    code = run.main(
        [
            "--mode",
            "pilot",
            "--pilot-size",
            "6",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out_json),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        loader=lambda _args: inputs,
        sec_lookup_factory=factory,
        git_dirty=lambda: False,
        contact_loader=lambda: DUMMY_CONTACT,
    )
    assert code == 0
    assert seen_clients and seen_clients[0] is not None

    captured = capsys.readouterr()
    json_text = out_json.read_text(encoding="utf-8")
    csv_text = out_csv.read_text(encoding="utf-8")

    for surface in (json_text, csv_text, captured.out, captured.err):
        assert DUMMY_CONTACT not in surface
        assert DUMMY_EMAIL not in surface
        assert "Test Agent" not in surface


def test_pilot_mode_picks_seeded_rows_and_writes_provenance_and_csv(tmp_path: Path) -> None:
    inputs = _make_synthetic_inputs(n_stocks=30, n_deaths=24)
    out_a = tmp_path / "run_a" / "_pilot.json"
    out_b = tmp_path / "run_b" / "_pilot.json"

    for out_path in (out_a, out_b):
        code = run.main(
            [
                "--mode",
                "pilot",
                "--pilot-size",
                "10",
                "--release-id",
                "1",
                "--crsp-load",
                "1",
                "--out",
                str(out_path),
            ],
            loader=lambda _args: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: False,
        )
        assert code == 0

    csv_a = pd.read_csv(out_a.parent / "_pilot_labels.csv")
    csv_b = pd.read_csv(out_b.parent / "_pilot_labels.csv")
    assert len(csv_a) == 10
    assert list(csv_a["permno"]) == list(csv_b["permno"])

    all_permnos = sorted(inputs["span_rows"]["permno"].tolist())
    expected_idx = np.random.default_rng(20261125).permutation(len(all_permnos))[:10]
    assert list(csv_a["permno"]) == [all_permnos[i] for i in expected_idx]

    payload = json.loads(out_a.read_text(encoding="utf-8"))
    assert payload["study"] == "dead-stock-early-filings"
    assert payload["mode"] == "pilot"
    assert payload["window"] == "early"
    assert payload["release_id"] == 1
    assert payload["crsp_load"] == 1
    assert payload["roster_counts"]["full_roster_rows"] == 24
    assert payload["roster_counts"]["selected_rows"] == 10
    assert payload["summary"]["total"] == 10
    assert sum(payload["summary"]["by_label"].values()) == 10
    assert set(payload["summary"]["by_label"]) == set(ec.REGISTERED_LABELS)
    # Minor finding: report fulfilled fetches and recorded 404s separately, not "requests made".
    assert "sec_fetches" in payload
    assert "fulfilled_fetches" in payload["sec_fetches"]
    assert "recorded_not_found" in payload["sec_fetches"]
    assert "network_requests" not in payload["sec_fetches"]
    assert "provenance" in payload
    prov = payload["provenance"]
    assert prov["release_id"] == 1
    assert prov["crsp_load"] == 1
    assert prov["window"] == "early"
    assert prov["data_release"] == {"release_id": 1, "release_sha256": "relhash001"}
    assert prov["bars_sha256"] == ["barhash001"]
    assert "code_sha256" in prov
    assert "labels_csv_sha256" in prov
    assert "command" in prov


def test_refusals_wrong_pins_dirty_tree_existing_output_and_drift(tmp_path: Path) -> None:
    inputs = _make_synthetic_inputs(n_stocks=20, n_deaths=5)
    base_args = [
        "--mode",
        "pilot",
        "--pilot-size",
        "5",
        "--release-id",
        "1",
        "--crsp-load",
        "1",
        "--out",
        str(tmp_path / "_pilot.json"),
    ]

    assert (
        run.main(
            [
                "--mode",
                "pilot",
                "--release-id",
                "2",
                "--crsp-load",
                "1",
                "--out",
                str(tmp_path / "bad_rel.json"),
            ],
            loader=lambda _a: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )

    assert (
        run.main(
            [
                "--mode",
                "pilot",
                "--release-id",
                "1",
                "--crsp-load",
                "9",
                "--out",
                str(tmp_path / "bad_load.json"),
            ],
            loader=lambda _a: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )

    assert (
        run.main(
            [
                "--mode",
                "pilot",
                "--pilot-size",
                "0",
                "--release-id",
                "1",
                "--crsp-load",
                "1",
                "--out",
                str(tmp_path / "bad_size.json"),
            ],
            loader=lambda _a: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )

    assert (
        run.main(
            base_args,
            loader=lambda _a: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: True,
        )
        == run.EXIT_REFUSED
    )

    def drifted(_args: Any) -> dict[str, Any]:
        copy = dict(inputs)
        copy["data_provenance"] = {
            "candidates": {"drift": True, "drift_symbols": ["PERMNO-10000"]},
        }
        return copy

    assert (
        run.main(
            base_args,
            loader=drifted,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )
    assert not (tmp_path / "_pilot.json").exists()
    assert not (tmp_path / "_pilot_labels.csv").exists()

    (tmp_path / "_pilot.json").write_text("{}", encoding="utf-8")
    assert (
        run.main(
            base_args,
            loader=lambda _a: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )
    (tmp_path / "_pilot.json").unlink()

    (tmp_path / "_pilot_labels.csv").write_text("column\n", encoding="utf-8")
    assert (
        run.main(
            base_args,
            loader=lambda _a: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )


def test_full_mode_roster_check_exits_3_on_mismatch_and_passes_on_251_33(tmp_path: Path) -> None:
    small_inputs = _make_synthetic_inputs(n_stocks=20, n_deaths=10)
    called = False

    def fail_if_called(_roster: pd.DataFrame, _client: Any) -> Any:
        nonlocal called
        called = True
        return _fake_lookup_factory(_roster, _client)

    code_bad = run.main(
        [
            "--mode",
            "full",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(tmp_path / "bad_full.json"),
        ],
        loader=lambda _a: small_inputs,
        sec_lookup_factory=fail_if_called,
        git_dirty=lambda: False,
    )
    assert code_bad == run.EXIT_CHECK_FAILED
    assert not called
    assert not (tmp_path / "bad_full.json").exists()
    assert not (tmp_path / "_early_labels.csv").exists()

    exact_inputs = _make_synthetic_inputs(n_stocks=310, n_deaths=251, n_held_deaths=33)
    out_full = tmp_path / "full" / "_early.json"
    code_ok = run.main(
        [
            "--mode",
            "full",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out_full),
        ],
        loader=lambda _a: exact_inputs,
        sec_lookup_factory=_fake_lookup_factory,
        git_dirty=lambda: False,
    )
    assert code_ok == 0
    payload = json.loads(out_full.read_text(encoding="utf-8"))
    csv_df = pd.read_csv(out_full.parent / "_early_labels.csv")
    assert len(csv_df) == 251
    assert int(csv_df["held_by_rule"].sum()) == 33
    assert payload["roster_counts"]["full_roster_rows"] == 251
    assert payload["roster_counts"]["full_held_by_rule"] == 33
    assert payload["summary"]["total"] == 251
    assert payload["summary"]["held_by_rule"]["total"] == 33
    assert sum(payload["summary"]["by_label"].values()) == 251
    assert sum(payload["summary"]["held_by_rule"]["by_label"].values()) == 33


def _make_master_zip(rows: list[tuple[str, str, str, str, str]]) -> bytes:
    buf = io.BytesIO()
    lines = [
        "Description: Master Index of EDGAR Dissemination Feed",
        early_sec.MASTER_INDEX_HEADER,
        "-" * 60,
        *[f"{cik}|{name}|{form}|{filed}|{path}" for cik, name, form, filed, path in rows],
    ]
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(early_sec.MASTER_INDEX_FILENAME, "\n".join(lines) + "\n")
    return buf.getvalue()


def _explicit_quarter_responses(
    start_q: tuple[int, int],
    end_q: tuple[int, int],
    per_quarter_rows: dict[tuple[int, int], list[tuple[str, str, str, str, str]]],
) -> dict[str, bytes]:
    """Build an explicit master.zip map for EVERY required quarter (no wildcard fallback)."""
    quarters = early_sec.quarters_covering(
        date(start_q[0], (start_q[1] - 1) * 3 + 1, 1),
        date(end_q[0], (end_q[1] - 1) * 3 + 1, 1),
    )
    return {
        early_sec.master_index_url(y, q): _make_master_zip(per_quarter_rows.get((y, q), []))
        for y, q in quarters
    }


def _header_bytes(
    *,
    form: str,
    filed_yyyymmdd: str,
    company: str,
    cik: str,
    items: tuple[str, ...] = (),
    former_names: tuple[str, ...] = (),
) -> bytes:
    lines = [
        "<SEC-HEADER>",
        f"CONFORMED SUBMISSION TYPE:\t{form}",
        f"FILED AS OF DATE:\t\t{filed_yyyymmdd}",
        *[f"ITEM INFORMATION:\t\t{item}" for item in items],
        f"COMPANY CONFORMED NAME:\t\t{company}",
        f"CENTRAL INDEX KEY:\t\t{cik}",
        *[f"FORMER CONFORMED NAME:\t\t{fn}" for fn in former_names],
        "</SEC-HEADER>",
        "<DOCUMENT>",
        "body",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)
        self.status = 200
        self.headers = {"Content-Type": "application/octet-stream"}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _strict_opener(responses: dict[str, bytes]) -> Any:
    """Strict fake opener: returns 404 for ANY URL not explicitly listed in ``responses``."""

    def opener(req: Any, timeout: float = 30.0) -> _FakeResponse:
        url = req.full_url
        if url in responses:
            return _FakeResponse(responses[url])
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    return opener


def test_missing_required_master_index_quarter_stops_run_with_exit_4_live_and_cached(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing required quarter (both live 404 and cached 404) stops run.main with exit 4."""
    inputs = _make_synthetic_inputs(n_stocks=10, n_deaths=2)
    # Supply every quarter from 2000 Q1..2006 Q4 EXCEPT 2003 Q2 (which is missing -> 404).
    responses = _explicit_quarter_responses((2000, 1), (2006, 4), {})
    missing_url = early_sec.master_index_url(2003, 2)
    del responses[missing_url]

    cache_dir = tmp_path / "sec_cache"
    client1 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_dir,
        min_interval=0.0,
        opener=_strict_opener(responses),
        sleep=lambda _s: None,
    )
    out1 = tmp_path / "run1" / "_pilot.json"
    code1 = run.main(
        [
            "--mode",
            "pilot",
            "--pilot-size",
            "2",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out1),
        ],
        loader=lambda _a: inputs,
        sec_client=client1,
        git_dirty=lambda: False,
    )
    err1 = capsys.readouterr().err
    assert code1 == run.EXIT_MISSING_INDEX
    assert "2003" in err1 and "QTR2" in err1
    assert DUMMY_CONTACT not in err1 and DUMMY_EMAIL not in err1
    assert not out1.exists()
    assert not (out1.parent / "_pilot_labels.csv").exists()
    assert client1.is_not_found(missing_url)

    # Second run sees the cached 404 marker: still exits 4 with zero new fulfilled fetches.
    client2 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_dir,
        min_interval=0.0,
        opener=_strict_opener(responses),
        sleep=lambda _s: None,
    )
    out2 = tmp_path / "run2" / "_pilot.json"
    code2 = run.main(
        [
            "--mode",
            "pilot",
            "--pilot-size",
            "2",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out2),
        ],
        loader=lambda _a: inputs,
        sec_client=client2,
        git_dirty=lambda: False,
    )
    err2 = capsys.readouterr().err
    assert code2 == run.EXIT_MISSING_INDEX
    assert "2003" in err2 and "QTR2" in err2
    assert DUMMY_CONTACT not in err2 and DUMMY_EMAIL not in err2
    assert client2.network_fetches == 0
    assert client2.not_found_hits == 1
    assert not out2.exists()


def test_sec_lookup_cik_binding_pass_b_ambiguity_unreadability_and_precedence(
    tmp_path: Path,
) -> None:
    """Exercise SecLookup end-to-end with strict per-quarter archives (no wildcard fallback)."""
    last_day = date(2001, 3, 10)
    day_180 = last_day + timedelta(days=180)  # 2001-09-06 (Q3)
    day_181 = last_day + timedelta(days=181)  # 2001-09-07 (Q3)

    per_quarter: dict[tuple[int, int], list[tuple[str, str, str, str, str]]] = {
        (2001, 1): [
            # 1. Late-discovered CIK 100001: March 8-K indexed under OLD NAME in Q1;
            # roster name LATE RENAMED CORP first appears in Q2! Pass B must recover this 8-K.
            (
                "100001",
                "VERY OLD PREDECESSOR INC",
                "8-K",
                "2001-03-15",
                "edgar/data/100001/0000100001-01-000001.txt",
            ),
            # 2. Header 404 -> evidence_unresolved
            (
                "100002",
                "HDR404 CORP",
                "8-K",
                "2001-03-20",
                "edgar/data/100002/0000100002-01-000001.txt",
            ),
            # 3. Unreadable 200 header body (garbage text) -> evidence_unresolved
            (
                "100003",
                "UNREADABLE CORP",
                "8-K",
                "2001-03-20",
                "edgar/data/100003/0000100003-01-000001.txt",
            ),
            # 4. Header CIK mismatch -> evidence_unresolved
            (
                "100004",
                "BADCIK CORP",
                "8-K",
                "2001-03-20",
                "edgar/data/100004/0000100004-01-000001.txt",
            ),
            # 5. Header form mismatch -> evidence_unresolved
            (
                "100005",
                "BADFORM CORP",
                "8-K",
                "2001-03-20",
                "edgar/data/100005/0000100005-01-000001.txt",
            ),
            # 6. Ambiguous match: two CIKs share cleaned name "DUPLICATE"; only wrong CIK has 8-K.
            (
                "100006",
                "DUPLICATE INC",
                "8-K",
                "2001-03-15",
                "edgar/data/100006/0000100006-01-000001.txt",
            ),
            (
                "100007",
                "DUPLICATE CORP",
                "10-Q",
                "2001-03-15",
                "edgar/data/100007/0000100007-01-000001.txt",
            ),
            # 7. Acquisition + day 181 filing -> kept_filing (evidence still notes acquisition)
            (
                "100008",
                "ACQ KEPT CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/100008/0000100008-01-000001.txt",
            ),
            # 8. Acquisition + day 180 filing -> acquisition_type (suggested_loss == 0.0)
            (
                "100009",
                "ACQ DAY180 CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/100009/0000100009-01-000001.txt",
            ),
            # 9. Bankruptcy + day 181 filing + unresolved 404 8-K -> bankruptcy wins
            (
                "100010",
                "BK KEPT CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/100010/0000100010-01-000001.txt",
            ),
            (
                "100010",
                "BK KEPT CORP",
                "8-K",
                "2001-03-18",
                "edgar/data/100010/0000100010-01-000002.txt",
            ),
            # 10. Former name in header must NOT substitute another CIK or match a different stock.
            (
                "100011",
                "OTHER FILER CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/100011/0000100011-01-000001.txt",
            ),
        ],
        (2001, 2): [
            # CIK 100001 first appears under the roster name in Q2!
            (
                "100001",
                "LATE RENAMED CORP",
                "10-Q",
                "2001-05-10",
                "edgar/data/100001/0000100001-01-000002.txt",
            ),
        ],
        (2001, 3): [
            (
                "100008",
                "ACQ KEPT CORP",
                "10-Q",
                day_181.isoformat(),
                "edgar/data/100008/0000100008-01-000002.txt",
            ),
            (
                "100009",
                "ACQ DAY180 CORP",
                "10-Q",
                day_180.isoformat(),
                "edgar/data/100009/0000100009-01-000002.txt",
            ),
            (
                "100010",
                "BK KEPT CORP",
                "10-Q",
                day_181.isoformat(),
                "edgar/data/100010/0000100010-01-000003.txt",
            ),
        ],
    }

    responses = _explicit_quarter_responses((2001, 1), (2006, 4), per_quarter)
    responses.update(
        {
            early_sec.filing_url("edgar/data/100001/0000100001-01-000001.txt"): _header_bytes(
                form="8-K",
                filed_yyyymmdd="20010315",
                company="VERY OLD PREDECESSOR INC",
                cik="0000100001",
                items=("Acquisition or disposition of assets",),
            ),
            # 100002 intentionally omitted -> 404
            early_sec.filing_url("edgar/data/100003/0000100003-01-000001.txt"): (
                b"plain text garbage without any SGML header fields\n"
            ),
            early_sec.filing_url("edgar/data/100004/0000100004-01-000001.txt"): _header_bytes(
                form="8-K",
                filed_yyyymmdd="20010320",
                company="WRONG COMPANY",
                cik="0000999999",
                items=("Bankruptcy or receivership",),
            ),
            early_sec.filing_url("edgar/data/100005/0000100005-01-000001.txt"): _header_bytes(
                form="10-K",
                filed_yyyymmdd="20010320",
                company="BADFORM CORP",
                cik="0000100005",
                items=("Acquisition or disposition of assets",),
            ),
            early_sec.filing_url("edgar/data/100006/0000100006-01-000001.txt"): _header_bytes(
                form="8-K",
                filed_yyyymmdd="20010315",
                company="DUPLICATE INC",
                cik="0000100006",
                items=("Acquisition or disposition of assets",),
            ),
            early_sec.filing_url("edgar/data/100008/0000100008-01-000001.txt"): _header_bytes(
                form="8-K",
                filed_yyyymmdd="20010315",
                company="ACQ KEPT CORP",
                cik="0000100008",
                items=("Acquisition or disposition of assets",),
            ),
            early_sec.filing_url("edgar/data/100009/0000100009-01-000001.txt"): _header_bytes(
                form="8-K",
                filed_yyyymmdd="20010315",
                company="ACQ DAY180 CORP",
                cik="0000100009",
                items=("Acquisition or disposition of assets",),
            ),
            early_sec.filing_url("edgar/data/100010/0000100010-01-000001.txt"): _header_bytes(
                form="8-K",
                filed_yyyymmdd="20010315",
                company="BK KEPT CORP",
                cik="0000100010",
                items=("Bankruptcy or receivership",),
            ),
            # 100010-01-000002 intentionally omitted -> 404 (still beaten by resolved bankruptcy!)
            early_sec.filing_url("edgar/data/100011/0000100011-01-000001.txt"): _header_bytes(
                form="8-K",
                filed_yyyymmdd="20010315",
                company="OTHER FILER CORP",
                cik="0000100011",
                items=("Acquisition or disposition of assets",),
                former_names=("FORMER ONLY STOCK INC",),
            ),
        }
    )

    spec = [
        ("LATE", 1, "LATE RENAMED INC"),
        ("H404", 2, "HDR404 INC"),
        ("UNRD", 3, "UNREADABLE INC"),
        ("BCIK", 4, "BADCIK INC"),
        ("BFRM", 5, "BADFORM INC"),
        ("AMBG", 6, "DUPLICATE LLC"),
        ("K181", 7, "ACQ KEPT INC"),
        ("A180", 8, "ACQ DAY180 INC"),
        ("BKKP", 9, "BK KEPT INC"),
        ("OTHR", 10, "OTHER FILER INC"),
        ("FRMR", 11, "FORMER ONLY STOCK CORP"),
    ]
    roster = pd.DataFrame(
        [
            {
                "column": col,
                "permno": p,
                "symbol": col,
                "last_bar": pd.Timestamp(last_day),
                "decision_date": pd.Timestamp("2001-02-28"),
                "rank_band": "1-50",
                "held_by_rule": True,
            }
            for col, p, _name in spec
        ]
    )
    roster["permno"] = roster["permno"].astype("Int64")
    names = roster.copy()
    names["company_name"] = [name for _col, _p, name in spec]
    names["name_source"] = "mcgill"
    names["exchcd"] = pd.Series([1] * len(spec), dtype="Int64")

    cache_dir = tmp_path / "sec_cache"
    client1 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_dir,
        min_interval=0.0,
        opener=_strict_opener(responses),
        sleep=lambda _s: None,
    )
    lookup1 = run.SecLookup(client1, names)
    classified = ec.classify_roster(roster, names, lookup1).set_index("column")

    # 1. Late-discovered CIK 100001 recovers Q1 March notice via Pass B!
    assert classified.loc["LATE", "label"] == "acquisition_type"
    assert classified.loc["LATE", "confidence"] == "checked"
    assert classified.loc["LATE", "suggested_loss"] == 0.0

    # 2-5. Every header failure yields evidence_unresolved with explicit reason, no suggested loss.
    assert classified.loc["H404", "label"] == "evidence_unresolved"
    assert "header_not_found" in classified.loc["H404", "evidence"]
    assert classified.loc["UNRD", "label"] == "evidence_unresolved"
    assert "header_unreadable" in classified.loc["UNRD", "evidence"]
    assert classified.loc["BCIK", "label"] == "evidence_unresolved"
    assert "header_cik_mismatch" in classified.loc["BCIK", "evidence"]
    assert classified.loc["BFRM", "label"] == "evidence_unresolved"
    assert "header_form_mismatch" in classified.loc["BFRM", "evidence"]

    # 6. Multiple CIKs -> ambiguous_match (never resolves by event availability).
    assert classified.loc["AMBG", "label"] == "ambiguous_match"
    assert "2 CIKs" in classified.loc["AMBG", "evidence"]
    assert pd.isna(classified.loc["AMBG", "suggested_loss"])

    # 7. Acquisition + day 181 -> kept_filing (no suggested loss, evidence records notice).
    assert classified.loc["K181", "label"] == "kept_filing"
    assert pd.isna(classified.loc["K181", "suggested_loss"])
    assert "acquisition_type notice" in classified.loc["K181", "evidence"]
    assert "acquisition or disposition of assets" in classified.loc["K181", "evidence"]

    # 8. Acquisition + day 180 -> acquisition_type with 0.0 suggested loss.
    assert classified.loc["A180", "label"] == "acquisition_type"
    assert classified.loc["A180", "suggested_loss"] == 0.0

    # 9. Resolved bankruptcy outranks both kept_filing and unresolved 404 header.
    assert classified.loc["BKKP", "label"] == "bankruptcy"
    assert classified.loc["BKKP", "confidence"] == "checked"
    assert pd.isna(classified.loc["BKKP", "suggested_loss"])

    # 10-11. Former-name second pass is dropped: FRMR is no_company_match (not OTHER FILER's CIK).
    assert classified.loc["OTHR", "label"] == "acquisition_type"
    assert classified.loc["FRMR", "label"] == "no_company_match"
    assert pd.isna(classified.loc["FRMR", "suggested_loss"])

    stats1 = lookup1.stats()
    assert stats1["fulfilled_fetches"] > 0
    assert stats1["recorded_not_found"] == 2  # H404 + BKKP's second 8-K

    # Second run against the same cache_dir makes zero new network fetches.
    client2 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_dir,
        min_interval=0.0,
        opener=_strict_opener(responses),
        sleep=lambda _s: None,
    )
    lookup2 = run.SecLookup(client2, names)
    classified2 = ec.classify_roster(roster, names, lookup2).set_index("column")
    assert list(classified2["label"]) == list(classified["label"])
    assert list(classified2["confidence"]) == list(classified["confidence"])
    stats2 = lookup2.stats()
    assert stats2["fulfilled_fetches"] == 0
    assert stats2["not_found_cache_hits"] == 2

    # Verify strictness: removing ONE unreferenced quarter (2004 Q3) MUST fail loudly.
    incomplete_responses = dict(responses)
    del incomplete_responses[early_sec.master_index_url(2004, 3)]
    client3 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "incomplete_cache",
        min_interval=0.0,
        opener=_strict_opener(incomplete_responses),
        sleep=lambda _s: None,
    )
    with pytest.raises(run.MissingMasterIndexError, match="2004 QTR3"):
        run.SecLookup(client3, names)


def _single_window_inputs(specs: list[tuple[str, int, str, date]]) -> dict[str, Any]:
    """Build synthetic early-window inputs for specific dying stocks (all held in top 50)."""
    days = pd.bdate_range("1998-06-01", "2001-12-31", tz="UTC")
    n_stocks = max(len(specs), 5)
    t = np.arange(len(days), dtype=float)[:, None]
    drifts = (n_stocks - np.arange(n_stocks, dtype=float))[None, :] * 1e-4
    prices = 50.0 * np.exp(t * drifts)
    cols = [s[0] for s in specs] + [f"ALIVE-{i}" for i in range(n_stocks - len(specs))]
    close = pd.DataFrame(prices, index=days, columns=cols)
    spy = pd.Series(100.0 * np.exp(t[:, 0] * 2e-4), index=days)
    eligible = pd.DataFrame(True, index=days, columns=cols)

    span_records: list[dict[str, Any]] = []
    name_records: list[dict[str, Any]] = []
    for col, permno, comp_name, death_day in specs:
        bar_ts = close.index[close.index.searchsorted(pd.Timestamp(death_day, tz="UTC"))]
        close.loc[bar_ts + pd.Timedelta(days=1) :, col] = np.nan
        span_records.append(
            {
                "symbol": col,
                "permno": permno,
                "last_date": bar_ts.tz_localize(None),
                "has_successor": False,
                "total_loss": False,
            }
        )
        name_records.append(
            {
                "column": col,
                "company_name": comp_name,
                "name_source": "mcgill",
                "exchcd": 1,
            }
        )

    span_rows = pd.DataFrame(span_records)
    names = pd.DataFrame(name_records)
    names["exchcd"] = names["exchcd"].astype("Int64")
    return {
        "close": close,
        "spy_close": spy,
        "eligible": eligible,
        "span_rows": span_rows,
        "release_spans": span_rows,
        "names": names,
        "data_provenance": {
            "candidates": {
                "data_release": {"release_id": 1, "release_sha256": "relhash001"},
                "bars_sha256": ["barhash001"],
                "n_batches": 1,
                "drift": False,
                "drift_symbols": [],
            },
            "spy": {
                "data_release": {"release_id": 1, "release_sha256": "relhash001"},
                "bars_sha256": ["spyhash001"],
                "n_batches": 1,
                "drift": False,
                "drift_symbols": [],
            },
            "n_candidate_columns": len(cols),
        },
    }


def test_driver_truncated_headers_and_complete_counterparts(tmp_path: Path) -> None:
    """Fix round 2 (a)-(c): truncated raw 200 headers through run.main + real SecLookup."""
    death_day = date(2001, 3, 12)  # Monday in 2001 Q1
    day_181 = death_day + timedelta(days=181)  # 2001-09-09 (Q3)

    truncated_id_body = (
        b"<SEC-HEADER>\n"
        b"CONFORMED SUBMISSION TYPE:\t8-K\n"
        b"FILED AS OF DATE:\t\t20010315\n"
        b"COMPANY CONFORMED NAME:\t\tEXAMPLE CORP\n"
        b"CENTRAL INDEX KEY:\t\t0000200001\n"
    )
    truncated_id_kept_body = (
        b"<SEC-HEADER>\n"
        b"CONFORMED SUBMISSION TYPE:\t8-K\n"
        b"FILED AS OF DATE:\t\t20010315\n"
        b"COMPANY CONFORMED NAME:\t\tEXAMPLE KEPT CORP\n"
        b"CENTRAL INDEX KEY:\t\t0000200002\n"
    )
    truncated_acq_body = (
        b"<SEC-HEADER>\n"
        b"CONFORMED SUBMISSION TYPE:\t8-K\n"
        b"FILED AS OF DATE:\t\t20010315\n"
        b"ITEM INFORMATION:\t\tAcquisition or disposition of assets\n"
        b"COMPANY CONFORMED NAME:\t\tCUTOFF ACQ CORP\n"
        b"CENTRAL INDEX KEY:\t\t0000200003\n"
    )
    corrupt_binary_body = b"\xff\xfe\x00\x01corrupt non-sgml payload"

    complete_id_body = truncated_id_body + b"</SEC-HEADER>\n<DOCUMENT>\nbody\n"
    complete_id_kept_body = truncated_id_kept_body + b"</SEC-HEADER>\n<DOCUMENT>\nbody\n"
    complete_acq_body = truncated_acq_body + b"</SEC-HEADER>\n<DOCUMENT>\nbody\n"

    per_quarter: dict[tuple[int, int], list[tuple[str, str, str, str, str]]] = {
        (2001, 1): [
            ("200001", "EXAMPLE CORP", "8-K", "2001-03-15", "edgar/data/200001/0001-01-000001.txt"),
            (
                "200002",
                "EXAMPLE KEPT CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/200002/0001-01-000001.txt",
            ),
            (
                "200003",
                "CUTOFF ACQ CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/200003/0001-01-000001.txt",
            ),
            (
                "200004",
                "CORRUPT BODY CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/200004/0001-01-000001.txt",
            ),
        ],
        (2001, 3): [
            (
                "200002",
                "EXAMPLE KEPT CORP",
                "10-Q",
                day_181.isoformat(),
                "edgar/data/200002/0001-01-000002.txt",
            ),
        ],
    }

    inputs = _single_window_inputs(
        [
            ("EX_ID", 20001, "EXAMPLE INC", death_day),
            ("EX_KP", 20002, "EXAMPLE KEPT INC", death_day),
            ("EX_AQ", 20003, "CUTOFF ACQ INC", death_day),
            ("EX_CB", 20004, "CORRUPT BODY INC", death_day),
        ]
    )

    # Run 1: truncated & corrupt headers -> all evidence_unresolved, never normal labels.
    trunc_responses = _explicit_quarter_responses((2001, 1), (2006, 4), per_quarter)
    trunc_responses.update(
        {
            early_sec.filing_url("edgar/data/200001/0001-01-000001.txt"): truncated_id_body,
            early_sec.filing_url("edgar/data/200002/0001-01-000001.txt"): truncated_id_kept_body,
            early_sec.filing_url("edgar/data/200003/0001-01-000001.txt"): truncated_acq_body,
            early_sec.filing_url("edgar/data/200004/0001-01-000001.txt"): corrupt_binary_body,
        }
    )
    client_trunc = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "cache_trunc",
        min_interval=0.0,
        opener=_strict_opener(trunc_responses),
        sleep=lambda _s: None,
    )
    out_trunc = tmp_path / "trunc" / "_pilot.json"
    code1 = run.main(
        [
            "--mode",
            "pilot",
            "--pilot-size",
            "4",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out_trunc),
        ],
        loader=lambda _a: inputs,
        sec_client=client_trunc,
        git_dirty=lambda: False,
    )
    assert code1 == 0
    df_trunc = pd.read_csv(out_trunc.parent / "_pilot_labels.csv").set_index("column")
    for col in ("EX_ID", "EX_KP", "EX_AQ"):
        assert df_trunc.loc[col, "label"] == "evidence_unresolved"
        assert "header_truncated" in df_trunc.loc[col, "evidence"]
        assert pd.isna(df_trunc.loc[col, "suggested_loss"])
    assert df_trunc.loc["EX_CB", "label"] == "evidence_unresolved"
    assert "header_unreadable" in df_trunc.loc["EX_CB", "evidence"]
    assert pd.isna(df_trunc.loc["EX_CB", "suggested_loss"])

    # Run 2: same headers WITH closing </SEC-HEADER> -> classified normally.
    comp_responses = _explicit_quarter_responses((2001, 1), (2006, 4), per_quarter)
    comp_responses.update(
        {
            early_sec.filing_url("edgar/data/200001/0001-01-000001.txt"): complete_id_body,
            early_sec.filing_url("edgar/data/200002/0001-01-000001.txt"): complete_id_kept_body,
            early_sec.filing_url("edgar/data/200003/0001-01-000001.txt"): complete_acq_body,
            early_sec.filing_url("edgar/data/200004/0001-01-000001.txt"): corrupt_binary_body,
        }
    )
    client_comp = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "cache_comp",
        min_interval=0.0,
        opener=_strict_opener(comp_responses),
        sleep=lambda _s: None,
    )
    out_comp = tmp_path / "comp" / "_pilot.json"
    code2 = run.main(
        [
            "--mode",
            "pilot",
            "--pilot-size",
            "4",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out_comp),
        ],
        loader=lambda _a: inputs,
        sec_client=client_comp,
        git_dirty=lambda: False,
    )
    assert code2 == 0
    df_comp = pd.read_csv(out_comp.parent / "_pilot_labels.csv").set_index("column")
    assert df_comp.loc["EX_ID", "label"] == "no_event_found"
    assert df_comp.loc["EX_KP", "label"] == "kept_filing"
    assert df_comp.loc["EX_AQ", "label"] == "acquisition_type"
    assert df_comp.loc["EX_AQ", "suggested_loss"] == 0.0
    assert df_comp.loc["EX_CB", "label"] == "evidence_unresolved"


def test_driver_non_404_fetch_failure_and_parse_error_propagate_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix round 2 (d)-(e): non-404 SecFetchError, corrupt index, and parser failure propagate."""
    from ef_sec_client import SecFetchError

    death_day = date(2001, 3, 12)
    per_quarter: dict[tuple[int, int], list[tuple[str, str, str, str, str]]] = {
        (2001, 1): [
            (
                "300001",
                "FAILFETCH CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/300001/0001-01-000001.txt",
            ),
        ],
    }
    inputs = _single_window_inputs([("FF01", 30001, "FAILFETCH INC", death_day)])
    base_responses = _explicit_quarter_responses((2001, 1), (2006, 4), per_quarter)
    header_url = early_sec.filing_url("edgar/data/300001/0001-01-000001.txt")

    # (d1) HTTP 500 on one filing header raises SecFetchError loudly, contact-free, no output files.
    def opener_500(req: Any, timeout: float = 30.0) -> _FakeResponse:
        url = req.full_url
        if url == header_url:
            raise urllib.error.HTTPError(url, 500, "Server Error", hdrs=None, fp=None)  # type: ignore[arg-type]
        if url in base_responses:
            return _FakeResponse(base_responses[url])
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    client_500 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "cache_500",
        min_interval=0.0,
        opener=opener_500,
        sleep=lambda _s: None,
    )
    out_500 = tmp_path / "out_500" / "_pilot.json"
    with pytest.raises(SecFetchError) as exc_500:
        run.main(
            [
                "--mode",
                "pilot",
                "--pilot-size",
                "1",
                "--release-id",
                "1",
                "--crsp-load",
                "1",
                "--out",
                str(out_500),
            ],
            loader=lambda _a: inputs,
            sec_client=client_500,
            git_dirty=lambda: False,
        )
    captured = capsys.readouterr()
    for surface in (str(exc_500.value), captured.out, captured.err):
        assert DUMMY_CONTACT not in surface
        assert DUMMY_EMAIL not in surface
    assert not out_500.exists()
    assert not (out_500.parent / "_pilot_labels.csv").exists()

    # (d2) Corrupt cached header body whose refetch fails (HTTP 500) also propagates loudly.
    cache_corrupt = tmp_path / "cache_corrupt"
    good_responses = dict(base_responses)
    good_responses[header_url] = _header_bytes(
        form="8-K",
        filed_yyyymmdd="20010315",
        company="FAILFETCH CORP",
        cik="0000300001",
        items=("Other events",),
    )
    seed_client = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_corrupt,
        min_interval=0.0,
        opener=_strict_opener(good_responses),
        sleep=lambda _s: None,
    )
    seed_client.get(header_url)
    # Tamper with cached .body so checksum fails and client attempts refetch against opener_500.
    for body_file in cache_corrupt.glob("*.body"):
        body_file.write_bytes(b"corrupted bytes")
    client_corrupt_refetch = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_corrupt,
        min_interval=0.0,
        opener=opener_500,
        sleep=lambda _s: None,
    )
    out_corrupt = tmp_path / "out_corrupt" / "_pilot.json"
    with pytest.raises(SecFetchError) as exc_corrupt:
        run.main(
            [
                "--mode",
                "pilot",
                "--pilot-size",
                "1",
                "--release-id",
                "1",
                "--crsp-load",
                "1",
                "--out",
                str(out_corrupt),
            ],
            loader=lambda _a: inputs,
            sec_client=client_corrupt_refetch,
            git_dirty=lambda: False,
        )
    assert DUMMY_CONTACT not in str(exc_corrupt.value)
    assert DUMMY_EMAIL not in str(exc_corrupt.value)
    assert not out_corrupt.exists()
    assert not (out_corrupt.parent / "_pilot_labels.csv").exists()

    # (e1) Corrupt master-index zip raises ValueError loudly out of main, writing no files.
    corrupt_index_responses = dict(base_responses)
    corrupt_index_responses[early_sec.master_index_url(2001, 1)] = b"not a valid zip archive"
    client_bad_zip = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "cache_bad_zip",
        min_interval=0.0,
        opener=_strict_opener(corrupt_index_responses),
        sleep=lambda _s: None,
    )
    out_bad_zip = tmp_path / "out_bad_zip" / "_pilot.json"
    with pytest.raises(ValueError, match="Invalid master-index archive"):
        run.main(
            [
                "--mode",
                "pilot",
                "--pilot-size",
                "1",
                "--release-id",
                "1",
                "--crsp-load",
                "1",
                "--out",
                str(out_bad_zip),
            ],
            loader=lambda _a: inputs,
            sec_client=client_bad_zip,
            git_dirty=lambda: False,
        )
    assert not out_bad_zip.exists()
    assert not (out_bad_zip.parent / "_pilot_labels.csv").exists()

    # (e2) An unexpected parser exception on a header propagates loudly, never a normal label.
    client_parse_err = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "cache_parse_err",
        min_interval=0.0,
        opener=_strict_opener(good_responses),
        sleep=lambda _s: None,
    )
    out_parse_err = tmp_path / "out_parse_err" / "_pilot.json"

    def _boom(_text: str) -> early_sec.FilingHeader:
        raise ValueError("unexpected header parse failure")

    monkeypatch.setattr(run.early_sec, "parse_filing_header", _boom)
    with pytest.raises(ValueError, match="unexpected header parse failure"):
        run.main(
            [
                "--mode",
                "pilot",
                "--pilot-size",
                "1",
                "--release-id",
                "1",
                "--crsp-load",
                "1",
                "--out",
                str(out_parse_err),
            ],
            loader=lambda _a: inputs,
            sec_client=client_parse_err,
            git_dirty=lambda: False,
        )
    assert not out_parse_err.exists()
    assert not (out_parse_err.parent / "_pilot_labels.csv").exists()


def test_driver_quoted_closing_tag_acq_then_bankruptcy_pair_and_lowercase_close(
    tmp_path: Path,
) -> None:
    """Fix round 3 (a)-(e): driver raw-response regressions through run.main + real SecLookup."""
    death_day = date(2001, 3, 12)

    # (a) Quoted mention of </SEC-HEADER> with NO later <DOCUMENT> line -> header_truncated.
    quoted_no_doc = (
        b"<SEC-HEADER>\n"
        b"CONFORMED SUBMISSION TYPE:\t8-K\n"
        b"FILED AS OF DATE:\t\t20010315\n"
        b"COMPANY CONFORMED NAME:\t\tQUOTED NODOC CORP\n"
        b"CENTRAL INDEX KEY:\t\t0000400001\n"
        b"ITEM INFORMATION:\t\tAcquisition or disposition of assets\n"
        b'Response interrupted; expected "</SEC-HEADER>" before the remainder.\n'
    )
    # (b) Same quoted mention WITH a later <DOCUMENT> line -> still header_truncated.
    quoted_with_doc = (
        b"<SEC-HEADER>\n"
        b"CONFORMED SUBMISSION TYPE:\t8-K\n"
        b"FILED AS OF DATE:\t\t20010315\n"
        b"COMPANY CONFORMED NAME:\t\tQUOTED DOC CORP\n"
        b"CENTRAL INDEX KEY:\t\t0000400002\n"
        b"ITEM INFORMATION:\t\tAcquisition or disposition of assets\n"
        b'Response interrupted; expected "</SEC-HEADER>" before the remainder.\n'
        b"<DOCUMENT>\nbody text\n"
    )
    # (c) Truncated after acquisition item, before later Bankruptcy item & closing tag.
    acq_before_bk_trunc = (
        b"<SEC-HEADER>\n"
        b"CONFORMED SUBMISSION TYPE:\t8-K\n"
        b"FILED AS OF DATE:\t\t20010315\n"
        b"COMPANY CONFORMED NAME:\t\tACQ BK PAIR CORP\n"
        b"CENTRAL INDEX KEY:\t\t0000400003\n"
        b"ITEM INFORMATION:\t\tAcquisition or disposition of assets\n"
    )
    # (d) Genuine standalone </SEC-HEADER> counterparts:
    # - quoted_no_doc with genuine standalone </SEC-HEADER> before <DOCUMENT> -> acquisition_type
    # - acq_before_bk_trunc with later Bankruptcy item + standalone </SEC-HEADER> -> bankruptcy
    quoted_complete = quoted_no_doc + b"</SEC-HEADER>\n<DOCUMENT>\nbody text\n"
    acq_then_bk_complete = (
        acq_before_bk_trunc
        + b"ITEM INFORMATION:\t\tBankruptcy or receivership\n"
        + b"</SEC-HEADER>\n<DOCUMENT>\nbody text\n"
    )
    # (e) Standalone lowercase </sec-header> line (with surrounding whitespace) -> complete.
    lowercase_complete = (
        b"<SEC-HEADER>\n"
        b"CONFORMED SUBMISSION TYPE:\t8-K\n"
        b"FILED AS OF DATE:\t\t20010315\n"
        b"COMPANY CONFORMED NAME:\t\tLOWERCASE CLOSE CORP\n"
        b"CENTRAL INDEX KEY:\t\t0000400004\n"
        b"ITEM INFORMATION:\t\tAcquisition or disposition of assets\n"
        b"  </sec-header>\t\n"
        b"<DOCUMENT>\nbody text\n"
    )

    per_quarter: dict[tuple[int, int], list[tuple[str, str, str, str, str]]] = {
        (2001, 1): [
            (
                "400001",
                "QUOTED NODOC CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/400001/0001-01-000001.txt",
            ),
            (
                "400002",
                "QUOTED DOC CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/400002/0001-01-000001.txt",
            ),
            (
                "400003",
                "ACQ BK PAIR CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/400003/0001-01-000001.txt",
            ),
            (
                "400004",
                "LOWERCASE CLOSE CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/400004/0001-01-000001.txt",
            ),
        ],
    }
    inputs = _single_window_inputs(
        [
            ("Q_ND", 40001, "QUOTED NODOC INC", death_day),
            ("Q_WD", 40002, "QUOTED DOC INC", death_day),
            ("PAIR", 40003, "ACQ BK PAIR INC", death_day),
            ("LOWR", 40004, "LOWERCASE CLOSE INC", death_day),
        ]
    )

    # Run 1: truncated / quoted-only headers -> evidence_unresolved (header_truncated);
    # lowercase standalone </sec-header> -> acquisition_type (0.0).
    resp1 = _explicit_quarter_responses((2001, 1), (2006, 4), per_quarter)
    resp1.update(
        {
            early_sec.filing_url("edgar/data/400001/0001-01-000001.txt"): quoted_no_doc,
            early_sec.filing_url("edgar/data/400002/0001-01-000001.txt"): quoted_with_doc,
            early_sec.filing_url("edgar/data/400003/0001-01-000001.txt"): acq_before_bk_trunc,
            early_sec.filing_url("edgar/data/400004/0001-01-000001.txt"): lowercase_complete,
        }
    )
    client1 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "cache_r3_trunc",
        min_interval=0.0,
        opener=_strict_opener(resp1),
        sleep=lambda _s: None,
    )
    out1 = tmp_path / "r3_trunc" / "_pilot.json"
    code1 = run.main(
        [
            "--mode",
            "pilot",
            "--pilot-size",
            "4",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out1),
        ],
        loader=lambda _a: inputs,
        sec_client=client1,
        git_dirty=lambda: False,
    )
    assert code1 == 0
    df1 = pd.read_csv(out1.parent / "_pilot_labels.csv").set_index("column")
    for col in ("Q_ND", "Q_WD", "PAIR"):
        assert df1.loc[col, "label"] == "evidence_unresolved"
        assert "header_truncated" in df1.loc[col, "evidence"]
        assert pd.isna(df1.loc[col, "suggested_loss"])
    assert df1.loc["LOWR", "label"] == "acquisition_type"
    assert df1.loc["LOWR", "confidence"] == "checked"
    assert df1.loc["LOWR", "suggested_loss"] == 0.0

    # Run 2: genuine standalone </SEC-HEADER> counterparts (with both acq + bk on PAIR).
    resp2 = _explicit_quarter_responses((2001, 1), (2006, 4), per_quarter)
    resp2.update(
        {
            early_sec.filing_url("edgar/data/400001/0001-01-000001.txt"): quoted_complete,
            early_sec.filing_url("edgar/data/400002/0001-01-000001.txt"): (
                quoted_no_doc.replace(b"QUOTED NODOC CORP", b"QUOTED DOC CORP").replace(
                    b"0000400001", b"0000400002"
                )
                + b"</SEC-HEADER>\n<DOCUMENT>\nbody text\n"
            ),
            early_sec.filing_url("edgar/data/400003/0001-01-000001.txt"): acq_then_bk_complete,
            early_sec.filing_url("edgar/data/400004/0001-01-000001.txt"): lowercase_complete,
        }
    )
    client2 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=tmp_path / "cache_r3_comp",
        min_interval=0.0,
        opener=_strict_opener(resp2),
        sleep=lambda _s: None,
    )
    out2 = tmp_path / "r3_comp" / "_pilot.json"
    code2 = run.main(
        [
            "--mode",
            "pilot",
            "--pilot-size",
            "4",
            "--release-id",
            "1",
            "--crsp-load",
            "1",
            "--out",
            str(out2),
        ],
        loader=lambda _a: inputs,
        sec_client=client2,
        git_dirty=lambda: False,
    )
    assert code2 == 0
    df2 = pd.read_csv(out2.parent / "_pilot_labels.csv").set_index("column")
    assert df2.loc["Q_ND", "label"] == "acquisition_type"
    assert df2.loc["Q_ND", "suggested_loss"] == 0.0
    assert df2.loc["Q_WD", "label"] == "acquisition_type"
    assert df2.loc["Q_WD", "suggested_loss"] == 0.0
    assert df2.loc["PAIR", "label"] == "bankruptcy"
    assert df2.loc["PAIR", "confidence"] == "checked"
    assert pd.isna(df2.loc["PAIR", "suggested_loss"])
    assert df2.loc["LOWR", "label"] == "acquisition_type"
    assert df2.loc["LOWR", "suggested_loss"] == 0.0
