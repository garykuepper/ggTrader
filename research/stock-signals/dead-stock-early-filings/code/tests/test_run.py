"""End-to-end tests for the dead-stock-early-filings driver and SecLookup (no network, no DB)."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from datetime import date
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
    # Higher index -> lower drift so S000..S049 are always top 50 (held_by_rule=True).
    drifts = (n_stocks - np.arange(n_stocks, dtype=float))[None, :] * 1e-4
    prices = 50.0 * np.exp(t * drifts)
    cols = [f"PERMNO-{10000 + i}" for i in range(n_stocks)]
    close = pd.DataFrame(prices, index=days, columns=cols)
    spy = pd.Series(100.0 * np.exp(t[:, 0] * 2e-4), index=days)
    eligible = pd.DataFrame(True, index=days, columns=cols)

    # Choose which stocks die inside 2000-02..2001-11.
    month_dates = pd.bdate_range("2000-02-15", "2001-11-15", freq="BMS") + pd.Timedelta(days=10)
    span_records: list[dict[str, Any]] = []
    name_records: list[dict[str, Any]] = []
    if n_held_deaths is None:
        dying_plan = [(i, month_dates[i % len(month_dates)]) for i in range(n_deaths)]
    else:
        # Held deaths die in month 0 while ranked 1..n_held_deaths; indices
        # [n_held_deaths .. n_held_deaths + 50) stay alive so every later death ranks > 50.
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
            )
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(ec.FilingEvidence(filing, header),),
                all_filing_dates=(last_day,),
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
            )
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(ec.FilingEvidence(filing, header),),
                all_filing_dates=(last_day,),
            )
        if kind == 2:
            filing = early_sec.Filing(
                form="25-NSE",
                company_name=company_name,
                cik=1000 + num,
                filed=last_day,
                file_path=f"edgar/data/{1000 + num}/0001-00-000003.txt",
            )
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(ec.FilingEvidence(filing, None),),
                all_filing_dates=(last_day,),
            )
        if kind == 3:
            return ec.CompanyEvidence(
                cik=1000 + num,
                company_name=company_name,
                match_confidence=early_sec.CONFIDENCE_NAME_ONLY,
                window_filings=(),
                all_filing_dates=(date(last_day.year + 1, last_day.month, 15),),
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

    # Verify exact seeded shuffle order over permnos.
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
    assert "sec_requests" in payload
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

    # Wrong release_id.
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

    # Wrong crsp_load.
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

    # Non-positive pilot size.
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

    # Dirty tracked tree.
    assert (
        run.main(
            base_args,
            loader=lambda _a: inputs,
            sec_lookup_factory=_fake_lookup_factory,
            git_dirty=lambda: True,
        )
        == run.EXIT_REFUSED
    )

    # Yahoo drift in inputs.
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

    # Existing JSON file refuses overwrite.
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

    # Existing CSV file also refuses overwrite.
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
    # Mismatch: only 10 deaths -> exit 3, no files written, no SEC lookup called.
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

    # Exact 251 deaths and 33 held_by_rule -> exit 0 and writes _early.json + _early_labels.csv.
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


def test_sec_lookup_streams_indexes_fetches_headers_handles_former_names_and_caches(
    tmp_path: Path,
) -> None:
    """Exercise SecLookup end-to-end against an in-memory fake opener (zero real network)."""
    header_alpha = (
        "<SEC-HEADER>\n"
        "CONFORMED SUBMISSION TYPE:\t8-K\n"
        "FILED AS OF DATE:\t\t20010315\n"
        "ITEM INFORMATION:\t\tAcquisition or disposition of assets\n"
        "COMPANY CONFORMED NAME:\t\tALPHA WIDGET CORP\n"
        "CENTRAL INDEX KEY:\t\t0000100001\n"
        "FORMER CONFORMED NAME:\t\tOLD BETA SYSTEMS INC\n"
        "</SEC-HEADER>\n<DOCUMENT>\nbody\n"
    ).encode("utf-8")
    header_eps_other = (
        "<SEC-HEADER>\n"
        "CONFORMED SUBMISSION TYPE:\t8-K\n"
        "FILED AS OF DATE:\t\t20010316\n"
        "ITEM INFORMATION:\t\tOther events\n"
        "COMPANY CONFORMED NAME:\t\tEPSILON NEWCO CORP\n"
        "CENTRAL INDEX KEY:\t\t0000100004\n"
        "FORMER CONFORMED NAME:\t\tPREDECESSOR EPSILON INC\n"
        "</SEC-HEADER>\n<DOCUMENT>\nbody\n"
    ).encode("utf-8")
    header_eps_pred = (
        "<SEC-HEADER>\n"
        "CONFORMED SUBMISSION TYPE:\t8-K\n"
        "FILED AS OF DATE:\t\t20010318\n"
        "ITEM INFORMATION:\t\tBankruptcy or receivership\n"
        "COMPANY CONFORMED NAME:\t\tPREDECESSOR EPSILON INC\n"
        "CENTRAL INDEX KEY:\t\t0000100005\n"
        "</SEC-HEADER>\n<DOCUMENT>\nbody\n"
    ).encode("utf-8")

    # Quarterly archives: Q1 2001 has ALPHA 8-K, GAMMA 8-K (404 header), DELTA 10-Q,
    # EPSILON NEWCO 8-K (other events + former name), PREDECESSOR EPSILON 8-K (bankruptcy);
    # Q1 2006 has DELTA 10-K (>180d kept filing); other quarters are empty valid archives.
    empty_zip = _make_master_zip([])
    q1_2001_zip = _make_master_zip(
        [
            (
                "100001",
                "ALPHA WIDGET CORP",
                "8-K",
                "2001-03-15",
                "edgar/data/100001/0000100001-01-000001.txt",
            ),
            (
                "100002",
                "GAMMA MISSING HDR INC",
                "8-K",
                "2001-03-20",
                "edgar/data/100002/0000100002-01-000001.txt",
            ),
            (
                "100003",
                "DELTA KEPT CORP",
                "10-Q",
                "2001-03-10",
                "edgar/data/100003/0000100003-01-000001.txt",
            ),
            (
                "100004",
                "EPSILON NEWCO CORP",
                "8-K",
                "2001-03-16",
                "edgar/data/100004/0000100004-01-000001.txt",
            ),
            (
                "100005",
                "PREDECESSOR EPSILON INC",
                "8-K",
                "2001-03-18",
                "edgar/data/100005/0000100005-01-000001.txt",
            ),
        ]
    )
    q1_2006_zip = _make_master_zip(
        [
            (
                "100003",
                "DELTA KEPT CORP",
                "10-K",
                "2006-03-15",
                "edgar/data/100003/0000100003-06-000001.txt",
            ),
        ]
    )

    responses: dict[str, bytes] = {
        early_sec.master_index_url(2001, 1): q1_2001_zip,
        early_sec.master_index_url(2006, 1): q1_2006_zip,
        early_sec.filing_url("edgar/data/100001/0000100001-01-000001.txt"): header_alpha,
        early_sec.filing_url("edgar/data/100004/0000100004-01-000001.txt"): header_eps_other,
        early_sec.filing_url("edgar/data/100005/0000100005-01-000001.txt"): header_eps_pred,
    }
    missing_url = early_sec.filing_url("edgar/data/100002/0000100002-01-000001.txt")

    def fake_opener(req: Any, timeout: float = 30.0) -> _FakeResponse:
        url = req.full_url
        if url == missing_url:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]
        if url in responses:
            return _FakeResponse(responses[url])
        if url.endswith("/master.zip"):
            return _FakeResponse(empty_zip)
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    roster = pd.DataFrame(
        {
            "column": ["A", "B", "G", "D", "E"],
            "permno": pd.Series([1, 2, 3, 4, 5], dtype="Int64"),
            "symbol": ["A", "B", "G", "D", "E"],
            "last_bar": pd.to_datetime(
                ["2001-03-10", "2001-03-12", "2001-03-15", "2001-03-10", "2001-03-14"]
            ),
            "decision_date": pd.to_datetime(
                ["2001-02-28", "2001-02-28", "2001-02-28", "2001-02-28", "2001-02-28"]
            ),
            "rank_band": ["1-50", "1-50", "51-150", "51-150", "151-500"],
            "held_by_rule": [True, True, False, False, False],
        }
    )
    names = roster.copy()
    names["company_name"] = [
        "ALPHA WIDGET INC",
        "OLD BETA SYSTEMS CORP",  # Matches ALPHA's FORMER CONFORMED NAME in Pass 1/2!
        "GAMMA MISSING HDR CORP",  # 404 header -> name_only, no_event_found
        "DELTA KEPT INC",  # 10-K in 2006 -> kept_filing
        "EPSILON NEWCO INC",  # Own header former name -> Pass 2 finds CIK 100005 bankruptcy!
    ]
    names["name_source"] = "mcgill"
    names["exchcd"] = pd.Series([1, 3, 1, 3, 1], dtype="Int64")

    cache_dir = tmp_path / "sec_cache"
    client1 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_dir,
        min_interval=0.0,
        opener=fake_opener,
        sleep=lambda _s: None,
    )
    lookup1 = run.SecLookup(client1, names)
    classified = ec.classify_roster(roster, names, lookup1).set_index("column")

    assert classified.loc["A", "label"] == "acquisition_type"
    assert classified.loc["A", "confidence"] == "checked"
    assert classified.loc["A", "suggested_loss"] == 0.0

    # B matched via ALPHA's FORMER CONFORMED NAME -> former_name tier, never checked.
    assert classified.loc["B", "label"] == "acquisition_type"
    assert classified.loc["B", "confidence"] == "former_name"
    assert classified.loc["B", "suggested_loss"] == 0.0

    # G had a 404 filing header -> stays name_only, no_event_found.
    assert classified.loc["G", "label"] == "no_event_found"
    assert classified.loc["G", "confidence"] == "name_only"

    # D filed a 10-K in 2006 -> kept_filing.
    assert classified.loc["D", "label"] == "kept_filing"
    assert classified.loc["D", "confidence"] == "name_only"

    # E matched predecessor CIK 100005 in Pass 2 via its own header's former name -> former_name.
    assert classified.loc["E", "label"] == "bankruptcy"
    assert classified.loc["E", "confidence"] == "former_name"
    assert pd.isna(classified.loc["E", "suggested_loss"])

    stats1 = lookup1.stats()
    assert stats1["network_requests"] > 0
    assert stats1["not_found_recorded"] == 1

    # Second run against the same cache_dir makes zero network requests (even for the 404).
    client2 = early_sec.EarlySecClient(
        user_agent=DUMMY_CONTACT,
        cache_dir=cache_dir,
        min_interval=0.0,
        opener=fake_opener,
        sleep=lambda _s: None,
    )
    lookup2 = run.SecLookup(client2, names)
    classified2 = ec.classify_roster(roster, names, lookup2).set_index("column")
    assert list(classified2["label"]) == list(classified["label"])
    assert list(classified2["confidence"]) == list(classified["confidence"])
    stats2 = lookup2.stats()
    assert stats2["network_requests"] == 0
    assert stats2["not_found_cache_hits"] == 1
