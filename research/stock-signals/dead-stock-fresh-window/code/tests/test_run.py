"""Tests for the dead-stock fresh-window driver: anchor check, code binding, claim and refusals."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import fp_dead_stock_first_test as fp
import fresh_inputs
import numpy as np
import pandas as pd
import pytest

from ggTrader.utils.study import load_driver as load_driver_by_path
from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study

CODE = study_dir(__file__) / "code"
SYNTH_START = "2019-09-02"
SYNTH_END = "2020-01-31"


def load_driver() -> Any:
    """Load the study driver by file path so its stem never clashes with another study's."""
    return load_driver_by_path(CODE / "run.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_window() -> dict[str, Any]:
    """Tiny 6-stock synthetic window with 3 dying stocks for fast, DB-free driver tests."""
    dates = pd.bdate_range("2018-01-02", SYNTH_END)
    n = len(dates)
    t = np.arange(n, dtype=float)
    close = pd.DataFrame(
        {
            "ALIVE-1": 50.0 * (1.0005**t),
            "ALIVE-2": 40.0 * (1.0003**t),
            "ALIVE-3": 38.0 * (1.0004**t),
            "DEAD-1": 30.0 * (1.0008**t),
            "DEAD-2": 35.0 * (1.0006**t),
            "DEAD-3": 25.0 * (1.0007**t),
        },
        index=dates,
    )
    death_dates = {
        "DEAD-1": pd.Timestamp("2019-10-15"),
        "DEAD-2": pd.Timestamp("2019-11-15"),
        "DEAD-3": pd.Timestamp("2019-12-16"),
    }
    for sym, death in death_dates.items():
        close.loc[close.index > death, sym] = np.nan

    spy_close = pd.Series(100.0 * (1.0004**t), index=dates, name="SPY")
    eligible = pd.DataFrame(True, index=dates, columns=close.columns)
    for sym, death in death_dates.items():
        eligible.loc[eligible.index > death, sym] = False

    span_rows = pd.DataFrame(
        [
            {
                "symbol": sym,
                "last_date": death,
                "has_successor": False,
                "total_loss": False,
            }
            for sym, death in death_dates.items()
        ]
    )
    release_spans = pd.DataFrame(
        [
            {"symbol": "ALIVE-1", "permno": 1, "kind": "linked_yahoo"},
            {"symbol": "ALIVE-2", "permno": 2, "kind": "placeholder"},
            {"symbol": "ALIVE-3", "permno": 3, "kind": "placeholder"},
            {"symbol": "DEAD-1", "permno": 101, "kind": "placeholder"},
            {"symbol": "DEAD-2", "permno": 102, "kind": "placeholder"},
            {"symbol": "DEAD-3", "permno": 103, "kind": "placeholder"},
        ]
    )
    return {
        "close": close,
        "spy_close": spy_close,
        "eligible": eligible,
        "span_rows": span_rows,
        "release_spans": release_spans,
        "false_death_audit": {"n_checked": 0, "n_suspect": 0, "suspects": []},
        "data_provenance": {
            "candidates": {"drift": False, "drift_symbols": [], "n_batches": 1},
            "spy": {"drift": False, "drift_symbols": [], "n_batches": 1},
            "n_candidate_columns": 6,
        },
    }


@pytest.fixture(scope="module")
def synth_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build synthetic inputs and their matching saved first-test holdout dict once per module."""
    data = _synthetic_window()
    saved_result = fp.run_study(
        data["close"],
        data["spy_close"],
        data["eligible"],
        data["span_rows"],
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=data["release_spans"],
    )
    return data, saved_result


def _setup_workspace(
    tmp_path: Path,
    saved_result: dict[str, Any],
) -> dict[str, Path]:
    """Create temporary study, results and saved-holdout paths inside ``tmp_path``."""
    s_dir = tmp_path / "study_out"
    s_dir.mkdir(parents=True, exist_ok=True)
    r_dir = tmp_path / "results_out"
    saved_path = tmp_path / "saved_holdout.json"
    saved_path.write_text(
        json.dumps({"study": "dead-stock-first-test", "result": saved_result}, indent=2),
        encoding="utf-8",
    )
    return {
        "study_dir": s_dir,
        "results_dir": r_dir,
        "saved_holdout": saved_path,
        "anchor_file": s_dir / "_anchor-A.json",
        "fresh_file": s_dir / "_fresh-A.json",
        "claim_file": r_dir / "holdout-A.claim",
    }


def _patch_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fresh_inputs,
        "WINDOWS",
        {"anchor": (SYNTH_START, SYNTH_END), "fresh": (SYNTH_START, SYNTH_END)},
    )


def _run_anchor_ok(
    driver: Any,
    paths: dict[str, Path],
    data: dict[str, Any],
) -> int:
    return driver.main(
        ["--window", "anchor", "--release-id", "1"],
        loader=lambda _w, _r, _c: copy.deepcopy(data),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        saved_holdout=paths["saved_holdout"],
    )


def _assert_no_delete_claim_advice(text: str) -> None:
    lower = text.lower()
    assert "delete" not in lower
    assert "remove the claim" not in lower


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    driver = load_driver()
    with pytest.raises(SystemExit) as done:
        driver.main(["--help"])
    assert done.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage:" in help_text.lower()
    for flag in ("--window", "--release-id", "--crsp-load"):
        assert flag in help_text
    for forbidden in ("--start", "--end", "--eval-start", "--eval-end", "--out", "--universe"):
        assert forbidden not in help_text


def test_anchor_match_writes_anchor_file(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)

    code = _run_anchor_ok(driver, paths, data)
    assert code == 0
    assert paths["anchor_file"].is_file()
    assert not paths["claim_file"].exists()
    assert (paths["results_dir"] / "anchor-A-trades.csv.gz").is_file()

    payload = json.loads(paths["anchor_file"].read_text(encoding="utf-8"))
    assert payload["anchor_ok"] is True
    assert payload["window"] == "anchor"
    assert payload["release_id"] == 1
    assert payload["crsp_load"] == 1
    assert isinstance(payload["git_head"], str) and payload["git_head"]
    assert payload["anchor_commit"] == payload["git_head"]
    assert isinstance(payload["code_hashes"], dict) and "run.py" in payload["code_hashes"]
    assert all(not name.startswith("tests/") for name in payload["code_hashes"])
    assert payload["comparison"]["differences"] == []
    assert set(payload["headline"]) == {"worst", "middle", "best"}
    assert len(payload["gap_table"]) == 18
    assert payload["result"]["strategy"]
    assert payload["result"]["spy"]
    assert payload["binding"]
    assert payload["command"]
    assert isinstance(payload["provenance"], dict) and payload["provenance"]

    # Anchor has no claim and may repeat cleanly.
    assert _run_anchor_ok(driver, paths, data) == 0
    assert not paths["claim_file"].exists()


def test_anchor_mismatch_exits_3_and_writes_nothing(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    perturbed = copy.deepcopy(saved)
    perturbed["strategy"]["loss1_cost18.3bps_taxlow"]["after_tax_final"] += 0.01
    paths = _setup_workspace(tmp_path, perturbed)

    code = _run_anchor_ok(driver, paths, data)
    assert code == 3
    err = capsys.readouterr().err
    assert "loss1_cost18.3bps_taxlow" in err
    assert not paths["anchor_file"].exists()
    assert not paths["results_dir"].exists()


def test_fresh_refused_without_anchor_file(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    _, saved = synth_bundle
    paths = _setup_workspace(tmp_path, saved)

    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run without anchor"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "_anchor-A.json" in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()
    assert not paths["fresh_file"].exists()


def test_fresh_refused_when_anchor_not_ok(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    anchor = json.loads(paths["anchor_file"].read_text(encoding="utf-8"))
    anchor["anchor_ok"] = False
    paths["anchor_file"].write_text(json.dumps(anchor, indent=2), encoding="utf-8")

    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run when anchor_ok is false"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "anchor_ok" in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()
    assert not paths["fresh_file"].exists()


def test_fresh_refused_when_code_changed_since_anchor(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    # 1. Changed path under study code/, src/, or research/stock-signals/_lib/ refuses.
    for changed in (
        ["research/stock-signals/dead-stock-fresh-window/code/fresh_inputs.py"],
        ["src/ggTrader/data/bars.py"],
        ["research/stock-signals/_lib/shared_helper.py"],
    ):
        code = driver.main(
            ["--window", "fresh", "--release-id", "1"],
            loader=lambda *_a: pytest.fail("loader must not run when code changed"),
            results_dir=paths["results_dir"],
            study_dir_=paths["study_dir"],
            git_dirty=lambda: False,
            drift_preflight=lambda _r: [],
            changed_paths=lambda _c, c=changed: c,
        )
        assert code == 2
        err = capsys.readouterr().err
        assert changed[0] in err
        _assert_no_delete_claim_advice(err)
        assert not paths["claim_file"].exists()
        assert not paths["fresh_file"].exists()

    # 2. Tampering with a stored code hash in _anchor-A.json refuses.
    anchor = json.loads(paths["anchor_file"].read_text(encoding="utf-8"))
    anchor["code_hashes"]["fresh_inputs.py"] = "0" * 64
    paths["anchor_file"].write_text(json.dumps(anchor, indent=2), encoding="utf-8")

    code_hash = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run when stored code hash mismatches"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [],
    )
    assert code_hash == 2
    err = capsys.readouterr().err
    assert "fresh_inputs.py" in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()
    assert not paths["fresh_file"].exists()

    # 3. Tampering with anchor_commit (e.g. symbolic ref "HEAD") refuses without claiming.
    anchor = json.loads(paths["anchor_file"].read_text(encoding="utf-8"))
    anchor["anchor_commit"] = "HEAD"
    paths["anchor_file"].write_text(json.dumps(anchor, indent=2), encoding="utf-8")

    code_ref = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run on symbolic anchor_commit"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
    )
    assert code_ref == 2
    err = capsys.readouterr().err
    assert "commit" in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()
    assert not paths["fresh_file"].exists()


def test_fresh_allows_non_code_changes_since_anchor(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda _w, _r, _c: copy.deepcopy(data),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [
            "research/stock-signals/dead-stock-fresh-window/README.md",
            "docs/notes.md",
        ],
    )
    assert code == 0
    assert paths["fresh_file"].is_file()
    assert paths["claim_file"].is_file()


@pytest.mark.parametrize(
    ("release_id", "crsp_load", "expected_word"),
    [(2, 1, "release-id"), (1, 2, "crsp-load")],
)
def test_fresh_refused_on_wrong_release_or_load(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    release_id: int,
    crsp_load: int,
    expected_word: str,
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    code = driver.main(
        [
            "--window",
            "fresh",
            "--release-id",
            str(release_id),
            "--crsp-load",
            str(crsp_load),
        ],
        loader=lambda *_a: pytest.fail("loader must not run on wrong pins"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert expected_word in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()
    assert not paths["fresh_file"].exists()


@pytest.mark.parametrize("window", ["anchor", "fresh"])
def test_fresh_refused_on_dirty_tree(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    window: str,
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    if window == "fresh":
        assert _run_anchor_ok(driver, paths, data) == 0

    code = driver.main(
        ["--window", window, "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run on dirty tree"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: True,
        drift_preflight=lambda _r: [],
        saved_holdout=paths["saved_holdout"],
        changed_paths=lambda _c: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "working tree" in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()
    assert not paths["fresh_file"].exists()
    if window == "anchor":
        assert not paths["anchor_file"].exists()


def test_fresh_writes_result_and_claim_once(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    claim_existed_before_read: list[bool] = []
    trades_path = paths["results_dir"] / "fresh-A-trades.csv.gz"
    write_order: list[tuple[str, bool]] = []
    real_write_json = driver.write_json

    def checking_loader(window: str, release_id: int, crsp_load: int) -> dict[str, Any]:
        assert (window, release_id, crsp_load) == ("fresh", 1, 1)
        claim_existed_before_read.append(paths["claim_file"].exists())
        return copy.deepcopy(data)

    def watching_write_json(path: Path, payload: Any, **kwargs: Any) -> Path:
        write_order.append((Path(path).name, trades_path.exists()))
        return real_write_json(path, payload, **kwargs)

    monkeypatch.setattr(driver, "write_json", watching_write_json)
    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=checking_loader,
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
    )
    assert code == 0
    assert claim_existed_before_read == [True]
    assert write_order == [("_fresh-A.json", True)]
    assert paths["claim_file"].is_file()
    assert paths["fresh_file"].is_file()
    assert trades_path.is_file()

    payload = json.loads(paths["fresh_file"].read_text(encoding="utf-8"))
    for key in (
        "headline",
        "gap_table",
        "result",
        "window",
        "release_id",
        "crsp_load",
        "git_head",
        "binding",
        "command",
        "provenance",
    ):
        assert key in payload
    assert payload["study"] == "dead-stock-fresh-window"
    assert payload["window"] == "fresh"
    assert payload["release_id"] == 1
    assert payload["crsp_load"] == 1
    assert set(payload["headline"]) == {"worst", "middle", "best"}
    assert len(payload["gap_table"]) == 18
    assert len(payload["result"]["strategy"]) == 18
    assert len(payload["result"]["spy"]) == 6
    assert payload["result"]["false_death_audit"] == data["false_death_audit"]
    assert payload["death_price_paths"] == payload["result"]["death_price_paths"]

    with gzip.open(trades_path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "cell,date,symbol,qty,price,reason"
    assert len(lines) > 1


def test_second_fresh_refused_naming_file(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    assert (
        driver.main(
            ["--window", "fresh", "--release-id", "1"],
            loader=lambda _w, _r, _c: copy.deepcopy(data),
            results_dir=paths["results_dir"],
            study_dir_=paths["study_dir"],
            git_dirty=lambda: False,
            drift_preflight=lambda _r: [],
            changed_paths=lambda _c: [],
        )
        == 0
    )
    first_bytes = paths["fresh_file"].read_bytes()

    # Second run refuses naming an existing file without touching anything.
    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run on second fresh attempt"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert str(paths["fresh_file"]) in err or str(paths["claim_file"]) in err
    _assert_no_delete_claim_advice(err)
    assert paths["fresh_file"].read_bytes() == first_bytes

    # Even if only _fresh-A.json exists (no claim file), it refuses without creating a claim.
    paths["claim_file"].unlink()
    code_only_result = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run when _fresh-A.json exists"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [],
    )
    assert code_only_result == 2
    err = capsys.readouterr().err
    assert str(paths["fresh_file"]) in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()


def test_claim_left_after_crash_blocks_fresh(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    def crashing_loader(_w: str, _r: int, _c: int) -> dict[str, Any]:
        raise RuntimeError("simulated mid-read crash")

    with pytest.raises(RuntimeError, match="simulated mid-read crash"):
        driver.main(
            ["--window", "fresh", "--release-id", "1"],
            loader=crashing_loader,
            results_dir=paths["results_dir"],
            study_dir_=paths["study_dir"],
            git_dirty=lambda: False,
            drift_preflight=lambda _r: [],
            changed_paths=lambda _c: [],
        )
    assert paths["claim_file"].is_file()
    assert not paths["fresh_file"].exists()

    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run when claim exists after crash"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert str(paths["claim_file"]) in err
    _assert_no_delete_claim_advice(err)
    assert not paths["fresh_file"].exists()


def test_fresh_drift_before_read_refuses_without_claiming(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run after preflight drift"),
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: ["LNK1", "LNK2"],
        changed_paths=lambda _c: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "drift" in err and "LNK1" in err
    _assert_no_delete_claim_advice(err)
    assert not paths["claim_file"].exists()
    assert not paths["fresh_file"].exists()


def test_fresh_drift_after_read_keeps_claim_and_writes_no_result(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, saved = synth_bundle
    _patch_windows(monkeypatch)
    paths = _setup_workspace(tmp_path, saved)
    assert _run_anchor_ok(driver, paths, data) == 0

    def drifting_loader(_w: str, _r: int, _c: int) -> dict[str, Any]:
        loaded = copy.deepcopy(data)
        loaded["data_provenance"] = {
            "candidates": {"drift": True, "drift_symbols": ["XYZ"]},
            "spy": {"drift": False, "drift_symbols": []},
        }
        return loaded

    code = driver.main(
        ["--window", "fresh", "--release-id", "1"],
        loader=drifting_loader,
        results_dir=paths["results_dir"],
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        changed_paths=lambda _c: [],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "XYZ" in err and "drift" in err
    _assert_no_delete_claim_advice(err)
    assert paths["claim_file"].is_file()
    assert not paths["fresh_file"].exists()
    assert not (paths["results_dir"] / "fresh-A-trades.csv.gz").exists()


def test_output_inside_first_test_folder_refused(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    _, saved = synth_bundle
    paths = _setup_workspace(tmp_path, saved)

    protected_study_dirs = (
        driver.STUDY_DIR.parent / "dead-stock-first-test",
        driver.STUDY_DIR.parent / "dead-stock-payout-rerun",
        driver.STUDY_DIR.parent / "dead-stock-data-integrity",
    )
    for forbidden in protected_study_dirs:
        code = driver.main(
            ["--window", "anchor", "--release-id", "1"],
            loader=lambda *_a: pytest.fail("loader must not run for protected study dir"),
            results_dir=paths["results_dir"],
            study_dir_=forbidden,
            git_dirty=lambda: False,
            drift_preflight=lambda _r: [],
            saved_holdout=paths["saved_holdout"],
        )
        assert code == 2
        err = capsys.readouterr().err
        assert forbidden.name in err
        assert not (forbidden / "_anchor-A.json").exists()

    # Symlink resolving into dead-stock-first-test is also refused.
    link_dir = tmp_path / "alias_to_first_test"
    link_dir.symlink_to(driver.STUDY_DIR.parent / "dead-stock-first-test")
    code_link = driver.main(
        ["--window", "anchor", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run for symlink to first-test"),
        results_dir=paths["results_dir"],
        study_dir_=link_dir,
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        saved_holdout=paths["saved_holdout"],
    )
    assert code_link == 2
    assert "dead-stock-first-test" in capsys.readouterr().err

    # Protected results dir is refused too.
    forbidden_results = driver.ROOT / "results" / "dead_stock_first_test"
    code_res = driver.main(
        ["--window", "anchor", "--release-id", "1"],
        loader=lambda *_a: pytest.fail("loader must not run for protected results dir"),
        results_dir=forbidden_results,
        study_dir_=paths["study_dir"],
        git_dirty=lambda: False,
        drift_preflight=lambda _r: [],
        saved_holdout=paths["saved_holdout"],
    )
    assert code_res == 2
    assert "dead_stock_first_test" in capsys.readouterr().err
