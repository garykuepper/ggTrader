"""Tests for the dead-stock payout rerun driver: preconditions, gates, provenance and safety."""

from __future__ import annotations

import ast
import builtins
import copy
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import ft_dead_stock_first_test as ft
import numpy as np
import pandas as pd
import pytest

from ggTrader.utils.study import load_driver as load_driver_by_path
from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study

CODE = study_dir(__file__) / "code"
SYNTH_START = "2019-09-02"
SYNTH_END = "2020-01-31"

# Subset of repo MODULE_CALLS allowed at module import time (mirrors test_scripts_have_help.py).
ALLOWED_MODULE_CALLS: frozenset[str] = frozenset(
    {
        "bootstrap",
        "study_dir",
        "repo_root",
        "Path",
        "Path(__file__).resolve",
        "re.compile",
        "frozenset",
        "set",
        "tuple",
        "dict",
        "str",
        "logging.getLogger",
        "os.environ.get",
        "date",
        "pd.Timestamp",
    }
)
PRE_PARSE_CALLS: frozenset[str] = frozenset(
    {
        "argparse.ArgumentParser",
        "ArgumentParser",
        "parser.add_argument",
        "parser.parse_args",
        "build_parser",
        "sorted",
        "Path",
        "str",
    }
)


def load_driver() -> Any:
    """Load the study driver by file path so its stem never clashes with another study's."""
    return load_driver_by_path(CODE / "run.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_window() -> dict[str, Any]:
    """Tiny 6-stock synthetic window with 4 dying stocks (confirmed, name_only, blank, missing)."""
    dates = pd.bdate_range("2018-01-02", SYNTH_END)
    n = len(dates)
    t = np.arange(n, dtype=float)
    close = pd.DataFrame(
        {
            "ALIVE-1": 50.0 * (1.0005**t),
            "ALIVE-2": 40.0 * (1.0003**t),
            "PERMNO-101": 30.0 * (1.0008**t),
            "PERMNO-102": 35.0 * (1.0006**t),
            "PERMNO-103": 25.0 * (1.0007**t),
            "PERMNO-104~2": 45.0 * (1.0004**t),
        },
        index=dates,
    )
    death_dates = {
        "PERMNO-101": pd.Timestamp("2019-10-15"),
        "PERMNO-102": pd.Timestamp("2019-11-15"),
        "PERMNO-103": pd.Timestamp("2019-12-16"),
        "PERMNO-104~2": pd.Timestamp("2020-01-15"),
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
            {"symbol": "PERMNO-101", "permno": 101, "kind": "placeholder"},
            {"symbol": "PERMNO-102", "permno": 102, "kind": "placeholder"},
            {"symbol": "PERMNO-103", "permno": 103, "kind": "placeholder"},
            {"symbol": "PERMNO-104~2", "permno": 104, "kind": "spell"},
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


def _write_payouts_csv(path: Path, loss_101: str = "0.0") -> tuple[Path, str]:
    """Write a tiny synthetic payout table and return ``(path, sha256)``."""
    rows = [
        {"symbol": "PERMNO-101", "suggested_loss": loss_101, "match_confidence": "confirmed"},
        {"symbol": "PERMNO-102", "suggested_loss": "0.0", "match_confidence": "name_only"},
        {"symbol": "PERMNO-103", "suggested_loss": "", "match_confidence": "confirmed"},
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["symbol", "suggested_loss", "match_confidence"])
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path, _sha256(path)


@pytest.fixture(scope="module")
def synth_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build synthetic inputs and their matching first-test result once per module."""
    data = _synthetic_window()
    first_test = ft.run_study(
        data["close"],
        data["spy_close"],
        data["eligible"],
        data["span_rows"],
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=data["release_spans"],
    )
    return data, first_test


def _setup_fixture_files(
    tmp_path: Path,
    first_test: dict[str, Any],
    *,
    window: str = "dev",
    loss_101: str = "0.0",
) -> dict[str, Any]:
    """Create temporary payouts CSV, first-test JSON, results dir and output path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    payouts_path, payouts_hash = _write_payouts_csv(tmp_path / "payouts.csv", loss_101=loss_101)
    ft_dir = tmp_path / "first_test"
    ft_dir.mkdir(parents=True, exist_ok=True)
    ft_file = ft_dir / f"_{window}-A.json"
    ft_file.write_text(
        json.dumps({"study": "dead-stock-first-test", "result": first_test}, indent=2),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    out_path = tmp_path / f"_rerun-{window}-A.json"
    return {
        "payouts_path": payouts_path,
        "payouts_hash": payouts_hash,
        "ft_dir": ft_dir,
        "ft_file": ft_file,
        "results_dir": results_dir,
        "out_path": out_path,
    }


def _base_argv(paths: dict[str, Any], *, window: str = "dev") -> list[str]:
    return [
        "--window",
        window,
        "--release-id",
        "1",
        "--payouts",
        str(paths["payouts_path"]),
        "--out",
        str(paths["out_path"]),
    ]


def test_help_works_and_there_are_no_date_flags(capsys: pytest.CaptureFixture[str]) -> None:
    driver = load_driver()
    with pytest.raises(SystemExit) as done:
        driver.main(["--help"])
    assert done.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage:" in help_text.lower()
    for flag in (
        "--window",
        "--release-id",
        "--crsp-load",
        "--payouts",
        "--out",
        "--diagnostics",
        "--force",
        "--allow-dirty",
    ):
        assert flag in help_text
    for forbidden in ("--start", "--end", "--eval-start", "--eval-end", "--universe"):
        assert forbidden not in help_text
    parser = driver.build_parser()
    for bad in (
        ["--window", "dev", "--release-id", "1", "--eval-start", "2005-01-03"],
        ["--window", "dev", "--release-id", "1", "--start", "2005-01-03"],
        ["--window", "dev", "--release-id", "1", "--universe", "A"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


@pytest.mark.parametrize("window", ["dev", "holdout"])
@pytest.mark.parametrize(
    ("release_id", "crsp_load", "expected_word"),
    [(2, 1, "release-id"), (1, 2, "crsp-load")],
)
def test_refuses_wrong_release_or_crsp_load(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    window: str,
    release_id: int,
    crsp_load: int,
    expected_word: str,
) -> None:
    driver = load_driver()
    _, first_test = synth_bundle
    paths = _setup_fixture_files(tmp_path, first_test, window=window)
    argv = [
        "--window",
        window,
        "--release-id",
        str(release_id),
        "--crsp-load",
        str(crsp_load),
        "--payouts",
        str(paths["payouts_path"]),
        "--out",
        str(paths["out_path"]),
    ]
    code = driver.main(
        argv,
        loader=lambda _a: pytest.fail("loader must not run when pins fail"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 2
    assert expected_word in capsys.readouterr().err
    assert not paths["out_path"].exists()
    assert not paths["results_dir"].exists()


def test_refuses_dirty_tree_unless_allowed(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, first_test = synth_bundle
    monkeypatch.setattr(ft, "WINDOWS", {**ft.WINDOWS, "dev": (SYNTH_START, SYNTH_END)})
    paths = _setup_fixture_files(tmp_path, first_test, window="dev")

    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run on dirty tree"),
        git_dirty=lambda: True,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 2
    assert "working tree" in capsys.readouterr().err
    assert not paths["out_path"].exists()

    ok_code = driver.main(
        [*_base_argv(paths), "--allow-dirty"],
        loader=lambda _a: copy.deepcopy(data),
        git_dirty=lambda: True,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert ok_code == 0
    payload = json.loads(paths["out_path"].read_text(encoding="utf-8"))
    assert payload["provenance"]["git_dirty"] is True
    assert payload["provenance"]["allow_dirty"] is True


def test_refuses_payouts_with_a_different_hash(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    _, first_test = synth_bundle
    paths = _setup_fixture_files(tmp_path, first_test, window="dev")

    # No expected_payouts_sha256 override: synthetic file hash differs from PAYOUTS_SHA256.
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run when payout hash fails"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
    )
    assert code == 2
    assert "SHA-256" in capsys.readouterr().err
    assert not paths["out_path"].exists()

    # Missing payouts file is also refused with exit 2.
    missing = tmp_path / "does_not_exist.csv"
    argv = [
        "--window",
        "dev",
        "--release-id",
        "1",
        "--payouts",
        str(missing),
        "--out",
        str(paths["out_path"]),
    ]
    code_missing = driver.main(
        argv,
        loader=lambda _a: pytest.fail("loader must not run when payouts missing"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
    )
    assert code_missing == 2
    assert "cannot read payout table" in capsys.readouterr().err


@pytest.mark.parametrize("dropped", ["symbol", "suggested_loss", "match_confidence"])
def test_refuses_payouts_missing_a_column(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    dropped: str,
) -> None:
    driver = load_driver()
    _, first_test = synth_bundle
    paths = _setup_fixture_files(tmp_path, first_test, window="dev")
    cols = [c for c in ("symbol", "suggested_loss", "match_confidence") if c != dropped]
    bad_csv = tmp_path / f"missing_{dropped}.csv"
    bad_csv.write_text(",".join(cols) + "\n" + ",".join(["x"] * len(cols)) + "\n", encoding="utf-8")
    bad_hash = _sha256(bad_csv)
    argv = [
        "--window",
        "dev",
        "--release-id",
        "1",
        "--payouts",
        str(bad_csv),
        "--out",
        str(paths["out_path"]),
    ]
    code = driver.main(
        argv,
        loader=lambda _a: pytest.fail("loader must not run on bad payouts columns"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=bad_hash,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "missing required column" in err and dropped in err


def test_refuses_missing_or_malformed_first_test_file(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    _, first_test = synth_bundle
    paths = _setup_fixture_files(tmp_path, first_test, window="dev")

    # 1. Missing file.
    paths["ft_file"].unlink()
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run when first-test file missing"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 2
    assert "cannot read first-test" in capsys.readouterr().err

    # 2. Invalid JSON.
    paths["ft_file"].write_text("{not-json", encoding="utf-8")
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run on bad JSON"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 2
    assert "not valid JSON" in capsys.readouterr().err

    # 3. Valid JSON missing result.strategy.
    paths["ft_file"].write_text(json.dumps({"study": "x", "result": {"spy": {}}}), encoding="utf-8")
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run without result.strategy"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 2
    assert "result.strategy" in capsys.readouterr().err


@pytest.mark.parametrize("window", ["dev", "holdout"])
def test_refuses_drift_on_both_windows(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    window: str,
) -> None:
    driver = load_driver()
    data, first_test = synth_bundle
    paths = _setup_fixture_files(tmp_path, first_test, window=window)

    # 1. Preflight drift refuses before loader runs.
    code_pre = driver.main(
        _base_argv(paths, window=window),
        loader=lambda _a: pytest.fail("loader must not run after preflight drift"),
        git_dirty=lambda: False,
        drift_preflight=lambda _a: ["LNK1", "LNK2"],
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code_pre == 2
    err = capsys.readouterr().err
    assert "drift" in err and "LNK1" in err
    assert not paths["out_path"].exists()
    assert not paths["results_dir"].exists()

    # 2. Drift reported inside loaded close-read provenance refuses and writes nothing.
    def drifting_loader(_a: Any) -> dict[str, Any]:
        loaded = copy.deepcopy(data)
        loaded["data_provenance"] = {
            "candidates": {"drift": True, "drift_symbols": ["XYZ"]},
            "spy": {"drift": False, "drift_symbols": []},
        }
        return loaded

    code_read = driver.main(
        _base_argv(paths, window=window),
        loader=drifting_loader,
        git_dirty=lambda: False,
        drift_preflight=lambda _a: [],
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code_read == 2
    err = capsys.readouterr().err
    assert "drift" in err and "XYZ" in err
    assert not paths["out_path"].exists()
    assert not paths["results_dir"].exists()


def test_refuses_existing_output_without_force(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, first_test = synth_bundle
    monkeypatch.setattr(ft, "WINDOWS", {**ft.WINDOWS, "dev": (SYNTH_START, SYNTH_END)})
    paths = _setup_fixture_files(tmp_path, first_test, window="dev")

    # 1. Existing result JSON without --force.
    paths["out_path"].write_text("KEEP ME", encoding="utf-8")
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run when output exists"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 2
    assert "--force" in capsys.readouterr().err
    assert paths["out_path"].read_text(encoding="utf-8") == "KEEP ME"

    # 2. Existing trades side file without --force.
    paths["out_path"].unlink()
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    trades_file = paths["results_dir"] / "dev-A-trades.csv.gz"
    trades_file.write_bytes(b"OLD TRADES")
    code_trades = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run when trades file exists"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code_trades == 2
    assert "--force" in capsys.readouterr().err
    assert trades_file.read_bytes() == b"OLD TRADES"

    # 3. With --force, both are overwritten cleanly.
    paths["out_path"].write_text("OLD JSON", encoding="utf-8")
    ok = driver.main(
        [*_base_argv(paths), "--force"],
        loader=lambda _a: copy.deepcopy(data),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert ok == 0
    assert json.loads(paths["out_path"].read_text(encoding="utf-8"))["study"] == (
        "dead-stock-payout-rerun"
    )
    assert trades_file.read_bytes() != b"OLD TRADES"


def test_refuses_output_inside_first_test_results_folder(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    _, first_test = synth_bundle
    paths = _setup_fixture_files(tmp_path, first_test, window="dev")

    # 1. --out inside results/dead_stock_first_test.
    forbidden_out = driver.FIRST_TEST_RESULTS_DIR / "should_not_exist.json"
    code = driver.main(
        [
            "--window",
            "dev",
            "--release-id",
            "1",
            "--payouts",
            str(paths["payouts_path"]),
            "--out",
            str(forbidden_out),
        ],
        loader=lambda _a: pytest.fail("loader must not run for forbidden output folder"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 2
    assert "dead_stock_first_test" in capsys.readouterr().err
    assert not forbidden_out.exists()

    # 2. Trades side file folder set to results/dead_stock_first_test.
    code_trades = driver.main(
        _base_argv(paths),
        loader=lambda _a: pytest.fail("loader must not run for forbidden trades folder"),
        git_dirty=lambda: False,
        results_dir=driver.FIRST_TEST_RESULTS_DIR,
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code_trades == 2
    assert "dead_stock_first_test" in capsys.readouterr().err

    # 3. Symlinked output path is refused even with --force.
    real_target = tmp_path / "real_target.json"
    real_target.write_text("UNTOUCHED", encoding="utf-8")
    link_out = tmp_path / "link_out.json"
    link_out.symlink_to(real_target)
    code_link = driver.main(
        [
            "--window",
            "dev",
            "--release-id",
            "1",
            "--payouts",
            str(paths["payouts_path"]),
            "--out",
            str(link_out),
            "--force",
        ],
        loader=lambda _a: pytest.fail("loader must not run on symlinked output"),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code_link == 2
    assert "symlink" in capsys.readouterr().err
    assert real_target.read_text(encoding="utf-8") == "UNTOUCHED"


def test_never_creates_reads_or_deletes_claim_files(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    data, first_test = synth_bundle
    monkeypatch.setattr(
        ft,
        "WINDOWS",
        {"dev": (SYNTH_START, SYNTH_END), "holdout": (SYNTH_START, SYNTH_END)},
    )

    # Create fake first-test results dir with .claim files and also put a .claim in first_test_dir.
    fake_ft_results = tmp_path / "fake_first_test_results"
    fake_ft_results.mkdir()
    claim_a = fake_ft_results / "holdout-A.claim"
    claim_b = fake_ft_results / "holdout-B.claim"
    claim_a.write_text('{"locked": "A"}\n', encoding="utf-8")
    claim_b.write_text('{"locked": "B"}\n', encoding="utf-8")
    before = {p: (p.read_bytes(), os.stat(p).st_mtime_ns) for p in (claim_a, claim_b)}
    monkeypatch.setattr(driver, "FIRST_TEST_RESULTS_DIR", fake_ft_results)

    # Guard every file open/read so any access to a .claim path fails immediately.
    real_open = builtins.open
    real_path_open = Path.open
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def guard_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file).endswith(".claim"):
            raise AssertionError(f"must never open claim file: {file}")
        return real_open(file, *args, **kwargs)

    def guard_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.suffix == ".claim":
            raise AssertionError(f"must never open claim file: {self}")
        return real_path_open(self, *args, **kwargs)

    def guard_read_bytes(self: Path) -> bytes:
        if self.suffix == ".claim":
            raise AssertionError(f"must never read claim file: {self}")
        return real_read_bytes(self)

    def guard_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.suffix == ".claim":
            raise AssertionError(f"must never read claim file: {self}")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard_open)
    monkeypatch.setattr(Path, "open", guard_path_open)
    monkeypatch.setattr(Path, "read_bytes", guard_read_bytes)
    monkeypatch.setattr(Path, "read_text", guard_read_text)

    for window in ("dev", "holdout"):
        paths = _setup_fixture_files(tmp_path / window, first_test, window=window)
        code = driver.main(
            _base_argv(paths, window=window),
            loader=lambda _a: copy.deepcopy(data),
            git_dirty=lambda: False,
            results_dir=paths["results_dir"],
            first_test_dir=paths["ft_dir"],
            expected_payouts_sha256=paths["payouts_hash"],
        )
        assert code == 0
        assert list(paths["results_dir"].glob("*.claim")) == []
        assert list((tmp_path / window).glob("*.claim")) == []

    monkeypatch.undo()
    for path, (old_bytes, old_mtime) in before.items():
        assert path.exists()
        assert path.read_bytes() == old_bytes
        assert os.stat(path).st_mtime_ns == old_mtime
    assert sorted(p.name for p in fake_ft_results.iterdir()) == [
        "holdout-A.claim",
        "holdout-B.claim",
    ]


def test_writes_side_file_then_result_with_provenance(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = load_driver()
    data, first_test = synth_bundle
    monkeypatch.setattr(ft, "WINDOWS", {**ft.WINDOWS, "dev": (SYNTH_START, SYNTH_END)})
    paths = _setup_fixture_files(tmp_path, first_test, window="dev")

    trades_file = paths["results_dir"] / "dev-A-trades.csv.gz"
    write_order: list[tuple[str, bool]] = []
    real_write_json = driver.write_json

    def watching_write_json(path: Path, payload: Any, **kwargs: Any) -> Path:
        write_order.append((Path(path).name, trades_file.exists()))
        return real_write_json(path, payload, **kwargs)

    monkeypatch.setattr(driver, "write_json", watching_write_json)
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: copy.deepcopy(data),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 0
    # Trades side file was already on disk when the result JSON was written.
    assert write_order == [("_rerun-dev-A.json", True)]

    payload = json.loads(paths["out_path"].read_text(encoding="utf-8"))
    assert payload["study"] == "dead-stock-payout-rerun"
    result = payload["result"]
    assert list(result.keys())[0] == "anchor"
    assert result["anchor_ok"] is True
    assert result["identity_check"]["identity_ok"] is True
    assert len(result["strategy"]) == 36
    assert len(result["spy"]) == 6
    assert result["false_death_audit"] == data["false_death_audit"]

    prov = payload["provenance"]
    expected_prov_keys = {
        "release_id",
        "crsp_load",
        "git_head",
        "git_dirty",
        "allow_dirty",
        "brief_commit",
        "window",
        "window_name",
        "universe",
        "crsp_end",
        "command",
        "payouts_sha256",
        "payouts_path",
        "first_test_file",
        "first_test_file_sha256",
        "data",
        "drift_symbols",
        "binding",
        "code_sha256",
    }
    assert expected_prov_keys <= set(prov.keys())
    assert prov["release_id"] == 1
    assert prov["crsp_load"] == 1
    assert prov["git_dirty"] is False
    assert prov["allow_dirty"] is False
    assert prov["universe"] == "A"
    assert prov["window_name"] == "dev"
    assert prov["window"] == {"start": SYNTH_START, "end": SYNTH_END}
    assert prov["crsp_end"] == ft.CRSP_END
    assert prov["payouts_sha256"] == paths["payouts_hash"]
    assert prov["first_test_file_sha256"] == _sha256(paths["ft_file"])
    assert prov["drift_symbols"] == []

    expected_code_files = {
        "run.py",
        "payout_prices.py",
        "payout_rerun.py",
        "ft_replay.py",
        "ft_ledger_run.py",
        "ft_bars_reader.py",
        "ft_universe_frames.py",
        "ft_dead_stock_first_test.py",
    }
    assert set(prov["code_sha256"].keys()) == expected_code_files
    for filename, digest in prov["code_sha256"].items():
        assert digest == _sha256(CODE / filename)

    with gzip.open(trades_file, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "cell,date,symbol,qty,price,reason"
    cells = {line.split(",", 1)[0] for line in lines[1:]}
    assert len(cells) == 42


def test_anchor_mismatch_exits_3_and_writes_nothing(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, first_test = synth_bundle
    monkeypatch.setattr(ft, "WINDOWS", {**ft.WINDOWS, "dev": (SYNTH_START, SYNTH_END)})

    perturbed = copy.deepcopy(first_test)
    perturbed["strategy"]["loss1_cost18.3bps_taxlow"]["after_tax_final"] += 0.01
    paths = _setup_fixture_files(tmp_path, perturbed, window="dev")

    # Without --diagnostics: exit 3, plain stderr, nothing written.
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: copy.deepcopy(data),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 3
    err = capsys.readouterr().err
    assert "loss1_cost18.3bps_taxlow" in err
    assert "the cause must be found before anything is published" in err
    assert not paths["out_path"].exists()
    assert not paths["results_dir"].exists()

    # With --diagnostics: still no result or side file, only diagnostics JSON written.
    diag_path = tmp_path / "diag_anchor.json"
    code_diag = driver.main(
        [*_base_argv(paths), "--diagnostics", str(diag_path)],
        loader=lambda _a: copy.deepcopy(data),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code_diag == 3
    assert not paths["out_path"].exists()
    assert not paths["results_dir"].exists()
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "anchor" in diag
    assert "identity_check" not in diag
    assert diag["anchor"]["strategy"]["loss1_cost18.3bps_taxlow"]["matches"] is False


def test_identity_mismatch_exits_3_and_writes_nothing(
    tmp_path: Path,
    synth_bundle: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = load_driver()
    data, first_test = synth_bundle
    monkeypatch.setattr(ft, "WINDOWS", {**ft.WINDOWS, "dev": (SYNTH_START, SYNTH_END)})
    paths = _setup_fixture_files(tmp_path, first_test, window="dev", loss_101="0.3")

    # Without --diagnostics: exit 3, plain stderr, nothing written.
    code = driver.main(
        _base_argv(paths),
        loader=lambda _a: copy.deepcopy(data),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code == 3
    err = capsys.readouterr().err
    assert "identity check failed" in err
    assert "the cause must be found before anything is published" in err
    assert not paths["out_path"].exists()
    assert not paths["results_dir"].exists()

    # With --diagnostics: still no result or side file, diagnostics carries anchor + identity_check.
    diag_path = tmp_path / "diag_identity.json"
    code_diag = driver.main(
        [*_base_argv(paths), "--diagnostics", str(diag_path)],
        loader=lambda _a: copy.deepcopy(data),
        git_dirty=lambda: False,
        results_dir=paths["results_dir"],
        first_test_dir=paths["ft_dir"],
        expected_payouts_sha256=paths["payouts_hash"],
    )
    assert code_diag == 3
    assert not paths["out_path"].exists()
    assert not paths["results_dir"].exists()
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert set(diag.keys()) == {"anchor", "identity_check", "first_test_range"}
    assert diag["identity_check"]["identity_ok"] is False
    assert len(diag["identity_check"]["mismatches"]) == 12


def test_real_payout_table_hash_matches_the_constant() -> None:
    driver = load_driver()
    real_path = driver.DEFAULT_PAYOUTS
    assert real_path.is_file(), f"expected real payout table at {real_path}"
    raw = real_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == driver.PAYOUTS_SHA256
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    assert driver.REQUIRED_PAYOUT_COLUMNS <= set(reader.fieldnames or ())


def test_driver_passes_the_repo_script_rules() -> None:
    """Verify run.py obeys static script rules: guard, ArgumentParser, parse-first, no work."""
    source = (CODE / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    def is_main_guard(node: ast.stmt) -> bool:
        return (
            isinstance(node, ast.If)
            and ast.unparse(node.test).replace('"', "'") == "__name__ == '__main__'"
        )

    assert any(is_main_guard(stmt) for stmt in tree.body)

    # Check import-time calls: top-level statements, decorators, defaults and annotations.
    import_calls: list[str] = []

    class ImportVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            import_calls.append(ast.unparse(node.func))
            self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            if is_main_guard(node):
                return
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for dec in node.decorator_list:
                self.visit(dec)
            for expr in [*node.args.defaults, *node.args.kw_defaults]:
                if expr is not None:
                    self.visit(expr)
            for arg in [*node.args.args, *node.args.kwonlyargs]:
                if arg.annotation is not None:
                    self.visit(arg.annotation)
            if node.returns is not None:
                self.visit(node.returns)

    ImportVisitor().visit(tree)
    disallowed = [call for call in import_calls if call not in ALLOWED_MODULE_CALLS]
    assert disallowed == []

    # Check main() reaches parse_args before any non-parser call.
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "main" in funcs
    all_calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(c.endswith("ArgumentParser") for c in all_calls)

    reached_parse = False
    for stmt in funcs["main"].body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        calls = [ast.unparse(n.func) for n in ast.walk(stmt) if isinstance(n, ast.Call)]
        if any(c.endswith("parse_args") for c in calls):
            reached_parse = True
            break
        assert all(c in PRE_PARSE_CALLS for c in calls), f"work before parse_args: {calls}"
    assert reached_parse
