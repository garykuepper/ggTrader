"""Check evidence comparisons."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/compare_evidence_json.py"


def run(a: Path, b: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(a), str(b)], capture_output=True, text=True
    )


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        ({"provenance": {"git_sha": "a"}}, {"provenance": {"git_sha": "b"}}, True),
        ({"provenance": {"tape_sha": "a"}}, {"provenance": {"tape_sha": "b"}}, False),
        ({"seconds": 1.5}, {"seconds": 2.5}, False),
        ({"a": 0.0}, {"a": -0.0}, False),
        ({"other": {"seconds": 1.5}}, {"other": {"seconds": 2.5}}, False),
        ({"other": {"wall_seconds": 1}}, {"other": {"wall_seconds": 2}}, False),
        ({"a": True}, {"a": 1}, False),
        ({"a": 1}, {"a": 1.0}, False),
        ({"a": 0.1}, {"a": 0.10000000000000002}, False),
        ({"verdict": "DIFFERENCES - classify"}, {"verdict": "DIFFERENCES - classify"}, True),
        ({"ok": False}, {"ok": False}, False),
    ],
)
def test_json_values(tmp_path: Path, left: object, right: object, equal: bool) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.json").write_text(json.dumps(left))
    (b / "x.json").write_text(json.dumps(right))
    assert (run(a, b).returncode == 0) is equal


def test_nested_and_files(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert run(a, b).returncode == 1
    (a / "x.json").write_text('{"a":{"b":[1,2]}}')
    (b / "x.json").write_text('{"a":{"b":[1,3]}}')
    assert "x.json: a.b[1]" in run(a, b).stdout
    (b / "x.json").write_text((a / "x.json").read_text())
    assert run(a, b).returncode == 0
    (a / "x.csv").write_bytes(b"a\n")
    assert run(a, b).returncode == 1
    (b / "x.csv").write_bytes(b"a\r\n")
    assert run(a, b).returncode == 1


def test_compare_peak(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "_p5_compare_x.json").write_text('{"peak_rss_mb":{"new":1}}')
    (b / "_p5_compare_x.json").write_text('{"peak_rss_mb":{"new":2}}')
    assert run(a, b).returncode == 0


def test_seconds_ignored_in_p7_json_only(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "p7.json").write_text('{"seconds": 1.5}')
    (b / "p7.json").write_text('{"seconds": 2.5}')
    assert run(a, b).returncode == 0


def test_nan_equals_nan_and_duplicate_keys_differ(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.json").write_text('{"a": NaN}')
    (b / "x.json").write_text('{"a": NaN}')
    assert run(a, b).returncode == 0
    (a / "x.json").write_text('{"a": 1, "a": 2}')
    (b / "x.json").write_text('{"a": 2}')
    assert run(a, b).returncode == 1


def test_tape_path_is_ignored_only_in_p5_result_files(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for directory, tape_path in ((a, "/a/t.parquet"), (b, "/b/t.parquet")):
        (directory / "_p5_run.json").write_text(
            json.dumps({"tape": {"path": tape_path, "sha256": "s1"}})
        )
    assert run(a, b).returncode == 0
    for directory in (a, b):
        (directory / "_p5_run.json").rename(directory / "x.json")
    assert run(a, b).returncode == 1


def test_tape_hash_difference_is_reported_in_p5_results(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for directory, digest in ((a, "s1"), (b, "s2")):
        (directory / "_p5_run.json").write_text(
            json.dumps({"tape": {"path": "/a", "sha256": digest}})
        )
    assert "tape.sha256" in run(a, b).stdout
