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
        ({"seconds": 1.5}, {"seconds": 2.5}, True),
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
