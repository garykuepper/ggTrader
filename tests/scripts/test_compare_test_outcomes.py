"""Check JUnit outcome comparisons."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/compare_test_outcomes.py"


def test_outcome_rules(tmp_path: Path) -> None:
    baseline, new = tmp_path / "base.xml", tmp_path / "new.xml"
    baseline.write_text('<testsuite><testcase classname="tests.test_a" name="test_a"/></testsuite>')

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(baseline), str(new), *args],
            capture_output=True,
            text=True,
        )

    new.write_text(baseline.read_text())
    assert run().returncode == 0
    new.write_text(
        '<testsuite><testcase classname="tests.test_a" name="test_a">'
        '<skipped message="why"/></testcase></testsuite>'
    )
    assert run().returncode == 1
    assert run("--must-pass", "tests/test_a.py::test_a").returncode == 1
    new.write_text("<testsuite/>")
    assert run().returncode == 1
    new.write_text(
        '<testsuite><testcase classname="tests.test_a" name="test_a">'
        "<failure/></testcase></testsuite>"
    )
    assert run().returncode == 1


def test_several_files_per_side_are_merged(tmp_path: Path) -> None:
    core = tmp_path / "core.xml"
    studies = tmp_path / "studies.xml"
    core.write_text('<testsuite><testcase classname="tests.test_a" name="test_a"/></testsuite>')
    studies.write_text(
        '<testsuite><testcase classname="research.x.code.tests.test_b" name="test_b"/></testsuite>'
    )
    both = f"{core},{studies}"

    def run(new: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), both, new, *args],
            capture_output=True,
            text=True,
        )

    assert run(both, "--must-pass", "research/x/code/tests/test_b.py::test_b").returncode == 0
    # A baseline node that only the second half held is missing when only one file is given.
    lost = run(str(core))
    assert lost.returncode == 1
    assert "test_b" in lost.stdout
