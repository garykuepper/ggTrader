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


def test_skip_reason_ignores_checkout_path_and_line(tmp_path: Path) -> None:
    def suite(root: str, line: int) -> str:
        return (
            '<testsuite><testcase classname="tests.test_a" name="test_a">'
            f'<skipped message="{root}/tests/test_a.py:{line}: no database"/>'
            "</testcase></testsuite>"
        )

    baseline, new = tmp_path / "b.xml", tmp_path / "n.xml"
    baseline.write_text(suite("/home/x/ggTrader-baseline", 10))
    new.write_text(suite("/home/x/ggTrader-refactor", 12))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(baseline), str(new)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout
    new.write_text(suite("/home/x/ggTrader-refactor", 12).replace("no database", "other"))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(baseline), str(new)], capture_output=True, text=True
    )
    assert result.returncode == 1


def _suite(*cases: str) -> str:
    return "<testsuite>" + "".join(cases) + "</testsuite>"


def _case(name: str, outcome: str = "") -> str:
    return f'<testcase classname="tests.test_a" name="{name}">{outcome}</testcase>'


def _run(tmp_path: Path, base: str, new: str) -> subprocess.CompletedProcess:
    (tmp_path / "b.xml").write_text(base)
    (tmp_path / "n.xml").write_text(new)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "b.xml"), str(tmp_path / "n.xml")],
        capture_output=True,
        text=True,
    )


def test_new_failing_test_is_a_breach(tmp_path: Path) -> None:
    base = _suite(_case("test_a"))
    assert _run(tmp_path, base, _suite(_case("test_a"), _case("test_b"))).returncode == 0
    new = _suite(_case("test_a"), _case("test_b", '<error message="collect"/>'))
    assert _run(tmp_path, base, new).returncode == 1


def test_baseline_xfail_that_now_fails_is_a_breach(tmp_path: Path) -> None:
    skipped = '<skipped type="pytest.xfail" message="known"/>'
    base = _suite(_case("test_a", skipped))
    assert _run(tmp_path, base, base).returncode == 0
    xpass = '<failure message="[XPASS(strict)] known"/>'
    assert _run(tmp_path, base, _suite(_case("test_a", xpass))).returncode == 1


def test_numbers_inside_a_skip_reason_are_content(tmp_path: Path) -> None:
    def skip(text: str) -> str:
        return _suite(_case("test_a", f'<skipped message="{text}"/>'))

    assert _run(tmp_path, skip("needs 12:30 window"), skip("needs 12:45 window")).returncode == 1


def test_must_pass_file_requires_every_listed_node(tmp_path: Path) -> None:
    base, new, required = tmp_path / "base.xml", tmp_path / "new.xml", tmp_path / "nodes.txt"
    base.write_text(_suite(_case("test_a")))
    required.write_text("# required\n\ntests/test_a.py::test_a\ntests/test_a.py::test_b\n")
    new.write_text(_suite(_case("test_a"), _case("test_b", '<skipped message="skip"/>')))
    command = [sys.executable, str(SCRIPT), str(base), str(new), "--must-pass-file", str(required)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and "tests/test_a.py::test_b" in result.stdout
    new.write_text(_suite(_case("test_a"), _case("test_b")))
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
