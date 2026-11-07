"""L0: guards from the P8 rehearsal of the upgrade runbook (P8b) that protect behaviour.

A bare pytest must not collect ``research/`` (R13), the evidence script must not overwrite committed
evidence by default (R5), a new indicator preset ships with an independent-reference test (G7), and
the adapter calls none of the vectorbt APIs the runbook's "not relevant" table dismisses (G7). The
runbook-wording pins of the same rehearsal were removed in S1.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".claude" / "skills" / "vbt-upgrade" / "SKILL.md"
WORKFLOWS = ROOT / ".github" / "workflows"
ENGINE_SRC = ROOT / "src" / "ggTrader" / "engine"

pytestmark = pytest.mark.tooling


def test_pytest_collects_tests_only_by_default() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert config["testpaths"] == ["tests"]


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_every_workflow_pytest_command_names_the_tests_folder(workflow: Path) -> None:
    for line in workflow.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not re.search(r"(^|[\s:])pytest(\s|$)", stripped):
            continue
        assert "tests" in stripped, (
            f"{workflow.name}: a bare pytest would collect research/: {line}"
        )


def test_the_parity_script_defaults_to_a_scratch_output() -> None:
    from scripts import engine_parity_evidence as script

    assert ROOT / "research" not in script.DEFAULT_OUT.parents
    tree = ast.parse((ROOT / "scripts" / "engine_parity_evidence.py").read_text())
    defaults = [
        ast.unparse(kw.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        and any(isinstance(a, ast.Constant) and a.value == "--out" for a in node.args)
        for kw in node.keywords
        if kw.arg == "default"
    ]
    assert defaults == ["DEFAULT_OUT"]


def test_every_public_indicator_preset_has_a_test() -> None:
    tree = ast.parse((ENGINE_SRC / "indicators.py").read_text())
    presets = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    assert presets, "indicators.py lost its presets"
    tests = "\n".join(
        path.read_text()
        for name in ("test_indicators.py", "test_defaults.py", "test_tripwires.py")
        if (path := ROOT / "tests" / "engine" / name).exists()
    )
    untested = [name for name in presets if not re.search(rf"\b{name}\(", tests)]
    assert not untested, f"add an independent-reference test for: {untested}"


def _skill() -> str:
    return SKILL.read_text()


def _section(text: str, heading: str) -> str:
    """The text from ``heading`` to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    start = text.index(heading)
    rest = text[start + len(heading) :]
    match = re.search(rf"(?m)^#{{1,{level}}} ", rest)
    return rest[: match.start()] if match else rest


def _not_relevant_rows() -> list[tuple[str, str]]:
    section = _section(_skill(), "### Not relevant to ggTrader")
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if not line.startswith("|") or len(cells) != 4 or cells[0] in ("Release", "---"):
            continue
        if set(cells[0]) <= {"-", "|"}:
            continue
        needle = cells[2].strip("`").replace("\\|", "|")
        rows.append((cells[1], needle))
    return rows


def test_the_not_relevant_table_has_its_rows() -> None:
    rows = _not_relevant_rows()
    assert len(rows) >= 10
    assert {"MSTD", "OBV", "expectancy", r"\.drawdowns"} <= {needle for _, needle in rows}


def test_the_adapter_calls_none_of_the_apis_the_not_relevant_table_dismisses() -> None:
    """If this fails the adapter started calling something the runbook says it never calls."""
    source = "\n".join(path.read_text() for path in sorted(ENGINE_SRC.glob("*.py")))
    # Comments and docstrings may mention a name; only code counts.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("#", '"""', "``"))
    )
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    called = [
        (change, needle) for change, needle in _not_relevant_rows() if re.search(needle, code)
    ]
    assert not called, f"engine/ now touches APIs the runbook says are out of surface: {called}"
