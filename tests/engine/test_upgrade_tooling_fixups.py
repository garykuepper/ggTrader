"""L0: fixes from the P8 rehearsal of the upgrade runbook (P8b), kept from regressing.

Each test names the P8 defect it protects (``R<n>`` in
``research/programme/vectorbt-engine/P8-completion.md`` section 7).
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


def _skill() -> str:
    return SKILL.read_text()


def _section(text: str, heading: str) -> str:
    """The text from ``heading`` to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    start = text.index(heading)
    rest = text[start + len(heading) :]
    match = re.search(rf"(?m)^#{{1,{level}}} ", rest)
    return rest[: match.start()] if match else rest


# --- R13: a bare pytest must not collect research/ ------------------------------------------


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


# --- R3 / R15: the workflow report and the runbook use uv's own delta ------------------------


def test_the_dependency_delta_comes_from_uv_not_from_a_blind_git_diff() -> None:
    watch = (WORKFLOWS / "vbt-upgrade-watch.yml").read_text()
    assert "grep -E '^(Added|Removed|Updated) ' lock.log" in watch
    assert "git diff -U0 -- uv.lock" not in watch
    skill = _skill()
    assert "grep -E '^(Added|Removed|Updated) ' lock.log" in skill
    assert "git diff -U0 -- uv.lock | grep" not in skill


def test_the_defaults_section_of_the_report_matches_the_test_message() -> None:
    watch = (WORKFLOWS / "vbt-upgrade-watch.yml").read_text()
    pattern = re.search(r"sed -n '/(vectorbt defaults changed[^/]*)/", watch)
    assert pattern, "the report no longer extracts the defaults section"
    assert pattern.group(1) in (ROOT / "tests" / "engine" / "test_defaults.py").read_text()


# --- R5: the evidence script must not overwrite committed evidence by default ---------------


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


# --- the runbook text ----------------------------------------------------------------------


def test_the_runbook_has_the_sections_the_rehearsal_found_missing() -> None:
    skill = _skill()
    for needle in (
        "### Pre-release look-ahead",  # R7
        "### 3b. Tape pass",  # R5
        "scripts/engine_parity_evidence.py --symbols 150 --out",  # R5
        "gh pr close <n> --delete-branch",  # R8
        "gh pr ready <n>",  # R8
        "**Create** `docs/vbt_upgrade_rejected.txt`",  # R9
        "(**create** it on first use",  # R9
        "2b. The new vectorbt value **agrees**",  # R16
        "A failure whose only difference is a version string",  # R14
        "Collection error, exit code 4",  # R14
        "Not relevant to ggTrader",  # G7
        "Record the sha",  # R7
    ):
        assert needle in skill, needle


def test_the_oracle_step_names_real_artifacts_and_never_the_old_folder() -> None:
    skill = _skill()
    assert "research/audits/engine-migration" not in skill  # R4: never existed
    assert "read-only database session" not in skill  # R4: the drivers read only the tape
    for path in (
        "research/programme/vectorbt-engine/p5-evidence/_p5_compare_",
        "p7-evidence/p7_live_golden.json",
        "--out-dir",
        '--out "$SCRATCH/p7_live_golden.json"',
    ):
        assert path in skill, path
    assert list(
        (ROOT / "research/programme/vectorbt-engine/p5-evidence").glob("_p5_compare_*.json")
    )
    assert (ROOT / "research/programme/vectorbt-engine/p7-evidence/p7_live_golden.json").exists()


def test_the_step_one_symbol_list_has_no_mstd_and_the_locked_version_is_used() -> None:
    step_one = _section(_skill(), "## 1. Read the release notes")
    assert "`engine/indicators.py` (RSI, MA, BBANDS, MACD, ATR)" in step_one  # R6
    assert "(RSI, MA, MSTD" not in step_one
    assert step_one.count("${LOCKED}") >= 2 or step_one.count("$LOCKED") >= 1  # R11


def test_the_promotion_step_does_not_claim_the_deploy_skill_builds_the_image() -> None:
    step_seven = _section(_skill(), "## 7. Promotion")
    assert "(it builds the image" not in step_seven  # R10
    assert "Dockerfile" in step_seven and "docker-build.yml" in step_seven
    assert (ROOT / ".github/workflows/docker-build.yml").exists()


def test_every_test_the_runbook_names_exists() -> None:
    """The failure table must not send an agent to a test that was renamed or never written."""
    missing = []
    for filename, name, glob in re.findall(r"(test_\w+\.py)::(test_\w+)(\*|\.\.\.|)", _skill()):
        if glob:  # a prefix such as test_dsr_* is checked by the next test
            continue
        text = (ROOT / "tests" / "engine" / filename).read_text()
        if f"def {name}(" not in text:
            missing.append(f"{filename}::{name}")
    assert not missing, missing


def test_the_names_a_prefix_glob_in_the_runbook_still_match_something() -> None:
    prefixes = (
        "test_dsr_",
        "test_the_two_engines_agree_",
        "test_the_adapter_gives_the_same_answer_",
        "test_rolling_std_is_numerically_stable_",
        "test_max_drawdown_of_a_series_that_ends_in_nan_",
        "test_window_stats_of_a_window_that_ends_in_nan_",
    )
    text = (ROOT / "tests" / "engine" / "test_tripwires.py").read_text()
    for prefix in prefixes:
        assert re.search(rf"def {prefix}\w*\(", text), prefix


# --- G7: the "not relevant" table cannot go stale -------------------------------------------


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


# --- G7 rule: a new preset ships with an independent-reference test ----------------------------


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
