"""scripts/study.py: ``new`` scaffolds a study that ``check`` passes, and ``check`` has teeth.

Everything runs on a throwaway repository root under ``tmp_path`` (the tool takes ``--repo-root``),
so no test writes into this checkout. The planted violations are the ones the study rules exist
for: a vectorbt import, one study importing another's code, a missing evaluation date, an
``--eval-end`` that defaults to today, a repeated module name and a ``ggTrader.paper`` import.
``close`` and ``repro`` run end to end on a tiny git repository.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "study.py"
CATEGORY = "stock-signals"
SLUG = "zz-probe"
MOD = "zz_probe"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("study_script_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = _load()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A minimal repository root: the research templates and README, no studies."""
    (tmp_path / "research").mkdir()
    for name in ("README.md", "TEMPLATE-brief.md", "TEMPLATE-research-report.md"):
        shutil.copy(ROOT / "research" / name, tmp_path / "research" / name)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (tmp_path / ".git").mkdir()
    return tmp_path


def run(repo: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    """Run the tool against ``repo``; return (exit status, stdout + stderr)."""
    capsys.readouterr()
    code = study.main(["--repo-root", str(repo), *args])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def new(repo: Path, capsys: pytest.CaptureFixture[str], slug: str = SLUG) -> Path:
    """Scaffold a study and return its folder."""
    code, out = run(
        repo, capsys, "new", CATEGORY, slug, "--kind", "screen", "--placement", "9",
        "--date", "2026-11-07",
    )  # fmt: skip
    assert code == 0, out
    return repo / "research" / CATEGORY / slug


def check(repo: Path, capsys: pytest.CaptureFixture[str], folder: Path) -> tuple[int, str]:
    return run(repo, capsys, "check", str(folder))


def edit(path: Path, old: str, new_text: str) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new_text, 1))


# --- new + check ---------------------------------------------------------------------------


def test_new_scaffolds_the_planned_tree_and_check_passes(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert names == [
        "2026-11-07-zz-probe-brief.md",
        "README.md",
        "code/run.py",
        "code/tests/conftest.py",
        "code/tests/test_zz_probe.py",
        "code/zz_probe.py",
        "study.toml",
    ]
    card = (folder / "study.toml").read_text()
    assert "allow_paper_imports = false" in card and "placement_rule = 9" in card
    code, out = check(repo, capsys, folder)
    assert (code, out.strip()) == (0, f"OK research/{CATEGORY}/{SLUG}")


def test_check_resolves_a_repo_relative_path(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    new(repo, capsys)
    code, _ = run(repo, capsys, "check", f"research/{CATEGORY}/{SLUG}")
    assert code == 0


def test_the_brief_is_the_template_with_the_placeholders_filled(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    brief = (folder / "2026-11-07-zz-probe-brief.md").read_text()
    assert "$" not in brief
    assert "- eval_start: 2021-01-31" in brief and "- eval_end: 2026-04-30" in brief
    assert "# Brief: Zz Probe" in brief


def test_new_rejects_a_placement_row_that_makes_no_study_folder(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for row in ("1", "2"):
        code, out = run(repo, capsys, "new", CATEGORY, SLUG, "--kind", "screen", "--placement", row)
        assert code == 1 and "does not create a study folder" in out
    assert not (repo / "research" / CATEGORY).exists()


def test_new_rejects_a_placement_row_for_another_category(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(repo, capsys, "new", CATEGORY, SLUG, "--kind", "wfo", "--placement", "7")
    assert code == 1 and "maps to 'allocation'" in out


def test_new_rejects_a_slug_that_exists_in_any_category(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    new(repo, capsys)
    code, out = run(repo, capsys, "new", "allocation", SLUG, "--kind", "wfo", "--placement", "7")
    assert code == 1 and "already exists" in out
    assert not (repo / "research" / "allocation").exists()


@pytest.mark.parametrize("slug", ["Bad_Slug", "9lives", "two--dashes", "json", "scripts"])
def test_new_rejects_a_bad_or_shadowing_slug(
    repo: Path, capsys: pytest.CaptureFixture[str], slug: str
) -> None:
    code, _ = run(repo, capsys, "new", CATEGORY, slug, "--kind", "screen", "--placement", "9")
    assert code == 1
    assert not (repo / "research" / CATEGORY).exists()


def test_the_placement_table_in_the_real_readme_matches_the_default() -> None:
    assert study.placement_table(ROOT) == study.DEFAULT_PLACEMENT
    assert study.placement_table(ROOT / "docs") == study.DEFAULT_PLACEMENT  # no README: fallback


# --- check has teeth -----------------------------------------------------------------------


def test_check_fails_on_a_vectorbt_import(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = new(repo, capsys)
    edit(
        folder / "code" / f"{MOD}.py",
        "from typing import Any",
        "import vectorbt\nfrom typing import Any",
    )
    code, out = check(repo, capsys, folder)
    assert code == 1 and "imports vectorbt" in out


def test_check_fails_on_the_vbt_accessor(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = new(repo, capsys)
    (folder / "code" / "helper.py").write_text("def f(s):\n    return s.vbt.returns\n")
    code, out = check(repo, capsys, folder)
    assert code == 1 and ".vbt accessor" in out


def test_check_fails_on_an_import_from_another_study(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    new(repo, capsys, "other-study")
    (repo / "research" / CATEGORY / "other-study" / "code" / "other_helper.py").write_text(
        "X = 1\n"
    )
    edit(
        folder / "code" / f"{MOD}.py",
        "from typing import Any",
        "import other_helper\nfrom typing import Any",
    )
    code, out = check(repo, capsys, folder)
    assert code == 1 and "imports other_helper from research/stock-signals/other-study" in out


def test_check_allows_a_module_from_the_own_category_lib(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    lib = repo / "research" / CATEGORY / "_lib"
    lib.mkdir()
    (lib / "shared_math.py").write_text("X = 1\n")
    edit(
        folder / "code" / f"{MOD}.py",
        "from typing import Any",
        "import shared_math\nfrom typing import Any",
    )
    assert check(repo, capsys, folder)[0] == 0


def test_check_fails_on_a_lib_module_of_another_category(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    lib = repo / "research" / "allocation" / "_lib"
    lib.mkdir(parents=True)
    (lib / "alloc_math.py").write_text("X = 1\n")
    edit(
        folder / "code" / f"{MOD}.py",
        "from typing import Any",
        "import alloc_math\nfrom typing import Any",
    )
    code, out = check(repo, capsys, folder)
    assert code == 1 and "imports alloc_math from research/allocation/_lib" in out


def test_check_fails_on_a_missing_eval_date(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = new(repo, capsys)
    edit(folder / "2026-11-07-zz-probe-brief.md", "- eval_end: 2026-04-30\n", "")
    code, out = check(repo, capsys, folder)
    assert code == 1 and "missing registered field eval_end" in out


COSTS_LINE = (
    "- costs: 5 bps per side fees plus 5 bps per side slippage (confirm against the lab config)"
)
BENCHMARK_LINE = "- benchmark: SPY buy-and-hold, same window and same costs"
START = "- eval_start: 2021-01-31"


@pytest.mark.parametrize(
    ("old", "new_text", "message"),
    [
        (START, "- eval_start: 2027-01-31", "eval_start is not before eval_end"),
        (START, "- eval_start: yesterday", "not a YYYY-MM-DD date"),
        ("- max_leverage: 1.0\n", "", "missing registered field max_leverage"),
        (BENCHMARK_LINE, "- benchmark: QQQ", "name SPY"),
        (COSTS_LINE, "- costs: free", "must state a number"),
        ("- family_size: 1", "- family_size: many", "family_size must be a positive integer"),
    ],
)
def test_check_fails_on_each_malformed_brief_field(
    repo: Path, capsys: pytest.CaptureFixture[str], old: str, new_text: str, message: str
) -> None:
    folder = new(repo, capsys)
    edit(folder / "2026-11-07-zz-probe-brief.md", old, new_text)
    code, out = check(repo, capsys, folder)
    assert code == 1 and message in out


@pytest.mark.parametrize(
    "default",
    ["dt.date.today().isoformat()", "None", '"today"', "datetime.now().isoformat()"],
)
def test_check_fails_when_eval_end_defaults_to_today(
    repo: Path, capsys: pytest.CaptureFixture[str], default: str
) -> None:
    folder = new(repo, capsys)
    edit(folder / "code" / "run.py", "default=EVAL_END_17", f"default={default}")
    code, out = check(repo, capsys, folder)
    assert code == 1 and "--eval-end" in out


def test_check_fails_when_eval_end_has_no_default(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    edit(
        folder / "code" / "run.py",
        'parser.add_argument("--eval-end", default=EVAL_END_17, ',
        'parser.add_argument("--eval-end", ',
    )
    code, out = check(repo, capsys, folder)
    assert code == 1 and "no pinned default" in out


def test_check_fails_on_a_repeated_module_name(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    other = repo / "research" / "allocation" / "another-study" / "code"
    other.mkdir(parents=True)
    (other / f"{MOD}.py").write_text("X = 1\n")
    code, out = check(repo, capsys, folder)
    assert code == 1 and f"module name {MOD!r} is defined more than once" in out


def test_check_fails_on_a_module_that_shadows_the_standard_library(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    (folder / "code" / "json.py").write_text("X = 1\n")
    code, out = check(repo, capsys, folder)
    assert code == 1 and "shadows a standard-library module" in out


def test_check_fails_on_a_paper_import_unless_the_study_allows_it(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    edit(
        folder / "code" / f"{MOD}.py",
        "from typing import Any",
        "from ggTrader.paper import risk\nfrom typing import Any",
    )
    code, out = check(repo, capsys, folder)
    assert code == 1 and "imports ggTrader.paper" in out
    edit(folder / "study.toml", "allow_paper_imports = false", "allow_paper_imports = true")
    assert check(repo, capsys, folder)[0] == 0


def test_check_fails_on_a_hand_rolled_repo_root(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    (folder / "code" / "helper.py").write_text("ROOT = Path(__file__).parents[3]\n")
    code, out = check(repo, capsys, folder)
    assert code == 1 and "hand-rolled repo root" in out
    (folder / "code" / "helper.py").write_text("ROOT = Path(__file__).parents[3]  # root-ok\n")
    assert check(repo, capsys, folder)[0] == 0


def test_check_fails_on_a_driver_with_top_level_work_and_no_guard(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    (folder / "code" / "run.py").write_text("import os\nos.system('echo hi')\n")
    code, out = check(repo, capsys, folder)
    assert code == 1
    for part in ("module-level call os.system", "no `if __name__", "no main()", "ArgumentParser"):
        assert part in out


def test_check_fails_on_bad_study_toml(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = new(repo, capsys)
    card = folder / "study.toml"
    edit(card, 'kind = "screen"', 'kind = "nonsense"\nsurprise = 1')
    edit(card, "placement_rule = 9", "placement_rule = 7")
    edit(card, 'id = "zz-probe"', 'id = "other"')
    code, out = check(repo, capsys, folder)
    assert code == 1
    for part in (
        "unknown key surprise",
        "kind 'nonsense'",
        "placement_rule 7 does not map",
        "id 'other'",
    ):
        assert part in out
    card.write_text("this is = = not toml")
    assert "not valid TOML" in check(repo, capsys, folder)[1]


def test_check_fails_on_a_misnamed_report_or_result_file(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    (folder / "notes.md").write_text("# notes\n")
    (folder / "results.json").write_text("{}\n")
    code, out = check(repo, capsys, folder)
    assert code == 1 and "notes.md: expected" in out and "results.json: results JSON" in out


def test_check_fails_on_a_cited_path_that_does_not_exist(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    with (folder / "2026-11-07-zz-probe-brief.md").open("a") as handle:
        handle.write("\nSee `research/stock-signals/missing/report.md` for the earlier work.\n")
    code, out = check(repo, capsys, folder)
    assert (
        code == 1 and "cited path does not exist: research/stock-signals/missing/report.md" in out
    )


def test_check_fails_on_results_without_a_provenance_block(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    (folder / f"_{MOD}_results.json").write_text('{"result": 1}\n')
    code, out = check(repo, capsys, folder)
    assert code == 1 and "no provenance block" in out
    (folder / f"_{MOD}_results.json").write_text('{"result": 1, "provenance": {"git_sha": "x"}}\n')
    assert check(repo, capsys, folder)[0] == 0


def test_check_fails_when_a_listed_report_is_missing(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    edit(folder / "study.toml", "reports = []", 'reports = ["2026-11-08-zz-probe.md"]')
    code, out = check(repo, capsys, folder)
    assert code == 1 and "2026-11-08-zz-probe.md: listed in study.toml but missing" in out


# --- driver rules on their own -------------------------------------------------------------


GOOD_DRIVER = '''"""d."""
import argparse


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval-end", default="2026-04-30")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
'''


def test_a_pinned_literal_eval_end_passes() -> None:
    assert study.driver_problems(GOOD_DRIVER, "run.py") == []


@pytest.mark.parametrize(
    "default",
    ['"2026-4-30"', "None", "date.today()", "pd.Timestamp.now()", '"now"'],
)
def test_an_unpinned_eval_end_default_is_found(default: str) -> None:
    source = GOOD_DRIVER.replace('default="2026-04-30"', f"default={default}")
    problems = study.driver_problems(source, "run.py")
    assert any("--eval-end" in p for p in problems), problems


def test_an_eval_end_written_with_an_underscore_is_still_checked() -> None:
    source = GOOD_DRIVER.replace('"--eval-end", default="2026-04-30"', '"--eval_end", default=None')
    assert any("--eval-end" in p for p in study.driver_problems(source, "run.py"))


def test_a_required_eval_end_needs_no_default() -> None:
    source = GOOD_DRIVER.replace('default="2026-04-30"', "required=True")
    assert study.driver_problems(source, "run.py") == []


# --- close ---------------------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    done = subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A real git repository holding the templates, a runnable scaffolded study, one commit."""
    (tmp_path / "research").mkdir()
    for name in ("README.md", "TEMPLATE-brief.md", "TEMPLATE-research-report.md"):
        shutil.copy(ROOT / "research" / name, tmp_path / "research" / name)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    git(tmp_path, "init", "-q")
    return tmp_path


def make_runnable(folder: Path) -> None:
    """Replace the stub with a deterministic study so the driver can run."""
    (folder / "code" / f"{MOD}.py").write_text(
        "from __future__ import annotations\n\n\n"
        "def run_study(eval_start: str, eval_end: str) -> dict:\n"
        '    return {"sharpe": 0.5, "window": [eval_start, eval_end]}\n'
    )


def run_driver(repo: Path, folder: Path) -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, str(folder / "code" / "run.py")],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )


def closed_study(
    repo: Path, capsys: pytest.CaptureFixture[str], verdict: str = "NO-GO"
) -> tuple[Path, str]:
    """Scaffold, make runnable, commit, run the driver, close; return (folder, code commit)."""
    folder = new(repo, capsys)
    make_runnable(folder)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "study code")
    sha = git(repo, "rev-parse", "HEAD")
    run_driver(repo, folder)
    code, out = run(
        repo, capsys, "close", str(folder), "--verdict", verdict, "--date", "2026-11-08"
    )
    assert code == 0, out
    return folder, sha


def test_close_pins_the_run_and_writes_a_report_skeleton(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder, sha = closed_study(git_repo, capsys)
    pin = json.loads((folder / "PIN.json").read_text())
    assert pin["schema"] == 1
    assert pin["study"]["id"] == SLUG
    assert pin["engine"]["git_sha"] == sha and pin["engine"]["git_dirty"] is False
    assert pin["windows"] == {"eval_start": "2021-01-31", "eval_end": "2026-04-30"}
    assert pin["command"]["argv"][-1] == f"research/{CATEGORY}/{SLUG}/code/run.py"
    paths = {e["path"] for e in pin["code"]}
    assert f"research/{CATEGORY}/{SLUG}/code/run.py" in paths
    assert [e["path"] for e in pin["results"]] == [
        f"research/{CATEGORY}/{SLUG}/_{MOD}_results.json"
    ]
    report = folder / "2026-11-08-zz-probe-nogo.md"
    text = report.read_text()
    assert text.startswith("# Zz Probe: NO-GO") and "**Date:** 2026-11-08" in text
    assert "**Study:** stock-signals/zz-probe. **Verdict:** NO-GO." in text
    assert "## 6. Contrarian Evaluation & Parked Research" in text
    assert check(git_repo, capsys, folder)[0] == 0


def test_close_prints_the_index_row_and_the_changelog_stub(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(git_repo, capsys)
    make_runnable(folder)
    git(git_repo, "add", "-A")
    git(git_repo, "commit", "-qm", "code")
    run_driver(git_repo, folder)
    code, out = run(
        git_repo, capsys, "close", str(folder), "--verdict", "GO", "--date", "2026-11-08"
    )
    assert code == 0
    assert "| TODO family | Zz Probe | **GO**" in out
    assert "(stock-signals/zz-probe/2026-11-08-zz-probe.md)" in out
    assert "## 2026-11-08 (study: zz-probe)" in out
    assert (folder / "2026-11-08-zz-probe.md").is_file()  # no -nogo suffix for a GO


def test_close_refuses_twice_without_force(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder, _ = closed_study(git_repo, capsys)
    code, out = run(
        git_repo, capsys, "close", str(folder), "--verdict", "NO-GO", "--date", "2026-11-08"
    )
    assert code == 1 and "already exists" in out
    code, _ = run(
        git_repo,
        capsys,
        "close",
        str(folder),
        "--verdict",
        "NO-GO",
        "--date",
        "2026-11-08",
        "--force",
    )
    assert code == 0


def test_close_refuses_results_without_provenance_or_from_a_dirty_tree(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(git_repo, capsys)
    results = folder / f"_{MOD}_results.json"
    results.write_text('{"result": 1}\n')
    code, out = run(git_repo, capsys, "close", str(folder), "--verdict", "NO-GO")
    assert code == 1 and "no provenance block" in out
    results.write_text('{"provenance": {"git_sha": "a", "git_dirty": true}}\n')
    code, out = run(git_repo, capsys, "close", str(folder), "--verdict", "NO-GO")
    assert code == 1 and "dirty tree" in out
    results.unlink()
    code, out = run(git_repo, capsys, "close", str(folder), "--verdict", "NO-GO")
    assert code == 1 and "no results JSON" in out


def test_close_refuses_a_study_that_fails_check(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(git_repo, capsys)
    edit(folder / "code" / "run.py", "default=EVAL_END_17", "default=None")
    code, out = run(git_repo, capsys, "close", str(folder), "--verdict", "NO-GO")
    assert code == 1 and "check fails" in out and not (folder / "PIN.json").exists()


def test_a_changed_code_file_breaks_the_pin(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder, _ = closed_study(git_repo, capsys)
    with (folder / "code" / f"{MOD}.py").open("a") as handle:
        handle.write("\n# edited after the study closed\n")
    code, out = check(git_repo, capsys, folder)
    assert code == 1 and f"{MOD}.py changed since it was pinned" in out


def test_the_report_template_is_untouched_by_a_close(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = (git_repo / "research" / "TEMPLATE-research-report.md").read_bytes()
    closed_study(git_repo, capsys)
    assert (git_repo / "research" / "TEMPLATE-research-report.md").read_bytes() == before
    assert before == (ROOT / "research" / "TEMPLATE-research-report.md").read_bytes()


# --- repro ---------------------------------------------------------------------------------


def test_repro_print_only_lists_the_plan_and_runs_nothing(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder, sha = closed_study(git_repo, capsys)
    code, out = run(git_repo, capsys, "repro", str(folder), "--print-only")
    assert code == 0
    assert f"checkout --detach {sha}" in out and "uv sync --frozen" in out
    assert "worktree" not in out.replace("never git worktree", "")
    assert not list(git_repo.glob("ggt-repro-*"))


def test_repro_reruns_in_a_scratch_clone_and_matches(
    git_repo: Path, capsys: pytest.CaptureFixture[str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    folder, _ = closed_study(git_repo, capsys)
    git(git_repo, "add", "-A")
    git(git_repo, "commit", "-qm", "close")
    scratch = tmp_path_factory.mktemp("scratch") / "s"
    code, out = run(
        git_repo, capsys, "repro", str(folder), "--no-sync", "--scratch-dir", str(scratch)
    )
    assert code == 0, out
    assert f"SAME research/{CATEGORY}/{SLUG}/_{MOD}_results.json" in out
    assert (scratch / "repo" / ".git").exists()


def test_repro_reports_a_number_that_moved(
    git_repo: Path, capsys: pytest.CaptureFixture[str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    folder, _ = closed_study(git_repo, capsys)
    results = folder / f"_{MOD}_results.json"
    data = json.loads(results.read_text())
    data["result"]["sharpe"] = 0.9
    results.write_text(json.dumps(data))
    git(git_repo, "add", "-A")
    git(git_repo, "commit", "-qm", "tampered")
    scratch = tmp_path_factory.mktemp("scratch") / "s"
    code, out = run(
        git_repo, capsys, "repro", str(folder), "--no-sync", "--scratch-dir", str(scratch)
    )
    assert code == 1 and "DIFFERS" in out and "/result/sharpe: 0.9 vs 0.5" in out


def test_repro_of_a_legacy_pin_prints_the_environment_recipe(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder, _ = closed_study(git_repo, capsys)
    pin = json.loads((folder / "PIN.json").read_text())
    pin["engine"]["tag"] = "legacy-engine-vbt0285"
    (folder / "PIN.json").write_text(json.dumps(pin))
    code, out = run(git_repo, capsys, "repro", str(folder), "--no-sync")
    assert code == 0 and "scripts/legacy_env.sh" in out and "legacy-engine-vbt0285" in out


def test_repro_without_a_pin_fails(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = new(repo, capsys)
    code, out = run(repo, capsys, "repro", str(folder), "--print-only")
    assert code == 1 and "PIN.json is missing" in out


def test_numeric_differences_ignore_noise_and_tolerate_float_rounding() -> None:
    old = {"a": 1.0, "wall_seconds": 3, "provenance": {"x": 1}, "rows": [1, 2.0], "n": float("nan")}
    new_same = {
        "a": 1.0 + 1e-12,
        "wall_seconds": 9,
        "provenance": {"x": 2},
        "rows": [1, 2.0],
        "n": float("nan"),
    }
    assert study.numeric_differences(old, new_same) == []
    new_diff = {**new_same, "a": 1.1, "rows": [1, 2.0, 3], "extra": 1}
    found = study.numeric_differences(old, new_diff)
    assert any("/a:" in f for f in found) and any("rows" in f for f in found)
    assert any("/extra" in f for f in found)


# --- the generated code is clean -----------------------------------------------------------


def _ruff() -> str | None:
    return shutil.which("ruff", path=str(Path(sys.executable).parent)) or shutil.which("ruff")


@pytest.mark.skipif(_ruff() is None, reason="ruff (dev extra) not installed")
def test_generated_code_passes_ruff_check_and_format(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = new(repo, capsys)
    for path in sorted((folder / "code").rglob("*.py")):
        rel = f"research/{CATEGORY}/{SLUG}/{path.relative_to(folder).as_posix()}"
        for args in (["check"], ["format", "--check"]):
            done = subprocess.run(
                [_ruff(), *args, "--no-cache", "--stdin-filename", rel, "-"],
                input=path.read_text(),
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert done.returncode == 0, (rel, args, done.stdout[-600:])


def test_help_prints_usage_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(SystemExit) as raised:
            study.main(["--help"])
    finally:
        os.chdir(cwd)
    assert raised.value.code == 0 and "usage" in capsys.readouterr().out.lower()
    assert list(tmp_path.iterdir()) == []
