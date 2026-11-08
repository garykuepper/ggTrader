"""The research tree keeps study code in known places, and the linter treats each place right.

Layout (study isolation): a study may hold ``code/`` (with ``code/tests/``), a category may hold
one ``_lib/`` (with ``tests/``), and frozen Python lives only under
``research/programme/vectorbt-engine/`` or an ``artifacts-*`` folder. Ruff lints the first two and
skips the last two (ruff.toml). This file checks the placement, the import rules between studies
and the shape of the lint exclusion. The rule helpers take ``{path: source}`` so each rule is
tested on synthetic input as well as on the real tree.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Where a tracked research ``.py`` file may live.
ALLOWED_PLACES = (
    re.compile(r"^research/[^/]+/[^/]+/code/.+\.py$"),
    re.compile(r"^research/[^/]+/_lib/.+\.py$"),
    re.compile(r"^research/programme/vectorbt-engine/.+\.py$"),
    re.compile(r"^research/[^/]+/artifacts-[^/]+/.+\.py$"),
)
#: Library basenames that may repeat (entry points and test scaffolding, never imported by name).
REPEATABLE_NAMES = frozenset({"run.py", "conftest.py", "__init__.py"})
#: Folder names at the repo root; a library of the same name would shadow them on ``sys.path``.
REPO_ROOT_NAMES = frozenset(
    {"scripts", "tests", "data", "src", "docs", "research", "conftest", "ggt"}
)
#: Study code that may import the broker-facing ``ggTrader.paper`` (L18): the production replica
#: audit exercises the live trader's internals on purpose; nothing else in research may.
PAPER_IMPORT_ALLOWED = {"research/audits/production-replica/code"}
#: A line that reaches for the repo root by hand; study code uses ``bootstrap`` instead.
ROOT_LINE = re.compile(
    r"parents\[[0-9]+\]|\.parent\.parent|find_project_root|sys\.path\.(insert|append)|sys\.path\["
)
RUFF_BAD_SOURCE = "import os\nx=1\n"  # unused import (F401) and unformatted


def _research_python() -> list[str]:
    """Repo-relative research ``.py`` files, tracked or about to be (not git-ignored)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "research"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({line for line in out.splitlines() if line.endswith(".py")})


def _is_study_code(rel: str) -> bool:
    """True for files under ``research/<cat>/<study>/code`` or ``research/<cat>/_lib``."""
    return bool(ALLOWED_PLACES[0].match(rel) or ALLOWED_PLACES[1].match(rel))


def _owner(rel: str) -> str:
    """The folder that owns a code file: ``research/<cat>/<study>`` or ``research/<cat>/_lib``."""
    parts = PurePosixPath(rel).parts
    return "/".join(parts[:3])


def _category(rel: str) -> str:
    return PurePosixPath(rel).parts[1]


def _library_modules(paths: Iterable[str]) -> dict[str, list[str]]:
    """Importable basenames of study code: ``stem -> [paths]``, minus names that may repeat."""
    found: dict[str, list[str]] = {}
    for rel in paths:
        name = PurePosixPath(rel).name
        if not _is_study_code(rel) or name in REPEATABLE_NAMES or name.startswith("test_"):
            continue
        found.setdefault(PurePosixPath(rel).stem, []).append(rel)
    return found


def misplaced(paths: Iterable[str]) -> list[str]:
    """Research ``.py`` files outside every allowed place."""
    return [p for p in paths if not any(rule.match(p) for rule in ALLOWED_PLACES)]


def artifacts_inside_code(paths: Iterable[str]) -> list[str]:
    """Study-code paths with an ``artifacts-`` segment (ruff's ``*`` would silently skip them)."""
    return [
        p
        for p in paths
        if _is_study_code(p) and any(seg.startswith("artifacts-") for seg in p.split("/"))
    ]


def duplicate_basenames(paths: Iterable[str]) -> dict[str, list[str]]:
    """Library basenames defined more than once across study code."""
    return {k: v for k, v in _library_modules(paths).items() if len(v) > 1}


def shadowing_basenames(paths: Iterable[str]) -> dict[str, str]:
    """Library basenames that equal a stdlib module, an installed module or a repo-root name."""
    installed = set(importlib.metadata.packages_distributions())
    out: dict[str, str] = {}
    for stem in _library_modules(paths):
        if stem in sys.stdlib_module_names:
            out[stem] = "stdlib module"
        elif stem in installed:
            out[stem] = "installed top-level module"
        elif stem in REPO_ROOT_NAMES:
            out[stem] = "repo-root name"
    return out


def _imported_modules(source: str) -> set[str]:
    """Dotted module names a source imports (``from a.b import c`` gives ``a.b`` and ``a.b.c``)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def cross_study_imports(sources: dict[str, str]) -> list[str]:
    """Imports of a module that lives in another study's ``code/`` or another category's ``_lib``.

    A study may import its own ``code/`` modules and its own category's ``_lib``; anything else
    is a hidden dependency between studies (promote the module to ``ggTrader.utils`` instead).
    """
    libs = _library_modules(sources)
    bad: list[str] = []
    for rel, source in sorted(sources.items()):
        if not _is_study_code(rel):
            continue
        allowed = {_owner(rel), f"research/{_category(rel)}/_lib"}
        for name in sorted({m.split(".")[0] for m in _imported_modules(source)}):
            for target in libs.get(name, []):
                if _owner(target) not in allowed:
                    bad.append(f"{rel} imports {name} from {_owner(target)}")
    return bad


def paper_imports(sources: dict[str, str]) -> list[str]:
    """Study code importing ``ggTrader.paper`` outside ``PAPER_IMPORT_ALLOWED``."""
    bad: list[str] = []
    for rel, source in sorted(sources.items()):
        if not _is_study_code(rel) or any(rel.startswith(f"{a}/") for a in PAPER_IMPORT_ALLOWED):
            continue
        names = _imported_modules(source)
        if any(n == "ggTrader.paper" or n.startswith("ggTrader.paper.") for n in names):
            bad.append(rel)
    return bad


def hand_rolled_roots(sources: dict[str, str]) -> list[str]:
    """Lines of study code that find the repo root or touch ``sys.path`` without ``# root-ok``."""
    bad: list[str] = []
    for rel, source in sorted(sources.items()):
        if not _is_study_code(rel):
            continue
        for number, line in enumerate(source.splitlines(), start=1):
            if ROOT_LINE.search(line) and "# root-ok" not in line:
                bad.append(f"{rel}:{number}: {line.strip()}")
    return bad


def _real_sources() -> dict[str, str]:
    return {rel: (ROOT / rel).read_text() for rel in _research_python() if _is_study_code(rel)}


# --- the real tree ------------------------------------------------------------------------


def test_every_research_python_file_is_in_an_allowed_place() -> None:
    assert misplaced(_research_python()) == []


def test_no_artifacts_folder_hides_inside_study_code() -> None:
    assert artifacts_inside_code(_research_python()) == []


def test_library_basenames_are_unique_and_shadow_nothing() -> None:
    paths = _research_python()
    assert duplicate_basenames(paths) == {}
    assert shadowing_basenames(paths) == {}


def test_no_study_imports_another_studys_code() -> None:
    assert cross_study_imports(_real_sources()) == []


def test_research_code_imports_ggtrader_paper_only_where_allowed() -> None:
    assert paper_imports(_real_sources()) == []


def test_research_code_finds_the_repo_root_only_through_bootstrap_or_root_ok() -> None:
    assert hand_rolled_roots(_real_sources()) == []


def test_the_paper_allow_list_names_real_places_or_nothing_yet() -> None:
    for place in PAPER_IMPORT_ALLOWED:
        assert re.match(r"^research/[^/]+/[^/]+/code$", place), place


# --- the rules have teeth -----------------------------------------------------------------


def test_misplaced_files_are_found() -> None:
    good = [
        "research/allocation/static-leverage/code/run.py",
        "research/stock-signals/_lib/french_data.py",
        "research/programme/vectorbt-engine/p5-evidence/x.py",
        "research/intraday/artifacts-2026-10-26-intraday-readiness/x.py",
    ]
    bad = ["research/allocation/static-leverage/x.py", "research/allocation/notes/y.py"]
    assert misplaced([*good, *bad]) == bad


def test_an_artifacts_segment_below_code_is_found() -> None:
    inside = "research/a/s/code/artifacts-1/x.py"
    outside = "research/a/artifacts-1/x.py"
    assert artifacts_inside_code([inside, outside, "research/a/_lib/artifacts-2/y.py"]) == [
        inside,
        "research/a/_lib/artifacts-2/y.py",
    ]


def test_duplicate_basenames_ignore_entry_points_and_tests() -> None:
    paths = [
        "research/a/s1/code/run.py",
        "research/a/s2/code/run.py",
        "research/a/s1/code/tests/test_x.py",
        "research/a/s2/code/tests/test_x.py",
        "research/a/s1/code/tests/conftest.py",
        "research/a/s2/code/tests/conftest.py",
        "research/a/s1/code/helper.py",
        "research/b/_lib/helper.py",
    ]
    assert duplicate_basenames(paths) == {
        "helper": ["research/a/s1/code/helper.py", "research/b/_lib/helper.py"]
    }


def test_shadowing_names_are_found() -> None:
    paths = [
        "research/a/s/code/json.py",
        "research/a/s/code/scripts.py",
        "research/a/s/code/numpy.py",
        "research/a/s/code/fine_name.py",
    ]
    assert set(shadowing_basenames(paths)) == {"json", "scripts", "numpy"}


def test_cross_study_import_is_found_but_own_code_and_own_lib_are_fine() -> None:
    sources = {
        "research/a/s1/code/run.py": "import helper\nfrom lib_mod import f\nimport mine\n",
        "research/a/s1/code/mine.py": "x = 1\n",
        "research/a/s2/code/helper.py": "x = 1\n",
        "research/a/_lib/lib_mod.py": "def f(): ...\n",
        "research/b/_lib/other_lib.py": "x = 1\n",
        "research/a/s3/code/run.py": "from other_lib import x\nfrom lib_mod import f\n",
    }
    assert cross_study_imports(sources) == [
        "research/a/s1/code/run.py imports helper from research/a/s2",
        "research/a/s3/code/run.py imports other_lib from research/b/_lib",
    ]


def test_paper_import_is_found_outside_the_allow_list() -> None:
    allowed = next(iter(PAPER_IMPORT_ALLOWED))
    sources = {
        f"{allowed}/run.py": "from ggTrader.paper import trader\n",
        "research/a/s/code/run.py": "import ggTrader.paper.risk\n",
        "research/a/s/code/two.py": "from ggTrader import paper\nfrom ggTrader.paper import x\n",
        "research/a/s/code/fine.py": "from ggTrader.lab import wfo\n",
    }
    assert paper_imports(sources) == ["research/a/s/code/run.py", "research/a/s/code/two.py"]


def test_hand_rolled_roots_need_a_root_ok_comment() -> None:
    sources = {
        "research/a/s/code/run.py": (
            "import sys\n"
            "sys.path.insert(0, 'src')\n"
            "ROOT = Path(__file__).resolve().parents[3]  # root-ok: stdlib-only family\n"
            "x = Path(__file__).parent.parent\n"
        )
    }
    found = hand_rolled_roots(sources)
    assert [f.split(":")[1] for f in found] == ["2", "4"]


# --- ruff lints study code and skips frozen artifacts --------------------------------------


def _ruff() -> str | None:
    return shutil.which("ruff", path=str(Path(sys.executable).parent)) or shutil.which("ruff")


def _lint(stdin_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_ruff(), "check", "--no-cache", "--stdin-filename", stdin_name, "-"],
        input=RUFF_BAD_SOURCE,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _format(stdin_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_ruff(), "format", "--check", "--no-cache", "--stdin-filename", stdin_name, "-"],
        input=RUFF_BAD_SOURCE,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(_ruff() is None, reason="ruff (dev extra) not installed")
def test_ruff_format_checks_a_code_path_and_skips_frozen_places() -> None:
    """The format gate (``ruff format --check .``) covers study code and spares frozen files."""
    assert _format("research/allocation/static-leverage/code/x.py").returncode == 1
    assert _format("research/stock-signals/_lib/x.py").returncode == 1
    assert _format("research/intraday/artifacts-2026-10-26-intraday-readiness/x.py").returncode == 0
    assert _format("research/programme/vectorbt-engine/p5-evidence/x.py").returncode == 0


@pytest.mark.skipif(_ruff() is None, reason="ruff (dev extra) not installed")
def test_ruff_skips_every_tracked_frozen_research_file_even_by_explicit_path() -> None:
    """Hooks pass explicit paths: ruff.toml (force-exclude) must still skip all frozen files."""
    frozen = [p for p in _research_python() if not _is_study_code(p)]
    assert frozen, "expected frozen research Python files (vectorbt-engine, artifacts-*)"
    for sub in ("check", "format"):
        args = [sub, "--no-cache", *(["--check"] if sub == "format" else []), *frozen]
        done = subprocess.run(
            [_ruff(), *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, (sub, done.stdout[-500:], done.stderr[-500:])
        assert "No Python files found" in done.stderr, (sub, done.stderr[-500:])


@pytest.mark.skipif(_ruff() is None, reason="ruff (dev extra) not installed")
def test_ruff_lints_a_code_path_and_skips_frozen_places() -> None:
    assert _lint("research/allocation/static-leverage/code/x.py").returncode == 1
    assert _lint("research/stock-signals/_lib/x.py").returncode == 1
    assert _lint("research/intraday/artifacts-2026-10-26-intraday-readiness/x.py").returncode == 0
    assert _lint("research/programme/vectorbt-engine/p5-evidence/x.py").returncode == 0


def test_ruff_toml_declares_the_exclusions_and_the_e402_ignore() -> None:
    config = tomllib.loads((ROOT / "ruff.toml").read_text())
    assert config.get("force-exclude") is True
    assert set(config["extend-exclude"]) == {
        "research/programme/vectorbt-engine",
        "research/*/artifacts-*",
    }
    assert config["lint"]["per-file-ignores"]["research/**/code/*.py"] == ["E402"]


def _study_marked(source: str) -> bool:
    """True when the source names the study marker as an attribute of ``pytest.mark``."""
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "study"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        for node in ast.walk(ast.parse(source))
    )


def test_no_file_under_tests_carries_the_study_marker() -> None:
    """Study tests live next to their study; the FAST tree must hold none (S8)."""
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    marked = [
        path
        for path in tracked
        if path.endswith(".py") and _study_marked((ROOT / path).read_text())
    ]
    assert marked == []


def test_the_study_marker_finder_sees_a_module_mark_and_a_decorator() -> None:
    mark = "pytest.mark." + "study"  # spelled in pieces so the folder-wide grep stays clean
    assert _study_marked(f"import pytest\npytestmark = {mark}\n")
    assert _study_marked(f"import pytest\n@{mark}\ndef test_x():\n    pass\n")
    assert not _study_marked("import pytest\npytestmark = pytest.mark.slow\nstudy = 1\n")
