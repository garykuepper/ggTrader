#!/usr/bin/env python3
"""Create, check, reproduce and close a research study (no database, no network).

A study lives in ``research/<category>/<slug>/`` with its brief, report, results and code. The
four subcommands keep that tree honest:

* ``new``    scaffold a study from the placement table in ``research/README.md``;
* ``check``  naming, ``study.toml``, brief fields, driver rules, import boundaries, provenance;
* ``repro``  re-run a closed study from its ``PIN.json`` in a scratch clone (never a worktree);
* ``close``  write ``PIN.json`` and a report skeleton, and print the index row and changelog stub.

``--repo-root`` points the tool at another checkout (a scratch clone, a test fixture). Only the
standard library is imported at start-up; ``repro`` loads ``ggTrader.engine.provenance`` lazily
to compare provenance blocks.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

KINDS = ("screen", "wfo", "event", "audit", "prospective")
VERDICTS = ("GO", "NO-GO", "UNRESOLVED", "METHODS")
STATUSES = ("open", "closed")
REPRODUCIBLE = ("PINNED", "PARTLY", "NOT")

#: Placement-test row -> category folder (rows 1 and 2 never create a study). The live table is
#: read from research/README.md; this is the fallback for a root without one.
DEFAULT_PLACEMENT = {
    3: "audits",
    4: "programme",
    5: "intraday",
    6: "crypto",
    7: "allocation",
    8: "paper-system",
    9: "stock-signals",
}
PLACEMENT_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|.*\|\s*`([a-z][a-z-]*)/`\s*\|\s*$")
SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
BRIEF_LINE_RE = re.compile(r"^\s*[-*]\s+([a-z_]+):\s*(.+?)\s*$")

#: Pinned 17-fold window (ggTrader.utils.paths.EVAL_START_17 / EVAL_END_17).
PINNED_EVAL_START = "2021-01-31"
PINNED_EVAL_END = "2026-04-30"
BRIEF_KEYS = ("eval_start", "eval_end", "max_leverage", "benchmark", "costs", "family_size")
TOML_KEYS = {
    "id",
    "category",
    "placement_rule",
    "question",
    "kind",
    "status",
    "verdict",
    "aliases",
    "brief",
    "reports",
    "results",
    "uses_lib",
    "allow_paper_imports",
}
#: Library modules that may repeat across studies (entry points and test scaffolding).
REPEATABLE_NAMES = frozenset({"run.py", "conftest.py", "__init__.py"})
#: Folder names at the repo root; a library of the same name would shadow them on sys.path.
REPO_ROOT_NAMES = frozenset(
    {"scripts", "tests", "data", "src", "docs", "research", "conftest", "ggt"}
)
#: Calls a driver's module level may make (constants and path set-up, never work).
MODULE_CALLS = frozenset(
    {
        "bootstrap", "study_dir", "repo_root", "Path", "Path(__file__).resolve", "re.compile",
        "frozenset", "set", "tuple", "dict", "str", "logging.getLogger", "os.environ.get",
        "matplotlib.use", "date", "pd.Timestamp",
    }
)  # fmt: skip
ROOT_LINE = re.compile(
    r"parents\[[0-9]+\]|\.parent\.parent|find_project_root|sys\.path\.(insert|append)|sys\.path\["
)
#: Result keys that change from run to run and are not numbers of the study.
NOISE_KEYS = frozenset({"provenance", "wall_seconds", "timings_s", "peak_rss_mb", "created_utc"})
PIN_SCHEMA = 1

README_TEMPLATE = """# $slug

**Question:** $question

**Kind:** $kind. **Status:** open. **Category:** $category (placement row $placement).

## Where things are

- Brief (registered plan): [$brief]($brief)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py), module [code/${mod}.py](code/${mod}.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/$category/$slug/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
"""

STUDY_TOML_TEMPLATE = """# Study card. Read by scripts/study.py check; edit by hand.
id = "$slug"
category = "$category"
placement_rule = $placement
question = "TODO: state the one question this study answers, in one sentence."
kind = "$kind"
status = "open"
verdict = ""
aliases = []
brief = "$brief"
reports = []
results = []
uses_lib = []
allow_paper_imports = false
"""

RUN_TEMPLATE = '''"""Driver for the $slug study: run it with --help for the options."""

from __future__ import annotations

import argparse
from pathlib import Path

from ggTrader.utils.binding import binding
from ggTrader.utils.io import write_json
from ggTrader.utils.paths import EVAL_END_17, EVAL_START_17
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import ${mod}  # noqa: E402  (found on sys.path through bootstrap)


def build_parser() -> argparse.ArgumentParser:
    """The command line: the evaluation window is pinned, never "today"."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-start", default=EVAL_START_17, help="first evaluation date")
    parser.add_argument("--eval-end", default=EVAL_END_17, help="last evaluation date")
    parser.add_argument(
        "--out",
        type=Path,
        default=study_dir(__file__) / "_${mod}_results.json",
        help="results JSON to write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the study and write its results JSON with a provenance block."""
    args = build_parser().parse_args(argv)
    result = ${mod}.run_study(args.eval_start, args.eval_end)
    payload = {
        "study": "$slug",
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "result": result,
        "provenance": binding(ROOT),
    }
    write_json(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

MODULE_TEMPLATE = '''"""Study logic for $slug: plain functions, no file or network access."""

from __future__ import annotations

from typing import Any


def run_study(eval_start: str, eval_end: str) -> dict[str, Any]:
    """Run the study over the window and return its numbers (replace this stub)."""
    raise NotImplementedError("write the $slug study here, test first")
'''

CONFTEST_TEMPLATE = '''"""Put the study's code and its category library on sys.path."""

from ggTrader.utils.study import bootstrap

bootstrap(__file__)
'''

TEST_TEMPLATE = '''"""Tests for the $slug study: write the failing test first."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.study

MODULE = importlib.import_module("${mod}")


def test_run_study_is_defined() -> None:
    assert callable(MODULE.run_study)
'''

REPORT_META = (
    "**Study:** $category/$slug. **Verdict:** $verdict. "
    "**Pinned run:** see PIN.json in the study folder."
)


class StudyError(Exception):
    """A user-facing failure: the message is printed and the exit status is 1."""


@dataclass(frozen=True)
class StudyRef:
    """A study folder located under a repository root."""

    root: Path
    path: Path
    category: str
    slug: str

    @property
    def rel(self) -> str:
        """The folder as a repo-relative POSIX path."""
        return self.path.relative_to(self.root).as_posix()


# --- small helpers ---------------------------------------------------------------------------


def fill(text: str, fields: dict[str, object]) -> str:
    """Substitute ``$name`` placeholders; a stray or unknown placeholder is an error."""
    try:
        return Template(text).substitute(fields)
    except (KeyError, ValueError) as exc:
        raise StudyError(f"template placeholder problem: {exc!r}") from exc


def snake(slug: str) -> str:
    """The module name for a slug: dashes become underscores."""
    return slug.replace("-", "_")


def title_of(slug: str) -> str:
    """A readable title for a slug."""
    return slug.replace("-", " ").title()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """Write text so a reader never sees half a file (temp file, then ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_file(path: Path, payload: Any) -> None:
    """Write indented JSON with a trailing newline."""
    write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def today() -> str:
    """Today's date, ISO formatted (called at run time, never as a default)."""
    return dt.date.today().isoformat()


def placement_table(root: Path) -> dict[int, str]:
    """Placement row -> category, read from the table in ``research/README.md``.

    Rows whose home is not a single category folder (rows 1 and 2) are absent. A root without a
    readable table uses ``DEFAULT_PLACEMENT``.
    """
    readme = root / "research" / "README.md"
    table: dict[int, str] = {}
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8").splitlines():
            match = PLACEMENT_ROW_RE.match(line)
            if match:
                table[int(match.group(1))] = match.group(2)
    return table or dict(DEFAULT_PLACEMENT)


def study_py_files(folder: Path) -> list[Path]:
    """Python files under a folder, skipping bytecode caches."""
    return sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)


def library_modules(root: Path) -> dict[str, list[str]]:
    """Importable stems of study code: ``stem -> [owner folder, ...]`` (repo-relative)."""
    found: dict[str, list[str]] = {}
    files = [
        *root.glob("research/*/*/code/**/*.py"),
        *root.glob("research/*/_lib/**/*.py"),
    ]
    for path in sorted(files):
        if "__pycache__" in path.parts:
            continue
        if path.name in REPEATABLE_NAMES or path.name.startswith("test_"):
            continue
        rel = path.relative_to(root).parts
        owner = "/".join(rel[:3])
        found.setdefault(path.stem, []).append(owner)
    return found


def locate_study(root: Path, raw: str) -> StudyRef:
    """Resolve a study folder given as an absolute path, cwd-relative or repo-relative path."""
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.insert(0, root / raw)
    for cand in candidates:
        if cand.is_dir():
            path = cand.resolve()
            try:
                parts = path.relative_to(root.resolve()).parts
            except ValueError as exc:
                raise StudyError(f"{path} is not inside the repo root {root}") from exc
            if len(parts) != 3 or parts[0] != "research":
                raise StudyError(f"{path} is not a study folder (research/<category>/<slug>)")
            return StudyRef(root.resolve(), path, parts[1], parts[2])
    raise StudyError(f"no such study folder: {raw}")


# --- new -------------------------------------------------------------------------------------


@functools.cache
def installed_modules() -> frozenset[str]:
    """Top-level module names of the installed distributions (looked up once)."""
    return frozenset(importlib.metadata.packages_distributions())


def shadow_reason(stem: str, taken: dict[str, list[str]]) -> str | None:
    """Why a library basename is unusable (shadows something, or repeats), else ``None``."""
    if stem in sys.stdlib_module_names:
        return "it shadows a standard-library module"
    if stem in installed_modules():
        return "it shadows an installed top-level module"
    if stem in REPO_ROOT_NAMES:
        return "it shadows a repo-root name"
    if stem in taken:
        return f"another study already has a module of that name ({', '.join(taken[stem])})"
    return None


def render_new(
    root: Path, category: str, slug: str, kind: str, placement: int, date: str
) -> dict[str, str]:
    """The files of a new study as ``{repo-relative path: text}`` (nothing is written)."""
    template = root / "research" / "TEMPLATE-brief.md"
    if not template.is_file():
        raise StudyError("research/TEMPLATE-brief.md is missing in this repo root")
    mod = snake(slug)
    brief = f"{date}-{slug}-brief.md"
    fields = {
        "slug": slug,
        "category": category,
        "kind": kind,
        "placement": placement,
        "date": date,
        "title": title_of(slug),
        "brief": brief,
        "mod": mod,
        "eval_start": PINNED_EVAL_START,
        "eval_end": PINNED_EVAL_END,
        "question": "TODO: state the one question this study answers, in one sentence.",
    }
    base = f"research/{category}/{slug}"
    return {
        f"{base}/README.md": fill(README_TEMPLATE, fields),
        f"{base}/study.toml": fill(STUDY_TOML_TEMPLATE, fields),
        f"{base}/{brief}": fill(template.read_text(encoding="utf-8"), fields),
        f"{base}/code/run.py": fill(RUN_TEMPLATE, fields),
        f"{base}/code/{mod}.py": fill(MODULE_TEMPLATE, fields),
        f"{base}/code/tests/conftest.py": CONFTEST_TEMPLATE,
        f"{base}/code/tests/test_{mod}.py": fill(TEST_TEMPLATE, fields),
    }


def cmd_new(args: argparse.Namespace) -> int:
    """Scaffold a new study folder."""
    root: Path = args.repo_root.resolve()
    if not (root / "research").is_dir():
        raise StudyError(f"{root} has no research/ folder: wrong --repo-root?")
    table = placement_table(root)
    if args.placement not in table:
        raise StudyError(
            f"placement row {args.placement} does not create a study folder "
            "(row 1 is a research-root document, row 2 a companion to an existing study); "
            f"rows that do: {', '.join(f'{k}={v}' for k, v in sorted(table.items()))}"
        )
    if table[args.placement] != args.category:
        raise StudyError(
            f"placement row {args.placement} maps to {table[args.placement]!r}, "
            f"not {args.category!r}"
        )
    if not SLUG_RE.match(args.slug):
        raise StudyError(f"slug {args.slug!r} must be lowercase words joined by single dashes")
    clash = sorted(root.glob(f"research/*/{args.slug}"))
    if clash:
        where = ", ".join(p.relative_to(root).as_posix() for p in clash)
        raise StudyError(
            f"slug {args.slug!r} already exists ({where}); a companion goes in that folder"
        )
    reason = shadow_reason(snake(args.slug), library_modules(root))
    if reason:
        raise StudyError(f"module name {snake(args.slug)!r} is unusable: {reason}")
    date = args.date or today()
    if not ISO_DATE_RE.match(date):
        raise StudyError(f"--date {date!r} is not YYYY-MM-DD")
    files = render_new(root, args.category, args.slug, args.kind, args.placement, date)
    for rel, text in files.items():
        write_text_atomic(root / rel, text)
    print(f"created research/{args.category}/{args.slug}/ ({len(files)} files)")
    for rel in files:
        print(f"  {rel}")
    print("next: fill in the brief (section 0 is machine-checked), then run:")
    print(f"  python scripts/study.py check research/{args.category}/{args.slug}")
    return 0


# --- check -----------------------------------------------------------------------------------


def _calls(node: ast.AST) -> list[str]:
    return [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]


def is_main_guard(node: ast.stmt) -> bool:
    """True for ``if __name__ == "__main__":``."""
    return (
        isinstance(node, ast.If)
        and ast.unparse(node.test).replace('"', "'") == "__name__ == '__main__'"
    )


def _eval_end_default_problem(call: ast.Call) -> str | None:
    """A problem with the ``default`` of an ``--eval-end`` argument, or ``None`` when fine."""
    required = any(
        k.arg == "required" and isinstance(k.value, ast.Constant) and k.value.value is True
        for k in call.keywords
    )
    default = next((k.value for k in call.keywords if k.arg == "default"), None)
    if default is None:
        return None if required else "--eval-end has no pinned default (argparse gives None)"
    text = ast.unparse(default)
    if isinstance(default, ast.Constant):
        value = default.value
        if value is None:
            return "--eval-end defaults to None (resolved to today later)"
        if isinstance(value, str) and not ISO_DATE_RE.match(value):
            return f"--eval-end default {value!r} is not a pinned YYYY-MM-DD date"
        return None
    if re.search(r"today|now|utcnow", text, re.IGNORECASE):
        return f"--eval-end defaults to today: {text}"
    return None


def driver_problems(source: str, label: str) -> list[str]:
    """AST rules for ``code/run.py``: argparse, ``main``, a guard, no side effects, pinned end."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{label}: does not parse ({exc.msg}, line {exc.lineno})"]
    problems: list[str] = []
    if not any(is_main_guard(n) for n in tree.body):
        problems.append(f'{label}: no `if __name__ == "__main__":` guard')
    if not any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
        problems.append(f"{label}: no main() function")
    calls = _calls(tree)
    if not any(c.endswith("ArgumentParser") for c in calls):
        problems.append(f"{label}: no argparse ArgumentParser")
    if not any(c.endswith("parse_args") for c in calls):
        problems.append(f"{label}: parse_args is never called")
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef | ast.ClassDef):
            continue
        if is_main_guard(node) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        ):
            continue
        for call in _calls(node):
            if call not in MODULE_CALLS:
                problems.append(f"{label}: module-level call {call} (top-level side effect)")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith("add_argument")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.replace("_", "-").lstrip("-") == "eval-end"
        ):
            issue = _eval_end_default_problem(node)
            if issue:
                problems.append(f"{label}: {issue}")
    return problems


def imported_modules(tree: ast.AST) -> set[str]:
    """Dotted names of absolute imports (``from a.b import c`` gives ``a.b`` and ``a.b.c``)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def boundary_problems(ref: StudyRef, allow_paper: bool, libs: dict[str, list[str]]) -> list[str]:
    """Import and root rules for every Python file under the study's ``code/`` folder."""
    problems: list[str] = []
    allowed_owners = {ref.rel, f"research/{ref.category}/_lib"}
    for path in study_py_files(ref.path / "code"):
        rel = path.relative_to(ref.root).as_posix()
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            problems.append(f"{rel}: does not parse ({exc.msg}, line {exc.lineno})")
            continue
        names = imported_modules(tree)
        tops = {n.split(".")[0] for n in names}
        if "vectorbt" in tops:
            problems.append(f"{rel}: imports vectorbt (only ggTrader.engine may)")
        if any(isinstance(n, ast.Attribute) and n.attr == "vbt" for n in ast.walk(tree)):
            problems.append(f"{rel}: uses the .vbt accessor (only ggTrader.engine may)")
        paper = any(n == "ggTrader.paper" or n.startswith("ggTrader.paper.") for n in names)
        if paper and not allow_paper:
            problems.append(f"{rel}: imports ggTrader.paper (study.toml allow_paper_imports=false)")
        for top in sorted(tops):
            for owner in libs.get(top, []):
                if owner not in allowed_owners:
                    problems.append(
                        f"{rel}: imports {top} from {owner} (another study or category)"
                    )
        for number, line in enumerate(source.splitlines(), start=1):
            if ROOT_LINE.search(line) and "# root-ok" not in line:
                problems.append(
                    f"{rel}:{number}: hand-rolled repo root or sys.path ({line.strip()})"
                )
    return problems


def basename_problems(ref: StudyRef, libs: dict[str, list[str]]) -> list[str]:
    """Library basenames of this study that repeat elsewhere or shadow an importable name."""
    problems: list[str] = []
    mine = {
        stem
        for stem, owners in libs.items()
        if ref.rel in owners or f"research/{ref.category}/_lib" in owners
    }
    for stem in sorted(mine):
        owners = libs[stem]
        if len(owners) > 1:
            problems.append(f"module name {stem!r} is defined more than once: {', '.join(owners)}")
        reason = shadow_reason(stem, {})
        if reason:
            problems.append(f"module name {stem!r}: {reason}")
    return problems


def parse_brief(text: str) -> dict[str, str]:
    """The ``- key: value`` lines of a brief for the registered keys."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = BRIEF_LINE_RE.match(line)
        if match and match.group(1) in BRIEF_KEYS and match.group(1) not in found:
            found[match.group(1)] = match.group(2)
    return found


def brief_problems(text: str, label: str) -> list[str]:
    """Missing or malformed registered fields in a brief."""
    fields = parse_brief(text)
    problems = [f"{label}: missing registered field {k}:" for k in BRIEF_KEYS if k not in fields]
    dates: dict[str, dt.date] = {}
    for key in ("eval_start", "eval_end"):
        if key in fields:
            try:
                dates[key] = dt.date.fromisoformat(fields[key])
            except ValueError:
                problems.append(f"{label}: {key} {fields[key]!r} is not a YYYY-MM-DD date")
    if len(dates) == 2 and dates["eval_start"] >= dates["eval_end"]:
        problems.append(f"{label}: eval_start is not before eval_end")
    if "max_leverage" in fields:
        try:
            if not float(fields["max_leverage"]) > 0:
                problems.append(f"{label}: max_leverage must be positive")
        except ValueError:
            problems.append(f"{label}: max_leverage {fields['max_leverage']!r} is not a number")
    if "benchmark" in fields and "SPY" not in fields["benchmark"]:
        problems.append(f"{label}: benchmark must name SPY")
    if "costs" in fields and not re.search(r"\d", fields["costs"]):
        problems.append(f"{label}: costs must state a number (bps per side)")
    if "family_size" in fields and not re.fullmatch(r"[1-9]\d*", fields["family_size"]):
        problems.append(f"{label}: family_size must be a positive integer")
    return problems


def toml_problems(card: dict[str, Any], ref: StudyRef, table: dict[int, str]) -> list[str]:
    """Schema problems of ``study.toml``."""
    problems = [f"study.toml: unknown key {k}" for k in sorted(set(card) - TOML_KEYS)]
    problems += [f"study.toml: missing key {k}" for k in sorted(TOML_KEYS - set(card))]

    def kind_of(key: str, want: type) -> bool:
        value = card.get(key)
        return isinstance(value, want) and not isinstance(value, bool)

    for key in ("id", "category", "question", "kind", "status", "verdict", "brief"):
        if key in card and not isinstance(card[key], str):
            problems.append(f"study.toml: {key} must be a string")
    for key in ("aliases", "reports", "results", "uses_lib"):
        if key in card and not (
            isinstance(card[key], list) and all(isinstance(v, str) for v in card[key])
        ):
            problems.append(f"study.toml: {key} must be a list of strings")
    if "allow_paper_imports" in card and not isinstance(card["allow_paper_imports"], bool):
        problems.append("study.toml: allow_paper_imports must be true or false")
    if kind_of("id", str) and card["id"] != ref.slug:
        problems.append(f"study.toml: id {card['id']!r} differs from the folder name {ref.slug!r}")
    if kind_of("category", str) and card["category"] != ref.category:
        problems.append(f"study.toml: category {card['category']!r} differs from the folder")
    if "placement_rule" in card:
        row = card["placement_rule"]
        if not kind_of("placement_rule", int):
            problems.append("study.toml: placement_rule must be an integer")
        elif table.get(row) != ref.category:
            problems.append(f"study.toml: placement_rule {row} does not map to {ref.category}")
    if kind_of("question", str) and not card["question"].strip():
        problems.append("study.toml: question is empty")
    if kind_of("kind", str) and card["kind"] not in KINDS:
        problems.append(f"study.toml: kind {card['kind']!r} not in {KINDS}")
    if kind_of("status", str) and card["status"] not in STATUSES:
        problems.append(f"study.toml: status {card['status']!r} not in {STATUSES}")
    if kind_of("verdict", str):
        if card["verdict"] and card["verdict"] not in VERDICTS:
            problems.append(f"study.toml: verdict {card['verdict']!r} not in {VERDICTS}")
        if card.get("status") == "closed" and not card["verdict"]:
            problems.append("study.toml: a closed study needs a verdict")
    return problems


def naming_problems(ref: StudyRef) -> list[str]:
    """Report and results file names at the top of the study folder."""
    problems: list[str] = []
    pattern = re.compile(
        rf"^\d{{4}}-\d{{2}}-\d{{2}}-{re.escape(ref.slug)}(-brief|-nogo|-amendment)?\.md$"
    )
    for path in sorted(ref.path.glob("*.md")):
        if path.name != "README.md" and not pattern.match(path.name):
            problems.append(
                f"{path.name}: expected YYYY-MM-DD-{ref.slug}[-brief|-nogo|-amendment].md"
            )
    for path in sorted(ref.path.glob("*.json")):
        if path.name != "PIN.json" and not path.name.startswith("_"):
            problems.append(f"{path.name}: results JSON files start with an underscore")
    return problems


def load_lens_checker() -> Any:
    """The ``check_report`` function of the sibling ``check_lens_report.py`` (loaded by path)."""
    path = Path(__file__).resolve().with_name("check_lens_report.py")
    spec = importlib.util.spec_from_file_location("study_check_lens_report", path)
    if spec is None or spec.loader is None:
        raise StudyError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.check_report


def provenance_problems(ref: StudyRef) -> list[str]:
    """Every results JSON must carry a non-empty provenance block."""
    problems: list[str] = []
    for path in sorted(ref.path.glob("_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: not readable JSON ({exc})")
            continue
        block = data.get("provenance") if isinstance(data, dict) else None
        if not isinstance(block, dict) or not block:
            problems.append(f"{path.name}: no provenance block")
    return problems


def pin_problems(ref: StudyRef) -> list[str]:
    """``PIN.json`` hashes must equal the files they name."""
    pin_path = ref.path / "PIN.json"
    if not pin_path.is_file():
        return []
    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"PIN.json: not valid JSON ({exc})"]
    problems: list[str] = []
    if pin.get("schema") != PIN_SCHEMA:
        problems.append(f"PIN.json: schema {pin.get('schema')!r}, expected {PIN_SCHEMA}")
    entries = [
        *pin.get("code", []),
        *pin.get("results", []),
        *pin.get("data", {}).get("inputs", []),
    ]
    for entry in entries:
        rel, want = entry.get("path", ""), entry.get("sha256", "")
        target = ref.root / rel
        if not target.is_file():
            problems.append(f"PIN.json: {rel} is missing")
        elif sha256_file(target) != want:
            problems.append(f"PIN.json: {rel} changed since it was pinned")
    return problems


def check_study(ref: StudyRef) -> list[str]:
    """All problems of a study folder (empty means it passes)."""
    problems: list[str] = []
    if not SLUG_RE.match(ref.slug):
        problems.append(f"folder name {ref.slug!r} must be lowercase words joined by dashes")
    table = placement_table(ref.root)
    if ref.category not in set(table.values()):
        problems.append(f"{ref.category!r} is not a study category ({sorted(set(table.values()))})")
    toml_path = ref.path / "study.toml"
    if not toml_path.is_file():
        return [*problems, "study.toml is missing"]
    try:
        card = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [*problems, f"study.toml: not valid TOML ({exc})"]
    problems += toml_problems(card, ref, table)
    problems += naming_problems(ref)

    check_report = load_lens_checker()
    documents = [card.get("brief"), *card.get("reports", [])]
    for name in [d for d in documents if isinstance(d, str) and d]:
        doc = ref.path / name
        if not doc.is_file():
            problems.append(f"{name}: listed in study.toml but missing")
            continue
        problems += [f"{name}: {p}" for p in check_report(doc, ref.root, paths_only=True)]
    brief_name = card.get("brief")
    if isinstance(brief_name, str) and (ref.path / brief_name).is_file():
        text = (ref.path / brief_name).read_text(encoding="utf-8")
        problems += brief_problems(text, brief_name)
    for name in card.get("results", []):
        if not (ref.path / name).is_file():
            problems.append(f"{name}: listed in study.toml results but missing")
    libs = library_modules(ref.root)
    for stem in card.get("uses_lib", []):
        if not ref.root.joinpath(f"research/{ref.category}/_lib/{stem}.py").is_file():
            problems.append(
                f"study.toml: uses_lib {stem!r} is not a module of this category's _lib"
            )

    run = ref.path / "code" / "run.py"
    if run.is_file():
        problems += driver_problems(run.read_text(encoding="utf-8"), f"{ref.rel}/code/run.py")
    else:
        problems.append(f"{ref.rel}/code/run.py is missing")
    allow_paper = card.get("allow_paper_imports") is True
    problems += boundary_problems(ref, allow_paper, libs)
    problems += basename_problems(ref, libs)
    problems += provenance_problems(ref)
    problems += pin_problems(ref)
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    """Check a study and print every problem."""
    ref = locate_study(args.repo_root, args.path)
    problems = check_study(ref)
    if problems:
        print(f"FAIL {ref.rel}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK {ref.rel}")
    return 0


# --- close -----------------------------------------------------------------------------------


def results_files(ref: StudyRef, names: list[str] | None) -> list[Path]:
    """The results JSON files a close pins: those named, else every ``_*.json`` in the folder."""
    if names:
        paths = [ref.path / n if not Path(n).is_absolute() else Path(n) for n in names]
    else:
        paths = sorted(ref.path.glob("_*.json"))
    if not paths:
        raise StudyError(
            "no results JSON found (_*.json in the study folder); run the driver first"
        )
    for path in paths:
        if not path.is_file():
            raise StudyError(f"results file {path} does not exist")
    return paths


def engine_fields(provenance: dict[str, Any]) -> dict[str, Any]:
    """The PIN ``engine`` block from a result's provenance block."""
    return {
        "git_sha": provenance.get("git_sha"),
        "git_dirty": provenance.get("git_dirty"),
        "diff_sha256": provenance.get("diff_sha256"),
        "tag": provenance.get("tag"),
        "uv_lock_sha256": provenance.get("lock_sha256"),
        "python": provenance.get("python"),
        "vectorbt": provenance.get("vectorbt"),
        "vectorbt_rust": provenance.get("vectorbt_rust"),
        "contract_version": provenance.get("contract_version"),
        "baseline_epoch": provenance.get("baseline_epoch"),
    }


def used_lib_files(ref: StudyRef, card: dict[str, Any]) -> list[Path]:
    """The category ``_lib`` modules a study uses (declared in study.toml or imported)."""
    lib = ref.root / "research" / ref.category / "_lib"
    stems = set(card.get("uses_lib", []))
    for path in study_py_files(ref.path / "code"):
        stems |= {m.split(".")[0] for m in imported_modules(ast.parse(path.read_text("utf-8")))}
    return sorted(p for s in stems if (p := lib / f"{s}.py").is_file())


def build_pin(
    ref: StudyRef, card: dict[str, Any], args: argparse.Namespace, created: str
) -> dict[str, Any]:
    """The ``PIN.json`` payload of a study being closed."""
    results = results_files(ref, args.results)
    blocks: list[dict[str, Any]] = []
    for path in results:
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("provenance") if isinstance(data, dict) else None
        if not isinstance(block, dict) or not block:
            raise StudyError(f"{path.name} has no provenance block: cannot pin it")
        if block.get("git_dirty") and not block.get("diff_sha256"):
            raise StudyError(
                f"{path.name} was produced on a dirty tree with no stored diff hash: "
                "commit the code and re-run before closing"
            )
        blocks.append(block)
    first = json.loads(results[0].read_text(encoding="utf-8"))
    tape_block = first.get("tape") if isinstance(first.get("tape"), dict) else None
    tape: dict[str, Any] | None = None
    if args.tape:
        tape_path = Path(args.tape)
        if not tape_path.is_file():
            raise StudyError(f"--tape {args.tape} does not exist")
        tape = {"path": str(args.tape), "sha256": sha256_file(tape_path), "content_sha256": None}
    elif tape_block and tape_block.get("sha256"):
        tape = {
            "path": tape_block.get("path"),
            "sha256": tape_block["sha256"],
            "content_sha256": tape_block.get("content_sha256"),
        }
    inputs = []
    for raw in args.input or []:
        path = Path(raw)
        if not path.is_file():
            raise StudyError(f"--input {raw} does not exist")
        try:
            rel = path.resolve().relative_to(ref.root).as_posix()
        except ValueError as exc:
            raise StudyError(f"--input {raw} is outside the repo root") from exc
        inputs.append({"path": rel, "sha256": sha256_file(path)})
    code_files = [*study_py_files(ref.path / "code"), *used_lib_files(ref, card)]
    brief_text = (ref.path / card["brief"]).read_text(encoding="utf-8")
    fields = parse_brief(brief_text)
    argv = shlex.split(args.argv) if args.argv else [".venv/bin/python", f"{ref.rel}/code/run.py"]
    return {
        "schema": PIN_SCHEMA,
        "study": {"id": ref.slug, "category": ref.category, "path": ref.rel},
        "engine": engine_fields(blocks[0]),
        "data": {"tape": tape, "inputs": inputs},
        "code": [
            {"path": p.relative_to(ref.root).as_posix(), "sha256": sha256_file(p)}
            for p in sorted(set(code_files))
        ],
        "command": {"argv": argv, "cwd": args.cwd},
        "windows": {"eval_start": fields.get("eval_start"), "eval_end": fields.get("eval_end")},
        "seed": args.seed,
        "results": [
            {"path": p.resolve().relative_to(ref.root).as_posix(), "sha256": sha256_file(p)}
            for p in results
        ],
        "reproducible": {"status": args.reproducible, "note": args.note},
        "created_utc": created,
    }


def render_report(root: Path, ref: StudyRef, verdict: str, date: str) -> str:
    """The report skeleton: the existing report template with the title, date and study line."""
    template = root / "research" / "TEMPLATE-research-report.md"
    if not template.is_file():
        raise StudyError("research/TEMPLATE-research-report.md is missing in this repo root")
    lines = template.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith("# <Report Title>"):
            line = f"# {title_of(ref.slug)}: {verdict} (skeleton, fill in)"
        elif line.startswith("**Date:**"):
            line = f"**Date:** {date}"
        out.append(line)
        if line.startswith("**Audience:**"):
            out.append(
                fill(REPORT_META, {"category": ref.category, "slug": ref.slug, "verdict": verdict})
            )
    return "\n".join(out) + "\n"


def cmd_close(args: argparse.Namespace) -> int:
    """Pin a finished study, write its report skeleton and print what to paste elsewhere."""
    ref = locate_study(args.repo_root, args.path)
    problems = check_study(ref)
    if problems:
        raise StudyError("check fails, fix before closing:\n  - " + "\n  - ".join(problems))
    card = tomllib.loads((ref.path / "study.toml").read_text(encoding="utf-8"))
    date = args.date or today()
    if not ISO_DATE_RE.match(date):
        raise StudyError(f"--date {date!r} is not YYYY-MM-DD")
    suffix = "-nogo" if args.verdict == "NO-GO" else ""
    report = ref.path / f"{date}-{ref.slug}{suffix}.md"
    pin_path = ref.path / "PIN.json"
    for target in (pin_path, report):
        if target.exists() and not args.force:
            raise StudyError(f"{target.name} already exists (use --force to replace it)")
    created = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    pin = build_pin(ref, card, args, created)
    skeleton = render_report(ref.root, ref, args.verdict, date)
    write_json_file(pin_path, pin)
    write_text_atomic(report, skeleton)
    rel_dir = f"{ref.category}/{ref.slug}"
    print(f"wrote {ref.rel}/PIN.json and {ref.rel}/{report.name}")
    print()
    print("RESEARCH_INDEX.md row (edit the family, work and scope cells):")
    print(
        f"| TODO family | {title_of(ref.slug)} | **{args.verdict}**: TODO one-line scope. | "
        f"[Report]({rel_dir}/{report.name}), [pin]({rel_dir}/PIN.json) |"
    )
    print()
    print("docs/changelog.md stub:")
    print(f"## {date} (study: {ref.slug})\n")
    print(
        f"- **Closed the {ref.slug} study: {args.verdict}.** TODO what was tested and the number."
    )
    print(f"  - Record: {ref.rel}/{report.name}, pinned in {ref.rel}/PIN.json.")
    print()
    print("still to do by hand:")
    print('  - in study.toml set status = "closed", verdict, reports and results;')
    print("  - fill in the report skeleton and delete its template notes;")
    print("  - run the research-snapshot skill so RESEARCH_SNAPSHOT.md catches up;")
    print(f"  - python scripts/study.py check {ref.rel}")
    return 0


# --- repro -----------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def numeric_differences(old: Any, new: Any, path: str = "") -> list[str]:
    """Differences between two result trees, ignoring the run-to-run noise keys."""
    if isinstance(old, dict) and isinstance(new, dict):
        found: list[str] = []
        for key in sorted(set(old) | set(new)):
            if key in NOISE_KEYS:
                continue
            if key not in old or key not in new:
                found.append(f"{path}/{key}: present in only one result")
            else:
                found += numeric_differences(old[key], new[key], f"{path}/{key}")
        return found
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} vs {len(new)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(old, new))
            for d in numeric_differences(a, b, f"{path}[{i}]")
        ]
    if _is_number(old) and _is_number(new):
        same = math.isclose(old, new, rel_tol=1e-9, abs_tol=1e-12) or (old != old and new != new)
        return [] if same else [f"{path}: {old!r} vs {new!r}"]
    return [] if old == new else [f"{path}: {old!r} vs {new!r}"]


def run_checked(argv: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run a command, echoing it; raise ``StudyError`` on a non-zero exit."""
    print("+ " + " ".join(shlex.quote(a) for a in argv), flush=True)
    done = subprocess.run(argv, cwd=cwd, env=env, check=False)
    if done.returncode != 0:
        raise StudyError(f"command failed with exit {done.returncode}: {argv[0]}")


def compare_provenance(old: dict[str, Any] | None, new: dict[str, Any] | None) -> str:
    """Provenance flag between two blocks, using the engine's own comparison when importable."""
    try:
        from ggTrader.engine import provenance as engine_provenance
    except ImportError as exc:
        return f"not compared (ggTrader.engine unavailable: {exc})"
    try:
        verdict = engine_provenance.compare(old, new, allow_cross_epoch=True)
    except (KeyError, ValueError) as exc:
        return f"not compared ({exc})"
    return f"{verdict.flag} (comparable={verdict.comparable}) {verdict.note}".strip()


def cmd_repro(args: argparse.Namespace) -> int:
    """Re-run a pinned study in a scratch clone and compare it with the recorded results."""
    ref = locate_study(args.repo_root, args.path)
    pin_path = ref.path / "PIN.json"
    if not pin_path.is_file():
        raise StudyError(f"{ref.rel}/PIN.json is missing: the study is not closed")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    engine = pin.get("engine", {})
    if engine.get("tag"):
        print(f"legacy-engine PIN (tag {engine['tag']}): rebuild its environment instead:")
        print(f"  scripts/legacy_env.sh   # then run: {' '.join(pin['command']['argv'])}")
        return 0
    sha = engine.get("git_sha")
    if not sha:
        raise StudyError("PIN.json records no git_sha")
    argv = list(pin["command"]["argv"])
    steps = [
        f"git clone --no-checkout {ref.root} <scratch>/repo",
        f"git -C <scratch>/repo checkout --detach {sha}",
        "compare uv.lock sha256 with " + str(engine.get("uv_lock_sha256")),
        *([] if args.no_sync else ["uv sync --frozen   (in the scratch clone)"]),
        *(
            [f"verify the tape sha256 {pin['data']['tape']['sha256']}"]
            if pin.get("data", {}).get("tape")
            else []
        ),
        "run "
        + " ".join(shlex.quote(a) for a in argv)
        + f"   (cwd <scratch>/repo/{pin['command']['cwd']})",
        "compare each pinned results file with the recorded one (numeric diff) and provenance",
    ]
    if engine.get("git_dirty"):
        steps.insert(
            0, f"WARNING: the recorded run was on a dirty tree (diff {engine.get('diff_sha256')})"
        )
    print("repro plan:")
    for step in steps:
        print(f"  - {step}")
    if args.print_only:
        return 0

    scratch = (
        Path(args.scratch_dir) if args.scratch_dir else Path(tempfile.mkdtemp(prefix="ggt-repro-"))
    )
    scratch.mkdir(parents=True, exist_ok=True)
    if any(scratch.iterdir()):
        raise StudyError(f"scratch folder {scratch} is not empty")
    repo = scratch / "repo"
    run_checked(["git", "clone", "--no-checkout", "-q", str(ref.root), str(repo)], scratch)
    run_checked(["git", "checkout", "-q", "--detach", sha], repo)
    lock = repo / "uv.lock"
    want_lock = engine.get("uv_lock_sha256")
    if want_lock and (not lock.is_file() or sha256_file(lock) != want_lock):
        raise StudyError("uv.lock at the pinned commit differs from the recorded lock hash")
    python = sys.executable
    if not args.no_sync:
        run_checked(["uv", "sync", "--frozen"], repo)
        python = str(repo / ".venv" / "bin" / "python")
    tape = pin.get("data", {}).get("tape")
    if tape:
        tape_path = Path(tape["path"])
        source = tape_path if tape_path.is_absolute() else ref.root / tape_path
        if not source.is_file() or sha256_file(source) != tape["sha256"]:
            raise StudyError(f"tape {tape['path']} is missing or its sha256 differs from the pin")
        if not tape_path.is_absolute():
            (repo / tape_path).parent.mkdir(parents=True, exist_ok=True)
            (repo / tape_path).symlink_to(source)
    if argv and argv[0] in {"python", "python3", ".venv/bin/python"}:
        argv[0] = python
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    run_checked(argv, repo / pin["command"]["cwd"], env)

    different = 0
    for entry in pin.get("results", []):
        recorded_path, fresh_path = ref.root / entry["path"], repo / entry["path"]
        if not fresh_path.is_file():
            print(f"MISSING {entry['path']}: the re-run did not write it")
            different += 1
            continue
        old = json.loads(recorded_path.read_text(encoding="utf-8"))
        new = json.loads(fresh_path.read_text(encoding="utf-8"))
        diffs = numeric_differences(old, new)
        flag = compare_provenance(old.get("provenance"), new.get("provenance"))
        if diffs:
            different += 1
            print(f"DIFFERS {entry['path']} ({len(diffs)} differences; provenance {flag})")
            for line in diffs[:20]:
                print(f"    {line}")
        else:
            print(f"SAME {entry['path']} (provenance {flag})")
    print(f"scratch clone kept at {repo}")
    return 1 if different else 0


# --- command line ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The command line (building it does no work)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="repository root to work in (default: the checkout holding this script)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="scaffold a study from the placement table")
    new.add_argument("category", help="category folder under research/")
    new.add_argument("slug", help="study folder name, unique across categories")
    new.add_argument("--kind", required=True, choices=KINDS)
    new.add_argument("--placement", required=True, type=int, choices=range(1, 10), metavar="1-9",
                     help="row of the placement test in research/README.md")  # fmt: skip
    new.add_argument("--date", help="date for the brief file name (default: today)")
    new.set_defaults(func=cmd_new)

    check = sub.add_parser("check", help="check a study folder; prints every problem")
    check.add_argument("path", help="study folder")
    check.set_defaults(func=cmd_check)

    repro = sub.add_parser("repro", help="re-run a pinned study in a scratch clone")
    repro.add_argument("path", help="study folder")
    repro.add_argument("--print-only", action="store_true", help="print the plan, run nothing")
    repro.add_argument("--scratch-dir", help="empty folder for the clone (default: a temp folder)")
    repro.add_argument("--no-sync", action="store_true", help="skip uv sync; use this interpreter")
    repro.set_defaults(func=cmd_repro)

    close = sub.add_parser("close", help="write PIN.json and a report skeleton")
    close.add_argument("path", help="study folder")
    close.add_argument("--verdict", required=True, choices=VERDICTS)
    close.add_argument("--date", help="report date (default: today)")
    close.add_argument(
        "--results", action="append", help="results JSON to pin (default: all _*.json)"
    )
    close.add_argument("--argv", help="command that produced the results, as one quoted string")
    close.add_argument(
        "--cwd", default=".", help="working directory of that command, repo-relative"
    )
    close.add_argument("--tape", help="tape file the run read (default: the results' tape block)")
    close.add_argument("--input", action="append", help="other frozen input file to hash")
    close.add_argument("--seed", type=int, default=None, help="random seed, if any")
    close.add_argument("--reproducible", choices=REPRODUCIBLE, default="PARTLY")
    close.add_argument("--note", default="", help="why the reproducible status is what it is")
    close.add_argument("--force", action="store_true", help="replace an existing PIN.json/report")
    close.set_defaults(func=cmd_close)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; a ``StudyError`` prints its message and returns 1."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except StudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
