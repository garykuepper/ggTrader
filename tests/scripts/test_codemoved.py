"""research/CODEMOVED.csv is well formed, honest about hashes, and no code still names a moved file.

CODEMOVED is the append-only ledger of every ``src/``, ``scripts/`` or ``tests/`` file that the
study-isolation work moved or deleted. Three promises are tested here:

1. Schema: each row has the nine columns, valid hashes and a consistent ``kind``.
2. ``do_not_edit=yes`` rows (byte-identical moves that evidence pins) still hash to
   ``sha256_after`` at ``new_path``, and no stub is left at an ``old_path``.
3. Reference sweep: for every moved or deleted ``lab/X.py``, ``utils/X.py`` or ``scripts/X.py``
   row, nothing in the sweep scope still imports or runs it. Exempt: ``trader.py`` (comments
   that cite a deleted script), ``lens2_build.py`` (frozen prose), ``test_driver_default_paths``
   (historical names resolved through CODEMOVED), and lines marked ``# history-read``.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
CODEMOVED = ROOT / "research/CODEMOVED.csv"
HEADER = [
    "old_path",
    "new_path",
    "kind",
    "base_commit",
    "move_commit",
    "sha256_before",
    "sha256_after",
    "do_not_edit",
    "note",
]
KINDS = {"moved", "moved+edited", "deleted"}
SHA = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
OLD_PREFIXES = ("src/", "scripts/", "tests/")

#: The files the reference sweep reads (git pathspecs; research code only, never research docs).
SWEEP_SPECS = (
    "src",
    "scripts",
    "tests",
    "ggt.py",
    "pyproject.toml",
    "ruff.toml",
    "Dockerfile",
    ".github",
    ".claude",
    ":(glob)research/*/*/code/**",
    ":(glob)research/*/_lib/**",
)
SWEEP_EXEMPT = frozenset(
    {
        "src/ggTrader/paper/trader.py",
        "scripts/lens2_build.py",
        "research/audits/root-cause-2026-11-04/code/lens2_build.py",
        "tests/scripts/test_driver_default_paths.py",
    }
)
HISTORY_MARK = "# history-read"


def _rows() -> list[dict[str, str]]:
    with CODEMOVED.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True).returncode == 0


def schema_problems(rows: list[dict[str, str]], header: list[str]) -> list[str]:
    """Everything wrong with the shape of the ledger (empty when it is well formed)."""
    bad: list[str] = [] if header == HEADER else [f"header is {header}"]
    for row in rows:
        old = row["old_path"]
        where = f"{old}:"
        if not old.startswith(OLD_PREFIXES):
            bad.append(f"{where} old_path must be under src/, scripts/ or tests/")
        if row["kind"] not in KINDS:
            bad.append(f"{where} kind {row['kind']!r}")
        for column in ("base_commit", "move_commit"):
            if not COMMIT.match(row[column]):
                bad.append(f"{where} {column} is not a full commit id")
        if not SHA.match(row["sha256_before"]):
            bad.append(f"{where} sha256_before is not a sha256")
        if row["do_not_edit"] not in {"yes", "no"}:
            bad.append(f"{where} do_not_edit {row['do_not_edit']!r}")
        if row["kind"] == "deleted":
            if row["new_path"] or row["sha256_after"]:
                bad.append(f"{where} a deleted row has no new_path or sha256_after")
            if row["do_not_edit"] == "yes":
                bad.append(f"{where} a deleted file cannot be do_not_edit")
        else:
            if not row["new_path"]:
                bad.append(f"{where} a moved row needs a new_path")
            if not SHA.match(row["sha256_after"]):
                bad.append(f"{where} sha256_after is not a sha256")
            if row["kind"] == "moved" and row["sha256_after"] != row["sha256_before"]:
                bad.append(f"{where} kind moved but the hashes differ (use moved+edited)")
            if row["do_not_edit"] == "yes" and row["sha256_after"] != row["sha256_before"]:
                bad.append(f"{where} do_not_edit but the hashes differ")
    return bad


def reference_patterns(old_path: str) -> list[re.Pattern[str]] | None:
    """Regexes that name a moved or deleted module, or ``None`` when the rule does not cover it."""
    parts = PurePosixPath(old_path)
    if old_path.startswith("src/ggTrader/lab/") or old_path.startswith("src/ggTrader/utils/"):
        if parts.suffix != ".py" or len(parts.parts) != 4:
            return None
        pkg, stem = parts.parts[2], parts.stem
        return [
            re.compile(rf"ggTrader\.{pkg}\.{stem}\b"),
            re.compile(rf"{pkg}/{stem}\.py"),
        ]
    if old_path.startswith("scripts/"):
        rel = old_path[len("scripts/") :]
        patterns = [re.compile(rf"scripts/{re.escape(rel)}(?![\w])")]
        if "/" not in rel and parts.suffix == ".py":
            patterns.append(re.compile(rf"\bfrom scripts\.{parts.stem}\b"))
            patterns.append(re.compile(rf"\bimport scripts\.{parts.stem}\b"))
        return patterns
    return None


def _from_import_names(text: str, package: str) -> list[tuple[int, str]]:
    """``(line, name)`` for each name in ``from <package> import a, (b, c)`` statements."""
    found: list[tuple[int, str]] = []
    pattern = re.compile(rf"^[ \t]*from {re.escape(package)} import[ \t]*(\([^)]*\)|[^\n]*)", re.M)
    for match in pattern.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        body = re.sub(r"#[^\n]*", "", match.group(1)).strip("() \t\n")
        for token in re.split(r"[,\s]+", body):
            name = token.strip()
            if name and name != "as":
                found.append((line, name))
    return found


def find_references(rel: str, text: str, old_paths: list[str]) -> list[str]:
    """References in one file to any moved or deleted module (honouring ``# history-read``)."""
    lines = text.splitlines()
    hits: list[str] = []

    def record(number: int, old: str) -> None:
        if HISTORY_MARK not in lines[number - 1]:
            hits.append(f"{rel}:{number}: names {old}")

    for old in old_paths:
        patterns = reference_patterns(old)
        if patterns is None:
            continue
        for number, line in enumerate(lines, start=1):
            if any(p.search(line) for p in patterns):
                record(number, old)
        stem = PurePosixPath(old).stem
        if old.startswith("src/ggTrader/"):
            package = f"ggTrader.{PurePosixPath(old).parts[2]}"
        elif "/" not in old[len("scripts/") :]:
            package = "scripts"
        else:
            continue
        for number, name in _from_import_names(text, package):
            if name == stem:
                record(number, old)
    return hits


def _sweep_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *SWEEP_SPECS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(
        p for p in set(out.splitlines()) if p not in SWEEP_EXEMPT and (ROOT / p).is_file()
    )


# --- the real ledger ----------------------------------------------------------------------


def test_codemoved_has_the_schema() -> None:
    with CODEMOVED.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert schema_problems(_rows(), header) == []


def test_do_not_edit_rows_still_hash_to_the_recorded_value() -> None:
    for row in _rows():
        if row["do_not_edit"] == "yes":
            assert _sha(ROOT / row["new_path"]) == row["sha256_after"], row["new_path"]


def test_moved_rows_name_a_file_that_exists() -> None:
    for row in _rows():
        if row["kind"] != "deleted":
            assert (ROOT / row["new_path"]).is_file(), row["new_path"]


def test_no_stub_is_left_at_an_old_path() -> None:
    assert [r["old_path"] for r in _rows() if (ROOT / r["old_path"]).exists()] == []


def test_sha256_before_matches_the_base_commit() -> None:
    rows = _rows()
    base = rows[0]["base_commit"] if rows else ""
    if not base or not _git_ok("cat-file", "-e", f"{base}^{{commit}}"):
        pytest.skip("the base commit is not in this clone (shallow checkout)")
    for row in rows:
        done = subprocess.run(
            ["git", "show", f"{row['base_commit']}:{row['old_path']}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        assert hashlib.sha256(done.stdout).hexdigest() == row["sha256_before"], row["old_path"]


def test_nothing_in_the_sweep_scope_names_a_moved_or_deleted_module() -> None:
    old_paths = sorted({r["old_path"] for r in _rows()})
    hits: list[str] = []
    for rel in _sweep_files():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += find_references(rel, text, old_paths)
    assert hits == []


# --- the rules have teeth -----------------------------------------------------------------


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "old_path": "scripts/widget.py",
        "new_path": "research/a/s/code/run.py",
        "kind": "moved",
        "base_commit": "a" * 40,
        "move_commit": "b" * 40,
        "sha256_before": "c" * 64,
        "sha256_after": "c" * 64,
        "do_not_edit": "no",
        "note": "",
    }
    return {**row, **overrides}


def test_schema_accepts_a_good_row_and_rejects_each_kind_of_bad_row() -> None:
    assert schema_problems([_row()], HEADER) == []
    assert schema_problems([_row()], HEADER[:-1])
    bad_rows = [
        _row(old_path="docs/x.md"),
        _row(kind="renamed"),
        _row(base_commit="46d8ff75"),
        _row(sha256_before="xyz"),
        _row(do_not_edit="maybe"),
        _row(kind="deleted"),  # still has new_path and sha256_after
        _row(sha256_after="d" * 64),  # kind moved but hashes differ
        _row(kind="moved+edited", sha256_after="d" * 64, do_not_edit="yes"),
        _row(new_path=""),
    ]
    for row in bad_rows:
        assert schema_problems([row], HEADER), row


def test_reference_sweep_finds_each_spelling_and_honours_history_read() -> None:
    old = [
        "src/ggTrader/lab/widget.py",
        "src/ggTrader/utils/gadget.py",
        "scripts/tool.py",
        "scripts/one_off/once.py",
    ]
    text = "\n".join(
        [
            "import ggTrader.lab.widget as w",  # 1
            "from ggTrader.lab import other, widget",  # 2
            "from ggTrader.lab import (",  # 3
            "    other,",
            "    widget,",
            ")",
            "x = 'src/ggTrader/lab/widget.py'",  # 7
            "from ggTrader.utils import gadget",  # 8
            "from scripts import tool",  # 9
            "run('scripts/tool.py')",  # 10
            "run('scripts/one_off/once.py')",  # 11
            "import ggTrader.lab.widget  # history-read",  # 12
            "from ggTrader.lab import widget_two, other",  # 13  (different module)
            "import ggTrader.lab.widgets",  # 14 (different module)
            "run('scripts/tool.pyc')",  # 15 (different file)
        ]
    )
    lines = [int(h.split(":")[1]) for h in find_references("x.py", text, old)]
    assert sorted(lines) == [1, 2, 3, 7, 8, 9, 10, 11]


def test_reference_sweep_ignores_names_the_rules_do_not_cover() -> None:
    assert reference_patterns("tests/lab/test_widget.py") is None
    assert reference_patterns("src/ggTrader/lab/sub/deep.py") is None
