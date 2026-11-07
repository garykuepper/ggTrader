"""scripts/verify_pins.py resolves recorded code hashes: parsing, resolution, shallow refusal.

The resolution tests build a throwaway git repository, so they need no real history. The one run
against the real repository (all 22 pins, every code map) is marked ``slow``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_pins.py"
CODEMOVED_HEADER = (
    "old_path,new_path,kind,base_commit,move_commit,sha256_before,sha256_after,do_not_edit,note"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_pins_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vp = _load()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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


def commit(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def codemoved(repo: Path, *rows: tuple[str, str, str]) -> None:
    """Write research/CODEMOVED.csv with (old_path, new_path, kind) rows (hashes are not read)."""
    lines = [CODEMOVED_HEADER]
    lines += [
        f"{old},{new},{kind},{'a' * 40},{'b' * 40},{'c' * 64},,no," for old, new, kind in rows
    ]
    write(repo, "research/CODEMOVED.csv", "\n".join(lines) + "\n")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo where src/a.py was edited, src/b.py moved, and src/gone.py deleted."""
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "src/a.py", "a = 1\n")
    write(tmp_path, "src/b.py", "b = 1\n")
    write(tmp_path, "src/gone.py", "gone = 1\n")
    codemoved(tmp_path)
    commit(tmp_path, "first")
    write(tmp_path, "src/a.py", "a = 2\n")
    commit(tmp_path, "edit a")
    (tmp_path / "study/code").mkdir(parents=True)
    git(tmp_path, "mv", "src/b.py", "study/code/b.py")
    git(tmp_path, "rm", "-q", "src/gone.py")
    codemoved(tmp_path, ("src/b.py", "study/code/b.py", "moved"), ("src/gone.py", "", "deleted"))
    commit(tmp_path, "move b, delete gone")
    return tmp_path


# --- collecting pins ----------------------------------------------------------------------


def test_collect_pins_reads_path_sha_pairs_and_path_keyed_maps(tmp_path: Path) -> None:
    good, other = sha("one"), sha("two")
    tree = {
        "inputs": [{"path": "src/a.py", "sha256": good, "note": "x"}],
        "code": {"files": {"scripts/run.py": other, "tests/test_x.py": good}},
        "docs": {"docs/plan.md": good, "path": "docs/plan.md", "sha256": good},
        "badhash": {"path": "src/c.py", "sha256": "not-a-hash"},
        "nested": [[{"path": "scripts/go.sh", "sha256": other}]],
    }
    write(tmp_path, "research/s/_report.json", json.dumps(tree))
    write(tmp_path, "research/s/broken.json", "{not json")
    pins = vp.collect_pins(tmp_path)
    assert sorted((p.path, p.sha256) for p in pins) == [
        ("scripts/go.sh", other),
        ("scripts/run.py", other),
        ("src/a.py", good),
        ("tests/test_x.py", good),
    ]
    assert {p.source for p in pins} == {"research/s/_report.json"}


# --- resolving pins -----------------------------------------------------------------------


def _resolve(repo: Path, path: str, text: str) -> tuple[str, str]:
    result = vp.resolve_pin(repo, vp.Pin("r.json", "", path, sha(text)), vp.read_codemoved(repo))
    return result.status, result.detail


def test_a_pin_on_unchanged_bytes_is_a_match(repo: Path) -> None:
    assert _resolve(repo, "src/a.py", "a = 2\n") == ("MATCH", "")


def test_a_pin_on_an_earlier_version_is_a_history_match(repo: Path) -> None:
    status, detail = _resolve(repo, "src/a.py", "a = 1\n")
    assert (
        status == "HISTORY-MATCH" and detail == git(repo, "rev-list", "--max-parents=0", "HEAD")[:8]
    )


def test_a_pin_on_a_moved_file_is_a_moved_match(repo: Path) -> None:
    assert _resolve(repo, "src/b.py", "b = 1\n") == ("MOVED-MATCH", "study/code/b.py")


def test_a_pin_on_a_deleted_file_resolves_in_history_skipping_the_deletion_commit(
    repo: Path,
) -> None:
    status, _ = _resolve(repo, "src/gone.py", "gone = 1\n")
    assert status == "HISTORY-MATCH"


def test_a_pin_on_bytes_that_never_existed_fails(repo: Path) -> None:
    assert _resolve(repo, "src/a.py", "a = 99\n")[0] == "FAIL"
    assert _resolve(repo, "src/nowhere.py", "x\n")[0] == "FAIL"


def test_a_moved_file_edited_after_the_move_falls_back_to_history(repo: Path) -> None:
    write(repo, "study/code/b.py", "b = 2\n")
    commit(repo, "edit moved b")
    assert _resolve(repo, "src/b.py", "b = 2\n")[0] == "MOVED-MATCH"
    assert _resolve(repo, "src/b.py", "b = 1\n")[0] == "HISTORY-MATCH"


# --- code maps ----------------------------------------------------------------------------


def _map_hash(files: dict[str, str]) -> str:
    return vp.canonical_hash({name: sha(text) for name, text in files.items()})


def test_canonical_hash_is_sorted_compact_json() -> None:
    body = json.dumps({"b": "2", "a": "1"}, sort_keys=True, separators=(",", ":")).encode()
    assert vp.canonical_hash({"b": "2", "a": "1"}) == hashlib.sha256(body).hexdigest()


def test_a_code_map_resolves_at_the_named_commit_or_by_searching_history(repo: Path) -> None:
    first = git(repo, "rev-list", "--max-parents=0", "HEAD")
    recorded = _map_hash({"src/a.py": "a = 1\n", "src/b.py": "b = 1\n"})
    write(repo, "research/s/_run.json", json.dumps({"id": {"code": recorded}, "commit": first}))
    path = repo / "research/s/_run.json"
    files = ("src/a.py", "src/b.py")

    named = vp.MapSpec("", ("id", "code"), files, commit_pointer=("commit",))
    assert vp.resolve_map(repo, path, named).detail == first[:8]
    searched = vp.MapSpec("", ("id", "code"), files)
    assert vp.resolve_map(repo, path, searched).status == "HISTORY-MATCH"
    hinted = vp.MapSpec("", ("id", "code"), files, commit_hint=first[:8])
    assert vp.resolve_map(repo, path, hinted).detail == first[:8]


def test_a_code_map_that_matches_no_commit_fails(repo: Path) -> None:
    write(repo, "research/s/_run.json", json.dumps({"code": "0" * 64}))
    spec = vp.MapSpec("", ("code",), ("src/a.py",))
    assert vp.resolve_map(repo, repo / "research/s/_run.json", spec).status == "FAIL"
    missing = vp.MapSpec("", ("nothing", "here"), ("src/a.py",))
    assert vp.resolve_map(repo, repo / "research/s/_run.json", missing).status == "FAIL"


# --- the command line ---------------------------------------------------------------------


def _pins_repo(repo: Path) -> Path:
    pins = [
        {"path": "src/a.py", "sha256": sha("a = 2\n")},
        {"path": "src/a.py", "sha256": sha("a = 1\n")},
        {"path": "src/b.py", "sha256": sha("b = 1\n")},
    ]
    write(repo, "research/s/_report.json", json.dumps({"pins": pins}))
    commit(repo, "pins")
    return repo


def run_cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_summary_prints_one_line_with_the_counts(repo: Path) -> None:
    done = run_cli(_pins_repo(repo), "--summary")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == ("pins=3 MATCH=1 MOVED-MATCH=1 HISTORY-MATCH=1 FAIL=0 maps=0/0")


def test_expect_passes_on_exact_counts_and_fails_on_any_deviation(repo: Path) -> None:
    _pins_repo(repo)
    ok = run_cli(repo, "--summary", "--expect", "MATCH=1,MOVED-MATCH=1,HISTORY-MATCH=1,FAIL=0")
    assert ok.returncode == 0
    wrong = run_cli(repo, "--summary", "--expect", "MATCH=2")
    assert wrong.returncode == 1 and "expected MATCH=2, got 1" in wrong.stderr
    bad_syntax = run_cli(repo, "--summary", "--expect", "SOMETHING=1")
    assert bad_syntax.returncode == 2


def test_a_failing_pin_exits_one_even_without_expect(repo: Path) -> None:
    write(repo, "research/s/_report.json", json.dumps({"path": "src/a.py", "sha256": sha("zz")}))
    commit(repo, "bad pin")
    done = run_cli(repo)
    assert done.returncode == 1 and "FAIL" in done.stdout


def test_a_shallow_clone_is_refused_with_exit_two(repo: Path, tmp_path_factory) -> None:
    _pins_repo(repo)
    clone = tmp_path_factory.mktemp("shallow") / "clone"
    git(repo, "clone", "-q", "--depth", "1", f"file://{repo}", str(clone))
    assert vp.is_shallow(clone) and not vp.is_shallow(repo)
    done = run_cli(clone, "--summary")
    assert done.returncode == 2 and "shallow" in done.stdout.lower()


def test_codemoved_is_required(repo: Path) -> None:
    (repo / "research/CODEMOVED.csv").unlink()
    commit(repo, "remove ledger")
    with pytest.raises(FileNotFoundError):
        vp.read_codemoved(repo)


def test_read_codemoved_ignores_deleted_rows(repo: Path) -> None:
    with (repo / "research/CODEMOVED.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert vp.read_codemoved(repo) == {"src/b.py": "study/code/b.py"}


# --- the real repository ------------------------------------------------------------------


@pytest.mark.slow
def test_every_pin_and_code_map_in_the_real_research_tree_resolves() -> None:
    if vp.is_shallow(ROOT):
        pytest.skip("shallow clone: the history pins cannot be resolved")
    report = vp.verify(ROOT)
    assert len(report.pins) == 22
    assert len(report.maps) == 5
    assert [r.label for r in [*report.pins, *report.maps] if r.status == "FAIL"] == []
