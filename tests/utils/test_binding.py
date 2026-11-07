"""The code binding stamped on a result matches what the engine's provenance records."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ggTrader.engine.provenance import provenance
from ggTrader.utils import binding as binding_module
from ggTrader.utils.paths import REPO_ROOT


def test_binding_agrees_with_engine_provenance() -> None:
    block = provenance()
    mine = binding_module.binding()
    assert mine["git_sha"] == block["git_sha"]
    assert mine["lock_sha256"] == block["lock_sha256"]
    assert mine["git_dirty"] == block["git_dirty"]
    assert mine["python"] == block["python"]


def test_binding_works_outside_a_git_checkout(tmp_path: Path) -> None:
    mine = binding_module.binding(tmp_path)
    assert mine["git_sha"] is None
    assert mine["lock_sha256"] is None
    assert mine["diff_sha256"] is None
    assert mine["git_dirty"] is False


def test_diff_hash_is_none_for_a_clean_tree_and_a_hash_after_a_change(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "f.txt").write_text("one\n")
    (tmp_path / "uv.lock").write_text("lock\n")
    git("add", ".")
    git("commit", "-q", "-m", "x")
    assert binding_module.diff_sha256(tmp_path) is None
    assert binding_module.git_dirty(tmp_path) is False
    (tmp_path / "f.txt").write_text("two\n")
    digest = binding_module.diff_sha256(tmp_path)
    assert digest is not None and len(digest) == 64
    assert binding_module.git_dirty(tmp_path) is True
    assert binding_module.git_sha(tmp_path) is not None
    assert binding_module.lock_sha256(tmp_path) is not None


def test_importing_binding_does_not_import_vectorbt() -> None:
    code = (
        "import sys, ggTrader.utils.binding;"
        "bad = [m for m in sys.modules if m.split('.')[0] == 'vectorbt'];"
        "sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT)
    assert done.returncode == 0
