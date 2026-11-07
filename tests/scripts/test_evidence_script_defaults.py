"""Evidence-regenerating scripts default their output to scratch, never into research/.

Static (AST) checks: the scripts are not imported or run. Writing next to the committed
evidence needs an explicit ``--out`` / ``--out-dir`` that names a path under ``research/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: script -> (the option that sets the output, the module constant that holds its default)
EVIDENCE_SCRIPTS = {
    "p5_wfo_validation.py": ("--out-dir", "DEFAULT_OUT"),
    "p7_live_golden.py": ("--out", "OUT"),
    "p3_pit_parity.py": ("--out", "DEFAULT_OUT"),
    "engine_parity_evidence.py": ("--out", "DEFAULT_OUT"),
    "legacy_reference_compare.py": ("--out", "DEFAULT_OUT"),
    "legacy_reference_runs.py": ("--out-dir", "OUT_DIR"),
}


def _tree(script: str) -> ast.Module:
    return ast.parse((ROOT / "scripts" / script).read_text())


def _constant_source(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.unparse(node.value)
    raise AssertionError(f"module constant {name} not found")


def _option_default(tree: ast.Module, option: str) -> str:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and any(isinstance(a, ast.Constant) and a.value == option for a in node.args)
        ):
            for kw in node.keywords:
                if kw.arg == "default":
                    return ast.unparse(kw.value)
    raise AssertionError(f"no default for {option}")


@pytest.mark.parametrize("script", sorted(EVIDENCE_SCRIPTS))
def test_the_default_output_is_scratch(script: str) -> None:
    option, constant = EVIDENCE_SCRIPTS[script]
    tree = _tree(script)
    source = _constant_source(tree, constant)
    assert "tempfile.gettempdir()" in source, f"{script}: {constant} = {source}"
    assert "research" not in source and "REPO_ROOT" not in source, f"{script}: {source}"
    default = _option_default(tree, option)
    assert constant in default and "research" not in default, f"{script}: {option} {default}"
