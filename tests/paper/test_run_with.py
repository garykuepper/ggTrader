"""The implementation switch (`implementation`, `run_with`) and the `impl.py` list (Task 47).

Immutable behavioral set: hash-recorded at Task 51. It never names the frozen copy.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path

import pytest

from ggTrader.paper.trader import PaperTrader
from tests.paper.equivalence import impl
from tests.paper.equivalence import run_with as run_with_module
from tests.paper.transcript.scenarios import SCENARIOS

IMPL_FILE = Path(impl.__file__)


def _scenario():
    return next(iter(SCENARIOS.values()))


def test_run_with_rejects_an_implementation_not_in_impls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(impl, "IMPLS", ("new",))
    with pytest.raises(ValueError):
        run_with_module.run_with("legacy", _scenario())


def test_impl_py_holds_only_impls() -> None:
    tree = ast.parse(IMPL_FILE.read_text())
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # the module docstring
    assert len(body) == 1, "impl.py holds one statement besides its docstring"
    (stmt,) = body
    assert isinstance(stmt, ast.Assign | ast.AnnAssign)
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    assert [t.id for t in targets if isinstance(t, ast.Name)] == ["IMPLS"]
    assert not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Import)
        or isinstance(node, ast.ImportFrom)
        for node in ast.walk(tree)
    )


def test_implementation_new_changes_nothing() -> None:
    before = dict(vars(PaperTrader))
    with run_with_module.implementation("new"):
        assert dict(vars(PaperTrader)) == before
    assert dict(vars(PaperTrader)) == before


def test_implementation_rejects_an_impl_not_in_impls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(impl, "IMPLS", ("new",))
    before = dict(vars(PaperTrader))
    with pytest.raises(ValueError):
        with run_with_module.implementation("legacy"):
            pytest.fail("the body of a rejected implementation must not run")
    with pytest.raises(ValueError):
        with run_with_module.implementation("no-such-implementation"):
            pytest.fail("the body of a rejected implementation must not run")
    assert dict(vars(PaperTrader)) == before


def test_run_with_is_run_scenario_under_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple] = []
    real = run_with_module.implementation
    scenario = _scenario()

    @contextmanager
    def counting_implementation(name: str):
        events.append(("enter", name))
        with real(name):
            yield
        events.append(("exit", name))

    def counting_run_scenario(s):
        events.append(("run_scenario", s))
        return {"sentinel": 1}

    monkeypatch.setattr(run_with_module, "implementation", counting_implementation)
    monkeypatch.setattr(run_with_module, "run_scenario", counting_run_scenario)
    assert run_with_module.run_with("new", scenario) == {"sentinel": 1}
    assert events == [("enter", "new"), ("run_scenario", scenario), ("exit", "new")]
