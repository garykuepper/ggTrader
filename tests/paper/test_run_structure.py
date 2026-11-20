"""The size ratchet on `PaperTrader.run` and `PaperTrader._run` (Task 50).

Mutable structural set: reviewed per task, not hash-recorded.

Ratchet protocol. `RUN_MAX_LINES` bounds the line count of `PaperTrader._run` (its `def` line to its
last line, measured on the source with `ast`) and only ever goes down. A task that cuts a region out
of `_run` (Tasks 57-63) first lowers the constant to the size it expects, sees
`test_run_stays_an_orchestrator` fail, then makes the cut and sets the constant to the measured
line count rounded up to the next multiple of 5. `test_the_bound_is_tight` rejects a bound more than
4 lines above the measurement, so a cut that forgets to ratchet is caught on the next run. Expected
values after each task: 57 about 740, 58 about 650, 59 about 575, 60 about 470, 61 about 380, 62
about 300, 63 at most 45; Task 74 keeps 45. `run()` stays at 25 lines or fewer throughout.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.paper.test_seam_contract import GENEXP_ALLOWED

TRADER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "ggTrader" / "paper" / "trader.py"

#: Upper bound on `PaperTrader._run`, in lines. 826 at 73c35417.
RUN_MAX_LINES = 240
#: Upper bound on `PaperTrader.run`, in lines. 22 at 73c35417.
RUN_METHOD_MAX_LINES = 25
RATCHET_STEP = 5


def method_lines(source: str, class_name: str, method: str) -> int:
    """Lines from the `def` line to the last line of `class_name.method`, measured with `ast`."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method:
                    assert member.end_lineno is not None
                    return member.end_lineno - member.lineno + 1
    raise LookupError(f"{class_name}.{method} not found")


def _trader_source() -> str:
    return TRADER_SOURCE.read_text(encoding="utf-8")


def ratchet_problem(measured: int, bound: int) -> str | None:
    """Why `bound` is not a valid ratchet for a method of `measured` lines, or ``None`` if it is."""
    if bound < measured:
        return f"{measured} lines exceed the bound {bound}; a cut may only make _run smaller"
    if bound % RATCHET_STEP:
        return f"the bound {bound} is not a multiple of {RATCHET_STEP}"
    if bound >= measured + RATCHET_STEP:
        tight = -(-measured // RATCHET_STEP) * RATCHET_STEP
        return f"the bound {bound} is loose: set RUN_MAX_LINES to {tight}"
    return None


def test_run_stays_an_orchestrator():
    """`_run` fits the ratchet and `run()` stays a short wrapper."""
    source = _trader_source()
    run_lines = method_lines(source, "PaperTrader", "run")
    body_lines = method_lines(source, "PaperTrader", "_run")
    assert run_lines <= RUN_METHOD_MAX_LINES, (
        f"run() is {run_lines} lines (max {RUN_METHOD_MAX_LINES})"
    )
    assert body_lines <= RUN_MAX_LINES, (
        f"_run is {body_lines} lines (ratchet {RUN_MAX_LINES}); a cut may only make it smaller"
    )


def test_the_bound_is_tight():
    """The constant is the measured size rounded up to the next 5; an unratcheted cut fails."""
    measured = method_lines(_trader_source(), "PaperTrader", "_run")
    assert ratchet_problem(measured, RUN_MAX_LINES) is None


def test_the_ratchet_rule_rejects_each_wrong_bound():
    assert ratchet_problem(826, 830) is None
    assert ratchet_problem(825, 825) is None
    assert ratchet_problem(826, 825) is not None  # the code outgrew the bound
    assert ratchet_problem(826, 831) is not None  # not a multiple of 5
    assert ratchet_problem(826, 835) is not None  # a cut that forgot to ratchet
    assert ratchet_problem(740, 830) is not None  # the same, after a real cut
    assert ratchet_problem(45, 45) is None


def test_the_line_counter_measures_a_method_and_finds_a_missing_one():
    source = (
        "class A:\n"
        "    def run(self):\n"
        "        '''Doc.'''\n"
        "        return 1\n"
        "\n"
        "    def _run(self, *, stop):\n"
        "        x = 1\n"
        "        y = 2\n"
        "        return x + y\n"
        "\n"
        "class B:\n"
        "    def run(self):\n"
        "        pass\n"
    )
    assert method_lines(source, "A", "run") == 3
    assert method_lines(source, "A", "_run") == 4
    assert method_lines(source, "B", "run") == 2
    for class_name, method in (("A", "missing"), ("C", "run")):
        try:
            method_lines(source, class_name, method)
        except LookupError:
            continue
        raise AssertionError(f"{class_name}.{method} should not be found")


def test_the_listed_generator_expressions_come_from_the_run_being_replaced():
    """Each generator expression allowed in `reporter.py` is one `_run` has today.

    `GENEXP_ALLOWED` (test_seam_contract.py) lists the three of region R7 by `ast.unparse` text and
    by the line they have in trader.py at 73c35417. While the frozen copy of `_run` exists (until
    Task 74 deletes it, and this check with it) its generator expressions are those three exactly.
    """
    from ggTrader.paper.trader import PaperTrader

    if not hasattr(PaperTrader, "_run_legacy"):
        return
    tree = ast.parse(_trader_source())
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PaperTrader":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "_run_legacy":
                    found = [
                        ast.unparse(n) for n in ast.walk(member) if isinstance(n, ast.GeneratorExp)
                    ]
    assert sorted(found) == sorted(text for text, _line in GENEXP_ALLOWED)
