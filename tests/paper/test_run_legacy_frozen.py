"""The frozen copy of `PaperTrader._run` (`_run_legacy`) is a faithful, unedited oracle (Task 47).

Mutable-set file: Task 74 deletes it together with the copy and `golden/run_legacy.sha256`.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import textwrap
from pathlib import Path

import pytest
from tests.paper.equivalence.run_with import implementation

from ggTrader.paper.trader import PaperTrader

HASH_FILE = Path(__file__).parent / "golden" / "run_legacy.sha256"
FROZEN_AT = "73c35417"


def normalized_source(fn: object) -> str:
    """Dedented source, docstring line dropped, `def` name set to `_run`."""
    lines = textwrap.dedent(inspect.getsource(fn)).rstrip().splitlines()  # type: ignore[arg-type]
    head, body = lines[0], lines[1:]
    if body and body[0].strip().startswith('"""') and body[0].strip().endswith('"""'):
        body = body[1:]
    head = re.sub(r"^def _run_legacy\(", "def _run(", head)
    return "\n".join([head, *body]) + "\n"


def recorded_hash() -> str:
    """The sha256 recorded for `_run` at `73c35417` (first token of the one data line)."""
    lines = [
        line
        for line in HASH_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(lines) == 1, "run_legacy.sha256 must hold exactly one hash line"
    return lines[0].split()[0]


def test_run_legacy_exists_and_is_a_copy() -> None:
    legacy = PaperTrader.__dict__.get("_run_legacy")
    assert inspect.isfunction(legacy), "PaperTrader._run_legacy must be a plain function"
    original = PaperTrader.__dict__["_run"]
    assert legacy is not original
    assert list(inspect.signature(legacy).parameters.values()) == list(
        inspect.signature(original).parameters.values()
    )
    assert (
        inspect.signature(legacy).return_annotation == inspect.signature(original).return_annotation
    )
    assert FROZEN_AT in (legacy.__doc__ or "")


def test_run_legacy_source_is_frozen() -> None:
    digest = hashlib.sha256(normalized_source(PaperTrader._run_legacy).encode()).hexdigest()
    assert digest == recorded_hash()


def test_implementation_legacy_selects_run_legacy_and_restores_the_class() -> None:
    original = PaperTrader.__dict__["_run"]
    legacy = PaperTrader.__dict__["_run_legacy"]
    with implementation("legacy"):
        assert PaperTrader._run is PaperTrader._run_legacy
        assert PaperTrader._run is legacy
    assert PaperTrader.__dict__["_run"] is original
    assert PaperTrader.__dict__["_run_legacy"] is legacy


def test_implementation_legacy_restores_the_class_after_an_exception() -> None:
    original = PaperTrader.__dict__["_run"]
    with pytest.raises(KeyError, match="boom"):
        with implementation("legacy"):
            assert PaperTrader._run is PaperTrader._run_legacy
            raise KeyError("boom")
    assert PaperTrader.__dict__["_run"] is original
