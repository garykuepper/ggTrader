"""Tests of `PgPaperLedger` and of the one-line delegations of `paper/persist.py` (Task 38).

The SQL golden (`golden/persist_sql.json`) is the immutable reference recorded from the 31 module
functions before they moved into `PgPaperLedger`. These tests prove, statement for statement,
that the delegations and the class methods still emit exactly what it records, except the one
stray bind key explained in `normalize_reference`.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from ggTrader.paper import persist
from ggTrader.paper.ledger import PaperLedger
from tests.paper import test_persist_sql_golden as golden_module
from tests.paper.sql_recorder import PERSIST_ENGINE, RecordingEngine, run_case
from tests.paper.test_persist_sql_golden import normalize_reference

GOLDEN_PATH = Path(__file__).parent / "golden" / "persist_sql.json"

#: The longest a delegation body may be, in source lines (docstring excluded).
MAX_DELEGATION_BODY_LINES = 3

#: The two deliberate annotation differences between the protocol and the persist functions
#: (`ledger.py` module docstring): `log_snapshot`'s positions and `paper_run_lock`'s return.
KNOWN_ANNOTATION_DIFFERENCES = {("log_snapshot", "positions"), ("paper_run_lock", "return")}


def _pg_class() -> type:
    return persist.PgPaperLedger  # attribute lookup, so a missing class fails each test alone


def _public_persist_functions() -> dict[str, Callable]:
    return {
        name: fn
        for name, fn in inspect.getmembers(persist, inspect.isfunction)
        if not name.startswith("_") and fn.__module__ == persist.__name__
    }


def _protocol_methods() -> dict[str, Callable]:
    return {
        name: fn
        for name, fn in inspect.getmembers(PaperLedger, inspect.isfunction)
        if not name.startswith("_")
    }


def _ledger_methods() -> dict[str, Callable]:
    return {
        name: fn
        for name, fn in inspect.getmembers(_pg_class(), inspect.isfunction)
        if not name.startswith("_")
    }


def _params(fn: Callable) -> list[tuple[str, object, object, object]]:
    """Name, kind, default and annotation of each parameter, `self` dropped."""
    return [
        (p.name, p.kind, p.default, p.annotation)
        for p in inspect.signature(fn).parameters.values()
        if p.name != "self"
    ]


# ------------------------------------------------------------------- the class and the protocol


def test_pg_paper_ledger_satisfies_paper_ledger() -> None:
    methods = _ledger_methods()
    protocol = _protocol_methods()
    assert sorted(methods) == sorted(protocol)
    assert len(methods) == 31
    assert isinstance(_pg_class()(), PaperLedger)
    for name, method in methods.items():
        proto_sig = inspect.signature(protocol[name])
        sig = inspect.signature(method)
        assert list(sig.parameters)[0] == "self", name
        proto_params = [(p.name, p.kind, p.default) for p in proto_sig.parameters.values()]
        params = [(p.name, p.kind, p.default) for p in sig.parameters.values()]
        assert params == proto_params, name
        for key, parameter in sig.parameters.items():
            if key == "self" or (name, key) in KNOWN_ANNOTATION_DIFFERENCES:
                continue
            assert parameter.annotation == proto_sig.parameters[key].annotation, (name, key)
        if (name, "return") not in KNOWN_ANNOTATION_DIFFERENCES:
            assert sig.return_annotation == proto_sig.return_annotation, name


def test_ledger_methods_keep_the_module_function_signatures() -> None:
    """A method differs from its module function by `self` alone (annotations included)."""
    functions = _public_persist_functions()
    methods = _ledger_methods()
    assert sorted(methods) == sorted(functions)
    for name, method in methods.items():
        assert _params(method) == _params(functions[name]), name
        assert (
            inspect.signature(method).return_annotation
            == inspect.signature(functions[name]).return_annotation
        ), name


# ---------------------------------------------------------------------------- the delegations


def _body_lines(fn: Callable) -> int:
    """Source lines of a function body, its docstring excluded."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(fn))))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return sum(stmt.end_lineno - stmt.lineno + 1 for stmt in body)


def test_module_functions_delegate_and_keep_their_signatures() -> None:
    # `test_persist_signatures_are_frozen` (Task 31) runs unchanged and is the signature proof.
    functions = _public_persist_functions()
    assert len(functions) == 31
    methods = _ledger_methods()
    for name, fn in functions.items():
        assert fn.__module__ == "ggTrader.paper.persist", name
        assert inspect.getdoc(fn) == inspect.getdoc(methods[name]), name
        assert _body_lines(fn) <= MAX_DELEGATION_BODY_LINES, name
        source = inspect.getsource(inspect.unwrap(fn))
        assert f"_pg().{name}(" in source, name
    assert persist.PaperRunLock.__module__ == "ggTrader.paper.persist"


def test_the_ledger_instance_is_created_once_and_holds_no_engine() -> None:
    first = persist._pg()
    assert isinstance(first, _pg_class())
    assert persist._pg() is first
    assert vars(first) == {}


# ------------------------------------------------------------------------- engine at call time


def test_engine_is_looked_up_at_call_time() -> None:
    """Each call goes to the engine `_get_engine` returns at that moment, never a cached one."""
    first, second = RecordingEngine([None]), RecordingEngine([None])
    ledger = _pg_class()()
    with patch.object(persist, "_get_engine", return_value=first):
        persist.clear_split_correction("AAA")
    with patch.object(persist, "_get_engine", return_value=second):
        ledger.clear_split_correction("BBB")
    third = RecordingEngine([None])
    with patch.object(persist, "_get_engine", return_value=third):
        persist.clear_split_correction("CCC")
    for engine, symbol in ((first, "AAA"), (second, "BBB"), (third, "CCC")):
        statements = [e for e in engine.events if e["op"] == "execute"]
        assert [s["params"] for s in statements] == [{"symbol": symbol}]


def test_engine_patch_by_dotted_string_reaches_the_ledger() -> None:
    engine = RecordingEngine([None])
    with patch(PERSIST_ENGINE, return_value=engine):
        persist._pg().clear_pending_order("ord-1")
    assert [e["op"] for e in engine.events] == ["connect", "execute", "commit", "close"]


# ------------------------------------------------------- the statement-for-statement proof


def _hold_lock_through_ledger(body_raises: bool = False) -> str:
    with _pg_class()().paper_run_lock() as lock:
        if body_raises:
            raise ValueError("boom")
        return type(lock).__name__


#: Cases that go through helper functions rather than a persist function: the helper that holds
#: the lock through the module has a twin that holds it through the class; the verify helper
#: uses `PaperRunLock` itself, which stays module-level, so it is its own twin.
HELPER_TWINS: dict[str, Callable] = {
    "_hold_lock": _hold_lock_through_ledger,
    "_verify_on_connection": golden_module._verify_on_connection,
}


def _through_ledger(fn: Callable) -> Callable:
    if fn.__name__ in HELPER_TWINS:
        return HELPER_TWINS[fn.__name__]
    assert fn.__module__ == persist.__name__, fn
    return getattr(_pg_class()(), fn.__name__)


def _both(fn: Callable, args: tuple, kwargs: dict, script: list) -> dict:
    """Run one case through the module function and through the matching ledger method."""
    return {
        "module": run_case(fn, args, kwargs, copy.deepcopy(script)),
        "ledger": run_case(_through_ledger(fn), args, kwargs, copy.deepcopy(script)),
    }


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("name", sorted(golden_module.CASES))
def test_sql_recording_identical_to_the_pre_extraction_reference(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The statement-for-statement proof, for every case of Task 31.

    The delegation and the method must record the same events, return value and exception, and
    both must equal the unedited reference once its one stray key is dropped.
    """
    monkeypatch.setattr(golden_module, "run_case", _both)
    recorded = golden_module.CASES[name]()
    reference = normalize_reference(name, _golden()[name])
    assert recorded["module"] == recorded["ledger"], name
    assert recorded["module"] == reference, name
    for event in recorded["ledger"]["events"]:
        if event["op"] == "execute" and event["params"] is not None:
            assert "conn" not in event["params"], name


def test_reference_golden_still_records_the_stray_conn_key() -> None:
    """The unedited reference keeps `conn`, so it cannot be regenerated away, and the single
    normalization changes the `create_order_intent` cases and no other entry."""
    golden = _golden()
    with_conn = {
        name
        for name, entry in golden.items()
        if any(
            event["op"] == "execute" and event["params"] and "conn" in event["params"]
            for event in entry["events"]
        )
    }
    assert with_conn
    assert {name.split("__")[0] for name in with_conn} == {"create_order_intent"}
    stray = [
        event["params"]["conn"]
        for name in with_conn
        for event in golden[name]["events"]
        if event["op"] == "execute"
    ]
    assert stray and all(value == {"$type": "Connection"} for value in stray)
    changed = {name for name, entry in golden.items() if normalize_reference(name, entry) != entry}
    assert changed == with_conn
    for name in set(golden) - with_conn:
        assert normalize_reference(name, golden[name]) == golden[name], name
