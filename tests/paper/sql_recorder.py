"""A recording stand-in for the SQLAlchemy engine, used to pin `paper/persist.py` SQL.

The recorder captures what the code *asked* of the engine: the exact statement text, a copy of
the bind dictionary, and the context events (connect, begin, commit, rollback, close).

These are modeled context events, not SQLAlchemy's real transaction semantics. A real 2.0
``connect()`` block that ends without ``commit()`` is rolled back by SQLAlchemy when the
connection closes; the recorder cannot show that, so ``close`` with ``committed: false`` is the
marker for "this session ended without a commit" and no more. What the real engine does with it is
covered by SQLAlchemy and by the SQLite runs in ``test_paper_persist.py``.

Bind dictionary insertion order is NOT part of the contract (SQLAlchemy binds by name), so
`encode_params` writes keys in sorted order and the golden does not notice a reordered
dictionary. That is deliberate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import patch

PERSIST_ENGINE = "ggTrader.paper.persist._get_engine"


class ScriptExhausted(AssertionError):
    """A case ran more statements than its script provided results for."""


class FakeResult:
    """The slice of a SQLAlchemy result that `persist.py` uses.

    Rows are dicts (read by `.mappings()`) or tuples (read by index or unpacking).
    """

    def __init__(self, rows: list | tuple = (), scalar: Any = None) -> None:
        self._rows = list(rows)
        self._scalar = scalar

    def mappings(self) -> FakeResult:
        return self

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def all(self) -> list:
        return list(self._rows)

    def scalar_one(self) -> Any:
        return self._scalar

    def __iter__(self):
        return iter(self._rows)


class RecordingConnection:
    """Records `execute` and `commit` calls and answers from the script."""

    def __init__(self, engine: RecordingEngine) -> None:
        self._engine = engine
        self.committed = False

    def execute(self, statement: Any, params: dict | None = None) -> FakeResult:
        self._engine.events.append(
            {
                "op": "execute",
                "sql": statement.text,
                "params": None if params is None else encode_params(params),
            }
        )
        return self._engine._next_result()

    def commit(self) -> None:
        self.committed = True
        self._engine.events.append({"op": "commit"})


class RecordingEngine:
    """Engine whose `connect()` and `begin()` yield one `RecordingConnection`.

    `script` holds, in statement order, a `FakeResult`, `None` (an empty `FakeResult`) or an
    exception instance (raised from that `execute`). A statement beyond the script raises
    `ScriptExhausted`.
    """

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self._used = 0
        self.events: list[dict] = []

    def _next_result(self) -> FakeResult:
        if self._used >= len(self._script):
            raise ScriptExhausted(f"statement {self._used + 1} has no scripted result")
        item = self._script[self._used]
        self._used += 1
        if isinstance(item, BaseException):
            raise item
        return FakeResult() if item is None else item

    @contextmanager
    def connect(self):
        conn = RecordingConnection(self)
        self.events.append({"op": "connect"})
        try:
            yield conn
        finally:
            self.events.append({"op": "close", "committed": conn.committed})

    @contextmanager
    def begin(self):
        conn = RecordingConnection(self)
        self.events.append({"op": "begin"})
        try:
            yield conn
        except BaseException:
            self.events.append({"op": "rollback"})
            raise
        self.events.append({"op": "commit"})


def encode(value: Any) -> Any:
    """JSON-ready, type-tagged copy of `value` (floats keep `repr`; NaN and infinities tagged)."""
    if isinstance(value, RecordingConnection):
        return {"$type": "Connection"}
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$type": "float", "v": "nan"}
        if math.isinf(value):
            return {"$type": "float", "v": "inf" if value > 0 else "-inf"}
        return {"$type": "float", "v": repr(value)}
    if isinstance(value, datetime):
        return {"$type": "datetime", "v": value.isoformat()}
    if isinstance(value, date):
        return {"$type": "date", "v": value.isoformat()}
    if isinstance(value, Decimal):
        return {"$type": "Decimal", "v": str(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, set | frozenset):
        items = sorted((encode(v) for v in value), key=repr)
        return {"$type": "set", "v": items}
    if isinstance(value, tuple):
        return {"$type": "tuple", "v": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    return {"$type": type(value).__name__, "v": repr(value)}


def encode_params(params: dict) -> dict:
    """A bind dictionary with keys in sorted order and each value encoded."""
    return {key: encode(params[key]) for key in sorted(params)}


def run_case(fn: Callable, args: tuple, kwargs: dict, script: list[object]) -> dict:
    """Call `fn` against a recording engine and return its events, result and exception."""
    engine = RecordingEngine(script)
    returned: Any = None
    raised: str | None = None
    with patch(PERSIST_ENGINE, return_value=engine):
        try:
            returned = fn(*args, **kwargs)
        except Exception as exc:
            raised = f"{type(exc).__name__}: {exc}"
    return {"events": engine.events, "returned": encode(returned), "raised": raised}
