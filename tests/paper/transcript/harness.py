"""Records everything `PaperTrader.run` does against scripted collaborators.

The harness replaces only names in the ``ggTrader.paper.trader`` module namespace (never the
global ``time.time``/``time.sleep``: that pattern caused the 2026-09-22 hang), scripts every
broker, ledger and signal reply from a `Scenario`, and writes each call, its arguments (copied
at call time), the rendered Telegram text and any raise into one canonical JSON transcript.
Nothing in a transcript varies between runs: ids and the clock are pinned, sets are sorted,
non-finite numbers are tagged and an unknown type raises instead of printing a memory address.
"""

from __future__ import annotations

import copy
import copyreg
import json
import math
import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import patch
from uuid import UUID

from ggTrader.paper import trader as trader_module
from ggTrader.paper.gates import PT, SignalGateError
from ggTrader.paper.notifier import TelegramNotifier
from ggTrader.paper.split_check import compute_split_corrections

#: Every environment variable read by ``src/ggTrader/paper/*.py``. `run_scenario` removes them
#: all, then applies the scenario's own `env`; a test greps the source to keep this list whole.
ENV_VARS: tuple[str, ...] = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "CASH_SWEEP_ENABLED",
    "CATASTROPHE_STOP_ENABLED",
    "CATASTROPHE_STOP_PCT",
    "GGT_MEMBERSHIP",
    "MAX_POSITIONS",
    "SWEEP_BUY_TRIGGER_PCT",
    "SWEEP_CASH_RESERVE_PCT",
    "SWEEP_MIN_CLIP_USD",
    "SWEEP_SYMBOL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "UNTRADABLE_SYMBOLS",
)

#: A run that sleeps more often than this is looping: fail instead of hanging.
MAX_POLLS = 200

#: Fill price for an order whose status sequence is empty ("filled at once").
DEFAULT_FILL_PRICE = 100.0

#: Pinned start of the fake clock (seconds).
CLOCK_START = 1_000_000.0

#: The persist functions that the in-memory `IntentStore` answers.
INTENT_FUNCTIONS: tuple[str, ...] = (
    "create_order_intent",
    "attach_broker_order",
    "get_unresolved_order_intents",
    "settle_order_intent",
    "settle_legacy_pending_order",
)

_MISSING = object()
_TERMINAL = ("filled", "canceled", "rejected", "expired")


def _reduce_gate_error(exc: SignalGateError) -> tuple:
    """Let `copy.deepcopy` rebuild a `SignalGateError` (its two-argument constructor defeats it)."""
    return SignalGateError, (exc.code, str(exc.args[0]).split("] ", 1)[1])


@contextmanager
def _gate_error_copy() -> Iterator[None]:
    """Make `SignalGateError` deep-copyable for one run, then restore the global registry."""
    previous = copyreg.dispatch_table.get(SignalGateError)
    copyreg.pickle(SignalGateError, _reduce_gate_error)
    try:
        yield
    finally:
        if previous is None:
            copyreg.dispatch_table.pop(SignalGateError, None)
        else:
            copyreg.dispatch_table[SignalGateError] = previous


class HarnessError(BaseException):
    """The harness itself failed (a missing reply, an unserializable value, a runaway loop).

    It derives from ``BaseException`` so the trader's own ``except Exception`` blocks cannot
    swallow it and record it as a trading outcome.
    """


@dataclass(frozen=True)
class MembershipToken:
    """Serializable stand-in for the membership source the trader hands the signal engine."""

    spec: str


@dataclass(frozen=True)
class Seq:
    """A per-call sequence of persist replies; the last one repeats once the list runs out."""

    values: tuple[object, ...]

    def __init__(self, *values: object) -> None:
        object.__setattr__(self, "values", tuple(values))


@dataclass(frozen=True)
class Scenario:
    """Everything one recorded run needs. The contents are copied before every run."""

    name: str
    dry_run: bool = False
    read_only: bool | None = None
    membership: str = "db"
    env: dict[str, str] = field(default_factory=dict)
    today_et: date = date(2026, 6, 19)
    now_pt: datetime = datetime(2026, 6, 19, 12, 46, tzinfo=PT)
    #: Ordered replies (or exceptions) per broker method; the last repeats. For the submit
    #: methods an entry is an exception (raised) or ``None`` (the next ``ord-n`` id). For
    #: ``get_order`` and ``get_order_by_client_id`` an entry is an exception, a (possibly
    #: malformed) reply dict, or ``None`` to fall through to the order the broker holds.
    replies: dict[str, list[object]] = field(default_factory=dict)
    #: Orders the broker holds before the run: id, client_order_id, symbol, side, qty|notional,
    #: and ``statuses`` (the status sequence, each ``{status, filled_qty, filled_avg_price}``).
    existing_orders: list[dict] = field(default_factory=list)
    signals: Any = None
    #: Persist replies by function name: a value, an exception (raised) or a `Seq`.
    persist: dict[str, object] = field(default_factory=dict)
    #: Starting rows of the in-memory intent store.
    intents: list[dict] = field(default_factory=list)
    #: Status sequence per symbol (or ``"SYMBOL:SIDE"``) for orders submitted in this run.
    fills: dict[str, list[dict]] = field(default_factory=dict)
    untradable: set[str] = field(default_factory=set)
    splits: dict = field(default_factory=lambda: {"corp_splits": {}, "activity_applied": set()})
    dividends: dict = field(default_factory=lambda: {"corp_dividends": {}, "credited_keys": set()})
    #: Scripted notifier outcomes. A key is a substring of a `send` message, or the method name
    #: ``trade_alert`` / ``daily_summary``; the value is an exception (raised) or ``False``.
    notifier: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- serialization


def _canonical_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize(value: object) -> object:
    """Turn `value` into plain JSON data, or raise ``TypeError`` for an unknown type."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "NaN"}
        if math.isinf(value):
            return {"float": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, MembershipToken):
        return {"membership": value.spec}
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"transcript dict key must be a string, got {type(key).__name__}")
            out[key] = normalize(item)
        return out
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((normalize(item) for item in value), key=_canonical_key)
    raise TypeError(f"cannot serialize {type(value).__name__} into a transcript")


def to_json(transcript: object) -> str:
    """Canonical JSON text: sorted keys, two-space indent, one trailing newline."""
    return (
        json.dumps(
            normalize(transcript),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- recording


class Recorder:
    """The ordered list of calls; each entry is snapshotted when the call happens."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def begin(self, target: str, args: object, kwargs: object) -> dict:
        """Append one call (arguments deep-copied and normalized now) and return its entry."""
        try:
            entry = {
                "target": target,
                "args": normalize(copy.deepcopy(list(args))),  # type: ignore[call-overload]
                "kwargs": normalize(copy.deepcopy(dict(kwargs))),  # type: ignore[call-overload]
                "raised": None,
            }
        except TypeError as exc:
            raise HarnessError(f"cannot record a call to {target}: {exc}") from exc
        self.calls.append(entry)
        return entry


class FakeTime:
    """The trader's ``time`` module: one fake clock that only moves when the trader sleeps.

    Every read and every sleep is recorded, so the poll deadline and interval are part of the
    transcript. The runaway guard counts what happens at one instant of the fake clock (clock
    reads plus broker order polls) and resets whenever the trader sleeps: a loop that never
    sleeps never lets the clock advance, so it exceeds `MAX_POLLS` at one instant, while a long
    legitimate poll of many orders advances the clock every iteration. Sleeps are also capped.
    """

    def __init__(self, recorder: Recorder) -> None:
        self._now = CLOCK_START
        self._recorder = recorder
        self.sleeps = 0
        self.events_at_instant = 0

    def note_event(self, what: str) -> None:
        """Count a clock read or a broker poll at the current instant; raise on a runaway."""
        self.events_at_instant += 1
        if self.events_at_instant > MAX_POLLS:
            raise HarnessError(
                f"more than {MAX_POLLS} {what} events at one instant of the fake clock: "
                "runaway loop (a sleep is missing?)"
            )

    def _read(self, name: str) -> float:
        self.note_event("clock")
        self._recorder.begin(f"clock.{name}", [], {"returned": self._now})
        return self._now

    def time(self) -> float:
        return self._read("time")

    def monotonic(self) -> float:
        return self._read("monotonic")

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        if self.sleeps > MAX_POLLS:
            raise HarnessError(f"the run slept more than {MAX_POLLS} times: runaway poll loop")
        self._recorder.begin("clock.sleep", [seconds], {})
        self._now += seconds
        self.events_at_instant = 0


class FakeRunLock:
    """Stand-in for the advisory run lock: `verify` is recorded and fails only when scripted."""

    def __init__(self, recorder: Recorder, failure: object = None) -> None:
        self._recorder = recorder
        self._failure = failure

    def verify(self) -> None:
        entry = self._recorder.begin("lock.verify", [], {})
        if isinstance(self._failure, BaseException):
            entry["raised"] = _error_text(self._failure)
            raise self._failure


class IntentStore:
    """In-memory order-intent ledger (modeled on ``_intent_store`` in test_trader.py).

    Legacy pending orders are not mutated here: ``get_pending_orders`` is scripted per scenario
    and ``settle_legacy_pending_order`` only keeps what it was given in `settled_legacy`.
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows: dict[str, dict] = {}
        for row in rows:
            seeded = dict(row)
            seeded.setdefault("state", "submitted" if seeded.get("broker_order_id") else "intent")
            self.rows[seeded["client_order_id"]] = seeded
        self.settled: list[dict] = []
        self.settled_legacy: list[tuple] = []

    def create_order_intent(
        self,
        client_order_id: str,
        run_date: str,
        side: str,
        symbol: str,
        amount: float,
        amount_kind: str,
        reason: str | None,
    ) -> None:
        if client_order_id in self.rows:
            raise ValueError("duplicate client order id")
        self.rows[client_order_id] = {
            "client_order_id": client_order_id,
            "run_date": run_date,
            "side": side,
            "symbol": symbol,
            "amount": amount,
            "amount_kind": amount_kind,
            "reason": reason,
            "broker_order_id": None,
            "state": "intent",
        }

    def attach_broker_order(self, client_order_id: str, order_id: str) -> None:
        row = self.rows.get(client_order_id)
        if row is None or row["broker_order_id"] not in (None, order_id):
            raise ValueError("Missing intent or conflicting broker order")
        row["broker_order_id"] = order_id
        row["state"] = "submitted"

    def get_unresolved_order_intents(self) -> list[dict]:
        return [copy.deepcopy(row) for row in self.rows.values()]

    def settle_order_intent(
        self,
        client_order_id: str,
        order_id: str,
        filled_qty: float,
        filled_avg_price: float | None,
        terminal_status: str,
    ) -> None:
        row = self.rows.pop(client_order_id, None)
        if row is None:
            raise ValueError("Missing intent")
        row.update(
            state="resolved",
            settled_broker_order_id=order_id,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            terminal_status=terminal_status,
        )
        self.settled.append(row)

    def settle_legacy_pending_order(self, *args: object) -> None:
        self.settled_legacy.append(args)


class RecordingNotifier(TelegramNotifier):
    """The real message formatting of `TelegramNotifier`, with `send` recorded, never sent.

    Outcomes can be scripted (`Scenario.notifier`): a matching call returns ``False`` or raises.
    """

    def __init__(self, recorder: Recorder, outcomes: dict[str, object] | None = None) -> None:
        super().__init__()
        self._enabled = True  # formatting is never skipped for lack of credentials
        self._recorder = recorder
        self._outcomes = outcomes or {}

    def _outcome(self, entry: dict, key_matches: Callable[[str], bool]) -> bool:
        for key, outcome in self._outcomes.items():
            if not key_matches(key):
                continue
            if isinstance(outcome, BaseException):
                entry["raised"] = _error_text(outcome)
                raise outcome
            return bool(outcome)
        return True

    def send(self, message: str) -> bool:
        entry = self._recorder.begin("notifier.send", [message], {})
        return self._outcome(
            entry, lambda key: key not in ("trade_alert", "daily_summary") and key in message
        )

    def trade_alert(self, *args: Any, **kwargs: Any) -> bool:
        entry = self._recorder.begin("notifier.trade_alert", args, kwargs)
        outcome = self._outcome(entry, lambda key: key == "trade_alert")
        # The real formatting still runs (and its `send` is recorded); a scripted False wins.
        return super().trade_alert(*args, **kwargs) and outcome

    def daily_summary(self, *args: Any, **kwargs: Any) -> bool:
        entry = self._recorder.begin("notifier.daily_summary", args, kwargs)
        outcome = self._outcome(entry, lambda key: key == "daily_summary")
        return super().daily_summary(*args, **kwargs) and outcome


# --------------------------------------------------------------------------- the broker


def _default_fill(side: str, amount: float) -> dict:
    qty = amount if side == "SELL" else amount / DEFAULT_FILL_PRICE
    return {"status": "filled", "filled_qty": qty, "filled_avg_price": DEFAULT_FILL_PRICE}


class _Order:
    """One order the fake broker knows: its identity and its status sequence."""

    def __init__(
        self,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        kind: str,
        amount: float,
        statuses: list[dict],
    ) -> None:
        self.order_id = order_id
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.side = side.upper()
        self.kind = kind
        self.amount = amount
        self.statuses = statuses or [_default_fill(self.side, amount)]
        self.polls = 0
        self.last: dict | None = None

    def _info(self, entry: dict) -> dict:
        return {
            "id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.lower(),
            "qty": self.amount if self.kind == "qty" else None,
            "notional": self.amount if self.kind == "notional" else None,
            "filled_qty": entry.get("filled_qty", 0.0),
            "filled_avg_price": entry.get("filled_avg_price"),
            "status": entry["status"],
        }

    def poll(self) -> dict:
        entry = self.statuses[min(self.polls, len(self.statuses) - 1)]
        self.polls += 1
        self.last = self._info(entry)
        return copy.deepcopy(self.last)

    def peek(self) -> dict:
        return copy.deepcopy(self.last or self._info(self.statuses[0]))


class FakeBroker:
    """Implements every public method of ``AlpacaBroker`` and answers from a scenario."""

    def __init__(self, scenario: Scenario, recorder: Recorder, clock: FakeTime) -> None:
        self._clock = clock
        self._s = scenario
        self._recorder = recorder
        self._cursor: dict[str, int] = {}
        self._orders: dict[str, _Order] = {}
        self._by_client: dict[str, _Order] = {}
        self._submitted = 0
        for raw in scenario.existing_orders:
            kind = "qty" if raw.get("qty") is not None else "notional"
            order = _Order(
                raw["id"],
                raw["client_order_id"],
                raw["symbol"],
                raw["side"],
                kind,
                raw[kind],
                list(raw.get("statuses", [])),
            )
            self._register(order)

    def _register(self, order: _Order) -> None:
        self._orders[order.order_id] = order
        self._by_client[order.client_order_id] = order

    def _call(self, name: str, args: tuple, kwargs: dict) -> dict:
        return self._recorder.begin(f"broker.{name}", args, kwargs)

    def _reply(self, name: str, *, default: object = _MISSING) -> object:
        replies = self._s.replies.get(name)
        if not replies:
            if default is _MISSING:
                raise HarnessError(f"no reply scripted for broker.{name}")
            return copy.deepcopy(default)
        index = self._cursor.get(name, 0)
        self._cursor[name] = index + 1
        entry = replies[min(index, len(replies) - 1)]
        if isinstance(entry, BaseException):
            raise entry
        return copy.deepcopy(entry)

    def _answer(self, name: str, args: tuple, kwargs: dict, produce: Callable[[], object]) -> Any:
        entry = self._call(name, args, kwargs)
        try:
            return produce()
        except Exception as exc:
            entry["raised"] = _error_text(exc)
            raise

    def get_account(self) -> dict:
        return self._answer("get_account", (), {}, lambda: self._reply("get_account"))

    def get_positions(self) -> dict[str, dict]:
        return self._answer("get_positions", (), {}, lambda: self._reply("get_positions"))

    def get_clock(self) -> dict:
        return self._answer("get_clock", (), {}, lambda: self._reply("get_clock"))

    def get_untradable(self, symbols: list[str]) -> set[str]:
        def produce() -> object:
            if "get_untradable" in self._s.replies:
                return self._reply("get_untradable")
            return {s for s in symbols if s in self._s.untradable}

        return self._answer("get_untradable", (symbols,), {}, produce)

    def get_split_evidence(self, symbols: list[str], since: date) -> dict:
        def produce() -> object:
            if "get_split_evidence" in self._s.replies:
                return self._reply("get_split_evidence")
            return copy.deepcopy(self._s.splits)

        return self._answer("get_split_evidence", (symbols, since), {}, produce)

    def get_split_corrections(self, symbols: list[str], since: date) -> dict[str, float]:
        def produce() -> object:
            evidence = self._s.splits
            return compute_split_corrections(evidence["corp_splits"], evidence["activity_applied"])

        return self._answer("get_split_corrections", (symbols, since), {}, produce)

    def get_dividend_corrections(self, symbols: list[str], since: date | str) -> dict:
        def produce() -> object:
            if "get_dividend_corrections" in self._s.replies:
                return self._reply("get_dividend_corrections")
            return copy.deepcopy(self._s.dividends)

        return self._answer("get_dividend_corrections", (symbols, since), {}, produce)

    def _submit(self, method: str, side: str, symbol: str, amount: float, client_id: str) -> str:
        def produce() -> object:
            self._reply(method, default=None)  # raises when the scenario scripts a failure
            self._submitted += 1
            order_id = f"ord-{self._submitted}"
            statuses = self._s.fills.get(f"{symbol}:{side}", self._s.fills.get(symbol, []))
            kind = "notional" if side == "BUY" else "qty"
            self._register(
                _Order(order_id, client_id, symbol, side, kind, amount, copy.deepcopy(statuses))
            )
            return order_id

        return self._answer(method, (symbol, amount, client_id), {}, produce)

    def submit_buy(self, symbol: str, notional: float, client_order_id: str) -> str:
        return self._submit("submit_buy", "BUY", symbol, notional, client_order_id)

    def submit_sell(self, symbol: str, qty: float, client_order_id: str) -> str:
        return self._submit("submit_sell", "SELL", symbol, qty, client_order_id)

    def _lookup(self, name: str) -> tuple[bool, object]:
        """The scripted reply for a lookup, or ``(False, None)`` to use the broker's own order."""
        replies = self._s.replies.get(name)
        if not replies:
            return False, None
        index = self._cursor.get(name, 0)
        self._cursor[name] = index + 1
        entry = replies[min(index, len(replies) - 1)]
        if entry is None:
            return False, None
        if isinstance(entry, BaseException):
            raise entry
        return True, copy.deepcopy(entry)

    def get_order(self, order_id: str) -> dict:
        def produce() -> object:
            self._clock.note_event("get_order")
            scripted, reply = self._lookup("get_order")
            if scripted:
                return reply
            order = self._orders.get(order_id)
            if order is None:
                raise HarnessError(f"get_order for an order the scenario never created: {order_id}")
            return order.poll()

        return self._answer("get_order", (order_id,), {}, produce)

    def get_order_by_client_id(self, client_order_id: str) -> dict:
        def produce() -> object:
            scripted, reply = self._lookup("get_order_by_client_id")
            if scripted:
                return reply
            order = self._by_client.get(client_order_id)
            if order is None:
                raise HarnessError(f"no order with client id {client_order_id}")
            return order.peek()

        return self._answer("get_order_by_client_id", (client_order_id,), {}, produce)


# --------------------------------------------------------------------------- the run


def persist_function_names() -> list[str]:
    """Every ``ggTrader.paper.persist`` function the trader module imported, by inspection."""
    return sorted(
        name
        for name, obj in vars(trader_module).items()
        if callable(obj) and getattr(obj, "__module__", None) == "ggTrader.paper.persist"
    )


class _Run:
    """The collaborators of one scenario run."""

    def __init__(self, scenario: Scenario) -> None:
        self.s = scenario
        self.recorder = Recorder()
        self.clock = FakeTime(self.recorder)
        self.store = IntentStore(scenario.intents)
        self.broker = FakeBroker(scenario, self.recorder, self.clock)
        self.notifier = RecordingNotifier(self.recorder, scenario.notifier)
        self.lock = FakeRunLock(self.recorder, scenario.persist.get("lock.verify"))
        self._persist_calls: dict[str, int] = {}
        self._uuid_counter = 0

    # -- persist -----------------------------------------------------------------------------

    def _scripted(self, name: str) -> object:
        value = self.s.persist.get(name, _MISSING)
        if value is _MISSING:
            if name.startswith("get_"):
                raise HarnessError(f"no reply scripted for persist.{name}")
            return None
        if isinstance(value, Seq):
            index = self._persist_calls.get(name, 0)
            self._persist_calls[name] = index + 1
            value = value.values[min(index, len(value.values) - 1)]
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)

    def persist_wrapper(self, name: str) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            entry = self.recorder.begin(f"persist.{name}", args, kwargs)
            try:
                if name in self.s.persist:
                    return self._scripted(name)
                if name in INTENT_FUNCTIONS:
                    return getattr(self.store, name)(*args, **kwargs)
                return self._scripted(name)
            except Exception as exc:
                entry["raised"] = _error_text(exc)
                raise

        wrapper.__name__ = name  # `PaperTrader._write` records the real function's name
        wrapper.__qualname__ = name
        return wrapper

    @contextmanager
    def paper_run_lock(self) -> Iterator[FakeRunLock]:
        """Replaces ``persist.paper_run_lock``: a context manager yielding the one fake lock."""
        self.recorder.begin("lock.enter", [], {})
        try:
            yield self.lock
        finally:
            self.recorder.begin("lock.exit", [], {})

    # -- other collaborators -------------------------------------------------------------------

    def generate_core_signals(self, *args: object, **kwargs: object) -> object:
        entry = self.recorder.begin("signals.generate_core_signals", args, kwargs)
        if isinstance(self.s.signals, BaseException):
            entry["raised"] = _error_text(self.s.signals)
            raise self.s.signals
        return copy.deepcopy(self.s.signals)

    def live_membership_source(self, spec: str | None) -> MembershipToken:
        self.recorder.begin("membership.live_membership_source", [spec], {})
        return MembershipToken(str(spec))

    def refresh_benchmark_tape(self) -> list[str]:
        self.recorder.begin("tape.refresh_benchmark_tape", [], {})
        return ["SPY"]

    def uuid4(self) -> UUID:
        self._uuid_counter += 1
        return UUID(int=self._uuid_counter)


def _check_scenario(s: Scenario) -> None:
    pending = s.persist.get("get_pending_orders")
    values = pending.values if isinstance(pending, Seq) else (pending,)
    for value in values:
        for row in value if isinstance(value, list) else []:
            created = row.get("created_at")
            if created is not None and not isinstance(created, datetime):
                raise HarnessError("a pending order's created_at must be a datetime or None")


def _pinned_datetime(pinned: datetime) -> type:
    """A ``datetime`` for the trader's namespace whose ``now`` returns the scenario's instant."""

    class _Meta(type):
        def __instancecheck__(cls, obj: object) -> bool:
            return isinstance(obj, datetime)

    class PinnedDatetime(datetime, metaclass=_Meta):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return pinned.astimezone(tz) if tz is not None else pinned.replace(tzinfo=None)

    return PinnedDatetime


def run_scenario(scenario: Scenario) -> dict:
    """Run one scenario and return its normalized transcript (see `_run_scenario`)."""
    with _gate_error_copy():
        return _run_scenario(scenario)


def _run_scenario(scenario: Scenario) -> dict:
    """Run one scenario and return its normalized transcript.

    Returns:
        ``{"calls": [...], "result": dict | None, "raised": str | None, "skipped_writes": [...]}``.
        A run that raises records ``"raised"`` and a ``None`` result. A `HarnessError` is never
        recorded: it propagates and fails the test.
    """
    s = copy.deepcopy(scenario)
    _check_scenario(s)
    run = _Run(s)
    result: dict | None = None
    raised: str | None = None
    skipped: list[str] = []
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ))
        for var in ENV_VARS:
            os.environ.pop(var, None)
        os.environ.update(s.env)
        for name in persist_function_names():
            if name == "paper_run_lock":
                continue
            stack.enter_context(patch.object(trader_module, name, run.persist_wrapper(name)))
        replacements: dict[str, object] = {
            "paper_run_lock": run.paper_run_lock,
            "generate_core_signals": run.generate_core_signals,
            "refresh_benchmark_tape": run.refresh_benchmark_tape,
            "live_membership_source": run.live_membership_source,
            "_today_et": lambda: s.today_et,
            "_now_pt": lambda: s.now_pt,
            "uuid4": run.uuid4,
            "time": run.clock,
            "datetime": _pinned_datetime(s.now_pt),
        }
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(trader_module, name, replacement))
        trader: trader_module.PaperTrader | None = None
        try:
            trader = trader_module.PaperTrader(
                run.broker,  # type: ignore[arg-type]
                run.notifier,
                dry_run=s.dry_run,
                read_only=s.read_only,
                membership=s.membership,
            )
            result = trader.run()
        except Exception as exc:
            raised = _error_text(exc)
        if trader is not None:
            skipped = list(trader.skipped_writes)
    try:
        return {
            "calls": run.recorder.calls,
            "result": normalize(result) if result is not None else None,  # type: ignore[dict-item]
            "raised": raised,
            "skipped_writes": skipped,
        }
    except TypeError as exc:
        raise HarnessError(f"the run result cannot be serialized: {exc}") from exc
