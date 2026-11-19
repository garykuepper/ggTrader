"""Scenario builder and transcript readers for the order-flow rule tests (Task 48).

Immutable behavioral set (hash-recorded at Task 51). Scenarios and goldens are not added: every
input is `dataclasses.replace(SCENARIOS[name], ...)` over an existing scenario, and every
assertion reads the transcript's calls, never a golden file.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import patch

from ggTrader.paper import trader as trader_module
from ggTrader.paper.trader import PaperTrader
from tests.paper.equivalence import impl
from tests.paper.equivalence.run_with import implementation, run_with
from tests.paper.transcript import harness
from tests.paper.transcript.harness import ENV_VARS, Scenario
from tests.paper.transcript.scenarios import (
    _LIVE_ENV,
    _STOP_ENV,
    _SWEEP_ENV,
    SCENARIOS,
    _acct,
    _blend,
    _pos,
    _replies,
    _sleeve,
)

__all__ = [
    "IMPLS",
    "LIVE_ENV",
    "STOP_ENV",
    "SWEEP_ENV",
    "SWEEP_STOP_ENV",
    "acct",
    "blend",
    "calls",
    "RawRun",
    "client_id",
    "from_scenario",
    "intents_settled",
    "messages",
    "pos",
    "replies",
    "run",
    "run_raw",
    "sleeve",
    "submissions",
    "targets",
]

#: The implementations every rule test runs under (Task 74 edits only `impl.py`).
IMPLS: tuple[str, ...] = impl.IMPLS

LIVE_ENV: dict[str, str] = dict(_LIVE_ENV)
STOP_ENV: dict[str, str] = dict(_STOP_ENV)
SWEEP_ENV: dict[str, str] = dict(_SWEEP_ENV)
SWEEP_STOP_ENV: dict[str, str] = {**_STOP_ENV, **_SWEEP_ENV}

acct = _acct
pos = _pos
blend = _blend
replies = _replies
sleeve = _sleeve


def from_scenario(name: str, **changes: Any) -> Scenario:
    """A deep copy of the named scenario with `changes` applied (`dataclasses.replace`)."""
    return replace(copy.deepcopy(SCENARIOS[name]), **changes)


def run(impl_name: str, scenario: Scenario) -> dict:
    """The normalized transcript of `scenario` under the implementation `impl_name`."""
    return run_with(impl_name, scenario)


def targets(transcript: dict) -> list[str]:
    """Every call target of the transcript, in order."""
    return [c["target"] for c in transcript["calls"]]


def calls(transcript: dict, target: str) -> list[dict]:
    """The calls to `target`, in order."""
    return [c for c in transcript["calls"] if c["target"] == target]


def submissions(transcript: dict, side: str | None = None) -> list[tuple[str, str, float]]:
    """``(side, symbol, quantity or notional)`` of every order the broker received, in order."""
    out: list[tuple[str, str, float]] = []
    for call in transcript["calls"]:
        if call["target"] == "broker.submit_sell" and side in (None, "SELL"):
            out.append(("SELL", call["args"][0], call["args"][1]))
        elif call["target"] == "broker.submit_buy" and side in (None, "BUY"):
            out.append(("BUY", call["args"][0], call["args"][1]))
    return out


def messages(transcript: dict) -> list[str]:
    """Every text the notifier's `send` received, in order."""
    return [c["args"][0] for c in calls(transcript, "notifier.send")]


def client_id(n: int) -> str:
    """The pinned client order id of the n-th submission of a run (n counts from 1)."""
    return f"00000000-0000-0000-0000-{n:012d}"


def intents_settled(transcript: dict) -> list[tuple[str, float, float | None, str]]:
    """``(client id, filled qty, price, status)`` of every ``settle_order_intent``, in order."""
    return [
        (c["args"][0], c["args"][2], c["args"][3], c["args"][4])
        for c in calls(transcript, "persist.settle_order_intent")
    ]


# --------------------------------------------------------------------------- event-logging runs


@dataclass
class RawRun:
    """One run with the exception object kept (a transcript keeps only its text).

    ``run.recorder.calls`` holds the harness's calls interleaved with this environment's own
    ``hook.<name>`` entries, in the order they happened.
    """

    run: harness._Run
    trader: PaperTrader | None
    result: dict | None
    error: Exception | None

    @property
    def targets(self) -> list[str]:
        return [c["target"] for c in self.run.recorder.calls]

    @property
    def targets_before_release(self) -> list[str]:
        """`targets` without the closing ``lock.exit`` a live run records even when it raises."""
        return self.targets[:-1] if self.targets[-1:] == ["lock.exit"] else self.targets

    def calls(self, target: str) -> list[dict]:
        return [c for c in self.run.recorder.calls if c["target"] == target]

    def messages(self) -> list[str]:
        return [c["args"][0] for c in self.calls("notifier.send")]


@contextmanager
def event_environment(
    scenario: Scenario, *, membership_failure: BaseException | None = None
) -> Iterator[harness._Run]:
    """Patch what `harness._run_scenario` patches, with one extra log line per hook.

    A copy of `_trader_environment` in test_trader_ledger.py (copied, not imported, so that file
    stays untouched). The harness does not record `_today_et`, `_now_pt` or `uuid4`, and it
    replaces `live_membership_source` after any patch a caller makes, so the wrappers here log
    ``hook.membership_source``, ``hook.now_pt``, ``hook.today_et`` and ``hook.new_uuid`` into
    the run's recorder, between the calls it already records. A ``membership_failure`` makes
    the membership wrapper raise it.
    """
    with harness._gate_error_copy():
        s = copy.deepcopy(scenario)
    run = harness._Run(s)

    def log(name: str) -> None:
        run.recorder.begin(f"hook.{name}", [], {})

    def membership_source(spec: str | None) -> object:
        log("membership_source")
        if membership_failure is not None:
            raise membership_failure
        return run.live_membership_source(spec)

    def now_pt() -> object:
        log("now_pt")
        return s.now_pt

    def today_et() -> object:
        log("today_et")
        return s.today_et

    def new_uuid() -> object:
        log("new_uuid")
        return run.uuid4()

    with ExitStack() as stack, harness._gate_error_copy():
        stack.enter_context(patch.dict(os.environ))
        for var in ENV_VARS:
            os.environ.pop(var, None)
        os.environ.update(s.env)
        replacements: dict[str, object] = {
            "generate_core_signals": run.generate_core_signals,
            "refresh_benchmark_tape": run.refresh_benchmark_tape,
            "live_membership_source": membership_source,
            "_today_et": today_et,
            "_now_pt": now_pt,
            "uuid4": new_uuid,
            "time": run.clock,
            "datetime": harness._pinned_datetime(s.now_pt),
            "paper_run_lock": run.paper_run_lock,
        }
        for name in harness.persist_function_names():
            if name != "paper_run_lock":
                replacements[name] = run.persist_wrapper(name)
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(trader_module, name, replacement))
        yield run


def run_raw(
    impl_name: str,
    scenario: Scenario,
    *,
    membership_failure: BaseException | None = None,
) -> RawRun:
    """Run `scenario` under `impl_name` in the event environment; keep the exception object.

    Only `Exception` is caught, as `run_scenario` does: a `HarnessError` fails the test.
    """
    with event_environment(scenario, membership_failure=membership_failure) as run:
        s = run.s
        trader: PaperTrader | None = None
        result: dict | None = None
        error: Exception | None = None
        with implementation(impl_name):
            try:
                trader = PaperTrader(
                    run.broker,  # type: ignore[arg-type]
                    run.notifier,
                    dry_run=s.dry_run,
                    read_only=s.read_only,
                    membership=s.membership,
                )
                result = trader.run()
            except Exception as exc:
                error = exc
        return RawRun(run=run, trader=trader, result=result, error=error)
