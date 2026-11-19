"""Record seeded runs in one shared, isolated harness environment."""

import copy
import hashlib
import logging
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import patch

from ggTrader.paper import trader as trader_module
from ggTrader.paper.ledger import WRITE_METHODS
from ggTrader.utils import config
from tests.netguard import blocked_connections
from tests.paper.equivalence.generator import Case, build_case
from tests.paper.equivalence.run_with import implementation
from tests.paper.equivalence.witnesses import reached_witnesses, witness_projection
from tests.paper.transcript import harness

LOGGER_NAMES = (
    "ggTrader.paper.trader",
    "ggTrader.paper.cash_sweep",
    "ggTrader.paper.untradable",
    "ggTrader.paper.notifier",
)
FAST_SEEDS = 100
FULL_SEEDS = 600
SEED_PASS_DEADLINE_SECONDS = 300


@dataclass(frozen=True)
class Recorded:
    """A transcript and its interleaved log records."""

    transcript: dict
    logs: tuple[tuple[int, str, int, str, str | None, str | None], ...]


class KthFailLock(harness.FakeRunLock):
    """Fail precisely the scripted verification call."""

    def __init__(self, recorder: harness.Recorder, fail_on: int | None) -> None:
        super().__init__(recorder)
        self.fail_on = fail_on
        self.count = 0

    def verify(self) -> None:
        self.count += 1
        self._failure = (
            RuntimeError("scripted lock failure") if self.count == self.fail_on else None
        )
        super().verify()


def run_recorded(impl: str, case: Case) -> Recorded:
    """Run either implementation through the same recorder and injected collaborators."""
    with (
        harness._gate_error_copy(),
        implementation(impl),
        patch.object(config, "load_dotenv", lambda *args, **kwargs: False),
    ):
        s = copy.deepcopy(case.scenario)
        harness._check_scenario(s)
        if "lock.verify" in s.persist:
            raise harness.HarnessError("scenario scripts lock.verify; use Case.lock_fails_on")
        run = harness._Run(s)
        run.lock = KthFailLock(run.recorder, case.lock_fails_on)
        logs = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.name not in LOGGER_NAMES:
                    raise harness.HarnessError(f"unexpected logger {record.name}")
                try:
                    exc_type, exc_value = record.exc_info[:2] if record.exc_info else (None, None)
                    exc = exc_type.__name__ if exc_type is not None else None
                    exc_text = str(exc_value) if exc_value is not None else None
                    logs.append(
                        (
                            len(run.recorder.calls),
                            record.name,
                            record.levelno,
                            record.getMessage(),
                            exc,
                            exc_text,
                        )
                    )
                except Exception as error:
                    raise harness.HarnessError(f"cannot capture log record: {error}") from error

        def membership(spec: str | None) -> harness.MembershipToken:
            if case.membership_error is None:
                return run.live_membership_source(spec)
            entry = run.recorder.begin("membership.live_membership_source", [spec], {})
            exc = ValueError(case.membership_error)
            entry["raised"] = harness._error_text(exc)
            raise exc

        result = None
        raised = None
        skipped = []
        with ExitStack() as stack:
            disable = logging.root.manager.disable
            stack.callback(logging.disable, disable)
            logging.disable(logging.NOTSET)
            names = (
                set(LOGGER_NAMES)
                | {"ggTrader"}
                | {
                    name
                    for name, logger in logging.Logger.manager.loggerDict.items()
                    if name.startswith("ggTrader.") and isinstance(logger, logging.Logger)
                }
            )
            for name in sorted(names):
                logger = logging.getLogger(name)
                old = (
                    logger.level,
                    logger.handlers[:],
                    logger.propagate,
                    logger.disabled,
                    logger.filters[:],
                )

                def restore(log: logging.Logger = logger, state: tuple = old) -> None:
                    log.setLevel(state[0])
                    log.handlers[:] = state[1]
                    log.propagate = state[2]
                    log.disabled = state[3]
                    log.filters[:] = state[4]

                stack.callback(restore)
                logger.setLevel(logging.DEBUG)
                logger.handlers[:] = [Capture()]
                logger.filters.clear()
                logger.disabled = False
                logger.propagate = False
            stack.enter_context(patch.dict(os.environ))
            for var in harness.ENV_VARS:
                os.environ.pop(var, None)
            os.environ.update(s.env)
            for name in harness.persist_function_names():
                if name != "paper_run_lock":
                    stack.enter_context(
                        patch.object(trader_module, name, run.persist_wrapper(name))
                    )
            replacements = {
                "paper_run_lock": run.paper_run_lock,
                "generate_core_signals": run.generate_core_signals,
                "refresh_benchmark_tape": run.refresh_benchmark_tape,
                "live_membership_source": membership,
                "_today_et": lambda: s.today_et,
                "_now_pt": lambda: s.now_pt,
                "uuid4": run.uuid4,
                "time": run.clock,
                "datetime": harness._pinned_datetime(s.now_pt),
            }
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(trader_module, name, replacement))
            trader = None
            try:
                trader = trader_module.PaperTrader(
                    run.broker,
                    run.notifier,
                    dry_run=s.dry_run,
                    read_only=s.read_only,
                    membership=s.membership,
                )
                result = trader.run()
            except Exception as exc:
                raised = harness._error_text(exc)
            if trader is not None:
                skipped = list(trader.skipped_writes)
        try:
            transcript = {
                "calls": run.recorder.calls,
                "result": harness.normalize(result) if result is not None else None,
                "raised": raised,
                "skipped_writes": skipped,
            }
            return Recorded(transcript, tuple(logs))
        except TypeError as exc:
            raise harness.HarnessError(f"the run result cannot be serialized: {exc}") from exc


def normalize_log(message: str) -> str:
    """Keep raw log text; the four-process investigation found no unstable records."""
    return message


def digest(rec: Recorded) -> str:
    """Hash the canonical transcript and ordered log tuples."""
    logs = tuple(
        (
            position,
            name,
            level,
            normalize_log(message),
            exc,
            normalize_log(exc_text) if exc_text is not None else None,
        )
        for position, name, level, message, exc, exc_text in rec.logs
    )
    return hashlib.sha256(
        (harness.to_json(rec.transcript) + harness.to_json(logs)).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SeedSummary:
    """The bounded information retained from a single seed."""

    seed: int
    digest: str
    witnesses: frozenset[str]
    violations: tuple[str, ...]
    error: str | None
    projections: tuple[tuple[str, str], ...] = ()


def safety_violations(case: Case, transcript: dict) -> tuple[str, ...]:
    """Check the ordered submit and dry-run safety properties."""
    calls = transcript["calls"]
    violations = []
    submits = {"broker.submit_buy", "broker.submit_sell"}
    intents = {
        "persist.create_order_intent",
        "persist.attach_broker_order",
        "persist.settle_order_intent",
    }
    read_only = (
        case.scenario.read_only if case.scenario.read_only is not None else case.scenario.dry_run
    )
    halt = False
    failed = False
    seen = {}
    sleeves = (
        case.scenario.signals.get("sleeves", {}) if isinstance(case.scenario.signals, dict) else {}
    )
    sweep = case.scenario.env.get("SWEEP_SYMBOL", "SPY")
    for index, call in enumerate(calls):
        target = call["target"]
        if case.scenario.dry_run and (target in submits or target in intents):
            violations.append(f"a: dry run {target} at {index}")
        if read_only and target.startswith("persist.") and target[8:] in WRITE_METHODS:
            violations.append(f"f: read-only write {target} at {index}")
        if failed and (target in submits or target == "persist.create_order_intent"):
            violations.append(f"c: {target} after failure at {index}")
        if target == "notifier.send" and "HALTED buys" in call["args"][0]:
            halt = True
        if target in submits:
            symbol, amount, client = call["args"]
            prior = calls[index - 2 : index] if index >= 2 else []
            if (
                len(prior) != 2
                or prior[0]["target"] != "persist.create_order_intent"
                or prior[0]["args"][0] != client
                or prior[0]["raised"]
                or prior[1]["target"] != "lock.verify"
                or prior[1]["raised"]
            ):
                violations.append(f"b: unprotected submit at {index}")
            if not call["raised"]:
                after = calls[index + 1] if index + 1 < len(calls) else None
                if (
                    after is None
                    or after["target"] != "persist.attach_broker_order"
                    or after["args"][0] != client
                    or after["raised"]
                ):
                    violations.append(f"b: unattached submit at {index}")
            if halt and target == "broker.submit_buy":
                violations.append(f"d: buy after halt at {index}")
            if not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
                violations.append(f"g: invalid amount at {index}")
            if symbol != sweep:
                if symbol in seen:
                    candidates = sum(symbol in sleeve["buys"] for sleeve in sleeves.values())
                    if len(sleeves) == 1 or candidates < 2:
                        violations.append(f"e: duplicate {symbol} at {index}")
                seen[symbol] = index
        if call["raised"] and (target in submits or target == "lock.verify"):
            failed = True
    return tuple(violations)


def summarize_seed(seed: int) -> SeedSummary:
    """Record and summarize one new-implementation run without retaining its transcript."""
    try:
        case = build_case(seed)
        with blocked_connections():
            rec = run_recorded("new", case)
        reached = reached_witnesses(case, rec.transcript, rec.logs)
        projections = tuple(
            sorted(
                (
                    name,
                    hashlib.sha256(
                        harness.to_json(
                            witness_projection(name, case, rec.transcript, rec.logs)
                        ).encode()
                    ).hexdigest(),
                )
                for name in reached
            )
        )
        return SeedSummary(
            seed, digest(rec), reached, safety_violations(case, rec.transcript), None, projections
        )
    except harness.HarnessError as exc:
        return SeedSummary(seed, "", frozenset(), (), f"seed {seed}: {exc}")
    except Exception as exc:
        exc.add_note(f"while summarizing seed {seed}")
        raise


class SeedBudgetExceeded(RuntimeError):
    """The seed pass exceeded its time budget."""


class SeedCache:
    """Compute each seed once, enforcing the shared pass budget."""

    def __init__(
        self, deadline: float = SEED_PASS_DEADLINE_SECONDS, clock: Callable[[], float] | None = None
    ) -> None:
        self.deadline = deadline
        self.clock = clock or time.monotonic
        self.computed: Counter[int] = Counter()
        self.elapsed_seconds = 0.0
        self._summaries: dict[int, SeedSummary] = {}

    def fill(self, seeds: Iterable[int]) -> None:
        start = self.clock()
        try:
            for seed in seeds:
                if seed in self._summaries:
                    continue
                if self.elapsed_seconds + self.clock() - start >= self.deadline:
                    raise SeedBudgetExceeded("seed pass exceeded its deadline")
                self._summaries[seed] = summarize_seed(seed)
                self.computed[seed] += 1
        finally:
            self.elapsed_seconds += self.clock() - start

    def get(self, seed: int) -> SeedSummary:
        return self._summaries[seed]

    def __len__(self) -> int:
        return len(self._summaries)
