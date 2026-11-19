"""Exercise a fake authority, without launching jobs or inspecting the host."""

from dataclasses import dataclass

import pytest

from scripts.whole_job_prototype import PhaseResult, run_job


@dataclass(frozen=True)
class Identity:
    boot: str = "boot-a"
    namespace: str = "host-namespace"
    cgroup: str = "/narrow/test/job"
    generation: int = 1


class FakeBroker:
    """Models the broker contract; its inputs are fixture data, not trusted evidence."""

    def __init__(self, outcomes: list[PhaseResult], storage: dict[str, bool] | None = None) -> None:
        self.registered = Identity()
        self.observed = Identity()
        self.outcomes = iter(outcomes)
        self.started: list[tuple[str, int]] = []
        self.storage = storage if storage is not None else {"held": False}

    def run_phase(self, ticket: object, phase: str, seconds: int) -> PhaseResult:
        if self.storage["held"] or ticket != self.registered or self.observed != self.registered:
            raise PermissionError("held, stale or unverified job")
        self.started.append((phase, seconds))
        outcome = next(self.outcomes)
        if outcome.terminal:
            self.storage["held"] = True  # The real broker must persist this before stopping.
        return outcome


def test_interrupted_first_phase_persists_hold_and_blocks_replay() -> None:
    broker = FakeBroker([PhaseResult(137, terminal=True)])
    assert run_job(broker, broker.registered, ("core", "studies"), 30) == 137
    assert broker.started == [("core", 30)]
    restarted = FakeBroker([PhaseResult(0)], storage=broker.storage)
    with pytest.raises(PermissionError):
        run_job(restarted, broker.registered, ("core", "studies"), 30)


def test_timeout_stops_before_second_phase() -> None:
    broker = FakeBroker([PhaseResult(143, terminal=True)])
    assert run_job(broker, broker.registered, ("core", "studies"), 10) == 143
    assert broker.storage["held"] and broker.started == [("core", 10)]


@pytest.mark.parametrize("first,expected", [(1, 1), (5, 0)])
def test_ordinary_failure_and_study_exit_five_continue(first: int, expected: int) -> None:
    broker = FakeBroker([PhaseResult(first), PhaseResult(0)])
    assert run_job(broker, broker.registered, ("core", "studies"), 15, studies=True) == expected
    assert broker.started == [("core", 15), ("studies", 15)]


@pytest.mark.parametrize(
    "changed",
    [
        Identity(boot="boot-b"),
        Identity(namespace="hidden"),
        Identity(cgroup="/outside"),
        Identity(generation=2),
    ],
)
def test_fake_authority_rejects_unverified_or_stale_identity(changed: Identity) -> None:
    broker = FakeBroker([PhaseResult(0)])
    broker.observed = changed
    with pytest.raises(PermissionError):
        run_job(broker, broker.registered, ("core",), 10)
    assert broker.started == []


def test_invalid_timeout_is_refused_before_broker_call() -> None:
    broker = FakeBroker([PhaseResult(0)])
    with pytest.raises(ValueError):
        run_job(broker, broker.registered, ("core",), 0)
    assert broker.started == []


def test_unheld_terminal_code_is_rejected() -> None:
    broker = FakeBroker([PhaseResult(124), PhaseResult(0)])
    with pytest.raises(RuntimeError, match="unheld terminal"):
        run_job(broker, broker.registered, ("core", "studies"), 10)
    assert broker.started == [("core", 10)]


def test_terminal_second_phase_overrides_first_failure() -> None:
    broker = FakeBroker([PhaseResult(1), PhaseResult(143, terminal=True)])
    assert run_job(broker, broker.registered, ("core", "studies"), 10) == 143


def test_study_half_exit_five_depends_on_collection_floor() -> None:
    for allowed, expected in [(True, 0), (False, 5)]:
        broker = FakeBroker([PhaseResult(0), PhaseResult(5)])
        assert (
            run_job(broker, broker.registered, ("core", "studies"), 10, empty_studies=allowed)
            == expected
        )
