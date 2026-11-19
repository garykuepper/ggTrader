"""Both implementations of `PaperTrader._run` give the same transcript for every golden scenario.

Mutable-set file (Task 47): Task 74 retires it with the frozen copy.
"""

from __future__ import annotations

from tests.paper.equivalence.run_with import run_with
from tests.paper.transcript.scenarios import EXPECTED_SCENARIO_COUNT, SCENARIOS


def test_both_implementations_give_identical_transcripts_for_every_golden_scenario() -> None:
    assert len(SCENARIOS) == EXPECTED_SCENARIO_COUNT == 103
    differing = [
        name
        for name, scenario in SCENARIOS.items()
        if run_with("new", scenario) != run_with("legacy", scenario)
    ]
    assert differing == []


def test_new_run_equals_old_run_for_100_seeds() -> None:
    from itertools import zip_longest

    from tests.paper.equivalence.generator import build_case
    from tests.paper.equivalence.runner import run_recorded

    for seed in range(100):
        case = build_case(seed)
        new = run_recorded("new", case)
        legacy = run_recorded("legacy", case)
        if new == legacy:
            continue
        calls = next(
            (
                (i, left, right)
                for i, (left, right) in enumerate(
                    zip_longest(new.transcript["calls"], legacy.transcript["calls"])
                )
                if left != right
            ),
            None,
        )
        logs = next(
            (
                (i, left, right)
                for i, (left, right) in enumerate(zip_longest(new.logs, legacy.logs))
                if left != right
            ),
            None,
        )
        raise AssertionError(
            f"seed {seed}: first differing call={calls!r}; first differing log={logs!r}; "
            f"new result={new.transcript['result']!r}; "
            f"legacy result={legacy.transcript['result']!r}; "
            f"new raised={new.transcript['raised']!r}; "
            f"legacy raised={legacy.transcript['raised']!r}; "
            f"skipped writes={new.transcript['skipped_writes']!r}/"
            f"{legacy.transcript['skipped_writes']!r}"
        )


def test_late_failure_and_membership_reach_map() -> None:
    from tests.paper.equivalence.generator import build_case
    from tests.paper.equivalence.runner import run_recorded
    from tests.paper.equivalence.witnesses import reached_witnesses

    for seed, witness in ((0, "W16"), (1, "W17"), (2, "W18"), (19, "W19")):
        case = build_case(seed)
        new = run_recorded("new", case)
        legacy = run_recorded("legacy", case)
        assert witness in reached_witnesses(case, new.transcript, new.logs), (seed, witness)
        assert new == legacy, (seed, witness, new, legacy)


def test_shared_submit_failure_script_is_differentially_equal():
    from tests.paper.equivalence.runner import run_recorded
    from tests.paper.test_run_digests import _submit_failure_with_broker_not_found

    case = _submit_failure_with_broker_not_found()
    assert run_recorded("new", case) == run_recorded("legacy", case)
