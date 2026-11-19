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
