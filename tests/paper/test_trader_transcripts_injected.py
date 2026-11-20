"""The injected transcript harness reproduces every recorded golden."""

from pathlib import Path

import pytest

from tests.paper.transcript.harness import to_json
from tests.paper.transcript.injected import run_scenario_injected
from tests.paper.transcript.scenarios import EXPECTED_SCENARIO_COUNT, SCENARIOS

GOLDEN_DIR = Path(__file__).parent / "golden" / "trader_transcripts"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_injected_transcript_matches_golden(name: str) -> None:
    assert len(SCENARIOS) == EXPECTED_SCENARIO_COUNT == 103
    expected = (GOLDEN_DIR / f"{name}.json").read_bytes()
    assert to_json(run_scenario_injected(SCENARIOS[name])).encode("utf-8") == expected
