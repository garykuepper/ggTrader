"""Check that a failed broker clock lookup is logged without changing the run."""

import logging
from pathlib import Path

from tests.paper.transcript.harness import run_scenario, to_json
from tests.paper.transcript.scenarios import SCENARIOS


def test_poll_clock_failure_is_logged_at_debug(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="ggTrader.paper.trader"):
        actual = to_json(run_scenario(SCENARIOS["poll_clock_failure"]))
    golden = Path(__file__).parent / "golden/trader_transcripts/poll_clock_failure.json"
    assert actual == golden.read_text()
    assert any("RuntimeError" in record.getMessage() for record in caplog.records)
