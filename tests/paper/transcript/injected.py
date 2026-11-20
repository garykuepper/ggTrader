"""Run transcript scenarios with an injected recording paper ledger."""

from __future__ import annotations

import copy
import os
from contextlib import ExitStack
from typing import cast
from unittest.mock import patch

from ggTrader.paper import trader as trader_module
from ggTrader.paper.ledger import PaperLedger
from tests.paper.transcript import harness
from tests.paper.transcript.harness import Scenario


class HarnessLedger:
    """Expose every ledger method through the scenario's recorder."""

    def __init__(self, run: harness._Run) -> None:
        for name, value in PaperLedger.__dict__.items():
            if not name.startswith("_") and callable(value):
                setattr(
                    self,
                    name,
                    run.paper_run_lock if name == "paper_run_lock" else run.persist_wrapper(name),
                )


def run_scenario_injected(scenario: Scenario) -> dict:
    """Run a scenario with an injected ledger and normalize the recorded transcript."""
    with harness._gate_error_copy():
        s = copy.deepcopy(scenario)
        harness._check_scenario(s)
        run = harness._Run(s)
        result: dict | None = None
        raised: str | None = None
        skipped: list[str] = []
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ))
            for var in harness.ENV_VARS:
                os.environ.pop(var, None)
            os.environ.update(s.env)
            replacements: dict[str, object] = {
                "generate_core_signals": run.generate_core_signals,
                "refresh_benchmark_tape": run.refresh_benchmark_tape,
                "live_membership_source": run.live_membership_source,
                "_today_et": lambda: s.today_et,
                "_now_pt": lambda: s.now_pt,
                "uuid4": run.uuid4,
                "time": run.clock,
                "datetime": harness._pinned_datetime(s.now_pt),
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
                    ledger=cast(PaperLedger, HarnessLedger(run)),
                )
                result = trader.run()
            except Exception as exc:
                raised = harness._error_text(exc)
            if trader is not None:
                skipped = list(trader.skipped_writes)
        try:
            return {
                "calls": run.recorder.calls,
                "result": harness.normalize(result) if result is not None else None,
                "raised": raised,
                "skipped_writes": skipped,
            }
        except TypeError as exc:
            raise harness.HarnessError(f"the run result cannot be serialized: {exc}") from exc
