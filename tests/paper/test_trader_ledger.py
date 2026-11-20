"""The trader reaches the database only through a `PaperLedger` (Phase 4, Task 39).

The tests pin the default ledger factory, late-bound injected methods, read-only logging and a
complete run against an `InMemoryLedger` without persistence patches.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper import trader as trader_module
from ggTrader.paper.ledger import InMemoryLedger, PaperLedger
from ggTrader.paper.persist import PgPaperLedger
from ggTrader.paper.trader import PaperTrader
from tests.paper.transcript import harness
from tests.paper.transcript.harness import ENV_VARS, run_scenario
from tests.paper.transcript.injected import HarnessLedger
from tests.paper.transcript.scenarios import SCENARIOS

TRADER_SOURCE = Path(trader_module.__file__)
GOLDEN_DIR = Path(__file__).parent / "golden" / "trader_transcripts"
TRADER_LOGGER = "ggTrader.paper.trader"

#: Calls a read-only run must never make (Deviation 11): the intent lifecycle and the legacy
#: pending-order and trade writers. A transcript target is ``persist.<name>``.
INTENT_AND_LEGACY_WRITES = (
    "create_order_intent",
    "attach_broker_order",
    "settle_order_intent",
    "settle_legacy_pending_order",
    "log_trade",
    "log_pending_order",
    "clear_pending_order",
    "mark_pending_order_stale",
)


@contextmanager
def _trader_environment(scenario_name: str, *, wrap_persist: bool) -> Iterator[harness._Run]:
    """Patch what `harness._run_scenario` patches in the trader module, for one scenario.

    With ``wrap_persist`` the default factory returns a recording ledger; otherwise the caller
    injects its own ledger.
    """
    s = SCENARIOS[scenario_name]
    run = harness._Run(s)
    with ExitStack() as stack, harness._gate_error_copy():
        stack.enter_context(patch.dict(os.environ))
        for var in ENV_VARS:
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
        if wrap_persist:
            replacements["PgPaperLedger"] = lambda: HarnessLedger(run)
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(trader_module, name, replacement))
        yield run


def _build_trader(run: harness._Run, **extra: object) -> PaperTrader:
    s = run.s
    return PaperTrader(
        run.broker,  # type: ignore[arg-type]
        run.notifier,
        dry_run=s.dry_run,
        read_only=s.read_only,
        membership=s.membership,
        **extra,  # type: ignore[arg-type]
    )


def _is_read_only(name: str) -> bool:
    s = SCENARIOS[name]
    return s.dry_run and (s.read_only is None or s.read_only)


def test_trader_module_binds_the_26_ledger_names() -> None:
    # The name is pinned by the phase 4 required-node list, so it stays; what it asserts now is
    # that the trader module binds PgPaperLedger and builds it lazily as the default ledger.
    assert trader_module.PgPaperLedger is PgPaperLedger
    with patch("ggTrader.paper.persist._get_engine", side_effect=AssertionError("database read")):
        assert isinstance(PgPaperLedger(), PgPaperLedger)
    assert all(
        callable(getattr(PgPaperLedger, name, None))
        for name in PaperLedger.__dict__
        if not name.startswith("_") and callable(PaperLedger.__dict__[name])
    )
    with patch.object(trader_module, "PgPaperLedger") as factory:
        trader = PaperTrader(MagicMock(), MagicMock(), dry_run=False, membership="db")
    factory.assert_called_once_with()
    assert trader._ledger is factory.return_value
    assert not factory.return_value.mock_calls


def test_trader_has_no_write_method_and_no_direct_persist_call() -> None:
    tree = ast.parse(TRADER_SOURCE.read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_write"]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "_write"]
    direct = sorted(
        {
            f"{node.func.id}@{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in PaperLedger.__dict__
        }
    )
    assert direct == []


def test_string_patch_after_construction_still_wins() -> None:
    with _trader_environment("plain_buys_and_sells", wrap_persist=True) as run:
        trader = _build_trader(run)
        # A method replaced after construction must be read from the injected ledger.
        assert trader._ledger.get_peak_value() is None
        late = MagicMock(return_value=123.0)
        with patch.object(trader._ledger, "get_peak_value", late):
            assert trader._ledger.get_peak_value() == 123.0
            trader.run()
    assert late.call_count == 2  # the direct call and the one inside the run
    reads = [c for c in run.recorder.calls if c["target"] == "persist.get_peak_value"]
    assert len(reads) == 1  # only the call made before the patch reached the harness wrapper
    # the other reads still went to the harness wrappers
    assert any(c["target"] == "persist.get_latest_snapshot" for c in run.recorder.calls)


def test_read_only_run_records_the_same_skipped_names_as_before() -> None:
    golden = json.loads((GOLDEN_DIR / "dry_run_read_only.json").read_text(encoding="utf-8"))
    expected = ["init_paper_schema", "refresh_benchmark_tape", "save_peak_value", "log_snapshot"]
    assert golden["skipped_writes"] == expected
    assert run_scenario(SCENARIOS["dry_run_read_only"])["skipped_writes"] == expected


def test_read_only_skip_is_logged_by_the_trader_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        run_scenario(SCENARIOS["dry_run_read_only"])
    skipped = [r for r in caplog.records if r.getMessage().startswith("read-only: skipped write")]
    assert [r.getMessage() for r in skipped] == [
        "read-only: skipped write init_paper_schema",
        "read-only: skipped write refresh_benchmark_tape",
        "read-only: skipped write save_peak_value",
        "read-only: skipped write log_snapshot",
    ]
    assert {r.name for r in skipped} == {TRADER_LOGGER}
    assert {r.levelno for r in skipped} == {logging.INFO}


@pytest.mark.parametrize("name", sorted(n for n in SCENARIOS if _is_read_only(n)))
def test_read_only_ledger_is_never_asked_to_run_an_intent_call(name: str) -> None:
    targets = {c["target"] for c in run_scenario(SCENARIOS[name])["calls"]}
    forbidden = {f"persist.{fn}" for fn in INTENT_AND_LEGACY_WRITES}
    assert targets & forbidden == set()


def test_trader_runs_on_an_in_memory_ledger() -> None:
    golden = json.loads((GOLDEN_DIR / "plain_buys_and_sells.json").read_text(encoding="utf-8"))
    ledger = InMemoryLedger()
    # The scenario scripts the two snapshot reads; seed the same values as stored snapshots.
    ledger.log_snapshot("2026-06-17", 100_000.0, 0.0, {})
    ledger.log_snapshot("2026-06-18", 99_500.0, 0.0, {})
    with _trader_environment("plain_buys_and_sells", wrap_persist=False) as run:
        trader = _build_trader(run, ledger=ledger)
        result = trader.run()
    assert harness.normalize(result) == golden["result"]
    assert result["skipped_writes"] == []
    assert ledger.schema_initialized
    assert ledger.lock_entries == 1
    assert ledger.lock_verifications == 3  # one per submitted order: a sell and two buys
    assert not ledger.get_unresolved_order_intents()
    settlements = [
        c["args"] for c in golden["calls"] if c["target"] == "persist.settle_order_intent"
    ]
    assert len(settlements) == len(ledger._intents) == 3
    for client_id, order_id, qty, price, status in settlements:
        row = ledger._intents[client_id]
        assert (row["client_order_id"], row["broker_order_id"], row["state"]) == (
            client_id,
            order_id,
            "resolved",
        )
        assert (row["filled_qty"], row["filled_avg_price"], row["terminal_status"]) == (
            qty,
            price,
            status,
        )
    settled = {(t["side"], t["symbol"]) for t in ledger.trades}
    assert settled == {("SELL", "AAPL"), ("BUY", "NVDA"), ("BUY", "GOOGL")}
    assert ledger.get_latest_snapshot() == pytest.approx(100_053.0)
    # nothing but the harness's fakes was patched: no persist function recorded a call
    assert not [c for c in run.recorder.calls if c["target"].startswith("persist.")]
