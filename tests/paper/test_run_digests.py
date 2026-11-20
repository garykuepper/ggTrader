"""Focused checks for the seeded comparison environment."""

import math
import os
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.paper.equivalence import impl
from tests.paper.equivalence.generator import build_case
from tests.paper.equivalence.runner import SeedCache, digest, run_recorded


def test_both_implementations_share_one_execution_environment():
    from tests.paper.transcript import harness

    lock_case = replace(build_case(3), lock_fails_on=2)
    membership_case = replace(build_case(3), membership_error="membership unavailable")
    log_case = build_case(15)
    for case in (build_case(0), lock_case, membership_case, log_case):
        recordings = []
        for name in impl.IMPLS:
            with (
                patch.object(harness, "_Run", wraps=harness._Run) as constructor,
                patch.object(harness, "run_scenario", side_effect=AssertionError("nested harness")),
            ):
                rec = run_recorded(name, case)
                assert constructor.call_count == 1
                recordings.append(rec)
            if case is lock_case:
                verifies = [c for c in rec.transcript["calls"] if c["target"] == "lock.verify"]
                assert len(verifies) >= 2 and not verifies[0]["raised"] and verifies[1]["raised"]
            if case is membership_case:
                assert any(
                    c["target"] == "membership.live_membership_source" and c["raised"]
                    for c in rec.transcript["calls"]
                )
            if case is log_case:
                assert any(r[4:] == ("RuntimeError", "notice failed") for r in rec.logs)
                assert all(0 <= r[0] <= len(rec.transcript["calls"]) for r in rec.logs)
                assert [r[0] for r in rec.logs] == sorted(r[0] for r in rec.logs)
        assert all(rec == recordings[0] for rec in recordings)
        assert all(digest(rec) == digest(recordings[0]) for rec in recordings)


def test_cache_reuses_completed_seeds():
    cache = SeedCache()
    cache.fill([0, 1])
    cache.fill([1, 0])
    assert dict(cache.computed) == {0: 1, 1: 1}
    assert cache.get(0).seed == 0


def test_safety_checks_detect_unprotected_submit_and_invalid_amount():
    from tests.paper.equivalence.runner import safety_violations

    case = build_case(3)
    transcript = {
        "calls": [
            {"target": "broker.submit_buy", "args": ["MSFT", 0, "id"], "kwargs": {}, "raised": None}
        ],
        "result": None,
        "raised": None,
        "skipped_writes": [],
    }
    violations = safety_violations(case, transcript)
    assert any(v.startswith("b:") for v in violations)
    assert any(v.startswith("g:") for v in violations)


def test_safety_invariants_for_100_seeds(fast_summaries):
    assert {s.seed: s.violations for s in fast_summaries if s.violations} == {}


def test_late_failure_and_membership_witnesses_require_observed_events():
    from tests.paper.equivalence.witnesses import reached_witnesses

    for seed, witness in [(0, "W16"), (1, "W17"), (2, "W18"), (3, "W20")]:
        case = build_case(seed)
        rec = run_recorded("new", case)
        assert witness in reached_witnesses(case, rec.transcript, rec.logs)
        assert witness not in reached_witnesses(case, {**rec.transcript, "calls": []}, ())
    case = replace(build_case(3), membership_error="membership unavailable")
    rec = run_recorded("new", case)
    assert "W19" in reached_witnesses(case, rec.transcript, rec.logs)


@pytest.fixture(scope="session")
def seed_cache():
    return SeedCache()


@pytest.fixture(scope="session")
def fast_summaries(seed_cache):
    from tests.netguard import blocked_connections

    with blocked_connections():
        seed_cache.fill(range(100))
    return tuple(seed_cache.get(seed) for seed in range(100))


@pytest.fixture(scope="session")
def full_seed_cache(seed_cache, fast_summaries):
    from tests.netguard import blocked_connections

    with blocked_connections():
        seed_cache.fill(range(600))
    return seed_cache


@pytest.fixture(scope="session")
def full_summaries(full_seed_cache):
    return tuple(full_seed_cache.get(seed) for seed in range(600))


@pytest.mark.slow
def test_witness_counts_for_first_600(full_seed_cache):
    from collections import Counter

    from tests.paper.equivalence.witnesses import WITNESSES

    cache = full_seed_cache
    for size in (100, 600):
        counts = Counter(w for seed in range(size) for w in cache.get(seed).witnesses)
        print(size, {name: counts[name] for name in WITNESSES})
        assert all(counts[name] >= (1 if size == 100 else 5) for name in WITNESSES)
    import pickle

    assert all(len(pickle.dumps(cache.get(seed))) < 2048 for seed in range(600))
    errors = {seed: cache.get(seed).error for seed in range(600) if cache.get(seed).error}
    violations = {
        seed: cache.get(seed).violations for seed in range(600) if cache.get(seed).violations
    }
    print("errors", errors, "violations", violations)
    assert not errors
    assert not violations


def test_structured_cases_reach_underrepresented_witnesses():
    from tests.paper.equivalence.witnesses import reached_witnesses

    for seed, witness in [(8, "W2"), (9, "W5"), (10, "W8"), (11, "W14"), (12, "W15"), (13, "W13")]:
        case = build_case(seed)
        rec = run_recorded("new", case)
        assert witness in reached_witnesses(case, rec.transcript, rec.logs), (seed, rec)


def test_cache_deadline_keeps_completed_summaries(monkeypatch):
    from tests.paper.equivalence import runner

    elapsed = iter((0.0, 0.1, 1.0, 1.0))

    def clock():
        return next(elapsed)

    monkeypatch.setattr(
        runner, "summarize_seed", lambda seed: runner.SeedSummary(seed, "", frozenset(), (), None)
    )
    cache = SeedCache(deadline=0.5, clock=clock)
    with pytest.raises(runner.SeedBudgetExceeded):
        cache.fill([1, 2])
    assert cache.get(1).seed == 1
    assert dict(cache.computed) == {1: 1}


def test_logger_state_is_restored_and_filters_cannot_hide_records():
    import logging

    logger = logging.getLogger("ggTrader.paper.trader")
    old = (logger.level, logger.disabled, logger.filters[:], logging.root.manager.disable)
    try:
        logger.setLevel(logging.CRITICAL)
        logger.disabled = True
        logger.addFilter(lambda record: False)
        logging.disable(logging.CRITICAL)
        rec = run_recorded("new", build_case(3))
        assert rec.logs
        assert (logger.level, logger.disabled, logger.filters, logging.root.manager.disable) == (
            logging.CRITICAL,
            True,
            [logger.filters[-1]],
            logging.CRITICAL,
        )
    finally:
        logger.setLevel(old[0])
        logger.disabled = old[1]
        logger.filters[:] = old[2]
        logging.disable(old[3])


def test_scripted_lock_failure_is_rejected():
    from tests.paper.transcript.harness import HarnessError

    case = build_case(3)
    scenario = replace(
        case.scenario, persist={**case.scenario.persist, "lock.verify": RuntimeError("existing")}
    )
    with pytest.raises(HarnessError, match="lock.verify"):
        run_recorded("new", replace(case, scenario=scenario))


@pytest.mark.parametrize("fault", ["delayed_attach", "failed_lock"])
def test_submit_guard_rejects_each_ordering_fault(fault):
    from tests.paper.equivalence.runner import safety_violations

    def call(target, args=(), raised=None):
        return {"target": target, "args": list(args), "kwargs": {}, "raised": raised}

    case = build_case(3)
    calls = [
        call("persist.create_order_intent", ("id",)),
        call("lock.verify", raised="failed" if fault == "failed_lock" else None),
        call("broker.submit_buy", ("MSFT", 100, "id")),
        call("persist.attach_broker_order", ("id", "order")),
    ]
    if fault == "delayed_attach":
        calls.insert(3, call("clock.time"))
    failures = safety_violations(case, {"calls": calls})
    assert any(v.startswith("b:") for v in failures)


def test_cache_failed_summary_is_not_counted():
    from tests.paper.equivalence import runner

    cache = SeedCache(clock=lambda: 0)
    with (
        patch.object(runner, "summarize_seed", side_effect=ValueError("fail")),
        pytest.raises(ValueError),
    ):
        cache.fill([1])
    assert not cache.computed


@pytest.mark.slow
def test_each_witness_has_five_distinct_recordings(full_seed_cache):
    from collections import defaultdict

    from tests.paper.equivalence.witnesses import WITNESSES

    cache = full_seed_cache
    distinct = defaultdict(set)
    for seed in range(600):
        summary = cache.get(seed)
        for witness, projection in summary.projections:
            distinct[witness].add(projection)
    counts = {name: len(distinct[name]) for name in WITNESSES}
    print("distinct witness projections", counts)
    assert all(count >= 5 for count in counts.values())


def test_log_capture_failure_escapes_trader_exception_handlers():
    import logging

    from tests.paper.transcript import harness

    def bad_log(run, *args, **kwargs):
        logging.getLogger("ggTrader.paper.trader").info("invalid integer %d", "bad")

    with patch.object(harness._Run, "generate_core_signals", bad_log):
        with pytest.raises(harness.HarnessError, match="cannot capture log record"):
            run_recorded("new", build_case(3))


def test_summarize_neutralizes_dotenv_during_harness_construction():
    from ggTrader.utils import config
    from tests.paper.equivalence.runner import summarize_seed
    from tests.paper.transcript import harness

    original = harness._Run
    seen = []

    def construct(scenario):
        seen.append(config.load_dotenv("anything"))
        return original(scenario)

    with (
        patch.object(harness, "_Run", side_effect=construct),
        patch.object(config, "load_dotenv", side_effect=AssertionError("dotenv was read")),
    ):
        summary = summarize_seed(3)
    assert summary.error is None
    assert seen == [False]


def test_stopped_buy_witness_requires_the_skip_log():
    from tests.paper.equivalence.witnesses import reached_witnesses

    case = build_case(14)
    rec = run_recorded("new", case)
    assert "W1" in reached_witnesses(case, rec.transcript, rec.logs)
    assert "W1" not in reached_witnesses(case, rec.transcript, ())


def test_generated_exception_log_includes_exception_text():
    recorded = run_recorded("new", build_case(15))
    assert any(r[4] == "RuntimeError" and r[5] == "notice failed" for r in recorded.logs)


def test_log_with_empty_exception_tuple_is_recorded():
    import logging

    from tests.paper.transcript import harness

    original = harness._Run.generate_core_signals

    def emit(run, *args, **kwargs):
        logging.getLogger("ggTrader.paper.trader").error(
            "empty exception", exc_info=(None, None, None)
        )
        return original(run, *args, **kwargs)

    with patch.object(harness._Run, "generate_core_signals", emit):
        recorded = run_recorded("new", build_case(3))
    assert any(r[3] == "empty exception" and r[4:] == (None, None) for r in recorded.logs)


def test_late_run_witness_rejects_dry_run():
    from tests.paper.equivalence.witnesses import WITNESSES

    case = replace(build_case(4), scenario=replace(build_case(4).scenario, dry_run=True))
    assert not WITNESSES["W12"]({"case": case, "transcript": {"result": {"gate": "late_run"}}})


def test_digest_changes_if_exception_text_changes():
    from tests.paper.equivalence.runner import Recorded

    recorded = run_recorded("new", build_case(15))
    index = next(i for i, row in enumerate(recorded.logs) if row[5] is not None)
    logs = list(recorded.logs)
    logs[index] = (*logs[index][:5], "different exception")
    assert digest(recorded) != digest(Recorded(recorded.transcript, tuple(logs)))


def test_every_generated_case_runs_without_a_harness_error(fast_summaries):
    assert {s.seed: s.error for s in fast_summaries if s.error} == {}


@pytest.mark.slow
def test_every_generated_case_runs_without_a_harness_error_600(full_summaries):
    assert {s.seed: s.error for s in full_summaries if s.error} == {}


def test_every_witness_is_reached_at_least_once_in_100_seeds(fast_summaries):
    from tests.paper.equivalence.witnesses import WITNESSES

    reached = set().union(*(s.witnesses for s in fast_summaries))
    assert set(WITNESSES) <= reached


@pytest.mark.slow
def test_every_witness_is_reached_at_least_five_times_in_600_seeds(full_summaries):
    from collections import Counter

    from tests.paper.equivalence.witnesses import WITNESSES

    counts = Counter(w for s in full_summaries for w in s.witnesses)
    assert {w: counts[w] for w in WITNESSES if counts[w] < 5} == {}


@pytest.mark.slow
def test_safety_invariants_for_600_seeds(full_summaries):
    assert {s.seed: s.violations for s in full_summaries if s.violations} == {}


def test_generated_sizing_is_positive():
    for seed in range(600):
        case = build_case(seed)
        assert case.scenario.signals["scale"] > 0, seed
        assert all(weight > 0 for weight in case.scenario.signals["weights"].values()), seed
        assert all(
            account["portfolio_value"] > 0
            for account in case.scenario.replies["get_account"]
            if isinstance(account, dict)
        ), seed


def test_summarize_seed_keeps_no_transcript(fast_summaries):
    import pickle

    for summary in fast_summaries:
        assert not any(isinstance(value, (dict, list)) for value in vars(summary).values())
        assert len(pickle.dumps(summary)) < 2048


@pytest.mark.slow
def test_the_cache_runs_each_seed_once(full_seed_cache):
    assert set(full_seed_cache.computed) == set(range(600))
    assert all(count == 1 for count in full_seed_cache.computed.values())
    before = full_seed_cache.computed.copy()
    full_seed_cache.fill(range(600))
    assert full_seed_cache.computed == before
    assert sum(before.values()) == len(full_seed_cache)


def _digest_record():
    import json
    from pathlib import Path

    return json.loads(
        (Path(__file__).parent / "golden" / "run_equivalence_digests.json").read_text()
    )


def test_new_run_matches_the_recorded_digests_for_100_seeds(fast_summaries):
    expected = _digest_record()["digests"]
    assert {
        s.seed: (expected[str(s.seed)], s.digest)
        for s in fast_summaries
        if expected[str(s.seed)] != s.digest
    } == {}


@pytest.mark.slow
def test_new_run_matches_the_recorded_digests_for_600_seeds(full_summaries):
    expected = _digest_record()["digests"]
    assert {
        s.seed: (expected[str(s.seed)], s.digest)
        for s in full_summaries
        if expected[str(s.seed)] != s.digest
    } == {}


def test_digest_file_pins_the_generator_and_the_runner():
    import hashlib
    import re
    from pathlib import Path

    from tests.paper.equivalence import generator, runner, witnesses

    record = _digest_record()
    assert record["production_base"] == "73c35417"
    assert re.fullmatch(r"[0-9a-f]{40}", record["recorder_commit"])
    assert record["normalization_version"]
    assert set(record["digests"]) == {str(seed) for seed in range(600)}
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in record["digests"].values())
    for module in (generator, witnesses, runner):
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        assert record[f"{module.__name__.rsplit('.', 1)[-1]}_sha256"] == actual


def test_generator_is_deterministic_across_hash_seeds():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = """
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
import ggTrader
root = Path(os.environ["GGT_TEST_ROOT"]).resolve()
assert Path(ggTrader.__file__).resolve().is_relative_to(root)
import ggTrader.utils.config as config
config.load_dotenv = lambda *args, **kwargs: False
os.environ["GGT_MEMBERSHIP"] = "fixture:sp500_v1_2026-06-02"
from tests.netguard import blocked_connections
from tests.paper.equivalence import generator
from tests.paper.transcript.harness import to_json
assert Path(generator.__file__).resolve().is_relative_to(root)
build_case = generator.build_case

def clean(value):
    if isinstance(value, BaseException):
        return {"exception": type(value).__name__, "message": str(value)}
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value

with blocked_connections():
    cases = [clean(asdict(build_case(seed))) for seed in range(600)]
print(hashlib.sha256(to_json(cases).encode()).hexdigest())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={
                "PATH": os.defpath,
                "PYTHONPATH": os.pathsep.join((str(root), str(root / "src"))),
                "GGT_TEST_ROOT": str(root),
                "PYTHONHASHSEED": seed,
                "GGT_MEMBERSHIP": "fixture:sp500_v1_2026-06-02",
            },
            text=True,
            timeout=20,
        )
        for seed in ("1", "2")
    ]
    assert outputs[0] == outputs[1]


def test_generated_sizing_and_buy_candidates_are_valid():
    for seed in range(600):
        case = build_case(seed)
        scenario = case.scenario
        assert not scenario.read_only or scenario.dry_run
        assert scenario.signals["scale"] > 0
        assert all(weight > 0 for weight in scenario.signals["weights"].values())
        assert all(
            account["portfolio_value"] > 0
            for account in scenario.replies["get_account"]
            if isinstance(account, dict)
        )
        for sleeve in scenario.signals["sleeves"].values():
            assert len(sleeve["buys"]) == len(set(sleeve["buys"]))
        buys = [
            symbol for sleeve in scenario.signals["sleeves"].values() for symbol in sleeve["buys"]
        ]
        if not scenario.name.startswith("duplicate_buy_"):
            assert len(buys) == len(set(buys))


def test_generated_cases_cover_scripted_inputs():
    generated = [build_case(seed) for seed in range(600)]
    cases = [case.scenario for case in generated]
    assert any(case.fills for case in cases)
    assert any(case.existing_orders for case in cases)
    assert any(case.splits["corp_splits"] for case in cases)
    assert any(case.dividends["corp_dividends"] for case in cases)
    assert any(case.notifier for case in cases)
    assert any("submit_buy" in case.replies for case in cases)
    assert any(isinstance(case.replies["get_clock"][0], Exception) for case in cases)
    assert any("UNTRADABLE_SYMBOLS" in case.env for case in cases)
    assert sum(case.membership_error is not None for case in generated) > 5
    assert any(case.lock_fails_on is not None for case in generated)
    assert any(isinstance(value, Exception) for case in cases for value in case.notifier.values())


def test_late_failure_cases_reach_their_first_order():
    from tests.paper.transcript.harness import run_scenario

    reached = {"late_stop": 0, "late_exit": 0, "late_sleeve": 0}
    for seed in range(100):
        scenario = build_case(seed).scenario
        kind = scenario.name.rsplit("_", 1)[0]
        if kind not in reached:
            continue
        transcript = run_scenario(scenario)
        orders = [
            call for call in transcript["calls"] if call["target"].startswith("broker.submit_")
        ]
        assert orders, (seed, transcript["raised"])
        expected = "KeyError: 'sleeve1'" if kind == "late_sleeve" else "KeyError: 'qty'"
        assert transcript["raised"] == expected, (seed, transcript["raised"])
        assert orders[0]["target"] == (
            "broker.submit_buy" if kind == "late_sleeve" else "broker.submit_sell"
        )
        reached[kind] += 1
    assert all(count >= 2 for count in reached.values()), reached


def test_deliberate_duplicate_buy_reaches_both_sleeves():
    from tests.paper.transcript.harness import run_scenario

    reached = 0
    for seed in range(100):
        scenario = build_case(seed).scenario
        if not scenario.name.startswith("duplicate_buy_"):
            continue
        transcript = run_scenario(scenario)
        buys = [call for call in transcript["calls"] if call["target"] == "broker.submit_buy"]
        assert len(buys) == 2, (seed, transcript["raised"])
        expected = scenario.signals["sleeves"]["sleeve0"]["buys"][0]
        assert buys[0]["args"][0] == buys[1]["args"][0] == expected
        reached += 1
    assert reached >= 2


@pytest.mark.parametrize("seed", range(4, 600, 80))
def test_structured_live_late_gate(seed):
    from tests.paper.transcript.harness import run_scenario

    transcript = run_scenario(build_case(seed).scenario)
    assert transcript["raised"] is None
    assert transcript["result"]["gate"] == "late_run"
    assert not [call for call in transcript["calls"] if call["target"].startswith("broker.submit_")]


@pytest.mark.parametrize("seed", range(5, 600, 80))
def test_structured_sweep_round_trip(seed):
    from tests.paper.transcript.harness import run_scenario

    sweep = run_scenario(build_case(seed).scenario)
    submits = [call for call in sweep["calls"] if call["target"].startswith("broker.submit_")]
    assert [(call["target"], call["args"][0]) for call in submits] == [
        ("broker.submit_sell", "SPY"),
        ("broker.submit_buy", "MSFT"),
        ("broker.submit_buy", "SPY"),
    ], (sweep["raised"], submits)


@pytest.mark.parametrize("seed", [6, 86])
def test_second_sell_failure_follows_successful_first_submit(seed):
    from tests.paper.transcript.harness import run_scenario

    scenario = build_case(seed).scenario
    transcript = run_scenario(scenario)
    sells = [call for call in transcript["calls"] if call["target"] == "broker.submit_sell"]
    expected = scenario.signals["sleeves"]["sleeve0"]["sells"]
    assert len(sells) == 2, transcript
    assert sells[0]["args"][0] == expected[0] and sells[0]["raised"] is None
    assert (
        sells[1]["args"][0] == expected[1] and sells[1]["raised"] == "RuntimeError: submit failed"
    )


@pytest.mark.parametrize("seed", [7, 87])
def test_unassessable_stop_halts_buys_but_allows_exit(seed):
    from tests.paper.transcript.harness import run_scenario

    scenario = build_case(seed).scenario
    transcript = run_scenario(scenario)
    held = scenario.replies["get_positions"][0]
    bad = next(symbol for symbol, position in held.items() if math.isnan(position["cost_basis"]))
    assert transcript["result"]["errors"] == [f"Cannot assess catastrophe stop for: {bad}"]
    submits = [call for call in transcript["calls"] if call["target"].startswith("broker.submit_")]
    expected = scenario.signals["sleeves"]["sleeve0"]["sells"][0]
    assert [(call["target"], call["args"][0]) for call in submits] == [
        ("broker.submit_sell", expected)
    ]


@pytest.mark.parametrize("rule", ["a", "c", "d", "e", "f"])
def test_safety_invariant_negative_controls(rule):
    from tests.paper.equivalence.runner import safety_violations

    def call(target, *args, raised=None):
        return {"target": target, "args": list(args), "kwargs": {}, "raised": raised}

    def buy(client):
        return [
            call("persist.create_order_intent", client),
            call("lock.verify"),
            call("broker.submit_buy", "MSFT", 100.0, client),
            call("persist.attach_broker_order", client, "order"),
        ]

    case = build_case(14)
    if rule in ("a", "f"):
        case = replace(case, scenario=replace(case.scenario, dry_run=True, read_only=True))
    if rule == "a":
        calls = buy("one")
    elif rule == "c":
        calls = [call("lock.verify", raised="failed"), *buy("one")]
    elif rule == "d":
        calls = [call("notifier.send", "HALTED buys"), *buy("one")]
    elif rule == "e":
        calls = [*buy("one"), *buy("two")]
    else:
        calls = [call("persist.log_snapshot")]
    assert any(v.startswith(f"{rule}:") for v in safety_violations(case, {"calls": calls}))


def test_partial_sell_cancellation_blocks_replacement_buy_in_seed_16():
    case = build_case(16)
    recorded = run_recorded("new", case)
    assert any("not fully filled" in error for error in recorded.transcript["result"]["errors"])
    sells = [c for c in recorded.transcript["calls"] if c["target"] == "broker.submit_sell"]
    assert len(sells) == 1 and sells[0]["args"][:2] == ["AAPL", 10.0]
    settlements = [
        c for c in recorded.transcript["calls"] if c["target"] == "persist.settle_order_intent"
    ]
    assert len(settlements) == 1 and settlements[0]["args"][2:] == [4.0, 150.0, "canceled"]
    assert not any(c["target"] == "broker.submit_buy" for c in recorded.transcript["calls"])
    assert any(sleeve["buys"] for sleeve in case.scenario.signals["sleeves"].values())


def _submit_failure_with_broker_not_found():
    """Script the broker's post-submit lookup reporting an unknown order."""
    from dataclasses import replace

    from tests.paper.equivalence.generator import build_case

    case = build_case(6)
    replies = {
        **case.scenario.replies,
        "get_order": [RuntimeError("broker order not found")],
        "get_order_by_client_id": [RuntimeError("broker order not found")],
    }
    return replace(case, scenario=replace(case.scenario, replies=replies))


def test_submit_failure_stops_before_broker_polling():
    from tests.paper.equivalence.runner import run_recorded

    recorded = run_recorded("new", _submit_failure_with_broker_not_found())
    assert recorded.transcript["raised"] == "RuntimeError: submit failed"
    assert not any(c["target"] == "broker.get_order" for c in recorded.transcript["calls"])


def test_submit_failure_script_matches_seed_six_recorded_digest():
    import json
    from pathlib import Path

    from tests.paper.equivalence.runner import digest, run_recorded

    expected = json.loads(
        (Path(__file__).parent / "golden" / "run_equivalence_digests.json").read_text()
    )["digests"]["6"]
    assert digest(run_recorded("new", _submit_failure_with_broker_not_found())) == expected


def test_write_methods_matches_frozen_set():
    from ggTrader.paper.ledger import WRITE_METHODS

    assert WRITE_METHODS == {
        "init_paper_schema",
        "create_order_intent",
        "attach_broker_order",
        "settle_order_intent",
        "settle_legacy_pending_order",
        "log_trade",
        "log_pending_order",
        "clear_pending_order",
        "mark_pending_order_stale",
        "log_snapshot",
        "save_peak_value",
        "save_rebalance_state",
        "log_dividend_accrual",
        "save_split_correction",
        "clear_split_correction",
    }


def test_safety_rejects_order_after_failed_submit():
    from tests.paper.equivalence.runner import safety_violations

    calls = [
        {
            "target": "broker.submit_sell",
            "args": ["AAPL", 1.0, "first"],
            "raised": "RuntimeError: failed",
        },
        {"target": "persist.create_order_intent", "args": ["second"], "raised": None},
    ]
    assert any(v.startswith("c:") for v in safety_violations(build_case(14), {"calls": calls}))


def test_safety_rejects_duplicate_not_shared_by_two_sleeves():
    from tests.paper.equivalence.runner import safety_violations

    case = build_case(3)
    signals = {
        **case.scenario.signals,
        "sleeves": {
            **case.scenario.signals["sleeves"],
            "sleeve1": {**case.scenario.signals["sleeves"]["sleeve1"], "buys": ["NVDA"]},
        },
    }
    case = replace(case, scenario=replace(case.scenario, signals=signals))
    calls = [
        {"target": "broker.submit_buy", "args": ["MSFT", 100.0, str(i)], "raised": None}
        for i in range(2)
    ]
    assert any(v.startswith("e:") for v in safety_violations(case, {"calls": calls}))


@pytest.mark.parametrize("qty", [float("nan"), -1.0])
def test_safety_rejects_nonfinite_or_negative_sell_quantity(qty):
    from tests.paper.equivalence.runner import safety_violations

    calls = [{"target": "broker.submit_sell", "args": ["AAPL", qty, "id"], "raised": None}]
    assert any(v.startswith("g:") for v in safety_violations(build_case(14), {"calls": calls}))


def test_w19_requires_signal_error_alert():
    from tests.paper.equivalence.witnesses import reached_witnesses

    case = replace(build_case(3), membership_error="membership unavailable")
    rec = run_recorded("new", case)
    assert "W19" in reached_witnesses(case, rec.transcript, rec.logs)
    transcript = {
        **rec.transcript,
        "calls": [c for c in rec.transcript["calls"] if c["target"] != "notifier.send"],
    }
    assert "W19" not in reached_witnesses(case, transcript, rec.logs)


def test_w15_requires_funding_sweep_sell():
    from tests.paper.equivalence.witnesses import reached_witnesses

    case = build_case(12)
    rec = run_recorded("new", case)
    assert "W15" in reached_witnesses(case, rec.transcript, rec.logs)
    account = {
        **case.scenario.replies["get_account"][0],
        "cash": 0.0,
        "portfolio_value": 1_000_000.0,
    }
    replies = {**case.scenario.replies, "get_account": [account]}
    reserve_only = replace(case, scenario=replace(case.scenario, replies=replies))
    assert "W15" not in reached_witnesses(reserve_only, rec.transcript, rec.logs)
