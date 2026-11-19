"""Contract checks for the seeded paper-trader scenario generator."""

import os
import subprocess
import sys

from tests.paper.equivalence.generator import build_case


def test_generator_is_deterministic_across_hash_seeds():
    script = """
import hashlib
from dataclasses import asdict
from tests.paper.equivalence.generator import build_case
from tests.paper.transcript.harness import to_json

def clean(value):
    if isinstance(value, BaseException):
        return {"exception": type(value).__name__, "message": str(value)}
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value

cases = [clean(asdict(build_case(seed))) for seed in range(600)]
print(hashlib.sha256(to_json(cases).encode()).hexdigest())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
        )
        for seed in ("1", "2")
    ]
    assert outputs[0] == outputs[1]


def test_generated_sizing_and_buy_candidates_are_valid():
    for seed in range(600):
        case = build_case(seed)
        scenario = case.scenario
        assert scenario.signals["scale"] > 0
        assert all(weight > 0 for weight in scenario.signals["weights"].values())
        assert scenario.replies["get_account"][0]["portfolio_value"] > 0
        for sleeve in scenario.signals["sleeves"].values():
            assert len(sleeve["buys"]) == len(set(sleeve["buys"]))
        buys = [
            symbol for sleeve in scenario.signals["sleeves"].values() for symbol in sleeve["buys"]
        ]
        if not scenario.name.startswith("duplicate_buy_"):
            assert len(buys) == len(set(buys))


def test_generated_cases_are_valid_harness_inputs():
    from tests.paper.transcript.harness import run_scenario

    for seed in range(600):
        run_scenario(build_case(seed).scenario)


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
    for seed in range(600):
        scenario = build_case(seed).scenario
        kind = scenario.name.rsplit("_", 1)[0]
        if kind not in reached:
            continue
        transcript = run_scenario(scenario)
        orders = [
            call for call in transcript["calls"] if call["target"].startswith("broker.submit_")
        ]
        assert orders, (seed, transcript["raised"])
        assert transcript["raised"].startswith("KeyError:"), (seed, transcript["raised"])
        reached[kind] += 1
    assert all(count > 5 for count in reached.values()), reached


def test_deliberate_duplicate_buy_reaches_both_sleeves():
    from tests.paper.transcript.harness import run_scenario

    reached = 0
    for seed in range(600):
        scenario = build_case(seed).scenario
        if not scenario.name.startswith("duplicate_buy_"):
            continue
        transcript = run_scenario(scenario)
        buys = [call for call in transcript["calls"] if call["target"] == "broker.submit_buy"]
        assert len(buys) == 2, (seed, transcript["raised"])
        assert buys[0]["args"][0] == buys[1]["args"][0] == "MSFT"
        reached += 1
    assert reached > 5
