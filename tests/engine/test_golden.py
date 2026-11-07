"""L2: seeded golden panels run on BOTH engines; numba equals Rust within tolerance.

Equity arrays are compared at rtol 1e-9 against the stored golden, trade counts, trade
indices and fold winners exactly. Regenerate only with
``pytest tests/engine --update-golden --golden-reason "<why>"`` (the reason must also be in
docs/changelog.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ggTrader.engine import config

from . import scenarios
from .golden_io import GOLDEN_DIR, check_or_update

REPO = Path(__file__).resolve().parents[2]
RTOL = 1e-9


@pytest.fixture(scope="module")
def golden(request: pytest.FixtureRequest) -> dict:
    """The stored golden; under --update-golden it is rewritten from a Rust run first."""
    before = config.active_engine()
    config.set_engine("rust")
    try:
        data = scenarios.run_all() if request.config.getoption("--update-golden") else {}
    finally:
        config.set_engine(before)
    return check_or_update(request, "engine_scenarios", data)


def _assert_sim(actual: dict, expected: dict) -> None:
    assert actual["n_trades"] == expected["n_trades"]
    assert actual["entry_idx"] == expected["entry_idx"]
    assert actual["exit_idx"] == expected["exit_idx"]
    assert actual["value"].keys() == expected["value"].keys()
    for group, curve in expected["value"].items():
        np.testing.assert_allclose(actual["value"][group], curve, rtol=RTOL)


@pytest.mark.parametrize(
    "name", ["signals_trailing", "signals_masked_time_stop", "weights_monthly"]
)
def test_simulation_goldens_hold_on_both_engines(engine: str, golden: dict, name: str) -> None:
    actual = getattr(scenarios, name)()
    _assert_sim(actual, golden[name])


def test_mini_wfo_golden_holds_on_both_engines(engine: str, golden: dict) -> None:
    actual = scenarios.mini_wfo()["folds"]
    expected = golden["mini_wfo"]["folds"]
    assert [f["winner"] for f in actual] == [f["winner"] for f in expected]
    assert [f["test_trades"] for f in actual] == [f["test_trades"] for f in expected]
    for a, e in zip(actual, expected, strict=True):
        for key in ("train_sharpe", "test_sharpe", "test_max_drawdown"):
            assert a[key] == pytest.approx(e[key], rel=RTOL, abs=1e-12), key


def test_numba_and_rust_agree_on_every_scenario() -> None:
    """The simulators are exact; indicators agree to 1e-10 (spec L3b)."""
    config.set_engine("rust")
    rust = scenarios.run_all()
    config.set_engine("numba")
    numba = scenarios.run_all()
    for name in ("signals_trailing", "signals_masked_time_stop", "weights_monthly"):
        assert rust[name]["n_trades"] == numba[name]["n_trades"]
        assert rust[name]["entry_idx"] == numba[name]["entry_idx"]
        for group, curve in rust[name]["value"].items():
            np.testing.assert_array_equal(curve, numba[name]["value"][group])
    for fr, fn in zip(rust["mini_wfo"]["folds"], numba["mini_wfo"]["folds"], strict=True):
        assert fr["winner"] == fn["winner"] and fr["test_trades"] == fn["test_trades"]
        assert fr["test_sharpe"] == pytest.approx(fn["test_sharpe"], rel=1e-9, abs=1e-12)


def test_golden_scenarios_exercise_something() -> None:
    """A golden of an empty run would prove nothing."""
    data = json.loads((GOLDEN_DIR / "engine_scenarios.json").read_text())["data"]
    for name in ("signals_trailing", "signals_masked_time_stop", "weights_monthly"):
        assert sum(data[name]["n_trades"].values()) > 50, name
    assert len(data["mini_wfo"]["folds"]) == 2
    assert all(f["n_finite_train_sharpes"] >= 3 for f in data["mini_wfo"]["folds"])


def test_goldens_are_changelogged() -> None:
    """Every golden's regeneration reason must appear in docs/changelog.md."""
    changelog = (REPO / "docs" / "changelog.md").read_text()
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        reason = json.loads(path.read_text())["provenance"]["reason"]
        assert reason, f"{path.name} has no reason"
        assert reason in changelog, f"golden reason for {path.name} not in docs/changelog.md"
