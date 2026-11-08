"""Verify synthetic WFO policy simulations."""

import json

import numpy as np
import pytest
from wfo_policy_synthetic import run_grid, simulate_run

from ggTrader.lab.wfo.breaker import (
    WfoPolicy,
    WfoState,
    check_circuit_breaker,
    check_shadow_reentry,
    compute_wfe,
)

pytestmark = pytest.mark.study


class Draws:
    """Return fixed per-fold draws."""

    def __init__(self, gate, ins, oos, nan):
        self.values = iter([np.array(gate), np.array(ins), np.array(oos), np.array(nan)])

    def random(self, size):
        return next(self.values)

    def standard_normal(self, size):
        return next(self.values)


def test_simulate_run_is_deterministic_for_seed():
    assert simulate_run(np.random.default_rng(1), 1, 0.7, WfoPolicy()) == simulate_run(
        np.random.default_rng(1), 1, 0.7, WfoPolicy()
    )


def test_zero_p_gate_means_anchor_share_one():
    assert simulate_run(np.random.default_rng(2), 1, 0, WfoPolicy())["anchor_share"] == 1


def test_policy_legacy_matches_direct_breaker_calls(monkeypatch):
    import wfo_policy_synthetic as synthetic

    seen = []
    original = synthetic.compute_wfe
    # Record the state entering each fold through its anchor decision.
    breaker = synthetic.check_circuit_breaker
    shadow = synthetic.check_shadow_reentry
    states = []

    def record_breaker(state, **kwargs):
        states.append(False)
        return breaker(state, **kwargs)

    def record_shadow(state, *args):
        states.append(True)
        return shadow(state, *args)

    monkeypatch.setattr(synthetic, "check_circuit_breaker", record_breaker)
    monkeypatch.setattr(synthetic, "check_shadow_reentry", record_shadow)
    monkeypatch.setattr(synthetic, "compute_wfe", original)
    gates = [True] * 5
    noises = [-1, -1, 1, 1, 1]
    simulate_run(Draws([0] * 5, [0] * 5, noises, [1] * 5), 1, 1, WfoPolicy(), 5)
    state = WfoState()
    for gate, noise in zip(gates, noises):
        seen.append(state.halted)
        oos = (0.5 if state.halted else 1) + 2 * noise
        wfe = compute_wfe(1.5, oos)
        state.wfe_history.append(wfe)
        state.oos_sharpes.append(oos)
        state = (
            check_shadow_reentry(state, gate, gate, wfe)
            if state.halted
            else check_circuit_breaker(state)
        )
    assert states == seen


def test_common_random_numbers_same_draws_across_policies():
    a = simulate_run(np.random.default_rng(7), 1, 0.7, WfoPolicy(), p_nan=0)
    b = simulate_run(np.random.default_rng(7), 1, 0.7, WfoPolicy(nan_wfe="exclude"), p_nan=0)
    assert a == b


def test_run_grid_output_schema():
    policies = {
        "legacy": WfoPolicy(),
        "nan_exclude": WfoPolicy(nan_wfe="exclude"),
        "anchor": WfoPolicy(anchor_fold_wfe="exclude_from_breaker"),
    }
    result = run_grid(1, 3, [0.0, 0.5, 1.0, 1.5], [0.3, 0.7, 1.0], policies)
    assert len(result) == 36
    assert all(0 <= cell["ever_halted"] <= 1 for cell in result.values())
    json.dumps(result, allow_nan=False)


def test_nan_exclude_halts_where_legacy_does_not():
    def run(policy):
        return simulate_run(
            Draws([0] * 5, [0] * 5, [0, 0, 0, 0, 0], [1, 1, 0, 1, 1]), 0, 1, policy, 5, 0.05
        )

    assert run(WfoPolicy())["ever_halted"] == 0
    assert run(WfoPolicy(nan_wfe="exclude"))["ever_halted"] == 1


def test_run_grid_pools_halt_length_over_episodes_not_runs(monkeypatch) -> None:
    """Mean halt length is halted folds / episodes pooled across runs, not a mean of run means."""
    import wfo_policy_synthetic as synthetic

    from ggTrader.lab.wfo.breaker import WfoPolicy

    zero = {
        "anchor_share": 0.0,
        "halted_fold_share": 0.0,
        "ever_halted": 0.0,
        "n_halt_episodes": 0.0,
        "mean_halt_length": 0.0,
        "reentries": 0.0,
        "never_reentered": 0.0,
    }
    halted = {
        **zero,
        "halted_fold_share": 6 / 17,  # 6 halted folds over 17
        "ever_halted": 1.0,
        "n_halt_episodes": 2.0,
        "mean_halt_length": 3.0,
        "never_reentered": 1.0,  # one halt still active at the end
    }
    runs = iter([halted, zero])
    monkeypatch.setattr(synthetic, "simulate_run", lambda *a, **k: next(runs))
    cell = synthetic.run_grid(1, 2, [1.0], [1.0], {"legacy": WfoPolicy()})["1.0|1.0|legacy"]
    # pooled: 6 halted folds / 2 episodes = 3.0 (a mean of run means would give 1.5)
    assert cell["mean_halt_length"] == 3.0
    # one run of two ends halted; one active halt among two episodes pooled -> 0.5
    assert cell["fraction_runs_ending_halted"] == 0.5
    assert cell["fraction_halts_active_at_end"] == 0.5


def test_fraction_halts_active_at_end_is_per_episode() -> None:
    """Two halt episodes per run on average and half the runs ending halted give 0.25."""
    import wfo_policy_synthetic as synthetic

    cell = {"n_halt_episodes": 2.0, "never_reentered": 0.5, "halted_fold_share": 0.0}
    out = synthetic.finalize_cell(cell, n_folds=17)
    assert out["fraction_halts_active_at_end"] == 0.25
    assert out["fraction_runs_ending_halted"] == 0.5
