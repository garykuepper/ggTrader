"""Simulate preregistered synthetic WFO policy paths."""

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ggTrader.lab.wfo import (
    WfoPolicy,
    WfoState,
    check_circuit_breaker,
    check_shadow_reentry,
    compute_wfe,
)

N_FOLDS = 17
P_NAN = 0.05
IS_BIAS = 0.5
OOS_NOISE_SD = 2.0
ANCHOR_MULTIPLIER = 0.5


def simulate_run(
    rng: np.random.Generator,
    mu: float,
    p_gate: float,
    policy: WfoPolicy,
    n_folds: int = N_FOLDS,
    p_nan: float = P_NAN,
) -> Dict[str, float]:
    """Draw a path and apply the production WFO state transition per fold."""
    gates = rng.random(n_folds) < p_gate
    is_noise = rng.standard_normal(n_folds)
    oos_noise = rng.standard_normal(n_folds)
    nan_flags = rng.random(n_folds) < p_nan
    state = WfoState()
    anchors = halted_folds = episodes = reentries = 0
    for gate, ins, noise, missing in zip(gates, is_noise, oos_noise, nan_flags):
        entering = state.halted
        anchor = not gate or entering
        anchors += int(anchor)
        halted_folds += int(entering)
        oos = (
            float("nan")
            if missing
            else mu * (ANCHOR_MULTIPLIER if anchor else 1) + OOS_NOISE_SD * noise
        )
        wfe = compute_wfe(mu + IS_BIAS + ins, oos)
        state.wfe_history.append(
            None if anchor and policy.anchor_fold_wfe == "exclude_from_breaker" else wfe
        )
        if np.isfinite(oos):
            state.oos_sharpes.append(oos)
        if not entering:
            state = check_circuit_breaker(state, nan_policy=policy.nan_wfe)
        else:
            state = check_shadow_reentry(state, bool(gate), bool(gate), wfe)
        episodes += int(not entering and state.halted)
        reentries += int(entering and not state.halted)
    return {
        "anchor_share": anchors / n_folds,
        "halted_fold_share": halted_folds / n_folds,
        "ever_halted": float(episodes > 0),
        "n_halt_episodes": float(episodes),
        "mean_halt_length": halted_folds / episodes if episodes else 0.0,
        "reentries": float(reentries),
        "never_reentered": float(state.halted),
    }


def run_grid(
    seed: int,
    n_reps: int,
    mus: Sequence[float],
    p_gates: Sequence[float],
    policies: Mapping[str, WfoPolicy],
) -> Dict[str, Any]:
    """Average paths; restore RNG state so each policy receives identical draws."""
    import copy

    rng = np.random.default_rng(seed)
    result = {}
    for mu in mus:
        for p_gate in p_gates:
            totals = {name: {} for name in policies}
            for _ in range(n_reps):
                start = copy.deepcopy(rng.bit_generator.state)
                for name, policy in policies.items():
                    rng.bit_generator.state = copy.deepcopy(start)
                    values = simulate_run(rng, mu, p_gate, policy)
                    for key, value in values.items():
                        totals[name][key] = totals[name].get(key, 0.0) + value
            for name in policies:
                result[f"{mu}|{p_gate}|{name}"] = {
                    key: value / n_reps for key, value in totals[name].items()
                }
    return result
