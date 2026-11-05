# Synthetic WFO Policy Study: halt, fallback and re-entry rates under stated assumptions

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-11-02
**Status:** Preregistered (commit `edc4d79`, before any run); results in section 5, with one aggregation correction recorded in 5.6. This is a **synthetic** study of the walk-forward (WFO) circuit-breaker bookkeeping. It does not test the real NDH or DSR gates, uses no market data, and says nothing about any real strategy.

Terms: **WFO** = walk-forward optimization. **WFE** = walk-forward efficiency (OOS Sharpe ÷ IS Sharpe). **IS / OOS** = in-sample / out-of-sample. **Anchor** = defensive fallback parameters. **Halt** = the circuit breaker has tripped and the anchor is deployed until re-entry.

## 1. Question

How often does the WFO decision process (gate pass, anchor fallback, circuit-breaker halt, shadow re-entry) deploy the anchor, halt, and re-enter, as a function of a synthetic strategy's true annual Sharpe `mu` and of the probability that a fold's gates pass? In particular, does the claim "a halt is close to certain for a positive-Sharpe strategy" hold under this model? (That claim came from a secondhand review and was not accepted as established.)

## 2. Model (assumptions, not measurements)

- 17 folds per run (the historical fold count). Each fold draws: gates pass with probability `p_gate`; training winner IS Sharpe ~ Normal(`mu` + 0.5, 1.0) (0.5 is an assumed selection-bias term); deployed OOS Sharpe ~ Normal(`mu`, 2.0) if the winner is deployed and ~ Normal(0.5·`mu`, 2.0) if the anchor is deployed (2.0 approximates the standard error of an annualized Sharpe over a 3-month window); OOS Sharpe is NaN with probability 0.05 (to exercise the NaN path).
- The anchor is deployed when gates fail or the system is halted entering the fold (as in `run_wfo`).
- WFE, circuit breaker and shadow re-entry use the real functions `compute_wfe`, `check_circuit_breaker`, `check_shadow_reentry` and `WfoState` from `ggTrader.lab.wfo`, with the shadow re-entry gates fed by the same Bernoulli gate draw.
- Not modelled: the real NDH/DSR statistics, parameter grids, transaction costs, regime structure, autocorrelation across folds.

## 3. Grid and sampling

`mu ∈ {0.0, 0.5, 1.0, 1.5}`, `p_gate ∈ {0.3, 0.7, 1.0}`, policies `{legacy, nan_wfe=exclude, anchor_fold_wfe=exclude_from_breaker}`, 5000 replications per cell, seed 20261102, common random numbers across policies within a cell.

## 4. Preregistered interpretation rules

1. **Halt likelihood claim.** Let `P_halt` be the fraction of runs with at least one halt at `mu = 1.0, p_gate = 1.0`, legacy policy. If `P_halt > 0.50` the statement "a halt is likely for a positive-Sharpe strategy" is *supported under this model*; otherwise it is *not supported under this model*. Report `P_halt` for all `mu` and `p_gate` regardless.
2. **Policy effect.** A policy is reported as *materially different* from legacy in a cell if its `P_halt` or its mean anchor share differs by more than 0.05 (absolute). Report all cells; do not select cells after seeing results.
3. **Re-entry.** Report the mean halt-episode length (folds) and the fraction of halts that never re-enter within the 17 folds, per cell.
4. Every statement is written "under these synthetic assumptions" and implies nothing about a real strategy, the real gates, or any past verdict.
5. No parameter in §2 or §3 may change after the first full run. If a defect is found, fix it, record the change, and re-run the entire grid.

## 5. Results

Run: seed 20261102, 5000 replications per cell, 36 cells, wall-clock about 29 seconds. Raw results: [`_wfo_policy_synthetic_results.json`](_wfo_policy_synthetic_results.json); code `src/ggTrader/lab/wfo_policy_synthetic.py`, `scripts/wfo_policy_synthetic.py`, tests `tests/lab/test_wfo_policy_synthetic.py`. No model or grid parameter changed after the preregistration commit (`edc4d79`). A 200-replication independent replay with a different seed gave 0.870 for the rule-1 cell, inside the Monte Carlo band.

### 5.1 Probability of at least one halt (P_halt), by policy

| mu | policy | p_gate 0.3 | p_gate 0.7 | p_gate 1.0 |
|---:|---|---:|---:|---:|
| 0.0 | legacy | 0.984 | 0.984 | 0.984 |
| 0.0 | nan_exclude | 0.986 | 0.986 | 0.986 |
| 0.0 | anchor_exclude_from_breaker | 0.966 | 0.979 | 0.984 |
| 0.5 | legacy | 0.970 | 0.961 | 0.951 |
| 0.5 | nan_exclude | 0.974 | 0.966 | 0.957 |
| 0.5 | anchor_exclude_from_breaker | 0.918 | 0.941 | 0.951 |
| 1.0 | legacy | 0.947 | 0.915 | 0.875 |
| 1.0 | nan_exclude | 0.956 | 0.925 | 0.888 |
| 1.0 | anchor_exclude_from_breaker | 0.859 | 0.857 | 0.875 |
| 1.5 | legacy | 0.913 | 0.818 | 0.729 |
| 1.5 | nan_exclude | 0.927 | 0.838 | 0.750 |
| 1.5 | anchor_exclude_from_breaker | 0.742 | 0.723 | 0.729 |

### 5.2 Legacy policy: fallback, halt length and re-entry

| mu | p_gate 0.3 | p_gate 0.7 | p_gate 1.0 |
|---:|---:|---:|---:|
| **mean anchor share** | | | |
| 0.0 | 0.897 | 0.738 | 0.597 |
| 0.5 | 0.885 | 0.677 | 0.475 |
| 1.0 | 0.868 | 0.605 | 0.341 |
| 1.5 | 0.850 | 0.542 | 0.240 |
| **mean halt length (folds per episode, pooled over episodes)** | | | |
| 0.0 | 10.934 | 9.245 | 7.877 |
| 0.5 | 10.244 | 7.653 | 6.065 |
| 1.0 | 9.412 | 6.472 | 4.857 |
| 1.5 | 8.623 | 5.939 | 4.424 |
| **fraction of halt episodes still active at the end of the run** | | | |
| 0.0 | 0.934 | 0.765 | 0.631 |
| 0.5 | 0.894 | 0.646 | 0.496 |
| 1.0 | 0.844 | 0.560 | 0.416 |
| 1.5 | 0.828 | 0.529 | 0.362 |
| **fraction of runs ending halted** | | | |
| 0.0 | 0.958 | 0.881 | 0.812 |
| 0.5 | 0.920 | 0.772 | 0.661 |
| 1.0 | 0.860 | 0.644 | 0.496 |
| 1.5 | 0.816 | 0.524 | 0.334 |

### 5.3 Preregistered rules applied

1. **Halt likelihood.** `P_halt` at `mu = 1.0, p_gate = 1.0`, legacy, is **0.875 (> 0.50)**. Under this model the statement "a halt is likely for a positive-Sharpe strategy" is **supported**. P_halt stays at 0.73 even at `mu = 1.5` and is 0.98 at `mu = 0`.
2. **Policy effect (differences above 0.05 in P_halt or mean anchor share).** `nan_wfe="exclude"` never crosses the threshold in any cell (largest P_halt difference 0.021, largest anchor-share difference 0.012). `anchor_fold_wfe="exclude_from_breaker"` crosses it in 5 of the 12 (mu, p_gate) cells, all at `p_gate < 1` and `mu >= 0.5`: P_halt lower by 0.052 (mu 0.5, p_gate 0.3), 0.088 (1.0, 0.3), 0.058 (1.0, 0.7), 0.171 (1.5, 0.3), 0.095 (1.5, 0.7). No anchor-share difference exceeds 0.05. At `p_gate = 1.0` the two policies are indistinguishable in P_halt.
3. **Re-entry.** Halts are long and often unresolved within a 17-fold run. Under legacy at `mu = 1.0`, `p_gate = 1.0` the pooled mean halt episode is 4.86 folds, 41.6% of halt episodes are still active at the end of the run and 49.6% of runs end halted; at `p_gate = 0.3` the figures are 9.41 folds, 84.4% and 86.0%. Halts that start late in a run have too few folds left to re-enter, so episode lengths understate how long a halt would last.

### 5.4 Exact calculation for one trigger (assumption-dependent)

With winner-deployed OOS Sharpe ~ Normal(`mu`, 2.0) and `p_gate = 1.0`, the two-consecutive-negative-OOS-Sharpe trigger alone fires at least once in 17 independent folds with probability (an exact two-state recurrence, not an independence approximation): 0.968 at `mu = 0`, 0.888 at 0.5, **0.728 at 1.0** and 0.512 at 1.5. Simulated legacy P_halt in the same cells is 0.984, 0.951, **0.875** and 0.729. The roughly 0.15 gap at `mu = 1.0` must come from other triggers or from state effects (chronic WFE decay, and the anchor being deployed after a halt), so this one trigger is a large but not a sole contributor. The simulation does not record the halt reason, so the cause split is unmeasured. (An earlier draft used an independence-of-pairs approximation giving 0.80; it overstated agreement with the simulation and was replaced.)

### 5.5 Limits

- The 2.0 standard deviation approximates the sampling error of an annualized Sharpe ratio over a three-month fold. If the real per-fold dispersion is much smaller, the halt rates fall.
- Gates are Bernoulli draws; the real NDH and DSR gates, parameter grids, costs, regimes and serial correlation are not modelled. Nothing here is evidence about a real strategy, a past run or the real gates.
- The legacy rates are a property of the existing circuit-breaker bookkeeping under the stated noise, not a measurement of historical behavior. Saved real runs (see [fold attribution](2026-11-02-wfo-fold-attribution.md)) show halts in some runs and none in others.
- Adopting an alternative policy remains an owner decision; this study does not recommend one.

### 5.6 Correction record (preregistered rule 5)

A final review found that the first run's cell-level `mean_halt_length` averaged per-run means (zero-episode runs included) and that the fraction labelled "halts never re-entered" was the share of runs ending halted. Both were aggregation and labelling defects, not simulation defects: `ever_halted`, anchor share, halted-fold share and episode counts are unchanged to machine precision after the fix. The aggregation was changed to pool episodes (`finalize_cell`), the entire grid was re-run with the same seed and parameters, and the tables above are from the re-run. First-run value at `mu = 1.0`, `p_gate = 1.0` for comparison: mean halt length 4.652 (now 4.857); the run-level share ending halted, 0.496, is unchanged and is now labelled correctly.
