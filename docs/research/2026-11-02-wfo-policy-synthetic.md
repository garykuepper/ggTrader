# Synthetic WFO Policy Study: halt, fallback and re-entry rates under stated assumptions

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-11-02
**Status:** PREREGISTRATION committed before any simulation was run. The Results section is filled in only after the run. This is a **synthetic** study of the walk-forward (WFO) circuit-breaker bookkeeping. It does not test the real NDH or DSR gates, uses no market data, and says nothing about any real strategy.

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

*Filled in after the run.*
