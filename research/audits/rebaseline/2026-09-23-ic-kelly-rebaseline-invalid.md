# ensemble_ic / ensemble_kelly Re-baseline: result invalid; WFO universe has no point-in-time membership

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-09-23
**Audience:** ggTrader research

> **Superseded 2026-09-23:** the PIT fix landed (`c9d1abe`) and the re-run
> is in `2026-09-23-pit-rebaseline-core-nogo.md` (core 0.59 < SPY 0.78).

## 1. Executive Summary & Core Engine Audit

`ensemble_ic` (Sharpe 1.01) and `ensemble_kelly` (0.98) were originally
rejected against the phantom 1.12 core baseline. The real baseline on the
corrected tape is 0.99, so both rejections lacked valid grounds. This
study re-ran both, together with `ensemble` as a same-night control. All
three used the identical pinned setup: sp500, 17 rolling 12/3-month folds,
2021-01-31 → 2026-04-30, full grid, gated WFO, and `run_core` from
`scripts/anchor_fix_reproduction_wfo.py`. Driver:
`scripts/ic_kelly_rebaseline_wfo.py`. Raw results:
`_ic_kelly_rebaseline_20260923.json`.

**Both variants fail the pre-registered criteria, but the measurement
itself is invalid, so neither is closed.** Kelly's 38.2% CAGR at Sharpe
0.60 comes from a single OOS day (2024-03-12, +170.8% equity). On that day
Kelly held **SIVB**, Silicon Valley Bank, which failed in March 2023,
through a garbage post-delisting bar: $0.0013 → $0.11, an 84× move. IC
covers the same folds and shows the same signature (34.0% CAGR at Sharpe
0.61). It is presumed to be the same artifact, but this was **not measured
per-day**.

**The root cause also reaches the core baseline.** The signal-strategy WFO
path in `lab/wfo.py` builds both in-sample and OOS equity from
`_sweep_fold_dispatch` → `sweep_signal_group` → `strategy.sweep_signals`.
That path runs over *every* symbol in the loaded window, which is the
2021–2026 union of S&P 500 members. It applies no point-in-time membership
mask; only `to_targets` applies `eligibility_mask`, and the WFO never calls
it. Measured on the core's own signal generator (`min_agree=3`), **381 of
2,711 entry signals (14.1%) fire on days the symbol was not an S&P 500
member**:
- before the stock joined the index (LITE, CVNA, TTD, WDAY): lookahead;
- after it left (SBNY, XRX, LEG): delisted or removed names.

Ticker renames (XYZ/SQ, PSKY/PARA) inflate that count somewhat, so 14.1% is
an upper bound. The direction and size of the bias on the 0.99 baseline
are **unmeasured**.

Live trading is **not** affected. `paper/` trades current constituents
only.

## 2. Quantitative Performance Context

Same run, same data, same window. SPY window-matched Sharpe 0.78, CAGR
13.0%, MaxDD -22.1%.

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Stability (top winner) |
|---|---|---|---|---|---|
| **ensemble (control)** | 0.99 | 8.0% | -7.7% | 12/17 | — |
| **ensemble_ic** | 0.61 | 34.0%* | -14.7% | 12/17 | 3/17 |
| **ensemble_kelly** | 0.60 | 38.2%* | -15.8% | 16/17 | 8/17 (`kelly_multiplier0.25`) |
| **SPY** | 0.78 | 13.0% | -22.1% | N/A | Buy-and-hold |

\* Artifact. Kelly: one SIVB bar is +170.8% on 2024-03-12. Kelly's
annualized OOS volatility is 84.5%, against 8.1% for the control. IC is
presumed the same.

The control reproduced the 2026-08-22 re-baseline exactly (0.99 / 8.0% /
-7.7%, 12/17 gates), so the tape has not moved since then.

Pre-registered criteria (2026-09-10 audit §5.3), fixed before any result
was seen:
- Sharpe > 1.05 and above the control;
- MaxDD ≥ -10%;
- the most common fold winner wins ≥ 8 of 17 folds.

| Criterion | IC | Kelly |
|---|---|---|
| Sharpe > 1.05 | ✗ | ✗ |
| Sharpe > control | ✗ | ✗ |
| MaxDD ≥ -10% | ✗ | ✗ |
| Stability ≥ 8/17 | ✗ | ✓ |

## 3. Actionable Research Directions

### Rank 1: Point-in-time membership mask in the WFO signal path

Apply the same membership mask the strategies already build in
`to_targets` (`eligibility_mask`) to the entries `sweep_signals` returns
inside `sweep_signal_group`. The mask must be derived from the
constituents history for each date. The fix belongs in `lab/sweep.py` or
`lab/wfo.py`, where every signal strategy's sweep passes through, not in
each strategy. Add a regression test: a symbol absent from the index on
day *d* must produce no entry on day *d*.

Then re-run, in order:
1. The core, which becomes the new baseline. Expect it to move in an
   unknown direction.
2. `ensemble_ic` and `ensemble_kelly` against that baseline, using the
   same driver: `--strategies ensemble ensemble_ic ensemble_kelly` with a
   new `--out`.

Every signal-strategy WFO number since the lab was built carries this
bias. That includes the closed NO-GOs in `RESEARCH_SNAPSHOT.md`; the
weight strategies use `_run_one_weight_combo`, which does call
`universe_fn`.

### Rank 2: Price-bar sanity filter

SIVB is one of 103 bars in the eval window with a daily move beyond ±40%;
most are SIVB's post-failure quotes. A PIT mask removes SIVB after its
index exit. An explicit data-quality guard is still worth adding: flag any
single-day return beyond about ±80% on non-event days. That catches the
next case the mask doesn't.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

Nothing closes on this study; its numbers are invalid. The prior
drawdown and instability findings for both variants still stand as
priors: IC -17.3% MaxDD with the winner in 3/17 folds, Kelly -17.0% MaxDD.
Today's run is consistent with them.

## 5. Operational Roadmap: Recommended First Action

Implement Rank 1, re-baseline the core, and then decide whether IC and
Kelly are worth re-running. Also re-run any closed signal NO-GO whose
margin was thin. None of this touches live trading.

## 6. Contrarian Evaluation & Parked Research

The PIT bias is not guaranteed to be positive. Reversion entries on
freshly-deleted names are often losers, and those would *depress* the
baseline; lookahead on future additions would *inflate* it. Until it is
measured, do not claim the core's edge over SPY is overstated. Claim only
that it is unverified on this point.

### Parked Direction: constrained `ensemble_ic` (voter weight floors)

This is the audit's suggested variant. It stays parked until a clean
unconstrained IC number exists. Building it against a contaminated harness
would repeat this study's problem.
