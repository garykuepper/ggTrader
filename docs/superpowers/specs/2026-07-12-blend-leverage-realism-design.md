# Leverage-Realistic 3-Sleeve Blend Verdict — Design

## Context

The roadmap currently carries two inconsistent numbers for the SP500+MidCap400+Nasdaq100
blend:

- **June 27** ("gate-honest 3-way harness"): Sharpe **1.05** < SP500-core baseline **1.12**.
  Produced by `multi_sleeve_research.py`/`portfolio_blend.py`, both retired June 29
  (commit `78d3f7f`) when `src/ggTrader/lab/blend.py` replaced them.
- **July 8** ("idealized blend", used as the baseline for the idio_vol 4-sleeve test):
  Sharpe **1.03**, MaxDD −10.59%. Produced by the current `blend.py` /
  `ggt lab --blend`, with `--max-leverage 2.0` (the CLI default).

These aren't a valid "idealized vs. honest" comparison — they're two different code
paths. Neither answers the question that actually decides whether this research arc
continues: **can a blend deployable on the live account (unlevered, flat 3% sizing) beat
the SP500 core's 1.12 Sharpe?** The 2.0x leverage default in the current tool means part
of the 1.03 number may depend on exposure the live paper trader cannot take.

Per-sleeve gating is not in question here — each sleeve already runs through the full
gated WFO (NDH + DSR gates, circuit breaker, anchor fallback) inside `run_wfo()` before
its OOS curve reaches the blend. The gap is entirely in the **overlay**
(`combine_sleeves` / `blend_curves`), which applies inverse-vol weighting and a
target-vol leverage scale with no gate of its own.

## Goal

Produce one clean, leverage-realistic Sharpe/MaxDD number for the 3-sleeve blend,
directly comparable to the SP500-core 1.12 baseline, and use it to make a single
written verdict that supersedes both the June-27 and July-8 entries.

## Non-goals

- No sweep of `target_vol` / `blend-window` / `max_leverage`. Every parameter-search
  lever tried on this system so far (IC-weighted voting, Kelly sizing, exit-rule sweep,
  overnight-gap, idio_vol) has failed with the same tell: a winning setting selected in
  only 3-4/17 folds (noise). No reason to expect an overlay-parameter sweep behaves
  differently, and it isn't needed to answer the go/no-go question.
- No new strategy or overlay code. `run_blend()` and its CLI flags
  (`--max-leverage`, `--target-vol`, `--blend-window`, `--eval-start`, `--eval-end`)
  already cover everything this needs.
- No investigation of *why* MidCap's standalone gate-pass rate is weak (6/17). Relevant
  only if the leverage-realistic blend turns out competitive — deferred to a follow-up.

## Protocol

1. **Leverage-realistic run.** `ggt lab --blend
   "ensemble@sp500,ensemble@midcap400,ensemble@nasdaq100" --max-leverage 1.0`, same
   `--eval-start`/`--eval-end` as the SP500-core 1.12 baseline (confirm the exact window
   used for that baseline before running — do not assume `2021-01-31` without checking).
2. **Reference run.** Re-run (or reuse, if the window matches) the current-default
   `--max-leverage 2.0` case to keep a direct within-tool comparison alongside (1).
3. **Leverage attribution.** Pull the persisted `diag`/summary for run (2) — `avg_leverage`,
   `max_leverage`, and the `scale` series's distribution (mean, max, fraction of days
   `scale > 1.0`) — via the `ggtrader` or `postgres` MCP. This quantifies how much of the
   2.0x-cap number depends on leverage the live account doesn't have.
4. **Verdict.** Compare run (1)'s Sharpe/MaxDD against the SP500-core 1.12/−11% baseline,
   same eval window. Write a single dated roadmap entry that:
   - States the leverage-realistic number plainly, superseding the June-27 (1.05,
     different tool) and July-8 (1.03, 2x leverage) entries with a note on why they
     weren't comparable.
   - **If run (1) < 1.12 (Sharpe) or doesn't meaningfully improve MaxDD:** close the
     diversification arc as NO-GO — for real this time, one number, one tool, leverage
     the live account can actually use. Redirect research effort to the crypto-carry
     sleeve (gated on >$10k capital + funding-data backfill).
   - **If run (1) is competitive (≥ ~1.10 Sharpe, or a materially shallower drawdown that
     survives the no-leverage constraint):** flag MidCap's weak standalone gate-pass rate
     as the next research question, scoped separately.

## Verification

- Both blend runs persist to `lab_runs`/`lab_periods` via the existing `run_blend()`
  persistence path — no new verification machinery needed. Confirm each run's `table`
  output includes SPY as a benchmark row for sanity (already built into `run_blend`).
- Cross-check: run (1)'s per-sleeve rows (SP500 alone, MidCap alone, Nasdaq alone) should
  match already-published per-sleeve WFO numbers for the same window, since sleeve gating
  is unchanged — only the overlay's leverage cap changes between runs (1) and (2).

## Deliverable

One roadmap.md update (§3, superseding the June-27 and July-8 blend entries) with the
leverage-realistic verdict and a clear go/no-go call. No code changes expected unless the
leverage attribution step surfaces a bug in `combine_sleeves`/`blend_curves` (out of scope
to fix here if found — would be logged as a separate follow-up).
