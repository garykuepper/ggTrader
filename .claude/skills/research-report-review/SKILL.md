---
name: research-report-review
description: Give a skeptical "second read" to a ggTrader research result before it is trusted, committed or acted on — a finished WFO report, a pasted summary from another Claude session, or a GO/NO-GO verdict. Checks quoted numbers against the raw results JSON, the verdict against the pre-registered pass bar, holdout contamination, post-hoc choices, single-leg carry, lookahead and point-in-time universe handling, leverage/sizing/cash traps, benchmark choice, significance, scope creep into live code, and real-money deployability on Alpaca. Use when the user pastes back a research session's output, asks "review this report", "is this GO real", "second opinion", "sanity check these results", "should we trust this", or before committing research artifacts — even if they just paste the summary with no question.
---

# Research report review

A research session grades its own homework. Most of what went wrong in this
project was caught only by a second read: an unrecoverable 1.12 Sharpe from an
unpinned window, a 38% CAGR from one garbage bar, a "holdout" that an earlier
sanity check had already seen, leverage defaulting to 2.0, survivorship from
today's index membership. This skill is that second read. Be the skeptical
senior quant from CLAUDE.md: confirm what holds, correct what doesn't, and
say plainly when a GO is thinner than it sounds. Agreeing is fine when the
work is sound — the point is that the agreement is checked.

## Gather

- The report (`docs/research/<date>-<slug>[-nogo].md`) and, if the user
  pasted a summary, the summary itself — they can disagree.
- The brief it ran from (`docs/research/briefs/`), whose **pre-registered
  design and pass bar** are the standard.
- The raw results JSON (`docs/research/_<slug>_results.json`) and the driver
  (`scripts/<slug>_wfo.py`).
- The code diff: `git status -s` and `git diff --stat` (work may be
  uncommitted), plus the new strategy file.

## Check

Work through these; skip any that can't apply and say why.

1. **Numbers match the artifact.** Every headline figure against the JSON:
   ```bash
   .venv/bin/python .claude/skills/research-report-review/scripts/flatten_results.py \
       docs/research/_<slug>_results.json sharpe max_dd cagr
   ```
2. **Verdict follows the pre-registered bar as written.** Compare criterion
   by criterion. Watch for a bar quietly reinterpreted, a criterion dropped,
   or "passes" that relies on an analysis added after the results. Post-hoc
   extras (e.g. a bootstrap CI) are welcome as *context*, labeled as such.
3. **Holdout contamination.** For any window called unseen, search earlier
   artifacts for runs that already covered it:
   `docs/research/artifacts-*/sanity_checks_output.txt`, earlier briefs and
   reports. If a sanity check saw it, it isn't a holdout.
4. **Post-hoc choices.** Instruments, legs, parameters or windows changed
   after seeing results? A leg that looks like a drag in-sample (e.g. TLT in
   the 2022 rate shock) is not evidence to drop it — that is fitting to the
   window. The pre-registered spec stays until a new test is registered.
5. **Concentration.** Does one leg, one year or one bar carry the result?
   Look at leave-one-out rows, yearly returns, and extreme single days (the
   SIVB $0.0013 → $0.11 bar made a 38% CAGR). Name what carries it.
6. **Lookahead / point-in-time.**
   - In the strategy, `select(asof, …)` must only use `data.loc[:asof]`, and
     `to_targets` must apply each plan on the bar *after* `asof`.
   - Index universes need `universe_fn` / the PIT entry mask (a missing mask
     prints a warning in `run_wfo`); `universe_members_asof(…, now)` is
     survivorship-biased.
   - Event dates must be known in advance, not the actual announcement date.
7. **Lab traps.**
   - `max_leverage` should be 1.0 (default 2.0 overstated results twice).
   - `SIGNAL_POSITION_SIZE` defaults to 0.03, which makes a signals sleeve
     ~3% invested. Check it was set explicitly.
   - Lab cash earns 0%. A sleeve that sits in cash is understated unless it
     used a T-bill ETF (BIL).
   - `--eval-start` and `--eval-end` must be pinned, and SPY's own Sharpe
     should match the reference for that window.
8. **Benchmarks.**
   - SPY is always a benchmark.
   - Timing or rotation rules also need the same instruments held **statically**
     (the 2026-09-28 sleeve's gain turned out to be diversification, not timing)
     and each instrument's own buy-and-hold.
   - Compare against the **live alternative** too (the deployed construction's
     Sharpe in RESEARCH_SNAPSHOT §1).
   - For any sleeve funded from SPY, compare against **SPY de-risked to the same
     equity share with the rest in T-bills** (e.g. 80% SPY / 20% BIL). A lower
     drawdown or a Sharpe bump can come purely from holding less stock. On
     2026-09-28 the 80/20 T-bill book cut drawdown as much as the cross-asset
     sleeve did. Also check each leg alone as the whole sleeve: 80/20 SPY/gold
     beat the three-leg book, so the result was mostly gold.
9. **Costs, risk-free basis and significance.**
   - The verdict must hold at 3 bp per side.
   - The lab's Sharpe treats cash as earning 0%. That flatters any book that
     is less volatile than SPY or sits partly in T-bills. Recompute the key
     comparisons on an **excess-of-T-bill basis** from the daily curves or
     JSON (BIL, or the 3-month rate in `fred_series` `TB3MS`), and rerun any
     leave-one-leg-out criterion on that basis. On 2026-09-28 the cross-asset
     sleeve passed "any leg removed" at 0% cash, but failed it for GLD and
     PDBC on T-bill excess. Report both; the pre-registered basis decides the
     formal verdict, and the excess basis decides how much to trust it.
   - A Sharpe gap of ~0.05 over ~5 years is inside noise. If the report has a
     bootstrap or CI, read it. If not, say the margin is unquantified.
10. **Scope.**
    - The diff must not touch `src/ggTrader/paper/`, `.env` or deployed config
      unless explicitly asked.
    - Registry, CLI and universe-snapshot edits are fine.
    - Tests: `scripts/run_tests.sh <new tests>`; lint: `ruff check`.
    - `ohlcv` writes (backfills) must be naive UTC.
11. **Deployability, if GO.**
    - The instruments must be Alpaca-tradable and fractionable, without K-1s
      (see the research-prompts skill's tradability note).
    - Taxes in a taxable account: gold is taxed as a collectible, and
      rebalancing realizes gains.
    - Any live-trader capability the rule needs but doesn't exist yet.

## Report

```
## Verdict
<agree / agree with corrections / disagree> — one paragraph, including how
strong the evidence really is.

## Confirmed
<what was checked and holds, briefly — with the numbers>

## Corrections
1. <issue> — <evidence> — <what to change>

## Next
<the concrete next step(s): e.g. pre-register a true holdout, a paper
shadow, commit (list files, excluding unrelated edits like AGENTS.md)>
```

Don't edit the report yourself unless asked; list the corrections so the user
(or the research session) applies them. If the user wants it committed, stage
only the research files, never unrelated working-tree edits.
