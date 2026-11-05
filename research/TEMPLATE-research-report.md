# <Report Title>: <One-line scope>

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** <YYYY-MM-DD>
**Audience:** <e.g. Principal Engineering Team & Quantitative Research Collaborators>

<!--
TEMPLATE NOTES (delete before publishing):
- This is the standard ggTrader research-report format. Keep the section order.
- Every quantitative claim must be either (a) measured and cited, or (b) explicitly
  labeled "unvalidated / expected". Never present a hoped-for number as a result.
- Anything that contradicts a completed research arc goes in §4, not §3.
- Verify data/infra assertions against the actual DB/repo before stating them.
- Each §3 direction must be executable under the CURRENT operational constraints
  (account size, available data). Non-executable ideas are parked in §6, not ranked.
- Save published reports as docs/research/<YYYY-MM-DD>-<slug>.md
-->

## 1. Executive Summary & Core Engine Audit

<2–4 paragraphs: what the system is, the current production configuration, the
headline OOS metrics with the validation method, and the one-sentence thesis of
this report. Include the pipeline ASCII diagram if architecture is relevant.>

<Close with a paragraph stating what is already exhausted/closed, so the reader
knows the report is not re-treading falsified ground.>

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **<Validated config>** | x.xx | xx.x% | -xx.x% | nn/nn folds | <sizing> |
| **<Benchmark>** | x.xx | ~xx% | -xx% | N/A | Buy-and-hold |

## 3. Actionable Research Directions

Ranked by `Expected Payoff × OOS Survival Probability / Implementation Effort`,
restricted to levers executable under current constraints (<state them: account
size, data coverage>).

```
Expected Payoff x OOS Survival Prob
----------------------------------  ===>  Strategic Research Priority Rank
      Implementation Effort
```

### Rank 1: <Name>

**Mechanism.** <Precise, reproducible spec. Define every term. If a metric is
applied to an object it doesn't natively fit, state the exact mapping. Include
formulas in fenced blocks.>

**Why it differs from rejected work.** <Cite the specific prior result — with its
real number — and explain why this is not the same lever. Reference §4 arcs.>

**WFO framework.** <Exact sweep ranges, selection metric, and which new trials the
DSR gate must account for. Note any prerequisite checks that gate the work.>

**Payoff / Effort / Failure.** <Payoff: magnitude + "unvalidated". Effort: S/M/L +
the file(s) touched. Primary risk: the single most likely failure mode, named
mechanistically.>

### Rank 2: <Name>
<same sub-structure>

### Rank 3: <Name>
<same sub-structure>

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

<One entry per falsified/closed lever. Each: REJECTED/CORRECTED tag + the
post-mortem evidence with real numbers. This section protects development time by
making it impossible to re-pitch dead ideas.>

**A. <Lever> — REJECTED.** <evidence + numbers>

**B. <Lever> — REJECTED.** <evidence + numbers>

## 5. Operational Roadmap: Recommended First Action

<The single highest-priority concrete action, stated imperatively and in bold.
Then the rationale, then the discipline rule (branches, gates, no live deploy
until WFO+NDH+DSR+significance over live baseline).>

## 6. Contrarian Evaluation & Parked Research

<Pose the strongest contrarian question against continuing this line of research.
Argue it honestly. Then resolve it — usually "run one decisive final experiment,
then close or pivot".>

### Parked Direction: <Name>
<Ideas that are sound but NOT executable now. State the explicit gate that opens
them (capital threshold, data backfill, infra). Include any verified blockers.>
