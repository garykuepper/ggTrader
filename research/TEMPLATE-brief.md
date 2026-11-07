<!--
TEMPLATE NOTES (delete before registering the brief):
- scripts/study.py new copies this file into a new study folder and fills the placeholders.
- The skeleton follows the October 2026 briefs (century leverage, static leverage, cross-asset
  sleeve): a fixed question, frozen data, a fixed model, the accounting, the output plan and the
  decision rule are all written down BEFORE anything is computed.
- Section 0 is read by the checker. Keep its seven "key: value" lines, one per line. The values
  filled in are the project defaults (the pinned 17-fold window, production leverage 1.0, SPY as
  the benchmark): change one only on purpose, and say why in section 4.
- Anything computed after registration that changes a number here needs a new dated brief or a
  dated amendment, never a silent edit.
-->

# Brief: $title

**Status:** DRAFT. Not registered: nothing below is fixed until the owner approves this brief.
**Study:** $category/$slug (kind: $kind), written $date.
**Decision fed:** TODO: name the one decision this study informs, and who makes it.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these
values or say which line it departs from.

- eval_start: $eval_start
- eval_end: $eval_end
- max_leverage: 1.0
- benchmark: SPY buy-and-hold, same window and same costs
- costs: 5 bps per side fees plus 5 bps per side slippage (confirm against the lab config)
- family_size: 1
- seed: none (state a fixed seed here if any step is random)

## 1. Question and fixed arms

TODO: one sentence for the question. Then list every arm (variant) that will be run. The number
of arms is the family size above; every arm counts toward the multiple-testing correction, run
or not.

## 2. Data and freeze before calculation

TODO: the exact data (tape file and its hash, or the free source and download date), the
universe rule (point in time, no survivorship), and the date the inputs are frozen. Inputs that
cannot be frozen need a stated reason.

## 3. Registered model

TODO: the signal or allocation rule with every parameter fixed. Define each term. Say which
lookahead guard applies (signals use data through the prior close only).

## 4. Portfolio and accounting

TODO: sizing, leverage cap, rebalance rule, costs and, for a taxable account, the tax treatment.
Explain any departure from the defaults in section 0.

## 5. Frozen output plan

TODO: the tables and charts the report will contain, the results JSON it will write, and the
metrics reported for the study and for SPY on the same window (CAGR, Sharpe, maximum drawdown,
gate pass rate).

## 6. Pre-registered decision rule

TODO: the pass bar, written as numbers, that turns the result into GO or NO-GO. A NO-GO is a
successful outcome of a well-run study. State what a failed bar means and what it does not.

## 7. Calls for the owner to check and execution boundary

TODO: the judgment calls the owner should confirm before the run. State what this task may
touch: new study code only, no engine, no paper-trading code, no database writes.
