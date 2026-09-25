# Core + SPY Sweep Blend: KEEP at tested size, live size untested

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-09-24
**Audience:** ggTrader owner & research collaborators

## 1. Executive Summary & Core Engine Audit

The point-in-time (PIT) re-baseline showed the SP500 core loses to SPY on its
own: Sharpe 0.59 / CAGR 3.1% vs 0.78 / 13.0%
(`docs/research/2026-09-23-pit-rebaseline-core-nogo.md`). That curve holds
idle cash at 0%. The live book doesn't. Idle cash above a 5% reserve is swept
into SPY, so what is deployed is core positions plus SPY. This study measures
that book directly, to answer the owner's open question from `next_steps.md`
item 5: should the sleeve shrink?

The test re-runs the identical pinned PIT core walk-forward optimization (WFO)
(sp500, 2021-01-31 → 2026-04-30, 17 folds) and records each out-of-sample
(OOS) fold's daily cash share. For each day t it then computes:

```
r(k) = k * r_core(t) + max(k * idle(t-1) + (1 - k) - 0.05, 0) * r_spy(t)
```

Here k scales the sleeve: k = 1 is the construction as backtested, and k = 0
is 95% SPY with 5% cash. Scaling is linear in the core's own returns. It
approximates a change in position size; it is not a re-simulation.

**The decision was pre-registered before any run:** KEEP if k = 1 has Sharpe
above SPY with a MaxDD no worse; else HALVE if k = 0.5 passes; else SHRINK.

**Result: KEEP.** It passes narrowly, and the test is weaker than the verdict
makes it sound:

- **The margin is noise-sized.** Sharpe is 0.80 vs 0.78, and CAGR is
  identical (12.96%). About 1 point of the 1.4-point drawdown improvement
  comes from the 5% cash reserve alone (k = 0: −21.05%), not from the sleeve.
  The honest reading is that the sleeve does no harm and diversifies slightly
  (daily correlation with SPY is 0.56). That is not an edge.
- **The tested sleeve is about 4× smaller than the live one.** The backtest is
  92.4% idle on average (median 97%) and was 100% idle for 2023 Q2–Q4, so
  only about 8% of the book is in stocks. The live account on 2026-09-24 held
  31 positions, roughly 32% of the book. The live size has never been
  validated.

## 2. Quantitative Performance Context

Raw results: `docs/research/_core_spy_blend_20260924.json`. Daily curves:
`docs/research/_core_spy_blend_curves_20260924.csv`. Driver:
`scripts/core_spy_blend.py`. Metrics cover the WFO OOS span (2022-01 →
2026-04) at lab default costs, with cash earning 0.

| Construction | Stocks / SPY | Sharpe | CAGR | MaxDD |
|---|---|---|---|---|
| **SPY buy-and-hold** | 0% / 100% | 0.78 | 12.96% | −22.09% |
| Core alone (idle cash at 0%) | ~8% / 0% | 0.59 | 3.07% | −5.28% |
| k = 0 (SPY + 5% reserve) | 0% / 95% | 0.78 | 12.36% | −21.05% |
| k = 0.5 | ~4% / 91% | 0.79 | 12.67% | −20.82% |
| **k = 1 (as backtested)** | ~8% / 87% | **0.80** | 12.96% | −20.72% |
| k = 1.5 | ~12% / 84% | 0.80 | 13.25% | −20.64% |

Inside the core, invested dollars earn about 17.4%/yr at 30.5% annualized
volatility. The core's low standalone CAGR is mostly idle cash. It is not
bad stock selection.

**Pre-registered bar:**

| Rule | Result | Pass? |
|---|---|---|
| k = 1: Sharpe > SPY and MaxDD ≥ SPY's | 0.80 > 0.78; −20.7% ≥ −22.1% | ✅ → KEEP |

## 3. Actionable Research Directions

### Rank 1: Explain the live-vs-backtest exposure gap, then re-test at live size

**Mechanism.** The two paths size entries differently. Both of these were
verified in code:

- Backtest (`lab/simulate.py`, `simulate_signals`): `size_type="percent"`
  with `SIGNAL_POSITION_SIZE` 0.03, which is **3% of remaining cash** per entry.
- Live (`paper/risk.py:13,62`): `position_pct` 0.033 × portfolio value, which
  is **3.3% of the whole portfolio** per entry.

**This does not explain the gap on its own.** At 3% of remaining cash, 31
concurrent positions would still leave about 60% of the book invested
(1 − 0.97³¹ ≈ 61%). An 8% average means the backtest holds only about 3
positions at a time, against 31 live. So the core issue is **entry count**,
not sizing. Live either takes far more entries than the backtest, or holds
them far longer. Candidates to check, all **unvalidated**:

1. Different parameters. Live signals may run pinned parameters that differ
   from the per-fold WFO winners.
2. The WFO's anchor fallback and circuit-breaker folds trade less.
3. Live exits never trigger (the stop is now armed, but the reversion exit
   may not fire).

**Payoff / Effort / Failure.** Payoff: settles whether the book as actually
run is validated at all. Effort: S–M (compare live `paper_trades` entry and
hold counts against backtest fold trades over overlapping dates). Primary
risk: the live size turns out to be what the lab should have tested, and it
fails the drawdown rule (see §6).

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**Core-vs-SPY sleeve sizing at backtested exposure (~8% stocks): KEEP,
2026-09-24.** Sharpe 0.80 vs 0.78, MaxDD −20.7% vs −22.1%. There is no case
for cutting the sleeve at that size, and no case for growing it on this
evidence.

## 5. Operational Roadmap: Recommended First Action

**No live change.** Keep the sleeve as deployed, and do Rank 1 before
anyone argues about sleeve size again. A KEEP verdict at 8% exposure does
not license the live book's roughly 32%.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question: if k = 1.5 scores higher than k = 1, doesn't that argue
for a larger sleeve?**

No. The gain from k = 1 to k = 1.5 is 0.005 Sharpe, far inside noise. An
extrapolation to live-like exposure (k ≈ 4, about 30% stocks and 69% SPY)
gave MaxDD about −24.4%, *worse* than SPY. That would fail the drawdown half
of the rule. Treat that point as **unvalidated**: it is well outside the
tested range, and linear scaling breaks down there (on busy days it implies
more than 100% invested). What it does suggest is that the live size could be
the one that fails, which is why Rank 1 comes before any sizing decision.
