# Month-End Treasury Duration Sleeve (A10): NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-09-24
**Audience:** ggTrader owner & research collaborators

## 1. Executive Summary & Core Engine Audit

Candidate A10 (Hartley & Schwarz, *"Predictable End-of-Month Treasury Returns"*,
SSRN 3440417) holds a Treasury ETF only over the last 3 trading days of each
month and holds cash otherwise. It was tested exactly as the brief
(`docs/research/briefs/2026-09-24-month-end-treasury-sleeve.md`) fixed it
before any run. There were three tests: the frozen paper rule on IEF, a
12-combo walk-forward optimization (WFO), and a deployable overlay that moves
20% of a SPY holding into IEF for those 3 days.

**Verdict: NO-GO.** It fails two of the five pre-registered criteria (3 and 4)
and therefore criterion 5 too.

- **The effect looks real as a standalone sleeve.** In each of three disjoint
  windows, IEF earned more per day on the rule's days than on all other days
  (2011–18: 6.9 vs 0.5 bp, Welch t 2.89; 2019–20: 9.6 vs 2.0 bp, t 1.99;
  2021–26: 7.9 vs −1.9 bp, t 2.75). On the pinned window the frozen rule
  returned Sharpe 0.96 at 1 bp per side and 0.78 at 3 bp, and every calendar
  year from 2021 to 2025 was positive.
- **It can't be deployed as specified.** The only construction the live
  account can actually run is funding the sleeve out of SPY, because idle cash
  is already swept into SPY. That construction gains nothing: Sharpe 0.909 vs
  0.896 at 1 bp, with a *worse* MaxDD (−24.9% vs −24.5%). The reason is that
  SPY earned **8.35 bp/day on those same month-end days**, more than IEF's
  7.86. The swap gives up one month-end premium (the equity turn-of-month
  effect) to collect another.
- **The WFO did not pick a stable parameter set.** Gates passed in 13/17
  folds, but the circuit breaker halted the run (folds 8–12, after back-to-back
  negative out-of-sample (OOS) Sharpes in 2023 H2). The most common winner won
  only 5/17 folds, against a bar of 8. OOS Sharpe was 0.61, CAGR 3.1% and
  MaxDD −7.4% at 5 bp, against SPY at 0.79 on the same span.

Already closed and not re-trodden here: slope/regime Treasury timing (A5),
pre-FOMC Treasury drift (A7), and all equity cross-sectional sleeves (see
`RESEARCH_SNAPSHOT.md` §2).

## 2. Quantitative Performance Context

Raw results: `docs/research/_month_end_treasury_results.json`. Driver:
`scripts/month_end_treasury_wfo.py`. Pinned window 2021-01-31 → 2026-04-30.
`SIGNAL_POSITION_SIZE` 1.0 (the sleeve is fully invested on its days), no
leverage, and cash earns 0.

| Configuration | Sharpe | CAGR | MaxDD | Gates | Notes |
|---|---|---|---|---|---|
| **Frozen rule, IEF last 3 days, 1 bp/side** | 0.96 | 2.6% | −2.2% | n/a | 14.3% exposure, 63 trades, 61.9% hit rate |
| Frozen rule, 3 bp/side | 0.78 | 2.1% | −2.3% | n/a | |
| Frozen rule, 5 bp/side (lab default) | 0.60 | 1.6% | −2.4% | n/a | |
| **WFO, 12 combos, 5 bp** (OOS 2022-01 → 2026-04) | 0.61 | 3.1% | −7.4% | 13/17, **halted** | top winner 5/17 |
| **80% SPY + 20% rotating to IEF, 1 bp** | 0.909 | 14.6% | −24.9% | n/a | vs 100% SPY below |
| 80/20 overlay, 3 bp | 0.898 | 14.4% | −25.0% | n/a | |
| **SPY buy-and-hold** (pinned window, full span) | 0.896 | 14.7% | −24.5% | n/a | |
| SPY buy-and-hold (WFO OOS span, lab's bar) | 0.79 | 13.2% | −22.1% | n/a | |
| IEF / TLT / EDV buy-and-hold (pinned) | −0.17 / −0.41 / −0.47 | −1.6% / −7.5% / −11.6% | −21% / −44% / −55% | n/a | |
| SP500 core (reference, 2026-09-23) | 0.59 | 3.1% | −5.3% | 17/17 | |

The frozen rule in other windows, at 1 bp:

| Window | Sharpe | CAGR | MaxDD | IEF B&H Sharpe |
|---|---|---|---|---|
| 2011-01 → 2018-12 (paper's in-sample, reference only) | 1.07 | 2.2% | −2.0% | 0.61 |
| 2019-01 → 2021-01 (post-publication) | 1.82 | 3.3% | −1.1% | 1.26 |
| 2021-02 → 2026-04 (pinned) | 0.96 | 2.6% | −2.2% | −0.17 |

**Pre-registered pass bar:**

| # | Criterion | Result | Pass? |
|---|---|---|---|
| 1 | Pinned Sharpe ≥ 0.5, positive return, ≥ 3/5 positive years | 0.96, +14.3%, 5/5 (0.8, 4.2, 1.1, 3.1, 3.3%) | ✅ |
| 2 | Positive over 2019-01 → 2021-01 | +6.9% | ✅ |
| 3 | WFO ≥ 12/17 gates, no halt, same winner ≥ 8/17 | 13/17 gates, **halted**, **5/17** | ❌ |
| 4 | Overlay Sharpe > SPY and MaxDD no worse | 0.909 > 0.896, but MaxDD −24.9% < −24.5% | ❌ |
| 5 | Criteria 1 and 4 hold at 3 bp | 1 holds (0.78); 4 fails | ❌ |

**Standalone 12-combo sweep on the pinned window at 5 bp** (context for the
WFO, not walk-forward): Sharpe ranges from 0.36 to 0.60. The best combo is IEF
at 3 days, the paper's rule. IEF beats TLT and EDV at every hold length except
2 days, where TLT edges it (0.40 vs 0.39). That is consistent with the paper's
finding that the best risk/reward sits at shorter maturities.

## 3. Actionable Research Directions

None ranked. The effect is real enough to note, but the only way this account
could use it has been tested and fails. Neither available lever is
executable without breaking pre-registration:

- **Fund the sleeve from T-bills instead of SPY.** This would add roughly the
  standalone sleeve's return on top of the cash yield. But the live account
  has no idle cash (`CASH_SWEEP_ENABLED`, SPY/5%). Holding a T-bill sleeve
  would be a new allocation decision, not a research result. **Unvalidated.**
- **Pick a different window or instrument after the fact** (e.g. the
  "moved one day earlier" TLT variant from Allocate Smartly). That is exactly
  the post-hoc tuning the frozen design exists to prevent. The WFO has already
  shown that no single combo wins consistently.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. A10 month-end Treasury sleeve (frozen rule + 12-combo WFO + SPY
overlay) — REJECTED, 2026-09-24.**
- *Standalone:* the premium held in every window tested (t 2.0–2.9) and the
  sleeve's Sharpe survives 3 bp of cost.
- *As a SPY overlay:* no benefit. SPY's own return on the same 3 days
  (8.35 bp/day on 2021–26) matches or beats IEF's (7.86).
- *WFO:* 13/17 gates, but the circuit breaker halted it and the top winner
  held only 5/17 folds. OOS Sharpe 0.61 vs SPY 0.79.

**Harness note, not a strategy finding.** The WFO anchor fallback
(`compute_anchor_set`) takes the least-drawdown combo among those with CAGR
above 4%. For a sleeve invested about 14–24% of the time, only the
long-duration combos clear 4%. So the fallback was TLT in every fold (3 days
in folds 1–4, then **5 days, MaxDD −11.2%**), a much higher-drawdown choice
than any IEF combo. Eight folds (5–7 and 9–13) deployed the TLT 5-day anchor,
and this is most of why WFO MaxDD
(−7.4%) is three times the frozen rule's (−2.4%). The fallback is miscalibrated
for low-exposure sleeves. It does not change this verdict: criterion 3 fails
on the halt and on winner stability either way.

## 5. Operational Roadmap: Recommended First Action

**Close A10 and move it into the `RESEARCH_SNAPSHOT.md` roster on the next
`research-snapshot` run. Do not re-run it with tweaked windows or
instruments.**

Rationale: every test the account could deploy has been run against a bar
fixed in advance. What's left would be post-hoc tuning, and the WFO's 5/17
winner stability says there is no stable choice to tune toward. A12
(month-end SPY → IEF rebalancing tilt) is mechanically almost the same as
the overlay tested here, so treat it as covered by this result (a lower
prior) unless it brings a different construction. No live change was made
and none is proposed.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question: the premium is statistically visible in three disjoint
windows, one of them wholly after publication. Isn't calling it NO-GO throwing
away the one real effect this lab has found?**

Partly, yes. This is the first candidate in a long while whose raw effect
survives out of sample and survives costs. But the pass bar asked whether the
account can *use* it, and it can't. The account is already long the equity
turn-of-month effect through its SPY sweep, and that effect falls on the same
days and is at least as large. A duration sleeve pays only when it is funded
from cash earning the T-bill rate. The lab can't model that (cash earns 0),
and the account doesn't hold it. That makes it a portfolio-construction
question, not a signal question. The owner is already weighing one
(core-versus-SPY sizing, `next_steps.md` item 5).

### Parked Direction: T-bill-funded duration sleeve
Hold a T-bill ETF (e.g. SGOV/BIL) as the base and switch to IEF for the last
3 trading days. **What would open it:** an owner decision to carry a
cash-equivalent allocation instead of sweeping everything into SPY, plus a
lab change so the cash leg earns the T-bill rate. Until both exist, the
~2.6%/yr standalone figure is an upper bound measured against 0% cash. It is
not a deployable return, and on invested days it forgoes the T-bill carry,
about 0.14 × 4–5% ≈ 0.6%/yr in 2023–25 (**unvalidated**).
