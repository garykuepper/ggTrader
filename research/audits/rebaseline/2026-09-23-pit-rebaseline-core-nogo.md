# Point-in-Time Re-baseline: the SP500 core does not beat SPY

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-09-23
**Audience:** ggTrader research

## 1. Executive Summary & Core Engine Audit

Commit `c9d1abe` added daily point-in-time (PIT) S&P 500 membership to
every signal-strategy sweep. Before it, the WFO generated entries for every
stock loaded over 2021–2026, whether or not it was an index member on that
day; 14.1% of core entries fell on non-member days
(`2026-09-23-ic-kelly-rebaseline-invalid.md`). This study re-ran the core,
`ensemble_ic` and `ensemble_kelly` with the fix in place. Everything else
matched the prior studies: 17 folds, 2021-01-31 → 2026-04-30, full grid,
gated WFO, leverage 1.0. Driver: `scripts/ic_kelly_rebaseline_wfo.py`. Raw
results: `_pit_rebaseline_20260923.json`.

**Verdict: the SP500 core loses to SPY on both Sharpe and CAGR once
membership is PIT.**

| | Sharpe | CAGR | Max drawdown |
|---|---|---|---|
| SP500 core, PIT | 0.59 | 3.1% | -5.3% |
| SPY | 0.78 | 13.0% | -22.1% |

The previously cited 0.99 / 8.0% came largely from trading stocks that
were not in the index at the time.

`ensemble_ic` and `ensemble_kelly` are now validly measured and **closed
NO-GO**. Both trail the core and SPY.

The core does still have one real property: low drawdown, with all 17
folds passing the gates. That makes it a low-risk sleeve, not a way to
beat the index.

## 2. Quantitative Performance Context

Same run, same data, same window.

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gates | Stability (top winner) |
|---|---|---|---|---|---|
| **SP500 core, PIT** | **0.59** | **3.1%** | **-5.3%** | 17/17 | — |
| SP500 core, pre-fix (2026-08-22) | 0.99 | 8.0% | -7.7% | 12/17 | — |
| `ensemble_ic`, PIT | 0.41 | 3.3% | -12.8% | 11/17 (anchor 15/17) | 4/17 |
| `ensemble_kelly`, PIT | 0.50 | 6.3% | -13.9% | 16/17 | 10/17 |
| **SPY** | **0.78** | **13.0%** | **-22.1%** | N/A | Buy-and-hold |

**Validity checks on the fix, so the drop isn't an artifact:**
- **Coverage:** with the driver's data start (2019-03-03), 483–501 stocks
  are eligible on every checked date against about 503 members. The gap
  is renames (ABC→COR, ANTM→ELV, FBHS→FBIN) and recent spin-offs under the
  400-bar history minimum (GEHC, KVUE, SOLV). That's 1–4% of the index, a
  coverage gap, not a directional bias.
- **Folds 6–8 (2023-04 → 2024-01) have NaN OOS Sharpe.** The WFO picked
  `min_agree=4` combos there, and they barely trade. In fold 6 the winner
  makes 2 entries (BIO, GIS), **identical with and without the mask**, so
  the mask is not what blanked those folds. The stitched OOS curve counts
  those nine months as flat cash, which is what the strategy would
  actually have done.
- **Kelly's SIVB spike is gone.** CAGR fell from 38.2% to 6.3%, and
  volatility is back in a normal range.

Pre-registered criteria (2026-09-10 audit §5.3): Sharpe > 1.05 and above
the control, MaxDD ≥ -10%, top winner ≥ 8/17 folds. IC fails all four.
Kelly fails every criterion except stability.

## 3. Actionable Research Directions

### Rank 1: Re-price the live decision against SPY, not the core

The deployed paper strategy is the core plus an idle-cash SPY sweep.
Paper NAV went from $102,459 on 2026-06-24 to $104,037 on 2026-09-22
(+1.5%, split-corrected). The honest backtest says the stock-picking sleeve
adds risk-adjusted value only as a low-drawdown diversifier. It does not
add return: it compounds at 3.1% against SPY's 13.0%. It's paper money,
so there's no capital at risk and keeping it running as a forward test is
cheap. **No real capital should follow this strategy.** Whether to shrink
the sleeve in favour of more SPY is a portfolio decision for the owner.
The evidence supports doing so.

### Rank 2: Re-screen closed signal NO-GOs against SPY under PIT

Every signal-strategy verdict before `c9d1abe` was computed without PIT
membership, on both sides of the comparison. Verdicts that failed by a
wide margin against SPY stay closed. Near-misses, and anything rejected
*only* for trailing the phantom core, deserve a PIT re-run against the one
benchmark that never moved: SPY.

### Rank 3: Cross-asset sleeve (TLT/GLD/DBC)

This was already next in the queue. The PIT result strengthens the case
for looking outside US large-cap stock selection.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

- **`ensemble_ic`**: closed NO-GO (PIT Sharpe 0.41, winner unstable at
  4/17, anchor used in 15/17 folds).
- **`ensemble_kelly`**: closed NO-GO (PIT Sharpe 0.50, MaxDD -13.9%).
- **Constrained IC with voter weight floors**: closed. The unconstrained
  version trails the core by 0.18 Sharpe with an unstable winner, so a
  floor has nothing to stabilize toward.

## 5. Operational Roadmap: Recommended First Action

Update the headline baseline everywhere to **core 0.59 / 3.1% (PIT)**:
`next_steps.md`, `roadmap.md` and `RESEARCH_SNAPSHOT.md`, the last via the
research-snapshot skill. Then the owner decides Rank 1. No code change to
live is implied by this report on its own.

## 6. Contrarian Evaluation & Parked Research

- **The mask costs 1–4% coverage through renames.** Those are mostly
  large, healthy names (COR, ELV), so fixing the rename mapping would add
  back ordinary members, not reversion winners. It is very unlikely to
  close a 0.19-Sharpe gap to SPY. It's worth doing for hygiene; it doesn't
  change the verdict.
- **The 17/17 gate pass with a *lower* Sharpe is worth noticing.** It
  suggests the pre-fix gate failures were partly caused by the leaked
  universe's noisier names. The gates themselves are fine.
