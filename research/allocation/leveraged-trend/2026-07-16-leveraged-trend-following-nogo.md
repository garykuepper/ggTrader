# Long-Only Leveraged-ETF Trend Following: NO-GO vs. Naive Buy-and-Hold

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-16
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

Following the NO-GO on breadth-driven leveraged/inverse rotation
(`docs/research/2026-07-16-leveraged-index-rotation-nogo.md`), this report tests a
deliberately simpler design: `src/ggTrader/lab/strategies/leveraged_trend.py`
holds one universe's leveraged-long ETF (2x or 3x — SSO/UPRO for SP500,
QLD/TQQQ for Nasdaq100, UWM/TNA for Russell2000) when the underlying,
unleveraged index is above its trailing SMA, otherwise sits in cash. No
inverse leg. A bar-level realized-vol-targeting overlay (already built into
the lab's `simulate_signals` path, `vol_cap=1.0` so it can only de-lever, never
add leverage) sits on top to reduce whipsaw/decay exposure during high-vol
regimes — the specific failure mode identified in the closed rotation arc.
Swept over `trend_window` (50–200 days), `leverage_tier` (2x/3x),
`min_hold_days` (1/5/10), and `vol_target` (0.10/0.15/0.20), 72 combos × 26
folds, `eval_start=2019-01-01`, same window as the closed arc.

Structurally, this design is much healthier than the rejected one: **no
persistent regime halt** on SP500, and gate pass rates of 14–18 out of 26
folds (vs. 3–4/26 for the breadth-driven rotation). But it does not beat its
own naive buy-and-hold baseline, let alone SPY, on absolute return — it trades
return for a genuinely lower drawdown, and not efficiently enough for that
trade to show up as a Sharpe improvement in two of three universes. Verdict:
**NO-GO** as designed. The mechanism is sound (trend-timing plus a vol
overlay behaves exactly as intended — see the wiring sanity check in §2's
footnote), but a 200-day SMA filter mostly keeps this strategy out of the
market during exactly the periods (2020 V-recovery, 2023–2024 AI-driven
Nasdaq melt-up) that made leveraged buy-and-hold so profitable over this
window, and being late to re-enter after whipsaw costs it further.

This closes a second leveraged-ETF research thread in as many days. Combined
with the rotation NO-GO, the evidence now points toward leveraged ETFs being
structurally hard to time profitably with either a breadth signal or a plain
trend filter — any further leveraged-ETF work should target a materially
different mechanism (see §6), not another variant of "gate long/cash with an
indicator."

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Regime Halt |
|---|---|---|---|---|---|
| **Trend-filtered — SP500 (SSO/UPRO)** | 0.62 | 4.0% | -10.7% | 17/26 | No |
| **Trend-filtered — Nasdaq100 (QLD/TQQQ)** | 0.53 | 4.7% | -16.5% | 18/26 | Yes |
| **Trend-filtered — Russell2000 (UWM/TNA)** | 0.40 | 2.9% | -16.8% | 14/26 | Yes |
| Buy&Hold UPRO (3x SP500) | 0.79 | 33.1% | -76.8% | N/A | N/A |
| Buy&Hold SSO (2x SP500) | 0.81 | 27.1% | -59.3% | N/A | N/A |
| Buy&Hold TQQQ (3x Nasdaq100) | 0.88 | 44.3% | -81.7% | N/A | N/A |
| Buy&Hold QLD (2x Nasdaq100) | 0.90 | 36.8% | -63.7% | N/A | N/A |
| Buy&Hold TNA (3x Russell2000) | 0.48 | 7.6% | -85.2% | N/A | N/A |
| Buy&Hold UWM (2x Russell2000) | 0.50 | 13.3% | -68.8% | N/A | N/A |
| **SPY benchmark** | 0.80 | 15.3% | -33.7% | N/A | N/A |

Aggregate WFE was 1.05 (SP500), 0.41 (Nasdaq100), and NaN (Russell2000 — a
fold with zero-variance OOS returns broke the ratio). Live-params stability
was weak everywhere: the WFO's "recommended" combo was selected in only 1–3
of 26 folds in every universe, meaning the winning parameter set churns
fold-to-fold rather than converging — a sign the trend/vol-target combination
isn't finding a stable regime-independent edge, just locally-good noise.

**Wiring sanity check (from the implementation-plan verification pass,
`scripts/leveraged_trend_research.py`'s underlying mechanism):** an extreme
`vol_target=0.01` smoke test on SP500 2019–2021 cut realized annualized vol
from 47.0% to 6.1% and max drawdown from -26.7% to -4.3% — the overlay is
correctly wired and behaves exactly as designed. The NO-GO here is a genuine
result of the mechanism, not a bug.

## 3. Actionable Research Directions

None ranked — this is a closure report. See §6 for the direction worth a
decisive look before abandoning long-only leveraged-ETF research entirely.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Long-only leveraged-ETF trend filter + vol-targeting overlay (all 3
universes) — REJECTED.** OOS Sharpe 0.40–0.62 vs. buy-and-hold-the-same-ETF
0.48–0.90 and SPY 0.80; OOS CAGR 2.9–4.7% vs. buy-and-hold 7.6–44.3% and SPY
15.3%. The SMA trend filter avoids most of the leveraged decay/whipsaw (MaxDD
-10.7% to -16.8% vs. buy-and-hold's -59.3% to -85.2%) but gives up far more
upside than it saves in downside over this bull-heavy 2019–2026 window — it
misses the exact rallies (2020 V-recovery, 2023–24 Nasdaq melt-up) that made
naive buy-and-hold leveraged exposure so profitable. Winning parameters are
unstable across folds (selected in 1–3/26). Confirmed the vol-targeting
overlay itself is correctly wired (extreme-value sanity check, see §2
footnote) — the rejection is about the trend-timing mechanism's opportunity
cost, not an implementation bug.

**B. Breadth-driven leveraged/inverse index rotation (all 3 universes) —
REJECTED 2026-07-16** (`docs/research/2026-07-16-leveraged-index-rotation-nogo.md`).
Separate, earlier-closed arc — OOS Sharpe -0.14 to -0.37 vs SPY, MaxDD -80%
to -89%, persistent regime halt. Not superseded by (A); both leveraged-ETF
timing approaches tried so far have failed, for different reasons (breadth
rotation collapsed to structural instability; trend-following survives fine
but is a drag on return).

## 5. Operational Roadmap: Recommended First Action

**Do not pursue long/cash trend-filtered leveraged-ETF timing further as a
return-maximizing strategy.** If low-drawdown leveraged exposure is
independently valuable (e.g. as a risk-budget-friendly satellite sleeve
rather than a return driver), that's a different question than "does this
beat SPY or buy-and-hold" — worth a deliberate ask to the user before
building anything further in that direction, since the report as measured
says no on the return dimension.

With two independent leveraged-ETF timing mechanisms now rejected in two
days, redirect research effort to non-leveraged-ETF levers (crypto pivot,
new equity signal families) rather than a third variant of "gate a
leveraged ETF with an indicator."

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** both rejected designs used *monthly-or-slower*
rebalancing logic dressed in different clothing (breadth confirmed over
month-end rebalances; here, `min_hold_days` of 1/5/10 is still coarse
relative to how fast leveraged ETFs can move) — is a genuinely faster or more
volatility-adaptive re-entry (not just de-levering, but re-levering quickly
after a confirmed reversal) the missing piece, rather than the trend concept
itself?

**Resolution:** Weak case, not worth building speculatively. The dominant
cost here wasn't slow re-entry after a whipsaw — it was being flat during
long, low-volatility uptrends where the SMA filter had no reason to exit in
the first place (2023–2024 Nasdaq specifically). A faster re-entry doesn't
address that; the strategy already held Nasdaq100 exposure most of that
period per the trend rule and still gave up 40 points of CAGR to the raw
ETF, so the loss is systemic to "timing at all," not a lag artifact. Closing
this line of inquiry.

### Parked Direction: Leverage as a risk-budget sleeve, not an alpha source
Reframe the question from "can timing beat buy-and-hold leveraged exposure"
to "does a small, permanently-sized leveraged sleeve improve a diversified
portfolio's risk-adjusted return, sized via `combine_sleeves`'
inverse-vol/target-vol blending (`src/ggTrader/lab/allocation.py`) alongside
the live SP500 core." Not executable as a re-run of this script — needs the
core equity strategy's own return stream as an input, and a specific target
allocation, not a new gating strategy. Gate: only pick this up if there's a
specific portfolio-construction ask; it's a different research question from
everything tried so far.
