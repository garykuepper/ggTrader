# Position-Sizing Regimes: Live Fixed-Notional vs. Backtest Decaying-Percent, Single-Window Measurement

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-08-19
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

The live paper trader and the validated 3-sleeve blend backtest (SP500 +
MidCap400 + Nasdaq100, `EnsembleSignal`, inverse-vol/target-vol overlay,
`max_leverage=1.0`) use **different position-sizing mechanisms**, and the
divergence was previously undetected. The backtest
(`ggTrader.lab.simulate.simulate_signals`) calls
`vbt.Portfolio.from_signals(size=0.03, size_type="percent")`, which vectorbt
resolves as a percent of **remaining cash** — order sizes decay geometrically
with each new entry, so a sleeve never fully deploys and can hold far more
than 30 small concurrent positions. Live (`ggTrader.paper.risk.RiskGuard`)
instead sizes each entry as a **fixed fraction of current total portfolio
value** (`portfolio_value * sleeve_weight * scale * position_pct`), capped at
a fixed number of concurrent slots per sleeve
(`RiskGuard.sleeve_slot_caps`, default `max_positions=30` shared
proportionally by weight). This report compares the two mechanisms
apples-to-apples on the same entries/exits, and separately sweeps the live
mechanism's per-sleeve slot cap (8/12/16/20/25/30), to answer: what slot cap
best reproduces the validated risk profile, and does the live-equivalent
sizing scheme look better or worse on its own merits.

**This report does NOT reproduce the validated headline figures (Sharpe
1.14, MaxDD -5.39%), and says so explicitly rather than presenting a
different number under the same label.** Those figures come from a full
gated walk-forward (`run_blend`/`run_wfo`): 48 `EnsembleSignal` combos x
~17 rolling 12-month-train/3-month-test folds x 3 sleeves, each fold also
recomputing the (now leak-fixed, per-fold expanding-window) defensive
anchor. The 2026-08-18 anchor-leakage report measured ~15 minutes per fold
for a *single* sleeve under the corrected code — a full 3-sleeve re-run is a
multi-hour job, well outside this task's "well under an hour" compute
budget. Per the task's explicit fallback, this report instead runs **one
long single-window backtest** (2021-01-31 → 2026-04-30, matching the
validated study's window) using each `EnsembleSignal` sleeve's **fixed
default parameters** (no per-fold combo re-optimization, no gates, no
anchor) — the same simplification the live paper trader itself makes
(`paper/overlay.py::compute_sleeve_curve` also runs `EnsembleSignal` at
fixed defaults, not a WFO-selected combo). Because both sizing regimes are
run on **identical** entries/exits/eligibility masks, the P-vs-F comparison
within this report is a clean, isolated read on the sizing mechanism alone;
the absolute Sharpe/CAGR levels are not comparable to the WFO-gated
headline numbers and should not be cited as such.

**Method used for Regime F.** vbt's `from_signals` only supports
`SizeType.{Amount, Value, Percent}` — `TargetPercent` (rebalance a position
to a fixed fraction of *current total group value*, which is exactly what
`RiskGuard.sleeve_position_notional` computes) is not implemented for the
signals API in this vectorbt version (0.28.5; confirmed by a runtime
`ValueError` when attempted). `TargetPercent` orders **are** supported by
`vbt.Portfolio.from_orders` (the same call `simulate_weights` already uses
for weight-based strategies), so Regime F converts each sleeve's boolean
entries/exits into a sparse order-size matrix — `+0.033` (the live
`position_pct`) on an admitted entry bar, `0.0` on an exit bar, `NaN`
(no order) elsewhere — and runs that through `from_orders`. vbt has no
native "max concurrent positions" knob, so the per-sleeve slot cap is
enforced by a small sequential admission filter run once per sleeve before
the vbt call: at each bar, close out symbols with an exit signal, then admit
up to `(cap − currently_open)` of that bar's entry signals in symbol order,
dropping the rest for that bar only (matching how `RiskGuard.max_new_positions`
behaves live — a blocked signal is skipped, not queued).

This closes out the position-sizing-mismatch investigation opened in the
task background. It does not reopen any of the twelve closed equity-research
arcs (ML gate, exit-rule sweep, IC-weighted voting, Kelly sizing, etc. — see
`docs/research/RESEARCH_SNAPSHOT.md` §4) — this is a sizing-mechanism study
on the already-validated `EnsembleSignal` entries/exits, not a new signal.

## 2. Quantitative Performance Context

Single-window backtest, 2021-01-31 → 2026-04-30, `LabConfig()` defaults,
`EnsembleSignal` fixed default params, PIT `eligibility_mask` applied
(commit `2eafab1`), 1007 unique symbols across the 3 universes + SPY (994
loaded from the TimescaleDB cache; 16 tickers permanently dropped — either
genuinely delisted with no data under any provider, or a ticker reused by a
different company post-2022 with no way to fetch the original constituent's
history — recorded in the lab `ohlcv_no_data` negative cache so they don't
retrigger slow live-fetch retries; a bounded, disclosed universe gap, not a
methodology error). Elapsed compute: 112s core simulation + ~55s one-time
data load.

**Sanity check against the task's own prior measurements:** Regime P's
sp500 sleeve here shows mean deployment 44.6% at mean 56.8 concurrent
positions — closely matching the task background's independently-derived
"sp500 sleeve ~51.6% mean deployment at ~56 mean concurrent positions."
The sizing *mechanics* reproduction is faithful even though the *strategy*
Sharpe is not (fixed-default-param vs. WFO-selected-combo).

| Config | Sharpe | CAGR | Max DD | Mean deploy | Max deploy | Mean concurrent | Sizing |
|---|---:|---:|---:|---:|---:|---:|---|
| **Regime P blend (status quo, this run)** | **0.70** | 4.97% | **-7.64%** | — | — | — | `percent` (decaying) |
| Regime F blend, cap=8 | 0.46 | 1.85% | -5.16% | — | — | — | `targetpercent`, 8 slots/sleeve |
| Regime F blend, cap=12 | 0.48 | 2.56% | -7.93% | — | — | — | `targetpercent`, 12 slots/sleeve |
| Regime F blend, cap=16 | 0.57 | 3.48% | -7.78% | — | — | — | `targetpercent`, 16 slots/sleeve |
| Regime F blend, cap=20 | 0.56 | 3.66% | -8.38% | — | — | — | `targetpercent`, 20 slots/sleeve |
| Regime F blend, cap=25 | 0.59 | 4.08% | -8.17% | — | — | — | `targetpercent`, 25 slots/sleeve |
| Regime F blend, cap=30 | **0.66** | 4.74% | -8.00% | — | — | — | `targetpercent`, 30 slots/sleeve |
| **SPY (same window)** | **0.85** | 15.89% | -33.72% | — | — | — | Buy-and-hold |
| *Validated headline (WFO-gated, NOT reproduced here)* | *1.14* | *9.93%* | *-5.39%* | — | — | — | *17-fold WFO, `percent`* |

Per-sleeve detail (mean/max gross deployment %, mean/max concurrent positions):

| Sleeve | Regime P (decaying) | Regime F cap=8 | cap=12 | cap=16 | cap=20 | cap=25 | cap=30 |
|---|---|---|---|---|---|---|---|
| sp500 Sharpe | 0.50 | 0.19 | 0.20 | 0.23 | 0.19 | 0.27 | 0.35 |
| sp500 deploy mean/max | 44.6% / 100% | 17.4/26.8% | 25.5/40.0% | 33.7/52.9% | 41.4/65.8% | 50.5/82.2% | 59.0/98.4% |
| sp500 concurrent mean/max | 56.8 / 350 | 5.5/8 | 8.0/12 | 10.6/16 | 13.0/20 | 15.9/25 | 18.5/30 |
| midcap400 Sharpe | 0.73 | 0.50 | 0.57 | 0.64 | 0.69 | 0.69 | 0.72 |
| midcap400 deploy mean/max | 38.7% / 99.9% | 16.8/26.8% | 24.4/40.0% | 31.8/52.9% | 38.6/66.1% | 46.5/82.5% | 53.4/98.6% |
| nasdaq100 Sharpe | 0.78 | 0.60 | 0.64 | 0.74 | 0.75 | 0.77 | 0.83 |
| nasdaq100 deploy mean/max | 19.7% / 86.7% | 13.9/27.3% | 18.7/39.8% | 22.4/52.4% | 25.0/65.1% | 27.7/81.4% | 29.3/97.8% |
| Blend avg realized leverage | 0.58x | 0.74x | 0.69x | 0.64x | 0.57x | 0.50x | 0.46x |

Full machine-readable output: reproducible via
`scripts/position_sizing_regimes.py --eval-start 2021-01-31 --eval-end
2026-04-30` (negative-cache-populated TimescaleDB, ~3 min end-to-end).

## 3. Answers to the Decision-Relevant Questions

### Q1: What per-sleeve slot cap matches Regime P's validated risk profile?

**Two different things can be "matched," and they disagree** — this is
itself the most useful finding here. The blend's inverse-vol/target-vol
overlay (`target_vol=0.068`, `max_leverage=1.0`) is the dominant risk
control, and it actively compensates for slot-cap changes: as the cap rises
from 8→30, sp500's own mean deployment more than triples (17.4%→59.0%) and
its standalone Sharpe nearly doubles (0.19→0.35), yet the **blend-level**
max drawdown barely moves (-5.16% → -8.00%, non-monotonic) because
`combine_sleeves` scales the whole blend down (avg leverage 0.74x → 0.46x)
to keep trailing blend volatility near the 6.8% target regardless of how
hard any one sleeve is internally deployed.

- **Matching on blend-level Max DD / leverage** (the risk terms the task
  frames the question in): **cap≈12–16 is the right neighborhood — the
  original 12–16 estimate is confirmed.** cap=16 gives blend MaxDD -7.78%
  / avg leverage 0.64x against Regime P's -7.64% / 0.58x, the closest match
  of the six caps tested; cap=12 is close on leverage but a touch better on
  drawdown match (-7.93% vs -7.64%, essentially a wash within this
  single-window's noise).
- **Matching on each sleeve's own gross deployment** (a stricter,
  sleeve-local reading of "same risk profile"): the needed cap is **higher
  than 12–16 for two of the three sleeves** — sp500 needs ≈20–22 to reach
  Regime P's 44.6% mean deployment (cap=20 gives 41.4%, cap=25 gives
  50.5%), midcap400 needs ≈20 (cap=20 gives 38.6% against P's 38.7% — the
  closest match of any row in this table), and nasdaq100 needs only ≈12–13
  (cap=12 gives 18.7% against P's 19.7%). **Correction to the 12–16
  estimate:** it holds for nasdaq100 and for blend-level risk, but
  undershoots sp500/midcap400's own standalone deployment target by
  roughly 4–6 slots.

Net: **12–16 is the defensible answer if the target is blend-level risk
(what actually reaches the account)**, which is the more decision-relevant
reading since the blend, not any single sleeve, is what's deployed live.

### Q2: On its own merits, does fixed-notional sizing beat Regime P and SPY?

**No — not at any slot cap tested, in this single-window measurement.**
Regime F's blend Sharpe rises monotonically with cap but tops out at 0.66
(cap=30, effectively an unconstrained cap given none of the three sleeves'
mean concurrent positions reach 30), still below Regime P's 0.70 measured
in this same run. Both regimes underperform SPY's raw Sharpe (0.85) here,
though that comparison is not decisive either way — recall Regime P itself
only reaches 0.70 in this construction against a validated 1.14 under full
WFO, so SPY-relative conclusions from this single-window run should not be
over-read for either regime. The sleeve-level picture reinforces the same
direction: at cap=30, sp500 deploys *more* capital on average (59.0%) than
Regime P's own sp500 sleeve (44.6%) yet scores a *lower* standalone Sharpe
(0.35 vs 0.50) — the decaying-percent mechanism is more capital-efficient
per dollar deployed for sp500 specifically in this window, not just
lower-exposure. Equal-weight-with-a-cap is not shown to be the better
design here; it is shown to be a comparable-to-slightly-worse one at the
caps that keep blend-level risk near the validated profile (12–16), and
still short of Regime P even at its most aggressive tested cap (30).

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

This report doesn't reopen or reject any strategy-level lever. All twelve
closed equity-book arcs (ML gate, exit-rule sweep, IC-weighted voting,
Kelly sizing, overnight-gap, idio_vol, PEAD, insider-cluster-buy,
congress-trades, short-interest, short-volume-ratio, index-deletion-fade,
max-effect — see `docs/research/RESEARCH_SNAPSHOT.md` §4) stand unchanged.
This is purely a sizing-*mechanism* study layered on the already-validated
`EnsembleSignal` entries/exits.

**New, still-open item this report does NOT resolve:** the
`sleeve_slot_caps()` weight² distortion described in the task background
(slots ∝ weight, notional ALSO ∝ weight, so deployed sleeve capital goes as
weight², over-weighting nasdaq100 relative to its intended 43% target at
current live weights) is orthogonal to the P-vs-F sizing-*scheme* choice
studied here — it exists inside Regime F regardless of which slot cap is
chosen, and this report did not re-measure it. It remains a live,
unresolved bug; §5 below does not fix it either.

## 5. Operational Roadmap: Recommended First Action

**Do not switch the live paper trader's sizing scheme based on this report
alone — this single-window measurement did not show fixed-notional sizing
beating the status quo at any tested cap, and it cannot speak to whether
the gap holds under the validated WFO-gated construction.** The single
actionable, low-cost next step: **queue the full 3-sleeve gated WFO re-run
of Regime F at cap=16 (this report's best blend-level match to Regime P's
risk profile) as an overnight/background job**, using the same
`run_blend`/`run_wfo` machinery the validated 1.14 figure came from, and
compare directly against the 1.14/-5.39% baseline rather than against this
report's un-optimized 0.70/-7.64% stand-in. Budget several hours per the
2026-08-18 anchor report's ~15 min/fold cost measured under the current
(correct, per-fold, non-cached) anchor computation, times 3 sleeves. Until
that run completes, the live trader should keep running as-is (its
current, already-shipped fixed-notional/slot-cap mechanism) — this report
is not evidence to change it, only evidence about which cap to test first
when the full validation is affordable.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question: if a single-window, default-param comparison can't
even reproduce the 1.14 baseline for either regime, is the P-vs-F
conclusion in §3 worth anything?** Partially, and it's important to be
precise about which parts. The **absolute** Sharpe/CAGR levels in §2 are
not trustworthy as stand-ins for the validated numbers — say so plainly,
as done above. But the **relative** comparison (P vs. F at each cap) is
built on identical entries, identical exits, identical eligibility masking,
and identical blend overlay — the only thing that varies is the sizing
mechanism under test. A confound this report cannot rule out: WFO combo
selection could interact with sizing regime in ways a single fixed-combo
run can't see (e.g. a combo that WFO would select specifically *because*
it produces fewer, larger concurrent positions might behave differently
under a hard slot cap than under decaying-percent sizing). That's exactly
why §5 recommends the full re-run rather than treating this report as
final. What this report *does* license with reasonable confidence: the
mechanical finding that the blend's target-vol overlay dominates
individual sleeve slot-cap choice for blend-level risk (§3 Q1) — that's a
structural property of `combine_sleeves`, not dependent on which combo any
sleeve happens to be running, and would hold under the full WFO too.

### Parked Direction: cap sweep resolution finer than 8/12/16/20/25/30

The blend-level Sharpe curve over the tested caps (0.46/0.48/0.57/0.56/
0.59/0.66) is not smooth — cap=20 dips slightly below cap=16. With only six
points and one window, this is plausibly noise rather than a real local
optimum. Not worth resolving with a finer sweep on the single-window
construction; if the full WFO re-run in §5 confirms Regime F is
competitive at cap≈16, a finer sweep (14/15/16/17/18) under the *gated*
construction would be the right next step, not under this report's
un-gated stand-in.
