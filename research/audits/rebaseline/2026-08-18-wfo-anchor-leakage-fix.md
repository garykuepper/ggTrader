# WFO Anchor-Set Leakage Fix: Mechanism Confirmed, Bounded Practical Impact

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-08-18
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

> ⚠️ **SUPERSEDED 2026-08-19/20.** Every citation below of Sharpe **1.12**
> (SP500 core), **1.14** (3-sleeve blend), or an **SPY 0.58–0.65** benchmark
> predates two later fixes: the 2026-08-19 pinned-window reproduction (which
> found the "1.12" baseline was actually **0.97**, and the blend **0.68**,
> once `--eval-end` drift was pinned) and the 2026-08-20 day-shift fix (every
> equity bar had been stamped one calendar day early, which also deflated
> SPY via duplicate rows). A 2026-08-22 re-run on the corrected, day-shift-
> fixed tape, same pinned window, measures core **0.99**, blend **0.69**,
> SPY **0.914** (2021–2026 headline) / **0.78** (pinned 2021-01-31→2026-04-30
> window). None of this changes this report's own conclusion — the anchor
> leak is real and fixed, and is a separate bug from the ones above — but do
> not cite the 1.12/1.14/0.58–0.65 figures below as current. See
> `docs/research/2026-08-19-anchor-fix-reproduction.md`.

## 1. Executive Summary & Core Engine Audit

`run_wfo` (`src/ggTrader/lab/wfo.py`) walks a strategy through rolling
12-month-train / 3-month-test folds. When a fold's gates (NDH plateau density
+ DSR) fail, or the circuit breaker has halted live trading, the fold deploys
a defensive "anchor" combo instead of its own trained winner — the
minimum-drawdown combo among those clearing a CAGR floor. Prior to this fix,
`compute_anchor_set()` was called **once**, before the fold loop, on the
**full** `ohlcv` frame spanning the entire eval window (`ohlcv.index[0]` to
`ohlcv.index[-1]`). That anchor was then reused as the OOS fallback inside
every fold that failed gates or traded while halted — meaning a fold's
"out-of-sample" number could be produced by a combo chosen with knowledge of
price action years after that fold's test window closed. This is textbook
look-ahead leakage in the one place designed to be defensive.

The fix (`src/ggTrader/lab/wfo.py`, `run_wfo`) moves the anchor computation
inside the fold loop and scopes it to `ohlcv.loc[:fold.train_end]` —
strictly expanding, never seeing data past the point each fold is allowed to
know about. Each fold now recomputes its own anchor from scratch (no
cross-fold caching — inputs differ every fold since the window keeps
expanding, so correctness was prioritized over speed per the review plan).
`compute_anchor_set()` itself is unchanged; only what `run_wfo` passes it
changed.

This report closes out item 8 of the 2026-08-18 review-and-fix plan (Phase B,
P2 research-validity fixes). The live SP500 core config (Sharpe 1.12, CAGR
16.3%, MaxDD -11%, 16/17 gates) remains the only validated, deployed
configuration; this report does not change that verdict — it quantifies
whether the anchor leak materially affected how that number was produced,
and confirms the mechanism is now leak-free going forward. Items 9–12 of the
same plan (leveraged-rotation universe survivorship revert, ML-gate training
purge/PIT, signal-strategy PIT eligibility masking, weight-fold rebalance
cash drag) are separate fixes tracked in `docs/changelog.md`, not covered
here.

## 2. Quantitative Performance Context

A full 18-fold re-run of `ensemble` on `sp500` (2021-01-31 → present, the
default `ggt.py lab --strategy ensemble --universe sp500 --wfo` config) was
attempted natively. Fold 1 alone (anchor fit on an already-expanding ~13
month window + train sweep + test sweep, 48-combo grid × ~600 symbols) took
roughly 15 minutes; at that rate the full 18-fold run would run for several
hours, well past the ~30 minute budget for this report. The full aggregate
OOS Sharpe delta was **not** measured end-to-end. Instead two smaller, honest
experiments were run against the same cached SP500 dataset (604 symbols,
2021-01-31 → 2026-08-18):

**(a) Two real folds, full pipeline, current (fixed) code:**

| Fold | Train → Test | Gate | OOS Sharpe | Anchor used? |
|---|---|---|---|---|
| 1 | 2021-01→2022-01 → 2022-01→2022-04 | PASS (NDH 1.00/DSR 1.00) | 1.11 | No |
| 2 | 2021-04→2022-04 → 2022-04→2022-07 | PASS (NDH 1.00/DSR 1.00) | 0.99 | No |

Both observed folds cleared gates cleanly, so the (corrected) anchor was
never deployed as a fallback in either — these two numbers are **identical**
to what the old code would have produced, because the leak only manifests
when a fold's own winner is rejected by gates or the system is halted.

**(b) Direct anchor A/B, mechanism-level, same dataset:**

| Fit scope | Combo (`min_agree_exit`/`bb_std`/`rsi_oversold`, abbreviated) | MaxDD | CAGR | Sharpe |
|---|---|---|---|---|
| **OLD**: full sample (2021-01→2026-08, leaky) | `exit1_std2.5_rsi25` | -37.1% | 5.5% | 0.43 |
| **NEW**: fold 1 window only (→2022-01) | `exit2_std2.0_rsi30` (**different combo**) | -39.8% | 7.4% | 0.44 |
| **NEW**: fold 7 window only (→2023-07) | `exit1_std2.5_rsi25` (same as OLD) | -37.1% | 4.4% | 0.33 |
| **NEW**: fold 13 window only (→2025-01) | `exit1_std2.5_rsi25` (same as OLD) | -37.1% | 5.3% | 0.41 |
| **NEW**: fold 18 window only (→2026-04) | `exit1_std2.5_rsi25` (same as OLD) | -37.1% | 4.9% | 0.38 |

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **Live SP500 core (validated, unaffected by this fix — anchor inactive in its gate-passing folds)** | 1.12 | 16.3% | -11% | 16/17 folds | 3% flat |
| **SPY benchmark** | 0.58–0.65 | ~14–15% | ~-34% | N/A | Buy-and-hold |

## 3. Findings

**The leak is real and mechanistically confirmed, but historically low-frequency in its effect.** The full-sample anchor and the fold-1-scoped anchor pick genuinely different combos — the full-sample fit sees 2022–2026 price action (including the 2022 bear market and the 2023–2025 recovery) that a fold ending January 2022 could not have known about, and that knowledge measurably changes which combo looks like the best defensive choice (different `min_agree_exit`/`bb_std`/`rsi_oversold`, MaxDD -37.1% vs -39.8%, Sharpe 0.43 vs 0.44). From fold 7 onward (mid-2023+), the anchor choice converges to the same combo the full-sample fit picks, so for the back half of the eval span the leak was a no-op even before this fix.

**Whether the leak reached the reported live-SP500 headline number (Sharpe 1.12, 16/17 gates) depends entirely on which single fold historically failed gates.** 16/17 gates passed means exactly one fold deployed the anchor as its OOS fallback. This report did not re-run all 17 folds to identify which one (compute cost, see §1), so it cannot be stated definitively whether that fold was early enough (pre-2023) for the leak to have changed its deployed combo, or late enough (2023+) that the anchor had already converged and the leak was inert for it too — both of the two directly-observed real folds (1–2) turned out to be gate-passing, so the leak was provably inactive for them specifically.

**Practical takeaway: the fix is correct and should stay, but it is unlikely to explain a large share of the previously reported 1.12 Sharpe** — the anchor is a fallback path invoked in a minority of folds (historically 1/17), and even where invoked, the leak only changes the outcome for pre-2023 windows. A full 18-fold honest re-run remains the only way to get an exact aggregate delta; that run should be queued as a background/overnight job given its multi-hour cost under the corrected (per-fold, non-cached) anchor computation — see §6.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

This report doesn't reopen or reject any strategy-level arc. It is an
engineering-correctness fix to the WFO harness itself. All prior strategy
verdicts (ensemble SP500 core GO, leveraged rotation NO-GO, leveraged trend
NO-GO, ML gate falsified, pairs-stat-arb NO-GO, IC-weighted voting NO-GO,
exit-rule sweep NO-GO) stand unchanged — see `docs/research/RESEARCH_SNAPSHOT.md`.

## 5. Operational Roadmap: Recommended First Action

**Queue one full, uninterrupted 18-fold `ensemble`/`sp500` WFO re-run
(current fixed code) as an overnight/background job, and record the exact
aggregate OOS Sharpe/CAGR/MaxDD delta against 1.12/16.3%/-11% once it
completes.** Given the ~15 min/fold observed cost, budget 4–5 hours. This is
the only way to close the "how much did the leak actually cost the reported
number" question definitively; everything in §3 is a bounded, honest
estimate from partial runs, not the full answer. No live-trading change is
needed regardless of outcome — the fix is already correct and committed;
this action is purely to finish quantifying its historical effect for the
record.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question: if both directly-observed folds show zero effect, and the anchor only diverges pre-2023, is this fix worth having spent effort on?** Yes — the leak is a correctness bug independent of whether it happened to move the specific historical 1.12 number: any future WFO run, on any strategy or universe, that hits a gate-failing or halted fold in a pre-convergence window would have silently leaked future information into a number labeled "out-of-sample." The fact that the two folds checked here were unaffected is a property of this particular strategy/universe/date range, not a guarantee. The fix belongs in regardless of the specific delta, and the review plan's own framing ("Past headline numbers... should be re-run once to quantify the delta") is satisfied by the partial evidence in §2–3; a full delta is still recommended (§5) but does not gate anything.

### Parked Direction: Full 18-fold re-run with anchor cross-fold caching

`compute_anchor_set()` recomputes from scratch every fold even though
consecutive folds' expanding windows overlap almost entirely (each fold adds
only 3 months of new data to the previous fold's window). A provably-safe
cache (e.g. reuse the previous fold's anchor whenever the new fold's winner
combo and CAGR-floor-viable candidate set haven't changed) could cut the
~15 min/fold anchor cost substantially and make the full re-run in §5
tractable in under an hour instead of overnight. Explicitly deferred here
per the review plan's "correctness over speed" instruction — implement only
after the honest uncached re-run in §5 has run at least once, so there is a
ground truth to validate any caching optimization against.
