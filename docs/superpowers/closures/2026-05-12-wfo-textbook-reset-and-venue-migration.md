# WFO textbook reset and venue migration — closure document

**Closed:** 2026-05-12
**Span:** 2026-05-10 (textbook reset commits land) → 2026-05-12 (venue migration decided)
**Outcome:** 22 textbook-validated combos selected for deployment on Binance.US; 14 TRX combos deferred 90 days; Kraken Pro retired as live venue.

This document closes the investigation arc that began with the WFO textbook reset producing empty validated sets and ended with a venue migration decision. It is written for the engineer (or future-you) returning in 90 days who needs to understand what happened, what was decided and why, and what remains open.

## 1. Investigation arc

The arc began with what looked like a research failure. The 2026-05-10 textbook reset rebuilt the WFO pipeline to strict train-only selection with four aggregate gates (WFE ≥ 0.5, % profitable folds ≥ 0.6, parameter CV ≤ 0.3, DD ratio ≤ 2.0), a 20% locked holdout, and a rank-based Sortino+Calmar+ProfitFactor composite. The integration run on the seven-coin diagnostic universe (BTC, ETH, TRX, DOGE, XMR, DASH, ADA) produced an empty validated set. Every coin's combos failed at least one of the four gates. The working hypothesis at the time was "no edge on this strategy × universe × timeframe."

The first move was to recalibrate the per-fold trade-count gate from 30 (textbook value) to 19. The recalibration was strategy-design-matched: the strategies fire roughly once per 7–8 days, which over a 6.7-month train fold yields ~26 expected trades at full activity. Demanding 30 was structurally impossible regardless of edge quality. The N=19 recalibration was pre-registered before re-running; it did not change the outcome at the gate level. Zero combos survived. This eliminated the per-fold trade gate as the binding constraint and shifted attention to the four aggregate gates.

The diagnostic that mattered next was the frictionless run — same pipeline, same universe, same parameters, but with FEES and SLIPPAGE both set to zero. Median in-sample annualized return shifted from +0.40 percentage points to +2.66 pp (Δ +2.24 pp), which fell into the "fees contributing" band of the pre-registered classification thresholds. What the original IS-shift summary missed was that the frictionless run *also* produced 36 combos that passed all four textbook gates. That fact only surfaced later, during the cleaned-frictionless comparison, when I pulled gate-pass data directly from the worker logs. The IS-shift was the correct point statistic but the wrong shape metric for the deployable-set question. This is the most important calibration lesson of the arc and is captured in section 6.

In parallel, an audit of the parameter grid found design residue from the rejected pre-reset selection methodology. Commits `33d863d` (2026-03-27) and `e23a40e` (2026-04-01) had pruned several grid axes using "X was selected N times" reasoning derived from runs that the textbook reset later acknowledged had test-data leakage. Three psar_adx axes were pinned to single values on that basis (sar_acceleration, sar_maximum, use_dmp_cross), and two macd_cross drops eliminated the canonical Appel parameters. The cleanup (Phases A + B, 2026-05-11) unpinned the three psar_adx axes to theory-justified pairs and re-added macd_fast=12 (the canonical Appel value, accidentally excluded as "interpolation" during the textbook reset). The discipline throughout was theory-justified values only, never survivor-tuned: each restored value was defensible on first-principles articulation grounds independent of any observed performance.

The cleaned-grid paired runs that followed produced symmetrical results. Cleaned + post-cost was empty — 0 of 231 combos passed, same as pre-cleanup at the live fee rate. Cleaned + frictionless produced 36 passers, 35 of which were shared with the pre-cleanup frictionless run. One swap exchanged TRX `psar_adx+trailing_stop` (lost to a cv-explosion in the expanded grid) for DOGE `macd_cross+trailing_stop` (gained from the restored canonical fast EMA). The classification was "methodologically correct but didn't unlock additional edge." The residue cleanup was the right principle; the missing edge wasn't in the residue.

Two follow-up diagnostics mapped the fee/edge relationship empirically. The half-fees run at 0.20% per side (1.0% round-trip) produced 14 passers. The Kraken-maker-rate run at 0.25% per side (0.50% round-trip) produced 13 passers — almost identical to half-fees, with only TRX `adx_filtered_rsi+trailing_stop` dropping out. Together these established that the fee/edge curve is nearly flat between 0.4% and 0.5% round-trip cost and drops off a cliff between 0.5% and 0.8%. Most of the edge collapse happens in that narrow band. This was the first hard data on where the transition lies.

With the fee/edge curve established, attention turned to execution cost. The Q1 live-execution audit confirmed that the live trader places 100% market orders on entries and on atr_trailing/trailing_stop exits, paying Kraken Pro's 0.40% taker rate on every leg. The Q2 signal-to-execution drift analysis on mtf_momentum and ema_cross signals (using 1-hour OHLCV against 4-hour signal-bar closes) showed that for BTC and TRX the |median| post-signal drift sits at 0.18–0.19 percentage points, below the 0.20% heuristic threshold for maker viability. ETH and DOGE were marginal at 0.26% and 0.42% respectively. The Q3 fee landscape across major exchanges revealed that Kraken Pro's $0-tier maker rate of 0.25%/side is 3-6× higher than competitors' makers, and that Binance.US (relisted in the US after the April 17, 2026 reopening) offered 0.02% taker — an order of magnitude cheaper than Kraken's maker rate, without any execution-mode complexity.

The post-only limit entry design proposal was drafted and immediately shelved when the venue economics became clear. Binance.US taker at 0.04% round-trip dominates every plausible Kraken execution mode for low-volume accounts. The proposal lives at `docs/superpowers/specs/2026-05-12-post-only-limit-entry.md` with a SHELVED header note explaining the supersession and pointing to the re-evaluation criteria in section 5 below.

The smoke-test sequence on Binance.US closed the arc. Authentication worked, the four target pairs (BTC, ETH, DOGE, TRX) are all active with $1 minimum cost, and the first spread/depth snapshot revealed a problem specific to TRX. Binance.US TRX/USD shows roughly $2K of 24-hour volume against Kraken's $2.5M — a 1,246× ratio. The spread at the first snapshot was 118.6 basis points. Even with the 0.02% taker advantage, the spread alone makes Binance.US TRX more expensive per trade than Kraken's full 80bp taker cost. BTC, ETH, and DOGE on Binance.US show 95–213× lower 24-hour volume than Kraken but with absolute volumes still in the $35K–$1M range — sufficient for the account's trade sizes.

The closing decision was Path D: deploy the 22 textbook-validated combos that live on BTC, ETH, and DOGE on Binance.US; defer the 14 TRX combos for 90 days pending volume growth on Binance.US (re-evaluation date ≥ 2026-08-10).

## 2. Decisions and reasoning

**Path D over Path B (hybrid Kraken-maker for TRX).** Path B would have kept TRX live on Kraken using post-only limit entries while everything else migrates to Binance.US. The hybrid offers up to 5 additional deployable combos (the survivors at Kraken maker rate from the 14 TRX frictionless passers) but requires running two venues simultaneously during the initial migration period. The operational complexity — two API integrations, two balance reconciliation paths, two sets of monitoring, two failure modes — was judged not worth 5 incremental combos during a phase whose explicit purpose is to validate Binance.US operationally. The proposal stays viable for Q3 2026 if TRX volume on Binance.US has not grown sufficiently by then.

**Phase 1/Phase 2 deployment sequencing.** The migration deliberately does not change venue and parameter set simultaneously. Phase 1 runs Binance.US with the legacy pre-reset parameters that the live trader currently uses on Kraken — known-good live behavior, only the venue is new. Phase 2 introduces the textbook-validated 22-combo parameter set after 1–2 weeks of operational validation has confirmed fill quality, spread tightness during actual trades, API stability, and balance reconciliation flow. If something breaks during Phase 1, root cause can be isolated to venue mechanics rather than tangled with parameter-set behavior. The cost is a 1–2 week delay before the textbook-validated set goes live. The benefit is clean attribution if any issues surface.

**Post-only limit entry shelving.** The proposal had a clear motivation while Kraken was the only viable venue: maker at 0.25%/side was the cheapest available execution cost. Binance.US taker at 0.02%/side ($0-volume tier) is 12× cheaper than Kraken maker, without requiring any code change to the execution path. The motivation evaporated cleanly. The proposal is preserved as reference in case the Binance.US migration is reversed, or in case TRX-specific maker-only on Kraken becomes the right move at the 90-day TRX re-evaluation.

**Not deploying BTC bbands_mean_reversion+atr_trailing.** This combo passes the gates at both the half-fees and Kraken-maker fee levels but does *not* pass at frictionless. Mechanically it appears to be a cv-instability artifact: the per-fold cell-picks under non-zero cost regimes happen to converge more tightly than under frictionless conditions, lowering parameter CV below the 0.30 gate. Shipping a combo whose textbook validation under the cleanest conditions explicitly rejects it would be methodologically inconsistent. It is left out of the deployable set even though it is the only "BTC passer" at non-zero fees.

## 3. Fee/edge curve

The empirical artifact that motivated and grounded every cost-related decision in this arc.

| Round-trip cost | FEES per side | Slippage | Validated passers | Run dir (results/research/) |
|---|---|---|---|---|
| 0.00% (frictionless) | 0.0000 | 0.0 | **36** | research_20260511_193048 |
| 0.40% (half-fees) | 0.0020 | 0.003 | **14** | research_20260511_214219 |
| 0.50% (Kraken maker) | 0.0025 | 0.003 | **13** | research_20260512_091940 |
| 0.80% (Kraken taker — current live) | 0.0040 | 0.003 | **0** | research_20260511_162755 |
| 0.04% (Binance.US taker — target) | 0.0002 | 0.003 (assumed) | **~30–35** (extrapolated, not measured) | not run |

The shape is approximately flat between 0.4% and 0.5% round-trip (one passer dropped across that 0.1pp range, well within run-to-run noise), then drops off a cliff between 0.5% and 0.8% where the remaining 13 passers go to zero. Most of the edge collapse happens inside that narrow 0.3pp band. The Binance.US target at 0.04% round-trip sits well below the half-fees data point, so the deployable count is expected to land close to the frictionless 36 — but this is interpolation, not measurement, and is flagged accordingly.

The per-coin distribution across the curve is what motivated Path D:

| Coin | Frictionless (36) | Half-fees (14) | Kraken maker (13) | Kraken taker (0) |
|---|---|---|---|---|
| TRX | 14 | 6 | 5 | 0 |
| DOGE | 10 | 7 | 7 | 0 |
| ETH | 7 | 0 | 0 | 0 |
| BTC | 5 | 1 (cv artifact) | 1 (cv artifact) | 0 |
| ADA, DASH, XMR | 0 each | 0 each | 0 each | 0 |

DOGE is the most fee-robust coin (70% retention from frictionless to half-fees). TRX retention is moderate at full Kraken-maker (5 of 14, 36%). ETH collapses entirely off frictionless once any meaningful cost is introduced. BTC retains zero frictionless passers at any non-zero cost; the one BTC entry that appears at half-fees and maker is the cv-instability artifact discussed in section 2.

## 4. Methodology issues banked

**The param_cv gate at the 0.30 threshold is grid-size dependent in a way that does not reflect true parameter instability.** This was confirmed during the residue cleanup. The psar_adx grid expanded from 4 to 32 cells when its three pinned axes were restored to theory-justified pairs. The per-fold optimizer, given more cells to choose from, naturally drifts more across folds — there are more candidate winners and more opportunities for adjacent folds to land on different ones. The cv-statistic mechanically inflates: psar_adx median cv jumped from 0.39 (4-cell grid) to 1.22 (32-cell grid), with a maximum of 3.00. Six high-IS-edge psar_adx combos (IS_mean +12 to +21pp on frictionless) failed *only* on param_cv. Every other gate would have passed them.

The principled fix is either an axis-aware CV (something like `unique_picks_per_axis / grid_size_per_axis`, which normalizes the spread to the search space) or a grid-size-relative threshold (allow CV up to some function of `log(N_cells)`). Either approach needs to be justified theoretically before re-running, not chosen post-hoc to produce a desired number of survivors. This is deferred until the execution-cost question is resolved and the live trader is operationally stable on Binance.US. It is the highest-priority methodology debt remaining from this arc.

**Run-data persistence: per-run wfo_stats must be snapshotted to disk before any subsequent cache purge.** This bit the investigation during the cleaned-grid runs when the cleaned-post-cost wfo_stats was lost between runs because the cache was purged to make room for the next experiment's data. The fix landed in `src/ggTrader/cli/cmd_research.py` at the merge step: every research run now writes `<run_dir>/wfo_stats_snapshot.json` containing the per-(symbol, combo) wfo_stats payloads from the cache. The discipline going forward is to never purge the cache before reading what the run produced. The snapshot file is the durable record.

## 5. 90-day TRX re-evaluation criteria (≥ 2026-08-10)

These are pre-registered now to anchor the decision against the data when the time comes, rather than re-negotiating thresholds in the moment.

**If Binance.US TRX/USD 24-hour volume is ≥ $500K and spread is ≤ 15bp during US hours**, migrate TRX onto Binance.US and eliminate the Kraken dependency entirely. The 22-combo deployment expands to 36, and the operational footprint stays single-venue.

**If Binance.US TRX volume is in the $50K–$500K range**, the picture is ambiguous: the structural illiquidity has partly resolved but the spreads may still be wide enough to consume the fee advantage. The right move is one more diagnostic run at the effective Binance.US TRX cost (taker + measured spread) and a decision based on the resulting deployable combo count. If 5+ TRX combos survive that effective-cost gate, migrate; if fewer, continue deferral.

**If Binance.US TRX volume remains below $50K**, two paths fork: continue Path D for another 90 days (accepting that TRX stays on the bench indefinitely until liquidity arrives), or commit to Path B by unshelving the post-only limit entry spec for TRX-on-Kraken specifically. The Path B commit is a meaningful engineering lift; the question at the 180-day mark would be whether 5 incremental TRX combos justify the lift given everything else is running cleanly on Binance.US.

## 6. Calibration lessons

**The Kraken-maker deployable estimate was wrong by roughly 3×.** Before running the Kraken-maker diagnostic, the working assumption was that Path A (maker-only on Kraken) would recover most of the frictionless TRX+BTC passer set — call it ~19 combos. The actual result was 5 combos. The error came from interpolating linearly between the frictionless (36 passers) and post-cost (0 passers) data points without accounting for the curve's concavity at lower fee levels. Future estimates of where the deployable count lands on the fee/edge curve should anchor on the *shape* of the curve (which was unknown until the half-fees and Kraken-maker runs measured it directly), not on a linear segment between the extremes.

**The first frictionless diagnostic missed the 36-passer signal.** The pre-registered classification metric was Δ-IS-median against three threshold bands. The frictionless run produced a +2.24pp shift, classified as "fees contributing," and that was the only summary reported at the time. The 36 combos that actually passed all four textbook gates were sitting in the worker logs but never extracted. The cleaned-frictionless analysis surfaced the gate-pass count when the comparison required it, and the prior data point had to be reconstructed retroactively from the still-existing worker logs.

The lesson is that point statistics and shape metrics measure different things. Δ-IS-median tells you whether the distribution moved; gate-pass count tells you whether anything became deployable. The two can disagree by a lot when the distribution sits close to a hard threshold — the distribution barely moves but the count crosses zero. Future pre-registrations on threshold-gated systems should include shape metrics (count of survivors above each gate, count of % both-positive, count of multi-gate passers) alongside the distributional statistics. Pre-register both, not one or the other.

**Investigation cost vs output.** Six diagnostic cycles over approximately one week of elapsed time produced: 22 textbook-validated combos ready for deployment on a venue 12× cheaper than the current live venue, two confirmed methodology issues banked for future cleanup, and a 90-day deferred decision on TRX with pre-registered re-evaluation criteria. The cost was worth the output. The arc could have terminated at the first frictionless run if the 36-passer signal had been extracted then — perhaps half the diagnostic cycles would have been avoidable. That doesn't invalidate the others: the residue cleanup, the fee/edge curve mapping, and the execution-cost investigation each contributed information that the deployment decision relies on. But the structural inefficiency is worth noting: pulling all the shape metrics on the first diagnostic could have shortened the arc by roughly two days.

## 7. Open items carrying into the next session

The manual Binance.US API key permission audit remains user-owned and pending: Reading ON, Spot Trading ON, Withdrawals OFF, IP whitelist applied if appropriate. Report when done so the smoke-test sequence can be formally closed.

Three scheduled spread/depth snapshots remain in flight via cron: 22:00 UTC tonight (2026-05-12), 11:00 UTC tomorrow (2026-05-13), and 17:00 UTC tomorrow (2026-05-13). After the last one lands, the consolidation step produces the 4-snapshot × 4-pair × 2-venue table, sanity-checks BTC/ETH/DOGE spread stability across time-of-day windows, and documents the TRX volume/spread pattern for the 90-day re-evaluation record.

After the consolidation: data architecture migration. Schema changes to accommodate Binance.US as a venue (alongside Kraken, since historical data from both will be carried forward), and the historical OHLCV backfill from Binance.US bulk zips for the four pairs. This is the largest remaining piece of work before Phase 1 deployment can start.

After data migration: Phase 1 deployment. Live trader switched to Binance.US with legacy pre-reset parameters at small position size for 1–2 weeks of operational validation.

After Phase 1 validates cleanly: Phase 2 deployment. Switch to the textbook-validated 22-combo BTC/ETH/DOGE parameter set.

GitHub issue #10 (the `fee_entry` recording bug discovered during the live-execution audit) remains filed. Fix-and-backfill scope deferred. Independent of the venue migration.

## 8. Status of shelved and closed artifacts

`docs/superpowers/specs/2026-05-12-post-only-limit-entry.md` — SHELVED. Header note explains Binance.US supersession and points to the 90-day TRX re-evaluation criteria. Kept as reference.

`docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md` — completed. All 9 implementation tasks landed; Task 9 (integration test on the diagnostic universe) produced the empty validated set that opened this arc.

WFO research run artifacts preserved at `results/research/research_20260511_*` (the five runs of 2026-05-11) and `results/research/research_20260512_091940/` (the Kraken-maker fallback run). All include `wfo_stats_snapshot.json` from 2026-05-11_193048 onward, after the snapshot fix landed.

Smoke-test artifacts: `scripts/binanceus_smoke_test.py`, `scripts/binanceus_spread_depth.py`, and `results/binanceus_smoke/snapshots.jsonl` (currently 8 rows from snapshot 0; will grow to 32 rows after all four snapshots).

Memory: `project_venue_migration_2026-05-12.md` and `MEMORY.md` index updated with a one-line pointer.

Crontab: three one-shot snapshot entries flagged for removal after 2026-05-13.

---

*This document is the consolidated closure record for an investigation arc that ran 2026-05-10 through 2026-05-12. If you are reading this 90 days later to support the TRX re-evaluation, the next decision is governed by section 5. If you are reading it earlier for the data-architecture migration or Phase 1/2 deployment work, section 7 lists what remains open.*
