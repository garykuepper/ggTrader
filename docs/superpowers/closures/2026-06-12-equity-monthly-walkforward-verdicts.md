# Equity monthly walk-forward — verdicts and closure

**Closed:** 2026-06-12
**Span:** 2026-06-08 (stocks pivot consideration) → 2026-06-12 (all three strategies evaluated out-of-sample)
**Outcome:** WFO tournament on S&P 500 is a clear NO-GO (+0.9% vs SPY +103.8% over 64 OOS months); momentum baselines are market-like (no risk-adjusted edge); Alpaca execution work shelved; signal-library redesign confirmed as the only live path in both crypto and equities. The strategy-agnostic monthly harness remains as the cheap, honest test bench for future candidates.

This document closes the arc that began with "crypto may be maxed out — would the same engine work on US equities?" and ended with an empirical answer. Written for future-you deciding whether (and how) to spend more time on equities.

## 1. Investigation arc

The arc started from the 2026-06-08 edge-search closure: gates, universe size, and venue/fees had all been swept on crypto and none produced a deployable edge with the current entry library. Two forward directions were named — redesign the signal library, or point the same engine at US equities where the methodology might have more room. The equities question needed an honest test before any execution work (Alpaca integration, market-hours handling, PDT rules) could be justified.

The first attempt failed review. Two draft research docs (2026-06-09) were rejected — one for selection bias (a Sharpe 2.15 result computed on a universe chosen with hindsight), one for fabricated vectorbt APIs. Both were archived with corrections preambles rather than deleted.

The honest rebuild (2026-06-10) addressed the failure modes structurally: point-in-time S&P 500 membership (committed history file, 1996→2026) eliminates survivorship bias; a rolling monthly walk-forward — select at month-end T with data ≤ T only, trade the forward month frozen, checkpoint, stitch — makes the equity curve out-of-sample by construction; a `--leak-check` verifies the selection layer can't see the future. The full run `sp500_monthly_v1` (64 months, 2021-02 → 2026-05, ~500 stocks/month, 11×3 entry×exit tournament per stock) took ~28 hours on 4 cores.

While it ran, the harness was generalized (plan: [`2026-06-10-monthly-strategy-interface.md`](../plans/2026-06-10-monthly-strategy-interface.md), executed 2026-06-12 in 8 commits): a `MonthlyStrategy` protocol (`select` + `simulate`), the WFO tournament moved behind it unchanged, and two literature-standard baselines added — `xs_momentum` (12-1 cross-sectional momentum, top-50 equal weight) and `dual_momentum` (absolute filter, cash fallback). The leak check became strategy-generic and gained a defense-in-depth case that feeds *untruncated* data to `select`.

That defense-in-depth case caught a real bug on first contact: both `select()` implementations indexed positionally from the end of the data frame (`iloc[-(lookback+1)]`, `.tail(lookback_bars)`), which silently reads post-T bars when given untruncated data. Fixed (`f4a23f0`) by self-truncating to `asof` inside every `select`, with regression tests. Harness-level pre-truncation alone is not a sufficient guarantee — strategies must be invariant to post-T data themselves.

The verdicts landed 2026-06-12, all over the same 64 selection dates.

## 2. Results

| Metric | WFO tournament | xs_momentum | dual_momentum | SPY |
|---|---|---|---|---|
| Total return | +0.88% | **+125.98%** | +125.98% | +104.52% |
| CAGR | 0.17% | 16.69% | 16.69% | 14.51% |
| Sharpe | 0.12 | 0.82 | 0.82 | **0.89** |
| Sortino | 0.15 | 1.13 | 1.13 | **1.22** |
| Ann. volatility | 1.49% | 21.83% | 21.83% | 16.89% |
| Max drawdown | −4.30% | −22.38% | −22.38% | −24.50% |
| Monthly hit rate vs SPY | 0.365 | 0.49 | 0.49 | — |
| Avg exposure | mostly cash | 1.00 | 1.00 | — |

Full detail: [`equity_monthly_walkforward.md`](../../equity_monthly_walkforward.md) §4. Checkpoints: `results/monthly_wf/sp500_monthly_v1/`, `sp500_xs_momentum/`, `sp500_dual_momentum/`.

Three readings:

**The WFO tournament does not transfer out-of-sample.** Holding days landed squarely in the target 2–10 band (median 4.9), so the signals trade — they just don't make money. The 1.49% annualized vol with 2% position caps means the per-stock gates rejected most candidates most months, and what passed didn't carry forward (hit rate 0.365 < coin flip). Monthly selection turnover of 0.83 says the tournament's "best combo per stock" is unstable month to month — in-sample robustness ranking is mostly noise.

**psar_adx's history was hindsight bias.** The old crypto in-sample favorite won 2 of ~3,200 tournament slots across 64 months. Mean-reversion entries with fixed stop/target exits dominated selections (consistent with the crypto edge-search), and still didn't profit forward.

**The momentum baselines validate the harness.** xs_momentum reproduced the textbook long-only equity momentum result — beats the index on raw return (+126% vs +105%) by taking ~30% more volatility, with Sharpe/Sortino slightly *below* SPY. A broken harness wouldn't produce a literature-consistent number; this is good evidence the tournament's failure is real, not an artifact. dual_momentum was byte-identical to xs_momentum: checkpoint audit shows the minimum selected 12-1 momentum across all 64 months was **+12.2%** (October 2022, the bear low) — at top-50 of ~500 names the stock-level absolute filter never has anything to drop.

## 3. Decisions and reasoning

**NO-GO on deploying the current methodology to equities; Alpaca execution shelved.** The §5 decision rule (pre-registered in the walk-forward doc before the run finished) requires beating SPY on risk-adjusted terms out-of-sample. Nothing did. Building Alpaca execution, market-hours handling, and PDT-rule logic for a strategy set that loses to buy-and-hold SPY would be effort spent on plumbing for an empty pipe. Roadmap §5 updated to ⏸ with re-entry condition: some strategy beats SPY on Sharpe/Sortino through this harness first.

**The binding constraint is the signal library, in both markets.** Crypto: fees + signals (2026-06-08). Equities: signals alone — fees are near-zero here and the tournament still earned nothing. This kills the residual hope that the crypto failure was venue/fee-specific. Roadmap North star updated accordingly; §2d (strategy-library redesign) is the only live research direction.

**The harness is the keeper.** Strategy evaluation on equities now costs ~1 minute per candidate (momentum-class) on cached data, with leak checking, point-in-time universes, and SPY benchmarking built in. Any future signal idea — equities or adapted from the crypto library — gets an honest OOS answer cheaply before anyone considers deployment.

## 4. Methodology lessons

1. **Defense in depth on lookahead is not paranoia.** The harness already pre-truncated data before `select`; the leak check's unmasked case still found that both strategies *would* have looked ahead if any future call path passed fuller data. Positional-from-end indexing (`iloc[-n]`, `.tail(n)`) is a lookahead hazard in any point-in-time selection code — index by timestamp (`loc[:asof]`) first, always.
2. **Verification steps in plans catch bugs the plan itself wrote.** The leak-check failure was in code copied verbatim from the reviewed plan. Execute the verification even when the implementation "matches the plan exactly" — especially then.
3. **Run baselines through the same machinery as the candidate.** The momentum baselines cost minutes and turned "the tournament failed" from a debugging question into a conclusion, because a known-good strategy produced known-good numbers through identical code.
4. **A vacuous filter looks identical to a working one.** dual_momentum "ran fine" and produced plausible numbers; only the checkpoint audit revealed the filter never fired. When a variant's results match its parent exactly, verify the differentiating mechanism actually engaged.

## 5. Next steps (pre-registered)

The pass criterion for every candidate below is fixed in advance: **beat SPY on Sharpe AND Sortino over the full 64-month OOS window**, drawdown tolerable, no per-candidate threshold adjustment afterward. Variants tried should be counted and reported — running many cheap experiments and keeping the best is exactly the data mining this harness exists to prevent.

| Priority | Candidate | Cost | Rationale |
|:---:|---|---|---|
| 1 | Concentrated dual momentum: `--top-n 10` (and 20) | minutes | The absolute filter only matters with a small book; concentration is also where the momentum literature finds the premium strongest. Both runs, both top-n values, report all. |
| 2 | Portfolio-level absolute filter (Antonacci): if SPY 12-1 momentum < T-bill return, hold cash for the month | ~1h (new strategy class) | The classic dual-momentum construction; sidesteps the vacuous stock-level filter. Cheap to add behind the existing protocol. |
| 3 | Volatility-targeted xs_momentum: scale month exposure to a vol target (e.g. 15% ann.) | ~half day | xs_momentum's failure mode was excess vol, not return. Vol targeting is the standard, theory-justified repair (roadmap §3.3.B). |
| 4 | Reversion-focused entry set through the harness (roadmap §2d) | days | The only crypto style with life; equities harness gives it an honest cross-market read before crypto re-runs. |

Not next: any Alpaca/execution work (blocked on a §5 pass), and any further tuning of the WFO tournament's gates or grids on equities (the failure is structural — selection instability — not calibration).

## 6. Reference

- Results doc: [`equity_monthly_walkforward.md`](../../equity_monthly_walkforward.md) (§4 results, §5 decision rule)
- Plan (executed): [`2026-06-10-monthly-strategy-interface.md`](../plans/2026-06-10-monthly-strategy-interface.md); spec: [`2026-06-10-monthly-strategy-interface-design.md`](../specs/2026-06-10-monthly-strategy-interface-design.md)
- Code: `src/ggTrader/research/monthly_strategies.py` (protocol + 3 strategies), `src/ggTrader/research/monthly_walkforward.py` (harness), `scripts/sp500_monthly_walkforward.py` (CLI, `--strategy`)
- Commit span: `cf99271` → `34ee57e` (2026-06-10 → 2026-06-12); lookahead fix `f4a23f0`
- Prior arcs: [`edge_search_report_2026-06-08.md`](../../archive/edge_search_report_2026-06-08.md) (crypto levers exhausted), [`2026-05-12-wfo-textbook-reset-and-venue-migration.md`](2026-05-12-wfo-textbook-reset-and-venue-migration.md)
