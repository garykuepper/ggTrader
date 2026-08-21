# Next Steps

This file is the **only-look-1-2-steps-ahead** worklist — literally the next
thing(s) to do, nothing further out. It exists so work can be handed to a
cheaper model without them needing to re-derive context from `roadmap.md`'s
full history. When a step here is done, delete it and add the next one (if
any) — don't let this file accumulate a backlog.

## Where new strategy ideas come from (2-stage pipeline)

1. **Discovery (external, no repo access)** — run
   `docs/research/prompts/web-strategy-research-prompt.md` in a web-research
   tool (Google Gemini or Claude's web-UI research feature). Paste the
   results back into a local session and ask to merge them into
   `docs/research/WEB_RESEARCH_CANDIDATES.md` (an accumulating backlog, not
   overwritten). `docs/research/RESEARCH_SNAPSHOT.md` §6 also holds a
   smaller set of internally-derived candidate ideas (reasoned from what's
   failed, not external research) — both are valid sources.
2. **Implementation (local, full repo access)** — pick ONE candidate from
   either source, copy it into
   `docs/research/prompts/local-implementation-prompt-TEMPLATE.md`'s
   "Candidate Strategy" section, and queue a concrete step below. That
   template's prompt is what actually drives the build-and-WFO-test work in
   this repo.

Do not skip straight to implementation-brainstorming here — this file is
for a single, already-scoped next step, not a list of ideas to pick from.

---

## ACTIVE STEP (2026-08-20) — ROOT CAUSE FOUND: every equity bar in `ohlcv` is stamped one calendar day early

**Priority: this outranks everything else in this file.** It is the root
cause behind at least three symptoms already "fixed" locally, and it
silently changes what date every piece of research thinks it is looking at.

### The finding

Every `venue='yfinance'`, `interval='1d'` row — **1,412 symbols, ~5.68M
rows, the entire equity store** — is stored one calendar day earlier than
the bar it actually contains.

Proof, day-of-week census over the whole table:

| Stamped weekday | Rows |
|---|---|
| Sun | 1,065,046 |
| Mon | 1,167,553 |
| Tue | 1,165,640 |
| Wed | 1,143,689 |
| Thu | 1,137,513 |
| **Fri** | **168** |

A daily US equity series cannot have 1.07M Sunday bars and 168 Friday bars.
The series runs Sun–Thu because it is Mon–Fri shifted back one day. (Those
168 Fridays are SPY's, from a second writer — see below.)

Ground-truth spot check, AAPL (DB vs. yfinance):

| DB timestamp | DB open/close/volume | Bar's true date |
|---|---|---|
| 2026-07-29 17:00 | 332.81 / 333.14 / 74,817,800 | **2026-07-30** |
| 2026-07-30 17:00 | 304.55 / 308.64 / 132,489,100 | **2026-07-31** |
| 2026-08-02 17:00 *(a Sunday)* | 309.31 / 303.16 / 75,052,000 | **2026-08-03** |
| 2026-08-03 17:00 | 302.47 / 309.11 / 68,001,000 | **2026-08-04** |

Exact OHLCV matches, uniformly one day early. AAPL's -8% earnings day
(actually 07-31, close 308.64) is filed under 07-30.

### Mechanism (single line)

`CachedYFinanceLoader._cache_to_db` builds `ts_list =
sub.index.to_pydatetime().tolist()` from a **tz-aware UTC** index and
inserts it into `ohlcv.timestamp`, which is `timestamp WITHOUT time zone`.
Postgres rebases the offset-carrying literal into the **session timezone**
— this box is `America/Los_Angeles` — so `2026-08-03 00:00:00+00:00` lands
as `2026-08-02 17:00:00`. The fetch path itself is correct (verified: it
returns `2026-08-03 00:00:00+00:00` with Aug-3's true OHLCV); only the
write is wrong.

The two stamp families are the same bug under DST: **17:00** = UTC-7 (PDT),
**16:00** = UTC-8 (PST). Both are "midnight UTC minus the Pacific offset".

Fix the writer by storing naive UTC:
`sub.index.tz_convert("UTC").tz_localize(None).to_pydatetime()`.

### Why SPY is the one symbol that looks broken

SPY additionally carries **826 correctly-dated rows** at 04:00/05:00
(midnight ET expressed in UTC, Mon–Fri) from a second, benchmark-only
writer active 2022-12-27 → 2026-06-08. SPY is therefore the only symbol
with **mixed alignment**: 642 calendar days hold two rows carrying
*different bars*, and 184 days exist only in the correct family.

That is why SPY alone shows duplicate days, and why its measured Sharpe is
unstable and deflated — 0.72 as-loaded vs **0.93** after collapsing to one
bar/day (2021-2026); 1.00 vs **1.46** (2023-2026). A ~29% benchmark
deflation, matching what the 2026-08-19 session observed.

**This is very likely the real explanation for "SPY scores 0.58 in the
cited runs vs 0.78 here"** — the blend-vs-core decision above rests on a
benchmark that changes value depending on which rows the window catches.
Do not settle that decision until this is fixed.

### Blast radius — read carefully, it is not uniform

- **Pure price-series backtests: returns are NOT invalidated.** The shift
  is uniform across all symbols, so bar-to-bar returns are unchanged. Only
  the labels are wrong.
- **SPY-benchmarked comparisons ARE invalidated** — SPY is internally
  inconsistent (above). Every "vs SPY" verdict in this repo is suspect.
- **Every join against a real-world calendar date is off by one, in the
  lookahead direction** — the bar labeled D holds D+1's outcome. This hits
  FOMC dates, earnings, Form 4 filings, congress PTRs, index add/delete,
  dividend ex-dates, short-volume. Event-study candidates were rejected on
  numbers computed against a shifted tape.
- **Already-patched symptoms that trace here** (all treated the symptom,
  not the cause):
  1. `fomc_drift` — "bars carry 16:00 while FOMC dates are midnight" was
     patched by matching on *calendar date*, but the calendar date is
     itself shifted, so that strategy tested the wrong day. **Its NO-GO is
     not trustworthy.**
  2. `9f69107 fix(paper): freshness gate froze all trading` — the gate saw
     a newest bar labeled a day stale and halted. Loosening the gate hid
     the shift.
  3. The 2026-08-19 "SPY duplicate rows" investigation.

### Status: writer fixed and data migrated (2026-08-20) — DONE

1. **Writer fixed.** `_cache_to_db` now strips the tz
   (`idx.tz_convert("UTC").tz_localize(None)`) so Postgres has no offset to
   rebase.
2. **Regression tests added** (`tests/data/test_cached_yfinance_loader.py`):
   4 unit tests plus an `@pytest.mark.integration` real-DB round-trip.
   Confirmed they fail on the old code with the exact production symptom
   (`['2026-08-02','2026-08-03',...]` — Sun–Thu) and pass on the new.
3. **Migrated.** `DELETE 5,678,783` / `INSERT 5,678,783`, zero
   conflict-drops, in one transaction. Needed
   `SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0`
   (compressed chunks). Backups kept: table
   `ohlcv_yf_preshift_backup_20260820` (5,679,609 rows) and
   `backups/ohlcv_yf_preshift_20260820.sql.gz` (206MB).
4. **SPY reconciled.** After the shift all 826 stray days had a
   correctly-dated counterpart (`stray_only_days = 0`), so the stale-vintage
   strays were dropped with zero coverage loss.

Verification: weekday census is now Mon 1,065,200 / Tue 1,167,570 /
Wed 1,165,638 / Thu 1,143,684 / Fri 1,137,517 — **zero weekend bars**
(Monday lowest, as it should be with Monday holidays). Duplicate days
across all 1,412 symbols: **0**. Every spot-checked bar now matches
yfinance exactly (AAPL's -8% earnings day sits on 07-31, not 07-30). A
fresh live fetch round-trips to the correct date. 922 tests pass; ruff
clean on both touched files.

**Benchmark, corrected:** SPY 2021-2026 Sharpe **0.914** (CAGR 15.06%,
1,340 bars, 0 dups), 2023-2026 **1.433**. Was 0.72/1.00 when read through
the duplicated tape.

### Still open, in priority order

1. **Re-measure the blend-vs-core decision below against the corrected
   benchmark.** Every cited SPY number in this repo (0.58, 0.74, 0.77,
   0.78) came off the shifted/duplicated tape and none of them match
   0.914. The deployed blend's 0.68 and the core's 0.97 both need re-running
   before that decision means anything.
2. **`fomc_drift`'s NO-GO is not trustworthy.** `fomc_drift.py:130-134`
   matched on calendar date to work around the non-midnight stamps — but
   the calendar date was itself shifted, so "the day before the
   announcement" selected the announcement day. That workaround is now
   inert and the logic is correct; re-run if the candidate is worth the
   cost.
3. **Other event-date candidates** (earnings/PEAD, Form 4 insider,
   congress PTR, index add/delete, short-volume) were all rejected on
   numbers computed against a one-day-lookahead tape. Their verdicts are
   suspect for the same reason — note this in any report that cites them.
4. **`9f69107`'s freshness-gate tolerance** was widened to absorb an
   apparent one-session lag that was partly this bug. It still allows lag,
   so nothing is broken, but `as_of` now advances a day — worth a look on
   the next live run.

---

## ACTIVE STEP (2026-08-19) — decide whether to revert live from the 3-sleeve blend to the SP500 core

**The July 17 note that stood here — "reconfirmed exactly (Sharpe 1.14,
MaxDD -5.39%) via fresh blend runs — no tooling drift" — is SUPERSEDED.**
A pinned-window 17-fold re-run on 2026-08-19
(`docs/research/2026-08-19-anchor-fix-reproduction.md`, driver
`scripts/anchor_fix_reproduction_wfo.py`, raw
`docs/research/_anchor_fix_reproduction_results.json`) reproduces neither
cited headline:

| Config | Sharpe | CAGR | MaxDD | Gates |
|---|---|---|---|---|
| SP500 core — cited | 1.12 | 16.3% | -11.0% | 16/17 |
| **SP500 core — measured** | **0.97** | **7.8%** | -7.6% | **12/17** |
| 3-sleeve blend @lev 1.0 — cited | 1.14 | 9.93% | -5.39% | — |
| **3-sleeve blend @lev 1.0 — measured (LIVE CONFIG)** | **0.68** | **4.76%** | -6.70% | — |
| SPY — same window | 0.78 | 13.0% | -22.1% | — |

**The decision to make:** the deployed blend measures **0.68, below SPY's
0.78**, and below its own SP500 core sleeve (0.97) while barely improving
drawdown (-6.70% vs -7.6%). On this evidence the blend overlay is
*subtracting* value. This is not new — the 2026-06-27 diversification work
independently found the 3-way blend at 1.05 vs the core's 1.12 and
concluded "deploy SP500 core, diversification arc closed"; the
leverage-realistic variant was then adopted anyway on a 1.14 that is now
unreproducible. **Two independent measurements, two months apart, both say
the blend is worse than the core.**

**Resolve these two before flipping live config — do not revert on this
run alone:**
1. **Regime split.** One window, and a bull tape (SPY 13.0% CAGR). A
   low-vol defensive book is *supposed* to lag here. Measure core vs blend
   vs SPY separately in up/down/high-vol regimes. If the blend earns its
   keep only in drawdowns, that is an allocation question, not a revert.
2. **Live ≠ either number.** `paper/overlay.py:68` and
   `paper/signal_runner.py:41-43` instantiate `EnsembleSignal` at **fixed
   defaults**, not WFO-selected combos, so neither row describes the
   trading account. Either wire WFO combo selection into live, or measure
   the fixed-default configuration directly and use *that* as the bar.

**Standing process fix (adopt now, cheap):** `ggt.py lab`'s `--eval-end`
defaults to "now" and drifts, which is why the window behind 1.12/1.14 is
unrecoverable — SPY itself scores 0.58 in the cited runs vs 0.78 here, and
SPY's returns cannot change, which proves the windows differ. **Pin
`--eval-start`/`--eval-end` explicitly on every run whose number will be
cited.** `scripts/position_sizing_wfo.py` now pins the production overlay
params (`target_vol=0.068, window=60, max_leverage=1.0`) and records them
into its results JSON; do the same for any new driver. Related trap, hit
twice: `run_blend`'s `max_leverage` default is **2.0**, not production's
1.0 (`src/ggTrader/lab/blend.py`, now commented).

Live trading continues unchanged in the meantime (Flynn's call, 2026-08-19)
— the account keeps collecting honest data, and the accounting corrections
for the broker's unapplied splits and uncredited dividends are deployed.

---

## RESOLVED / WITHDRAWN (2026-08-20) — the "split/dividend double-apply" bug report was itself wrong

**A previous session queued an ACTIVE STEP here demanding the split and
dividend corrections be ripped out as a double-apply bug. That report was
based on a false premise and has been withdrawn. Do not implement it — the
deployed correction code is correct, and removing it would introduce the
very error the report described.**

The report asserted MNST's position was "textbook post-split: qty 20.8041
(2x pre-split), avg_entry $90.71 (1/2 of $181.42)". MNST never traded at
$181. Re-verified 2026-08-20 against four independent sources:

| Evidence | Finding |
|---|---|
| Alpaca corp-actions feed | MNST forward split, 2-for-1, ex-date 2026-08-11 (confirmed real) |
| yfinance daily bars | close $90.36 on 08-07 -> $45.53 on 08-11; `Stock Splits = 2` on 08-11 |
| `paper_trades` | only two MNST buys, both 2026-08-06, $749.04 + $1138.07 = **$1,887.11** = today's `cost_basis` exactly |
| `paper_snapshots` qty across the ex-date | 08-10 = 20.8041, 08-11 = 20.8041, 08-12 = 20.8041 — **unchanged**, with no trades after 08-06 |

So the position was opened at 20.8041 shares x $90.71 = $1,887.11
**pre-split**, and the broker left `qty` and `avg_entry_price` at those
pre-split values while its price feed moved to the post-split $47.43.
`market_value` = 20.8041 x $47.43 = $986.67 is therefore **half the true
economic value**. The correct book is qty 41.608, market_value ~$1,973.5,
unrealized **+$86 (+4.6%)** — which is what the reporting script produces.
The broker's raw -$900 (-47.7%) is the artifact, not the corrected figure.

The report's reasoning inverted this because it read the *current* qty as
already-doubled without checking what the qty had been before the ex-date.
The snapshot history settles it: nothing about the position changed on
2026-08-11.

**The dividend accrual is also not "fabricated."** The three accrued events
(MPWR 06-30 $5.05, VZ 07-10 $54.51, PNR 07-24 $14.50 = $74.06) are real
corporate actions on positions genuinely held on their ex-dates. Confirmed
that Alpaca paper never credits them: `paper_snapshots.cash` is *flat across
every non-trade day* (63,291.91 on 08-17/18/19; 66,267.89 across 07-15 to
07-29), and `/v2/account/activities/DIV` returns `[]`. The accrual is a
reporting-only adjustment, is labeled as such in the Telegram summary
alongside the broker's uncorrected figure, and never touches cash or
buying power.

**Equity truth (opposite of the withdrawn report):** the broker figure
*understates* the account. Reported ~$104,921 (08-19) = broker $103,857
+ $987 unapplied MNST split + $74 dividend accrual.

### The one real defect this audit did surface (still open)

`AlpacaBroker._get_split_activity_symbols` reads
`/v2/account/activities/SPLIT` to decide whether a split was *already*
applied — and that endpoint returns `[]` on this paper account even for
MNST's real split. The guard is therefore a **no-op in paper**: it has
never once suppressed a correction, and its protective behavior is
completely untested. If Alpaca ever does apply a split correctly, today's
code would double it — the exact failure the withdrawn report imagined it
had found.

Harden it with evidence that actually exists in paper: compare `qty` in
`paper_snapshots` immediately before vs. after the ex-date, cross-checked
against `paper_trades` to rule out a trim. qty roughly x factor => applied,
suppress. qty unchanged => unapplied, correct it. Keep the activities feed
as a secondary signal, not the primary one.

**Standing lesson (cross-agent):** this bug report was written from a
single point-in-time position snapshot. A split is a *change*, and it
cannot be diagnosed without the before-state. Any future corporate-action
claim must cite `paper_snapshots` across the ex-date plus `paper_trades`,
not just today's position.

---

## Historical context (pre-2026-08-19)

Since the
July 16 leveraged-ETF closures, ten research arcs have closed NO-GO:
market-neutral pairs/stat-arb (July 17, first of `RESEARCH_SNAPSHOT.md`
§6's 4 internal candidates), the MAX-effect quintile filter (July 17,
candidate #11), the free-data-only short-interest cut (July 17, candidate
#3), PEAD (July 17, candidate #12), the S&P 500 index-deletion overshoot
fade (July 17, candidate #13 — the fastest build of the session, but also
the worst outcome: MaxDD -68.7%), insider cluster-buying (July 19,
candidate #1 — the highest-effort build, ~24-hour SEC Form 4 backfill),
Congressional trade mirroring (July 19, candidate #9 — see note below, the
third and clearest confirmation of the eval-window-drift pattern), and
short-volume-ratio/"stealthy shorts" free-data cut (July 19, candidate #5
— a clean "no signal at this fidelity" rejection, not an eval-window or
overfitting one). See `roadmap.md` §3 and
`docs/research/2026-07-16-leveraged-index-rotation-nogo.md` /
`docs/research/2026-07-16-leveraged-trend-following-nogo.md` /
`docs/research/2026-07-17-pairs-stat-arb-nogo.md` /
`docs/research/2026-07-17-max-effect-nogo.md` /
`docs/research/2026-07-17-short-interest-nogo.md` /
`docs/research/2026-07-17-pead-nogo.md` /
`docs/research/2026-07-17-index-deletion-fade-nogo.md` /
`docs/research/2026-07-19-insider-cluster-buy-nogo.md` /
`docs/research/2026-07-19-congress-trades-nogo.md` /
`docs/research/2026-07-19-short-volume-ratio-nogo.md`.

**The eval-window-drift pattern is now confirmed three times and should be
treated as a standing expectation, not a caveat.** PEAD (July 17) first
showed it: a long-window "beats SPY" result that erased on the deployed
blend's matched 2021-2026 window and hurt the blend (1.14→1.06).
Insider cluster-buying (July 19) confirmed it independently (long-window
near-tie + lowest core-correlation seen, matched-window Sharpe 0.39 vs SPY
0.58, blend 1.14→1.12). **Congressional trade mirroring (July 19) then
produced the STRONGEST long-window result of the entire session** (Sharpe
0.89 vs SPY 0.77, gate pass 85%, stability 33% — all highest seen) **and
still failed identically**: matched-window Sharpe 0.36 vs SPY 0.58, and
the worst blend degradation of the three (1.14→1.04). Three consecutive
diversification-sleeve candidates, three different mechanisms (earnings
drift, insider intent, political access), three identical failures.
**Any future candidate showing a standalone "beats SPY" or "low
correlation" result must be re-verified on the deployed blend's exact eval
window (and, if still promising, an actual blend test) before being
reported as promising, no exceptions** — and per the closure report's own
recommendation, consider whether the deployed 3-sleeve blend construction
itself may be near a local optimum that new equity-signal sleeves aren't
going to move, rather than treating each rejection as isolated evidence
about only that one candidate.

**Candidate #2 (analyst estimate-revision momentum) was checked and found
infeasible for an honest historical WFO** — every free source checked
(`yfinance`'s `eps_trend`/`eps_revisions`/`earnings_estimate`, this
project's other integrated sources) is current-snapshot-only, no queryable
point-in-time history; building it would need a paid I/B/E/S-style feed or
would introduce look-ahead bias. Deprioritized pending a paid-data decision
— skip past it in the effort ordering below.

**Candidate #7 (Anomaly-Driven Demand) was also checked and found
infeasible (2026-07-18)** — same class of blocker as #2. Verified against
the actual Chen & Zimmermann dataset source code
(`openassetpricing` package): the firm-level characteristics data is keyed
purely by CRSP `permno`, no ticker column, and the package's own pipeline
calls a WRDS connection directly for some signals. No free permno-to-ticker
crosswalk exists (confirmed via search — CRSP ticker-history identity data
is a WRDS subscription product). A name-matching heuristic crosswalk was
considered and rejected as a silent-data-corruption risk given ticker
reuse over the dataset's multi-decade span. Deprioritized pending a
WRDS-access decision — skip past it too.

**Candidate #8 (retail-attention factors) is built and tested but PAUSED,
not resolved (2026-07-19)** — `retail_attention` strategy + Google Trends
pipeline (`google_trends_data.py`) are complete, 19 tests passing. The
live backfill (`scripts/google_trends_backfill.py`) hit a Google rate-limit
lockout (429s that didn't clear on retry) partway through — a quick
feasibility spot-check beforehand under-sampled the real constraint. Retry
the backfill in a later session with more conservative pacing (current
script already uses 2s/request; consider longer, or spread across multiple
sessions) once the block has likely cleared. Do not attempt IP rotation.
Skip past it for now — it's not blocking the effort-ordered queue below,
just not runnable this session.

**Candidate #4 (crypto funding-rate carry) was checked and found
infeasible for an honest WFO (2026-07-20)** — same class of blocker as #2
and #7, closing the long-parked internal crypto-carry line
(`RESEARCH_SNAPSHOT.md` §6 Rank 4) on the same evidentiary basis. Kraken
Futures is the only real venue (Binance.US has no perpetual futures at
all, spot-only). Kraken's historical-funding-rates data — verified via
both `ccxt` and Kraken's own native API directly — only retains a rolling
~1 year of hourly records, nowhere near the 12mo-train + 3mo-test-per-fold,
many-folds depth this project's WFO requires. Free third-party aggregators
don't offer bulk historical download; paid ones would still fall short of
this project's usual eval-window convention. Deprioritized pending either
a much longer free-data source or a paid-data decision.

**Candidate #14 (options IV skew) was checked and found infeasible
(2026-07-19)** — same class of blocker as #2/#7/#4. Verified concretely:
`yfinance`'s `option_chain()` takes only a future-expiry selector, no
historical/as-of parameter (current-snapshot-only, like #2's
`eps_trend`/`eps_revisions`); Alpaca's option-bars API (already integrated
here) returned zero data for a real contract in both 2022 and 2024 windows
— Alpaca's options market data only starts in 2024, and even then there's
no historical listed-contracts feed to reconstruct past chains from. Deep
historical options-chain data remains a paid-vendor problem (OptionMetrics
IvyDB), exactly as originally flagged, and the signal may just be
re-deriving #3's already-NO-GO'd borrow-cost signal through a noisier
instrument anyway. No code built.

**First 14-candidate batch is fully resolved** (10 NO-GO, 4 infeasible —
#2/#7/#4/#14, 1 paused — #8). See prior paragraphs above for detail.

**2026-07-19 batch: 25 candidates, all non-US-equity asset classes** —
merged into `WEB_RESEARCH_CANDIDATES.md` from a discovery pass explicitly
scoped away from equities per the pivot recommendation in
`RESEARCH_SNAPSHOT.md` §4/§6 (nine consecutive equity diversification
sleeves had failed). Covers FX, commodities, Treasuries/rates, and crypto.

**Corrected and reorganized a second time (2026-07-19)** following an
independent review of the reformatted draft — the review caught real
citation errors (one paper wrongly called "fabricated," a venue
misattribution, a magnitude error, a reversed-then-re-reversed finding
now flagged contested) and replaced the single "Confidence" score with
four independent ratings (Evidence status / Rule correspondence /
Implementation class / Validation stage — see `WEB_RESEARCH_CANDIDATES.md`
for the full explanation). Candidates are now organized by strategy type
— **A. Active strategy replication queue** (A1-A9), **B. Risk/exposure
overlays** (B1-B4), **C. Portfolio-construction methods** (C1-C2), **D.
Parked hypotheses** — rather than by source-prestige tier. Every
candidate is still at Validation stage: Literature only.

**Most consequential change: the previous #1 pick (direct CIP/basis
harvesting) is explicitly demoted for this project's home-lab/ETF
workflow** — the literature validates the CIP deviation's existence and
cause, not "tilt an FX-ETF carry portfolio using the basis as a signal,"
which is a separate, untested extension. FX ETFs can't implement the
actual documented arbitrage (needs forwards/swaps/funding access this
project doesn't have). **The dynamic FX hedge overlay (A1) is now the
clear top pick** instead.

**A1 (dynamic FX hedge overlay) was built and tested — REJECTED
(2026-07-20)**, despite having the cleanest source-to-rule correspondence
in the whole register. Built `src/ggTrader/lab/fred_data.py` (free FRED
CSV data, no API key, point-in-time correct) + `fx_hedge_overlay` strategy
(carry+PPP-value+trend on EWJ/DXJ + EZU/HEZU hedged/unhedged pairs, the
only currently-active such pairs — others were found delisted 2023-2024).
Standalone WFO Sharpe 0.41 looked weak but SPY isn't the real benchmark
here — the decisive test compared the dynamic strategy against static
hedge-ratio baselines (100% unhedged / 100% hedged / 50-50) on the *same*
instruments, the paper's actual claim. **The dynamic strategy
underperformed every static alternative** (0.41 vs 0.53/0.66/0.73 Sharpe),
not an eval-window or overfitting rejection (WFE 1.10, healthy) — the
signal construction just doesn't add value over doing nothing clever;
simply staying 100% hedged was the best of the four FX configurations
over this window. Full report:
`docs/research/2026-07-20-fx-hedge-overlay-nogo.md`. `fred_data.py` and
the `fx_hedge` universe remain reusable infrastructure for A5/A7 (both
need FRED-adjacent macro data).

**A7 (pre-FOMC long-Treasury drift) was built and tested — REJECTED
(2026-07-20)**, despite a strong, current, directly-on-point citation
(Pan & Peng, June 2026). Built `src/ggTrader/lab/fomc_calendar.py` (free
Fed calendar scraper, 122 scheduled FOMC dates 2011-2026, two source
formats stitched together and cross-checked) + `fomc_drift` strategy
(long TLT/IEF/EDV the day before, exit at/after, `target_kind="signals"`
protocol for exact-bar entries/exits). Found and fixed a real bug on
first run: OHLCV bars carry a non-midnight time-of-day
(`16:00:00+00:00`) while FOMC dates are midnight-normalized, so an
exact-timestamp match silently matched zero events — fixed by comparing
on calendar date, regression-tested. Full 54-fold WFO: **OOS total
return 0.45% over 13.5 years (essentially flat)**, Sharpe 0.10, WFE 0.44
(below the 0.50 overfitting floor), regime halt active (winning combo
selected in only 14/54 folds) — a clean "no exploitable drift" result,
not an eval-window or data-quality artifact. Full report:
`docs/research/2026-07-20-fomc-drift-nogo.md`. `fomc_calendar.py` remains
reusable for any future macro-calendar-timed candidate.

**A3 (commodity medium-term trend) was built and tested — REJECTED
(2026-07-20)**. Built a 14-ETF single-commodity universe (metals/energy/
ags; BAL and JJC checked and found delisted 2023, excluded) +
`commodity_trend` strategy (12-1 cross-sectional momentum, reusing this
lab's established `xs_momentum` pattern, plus a market-wide realized-vol
regime filter — a distinct construction from the already-rejected equity
VIX-regime-throttling idea, tested on its own merits in a different asset
class). Full 50-fold WFO: OOS Sharpe 0.13 vs SPY 0.74, **MaxDD -37.0%
(worse than SPY's -33.7%)** despite the vol-regime filter being
specifically meant to avoid crash exposure, aggregate WFE undefined
(`nan`), regime halt active despite 82% fold-stability for the
recommended combo. Full report:
`docs/research/2026-07-20-commodity-trend-nogo.md`. The `commodity_trend`
universe and code remain reusable for A2 (commodity carry, same universe,
untested) if commodity exposure is revisited — A4 (short-term basis
reversal) is **not** ETF-approximable per the register's own note (needs
futures data this project doesn't have).

**A5 (Treasury term-structure factors, ETF-approximation) was built and
tested — REJECTED (2026-07-20)**. Built `TreasuryCurveStrategy` (FRED
10y-2y curve-slope regime signal, one-hot rotation across SHY/IEF/TLT).
Standalone WFO Sharpe 0.19 looked routinely weak, but per A1's lesson the
decisive test compared it against static duration baselines: **static
100% SHY beats the dynamic strategy on every metric** (Sharpe 0.90 vs
0.19, CAGR 1.5% vs 0.8%, MaxDD -5.7% vs -8.2%) — the same "dynamic loses
to static" pattern as A1, now confirmed twice. Full report:
`docs/research/2026-07-20-treasury-curve-nogo.md`. This closes the
approximation only — the paper's actual 4-factor model (needs cash
Treasuries/STRIPS/futures) remains untested and out of reach at this
project's current data-access tier.

**A8 (headline/LLM sentiment) was built, tested, and CLOSED NO-GO
(2026-07-24)** — provisional-confidence rejection per
`docs/research/2026-07-24-headline-sentiment-nogo.md` (only 3 usable
folds vs. 20-54 elsewhere, direction unfavorable). This closes the first
14-candidate + 25-candidate web-research batches: **all A-series and
first-round B-series items are now resolved or explicitly paused** — see
`docs/roadmap.md`'s 2026-07-24 entry for the B1/B2/B3 feasibility triage
(B1 stablecoin-stress: data exists but crypto trading is dormant, no
exposure to overlay; B2 FOMC country-ETF reaction: blocked project-wide,
every non-crypto symbol is daily-bar-only; B3 bond-ETF NAV dislocation:
no free NAV/premium-discount source found). None of B1-B3 started.

**pairs_stat_arb was re-run on corrected data and re-confirmed NO-GO
(2026-07-26)**, closing the book on that candidate — see
`docs/research/2026-07-17-pairs-stat-arb-nogo.md` §7. Also: a severe
SPY-timestamp data bug (deflating every equity Sharpe ~29% since 2023)
and two smaller bugs were found and fixed 2026-07-25 — see
`RESEARCH_SNAPSHOT.md` §5 for what's now settled vs. still open there.

**Queued next (2026-07-28): Rank 1 cross-asset trend sleeve
(TLT/GLD/DBC).** A new external research deliverable,
`docs/quant_strategy_research_report.md` (12 ranked candidates across FX,
Treasuries, commodities, intl equities, crypto), converges independently
with `RESEARCH_SNAPSHOT.md` §6 on the same pick: a long-only trend/
momentum overlay on liquid non-leveraged ETFs (TLT for duration, GLD/DBC
for commodities) as a 4th blend sleeve. Rationale for the pick: nine
consecutive equity-cross-sectional diversification sleeves have failed
(eval-window-drift or high correlation to core, see `RESEARCH_SNAPSHOT.md`
§4) — both sources agree the next lever needs to be a genuinely different
asset class, not another equity sort. Cheapest, most-corroborated
candidate available: free daily OHLCV only, reuses existing `ggt lab
--blend` infrastructure, and the already-tested `leveraged_trend_*`
strategies prove the trend-filter mechanism works in this codebase (their
NO-GO was leveraged-ETF decay specifically, not the trend logic — TLT/GLD/
DBC are unleveraged). **Not yet built** — next session should copy this
candidate into `docs/research/prompts/local-implementation-prompt-TEMPLATE.md`
and run it through the standard WFO/NDH/DSR gate, same discipline as every
prior candidate. The report's #2 (intl DM equity rotation) and #3
(re-test `xs_momentum`/`dual_momentum` on the current lab stack, a
zero-new-data-cost gap-closer per `RESEARCH_SNAPSHOT.md` §5) are the
next-best picks if #1 doesn't pan out. Also still open, lower priority:
retry #8's Google Trends backfill once the rate-limit has likely cleared.
