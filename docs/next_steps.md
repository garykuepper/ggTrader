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

## ACTIVE STEP (2026-09-22) — verify the core-revert deploy, then one change per week

**Deployed 2026-09-22 ~23:00 PT** (merge `8d9d3ef`, image
`ghcr.io/garykuepper/ggtrader:latest` built 2026-09-23 05:59 UTC, live
container recreated and checked):

- **Core revert** — live calls `generate_core_signals()` (SP500 core only).
  The only change that affects what gets traded in this deploy.
- **Persistent split state** — `paper_split_state` holds MNST (2:1, ex
  2026-08-11); the 14-day lookback had expired, so live had stopped
  correcting it on 08-26.
- **Failure trap** in `scripts/paper_trade.sh` (Telegram on nonzero exit)
  and the **benchmark/ETF tape keepalive** (SPY/TLT/GLD/DBC/IEF).
- **Held back:** sweep buy dead-band — `.env` sets
  `SWEEP_BUY_TRIGGER_PCT=0.05` (= reserve, i.e. pre-dead-band behavior; logs
  one expected "dead-band is empty" warning per run).
  `CATASTROPHE_STOP_ENABLED` still unset.
- **One-off data fix:** MNST restated in `paper_snapshots` for 2026-08-11 →
  2026-09-22 (market value, P&L, *and* `portfolio_value`); backup in
  `paper_snapshots_backup_20260922`. Restated NAV 2026-09-22: $104,037
  (broker figure $103,112).

**Next, in order (one per 12:45 PT verification cycle):**

1. **2026-09-23 run — verify the revert** per
   `docs/superpowers/plans/2026-09-16-ops-track-and-tape-restore.md` Task 1
   Step 5: log shows `Weights: sp500=100%`; strategy BUYs ≈ 3.3% of PV and
   SP500-only; MNST corrected in the snapshot; SPY/TLT/GLD/DBC/IEF `ohlcv`
   rows dated 2026-09-22; no failure page.
2. **Week of 2026-09-28 — enable the dead-band:** delete the
   `SWEEP_BUY_TRIGGER_PCT` line (default 0.08). Verify per plan Task 3.
   Expect the daily SPY sell/buy round-trip (every session since 08-25) to
   stop.
3. **Week after — catastrophe stop:** re-check the book first. On
   2026-09-22, HGV (-23.6%) and KNF (-23.1%) sat just inside the -25% floor;
   MNST is no longer a false trigger now that split state is persistent.
4. **Minor, open:** `paper_risk_state.peak_value` ($104,518) is below the
   restated high ($105,591, 2026-08-26), so drawdown reads ~1% shallow.
   Harmless against the halt threshold; decide whether to reset it.
5. **Research — blocked on a harness fix first (2026-09-23):** the
   ensemble_ic/kelly re-run came back invalid, and it exposed that the
   signal-strategy WFO path uses no point-in-time S&P 500 membership:
   14.1% of core entries fire on non-member days (lookahead on future
   additions, plus delisted names like SIVB, whose garbage bar alone
   produced Kelly's "38% CAGR"). Next research step: mask entries by
   constituents history in `sweep_signal_group`, with a regression test,
   then re-baseline the core. The 0.99 is unverified on this point until
   then. See `docs/research/2026-09-23-ic-kelly-rebaseline-invalid.md`.
   Research runs natively and does not interact with live steps 1–3.
---

## SUPERSEDED (2026-09-11 update, deployed 2026-09-22 — see ACTIVE STEP above)
 — fix MNST split state → catastrophe stop → core revert → sweep hysteresis: CODE-COMPLETE ON BRANCH, NOT YET DEPLOYED

**Status as of this run (2026-09-11): all four items below (split-state
persistence + re-verification, the catastrophe-stop changelog note, the
core revert, and cash-sweep hysteresis) are implemented on the
`worktree-paper-trading-remediation` branch**
(`docs/superpowers/plans/2026-09-11-paper-trading-remediation.md`) **and
have passed review, including a final whole-branch fix pass. The branch is
NOT yet merged to `main` and NOT yet deployed** — the live container is
still running the image built from pre-remediation code (still the
3-sleeve blend, still the 14-day rolling split-state lookback,
`CATASTROPHE_STOP_ENABLED` still unset). Deploying requires: merge to
`main` → CI builds `ghcr.io/garykuepper/ggtrader:latest` (~2.5 min) →
`docker compose pull && docker compose up -d` on the live container. Do not
skip this step when picking this file back up — treat "code-complete" and
"live" as distinct until that deploy has actually happened and been
verified (see `daily-trader-check`).

**Below is the original 2026-09-10 sequencing rationale**, kept for the
reasoning record now that the work it describes is done on-branch:

**The 2026-08-22 rollout stalled: only step 1 (sweep) executed.** The week-
of-08-31 core revert never happened — confirmed 2026-09-11 (before this
session's work), `trader.py` still called `generate_blended_signals()`,
three weeks past the planned date — and the 2026-09-09 ops review
(`docs/research/artifacts-2026-09-09-perf-review/ggtrader_paper_review_20260909.md`)
plus the 2026-09-10 full-repo audit
(`docs/research/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md`
§7) found a live-data-integrity blocker that the original sequence didn't
know about and that must go **first**, ahead of the revert:

**`_SPLIT_LOOKBACK_DAYS = 14` (`trader.py:46`) has expired while MNST is
still held.** MNST's broker-side quantity was never doubled for its
2026-08-11 2-for-1 split (Alpaca paper never applied it), and the display-
side correction that papered over this stopped applying on 2026-08-25. As
of the 09-09 review, MNST displays a **fictitious -52% unrealized loss**
(true economic loss is -4.8%; cost basis is split-invariant). Two
consequences: NAV has been understated ~0.87% since 8/26, and — critically
— **arming `CATASTROPHE_STOP_ENABLED` right now would force-sell MNST on
that fictitious loss.** The old "nothing in the book would trigger" note
below is stale as of today's book.

**The re-baseline still settles the blend question independently of the
above** — pinned 17-fold WFO (2021-01-31 → 2026-04-30) on the corrected
tape (`docs/research/_rebaseline_corrected_tape_20260822.json`): SP500
core **Sharpe 0.99 / CAGR 8.0% / MaxDD -7.7%**; 3-sleeve blend @ lev 1.0
(live config) **0.69 / 4.8%**; SPY window-matched **0.78**. Third
independent core-beats-blend result (June: 1.05 vs 1.12; August
pre-correction: 0.68 vs 0.97) — the blend trails SPY too. 11 weeks of live
data (2026-06-23→2026-09-08, +2.03% split-fair vs SPY +4.49%) is
*consistent* with this but is not independent evidence either way — its
Sharpe SE is ±2.21, indistinguishable from zero.

Revised sequence (still one change at a time so attribution stays clean):

1. **DONE 2026-08-22: `CASH_SWEEP_ENABLED=true`.** SPY, 5% reserve, $500
   min clip. Idle cash (~60% of the account) now earns index return. Cash
   sweep churn is real but low-priority (see step 4).
2. **DONE ON BRANCH 2026-09-11, NOT DEPLOYED — fix MNST split-state
   persistence and re-verification.** Replaced the 14-day rolling lookback
   with a durable `paper_split_state` table ("correct until the broker
   applies it or the position closes" semantics), then hardened it in this
   session's fix pass so a persisted, still-held correction keeps being
   re-verified against snapshot evidence even once its ex_date ages out of
   the broker feed's own lookback window (`get_open_split_states()` in
   `persist.py`, wired into `trader._compute_split_corrections`) — without
   this, a belated broker-side split apply could have double-corrected
   forever. One-off NAV restatement for 2026-08-26→09-08 is still pending
   the actual deploy (can't verify against a live table from a worktree).
   This is the acceptance gate for step 3.
3. **DONE ON BRANCH 2026-09-11, NOT ARMED — `CATASTROPHE_STOP_ENABLED`
   changelog note recorded** (commit `c118310`), gated explicitly on step
   2 landing live first. Still **unset** in the live environment — do not
   flip it until step 2's deploy is verified (post-fix, the book's worst
   true position is PNR at -8.1%, so this should be inert at arm time, but
   that has not been confirmed against live data yet).
4. **DONE ON BRANCH 2026-09-11, NOT DEPLOYED — revert live from the
   3-sleeve blend to the SP500 core** (evidence above; this closes the
   2026-08-19 ACTIVE STEP further below). `trader.py` now calls
   `generate_core_signals()`. The live container is still serving the
   pre-revert image until this branch is merged and deployed.
5. **DONE ON BRANCH 2026-09-11, NOT DEPLOYED — hysteresis added to the
   cash sweep.** 15 sweep trades / $82.4K notional across 12 sessions
   oscillating around the 5% reserve (pure spread friction, ~zero expected
   return) prompted a dead-band: buy only fires once cash exceeds 8% of
   portfolio value (`SWEEP_BUY_TRIGGER_PCT`, sweeps back down to the 5%
   reserve once triggered). A coherence warning was added this session
   (`cash_sweep.compute_sweep_buy`) so a misconfigured trigger-below-reserve
   setup logs instead of silently going inert.
6. **Once steps 2-5 are merged and deployed and verified live**, then
   research work can resume: re-run `ensemble_ic` (1.01) and
   `ensemble_kelly` (0.98) against the real 0.99 baseline — both were
   rejected only against the phantom 1.12, so their NO-GOs are void on
   their stated grounds (drawdown/fold-instability may still kill
   `ensemble_ic`). Then the queued TLT/GLD/DBC cross-asset sleeve, then
   `xs_momentum`/`dual_momentum` on the current lab stack (zero-cost gap
   closer, never re-tested post-rewrite). Event-date NO-GOs stay closed
   (they failed *with* a one-day lookahead advantage, see
   `docs/research/RESEARCH_SNAPSHOT.md` §4); `fomc_drift` is the only one
   whose workaround tested the outright wrong day — re-run if the compute
   is worth it. Full ranked list: `RESEARCH_SNAPSHOT.md` §6.

**Also flagged, not yet a queued step:** the 09-09 review found a real
alerting gap (`scripts/daily_pnl_report.sh` invokes a deleted CLI
subcommand and has been disabled since 2026-05-06) — a crash before the
Telegram notifier's first call currently pages nobody, which is how the
7/29-30 leverage-guard halt went unnoticed for a day. Worth a small
SQL-plus-notifier fix and a cron-level "alert on nonzero exit" wrapper
whenever there's a free cycle; not urgent enough to displace 1-5 above.

---

## RESOLVED (2026-08-20/22) — every equity bar in `ohlcv` was stamped one calendar day early

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

### Status: writer fixed, data migrated, **and deployed** — DONE

> **Deployed 2026-08-21 11:33 PDT.** The migration on 08-20 left the *live
> container* still running the pre-fix image (built 08-19 18:49). `src/` is
> not bind-mounted, so it would have re-poisoned the freshly-migrated table
> on that day's 12:45 cron run. Pushed `ef4e15f`, rebuilt via CI, pulled and
> recreated the container, and confirmed the fix is present in the running
> image. **Standing lesson: a data migration is not finished until the code
> that writes that data is deployed.** The `daily-trader-check` skill now
> compares image-created against the last `src/` commit for exactly this.

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

## SUPERSEDED (2026-08-22, decision made — see the top ACTIVE STEP) — (2026-08-19) decide whether to revert live from the 3-sleeve blend to the SP500 core

> **Resolution 2026-08-22:** the corrected-tape re-baseline (core 0.99 /
> blend 0.69 / SPY 0.78) is the third independent core-beats-blend result;
> revert is queued for the week of 2026-08-31 in the rollout above. Kept
> below for the reasoning record.

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
