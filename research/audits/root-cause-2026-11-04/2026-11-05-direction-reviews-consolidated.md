# Direction reviews of the root-cause investigation, consolidated

**Date:** 2026-11-05 · **Type:** editorial comparison of two direction reviews (no new computation,
no new evidence) · **Status:** reading copy; the originals stay as written and are the sources of
record.

**Abbreviations.** SPY = SPDR S&P 500 ETF Trust (the benchmark). 2x = twice SPY's total return over
the same period. pp/yr = percentage points a year. WFO = walk-forward optimization. OOS = out of
sample. MDE = minimum detectable effect (the smallest true effect a test finds 80% of the time at its
pass bar). PIT = point in time. CAGR = compound annual growth rate. TE = tracking error. DSR =
deflated Sharpe ratio. SSO = ProShares Ultra S&P 500 (2x daily SPY). A23 = the owner's adopted 70%
SPY / 30% SSO allocation. P2-nn = a numbered Phase 2 re-run item in the Phase 1 checkpoint. vbb = the
ensemble's Bollinger-band-plus-volume voter. "Decision 0" = whether re-runs may write to the shared
price table (the owner answered no). Citations below are `file:line` with the date prefix dropped;
line numbers are as committed with this report.

## 1. Sources and status

| Short name | File | Author | Status |
|---|---|---|---|
| Opus review | `research/audits/root-cause-2026-11-04/2026-11-05-senior-quant-direction-review.md` | Claude Opus, single reviewer, read-only (`senior-quant-direction-review.md:258-264`) | Reliable as a single review; its own estimates are labelled "(my estimate)" and unmeasured (`senior-quant-direction-review.md:17-18`) |
| Argon review | `research/audits/root-cause-2026-11-04/2026-11-05-argon-synthesis-review.md` | Gemini 4 Argon via agy (`argon-review-verification.md:3`); did not read the Opus review (`argon-synthesis-review.md:165`) | **Unverified as written.** 43 of 105 citations are broken or unsupported (`argon-review-verification.md:10`); several specifics are invented (`argon-review-verification.md:13`) |
| Verification | `research/audits/root-cause-2026-11-04/2026-11-05-argon-review-verification.md` | Claude Sonnet, did not read the Opus review (`argon-review-verification.md:3`) | States which Argon items survive (`argon-review-verification.md:90-105`) |

Both reviews assess the synthesis (`research/audits/root-cause-2026-11-04/2026-11-04-root-cause-investigation.md`)
and its challenge (`research/audits/root-cause-2026-11-04/2026-11-04-root-cause-challenge.md`). Below, an Argon
statement is used only where the verification marks it as surviving or corrects it; the corrected
form is the one shown. Anything else is labelled **not verified**.

## 2. Side by side

| Topic | Opus | Argon | Verification disposition | Status |
|---|---|---|---|---|
| 2x with the built design (about 85% SPY) | Negative, high confidence: the replica could detect 4.24 pp/yr against gaps of 9.57 and 6.14 (`senior-quant-direction-review.md:37-45`) | Same: the replica was well powered against 2x (`argon-synthesis-review.md:32`) | True but already in Lens 9 and the synthesis (`argon-review-verification.md:64`) | agree |
| Paper-trader reliability | First priority: measure the partial bar, propose an input archive and a broker-based halt base, document MNST/AVB as broker artifacts, leave the cap, sizing and parameters alone (`senior-quant-direction-review.md:93-111`) | A parity and tax-hygiene patch, a dated freeze and a clock reset (`argon-synthesis-review.md:76-86`) | Divergences real (Lens 6 rows 1, 4, 8, 14); "recovers 0.85-1.00 pp/yr" wrong; freeze and reset dates invented (`argon-review-verification.md:99`, `argon-review-verification.md:54`) | agree on direction; Argon specifics dropped |
| The 29 unrun Phase 2 checks | No-database items first; defer price-table writers until Decision 0 (`senior-quant-direction-review.md:113-123`) | Same intent, but wrong item numbers and a non-existent `write_db` guard (`argon-synthesis-review.md:67-68`) | Intent matches Decision 0; ids mislabelled; no guard exists (`argon-review-verification.md:97`, `argon-review-verification.md:74-84`) | agree on intent; Argon ids dropped |
| Stock-picking power | "Undecided" at 1 to 4 pp/yr, not at 2x; detecting +1.0 pp/yr at TE 6.5% needs about 620 years (Opus estimate, not verified) (`senior-quant-direction-review.md:54-59`, `senior-quant-direction-review.md:126-136`) | Re-power with a 1994-2026 concentrated sleeve; +1.5 to +4.0 pp/yr; 15-25% chance of t ≥ 3 (`argon-synthesis-review.md:88-99`) | sqrt(T) arithmetic verified; payoff odds dropped (about 1% to 9%); free-data plan fails (`argon-review-verification.md:57`, `argon-review-verification.md:100-101`) | differ |
| Gated-WFO machinery | Lens 2's rules came in one commit; other rules could create a few reliable NOs, not a GO (`senior-quant-direction-review.md:85-89`) | The synthesis's "biggest blind spot" (`argon-synthesis-review.md:50-53`) | Survives, but it is already the synthesis's position; synthetic-model caveat (`argon-review-verification.md:92`) | differ in weight |
| Leverage | Exposure, not skill; 2.0x at today-like funding about +1.65 pp/yr (Opus estimate, not verified); no allocation advice (`senior-quant-direction-review.md:61-70`, `senior-quant-direction-review.md:148-162`) | Same framing; 2.0x "+1.5 to +2.5" (`argon-synthesis-review.md:106-108`) | 1.3x figures verified; 2.0x is inference, "about +1.7, unmeasured" (`argon-review-verification.md:58-59`) | agree |
| Tax | After-tax as the primary metric for short-holding designs (`senior-quant-direction-review.md:178-179`); annual rebalancing realizes fewer short-term gains (inference, `senior-quant-direction-review.md:140-143`) | 52 SPY sweep trades churn a taxable account; holds over 366 days cut the hurdle (`argon-synthesis-review.md:48`, `argon-synthesis-review.md:92`) | Trade count verified; wash-sale harm unestablished; "1-2 pp/yr" unsourced; holding-period saving about 1.9 to 4.3 pp/yr (`argon-review-verification.md:50`, `argon-review-verification.md:102`) | Argon-only (sweep, corrected); agree in kind (holding period) |
| Partial-bar volume | Measure with P2-14 (`senior-quant-direction-review.md:99-100`) | Partial volume suppresses vbb entries (`argon-synthesis-review.md:56`) | Direction plausible; 24-28% of volume missing in two samples; effect unmeasured (`argon-review-verification.md:62`) | agree on measuring |
| Crypto | Did not beat BTC over three years; low priority (`senior-quant-direction-review.md:78-79`, `senior-quant-direction-review.md:171-173`) | Park active crypto (`argon-synthesis-review.md:116`) | Figures verified (`argon-review-verification.md:60`) | agree |
| Intraday | Let the observer finish; do not read early (`senior-quant-direction-review.md:164-167`) | Leave it running; dates, timer and trade count stated (`argon-synthesis-review.md:114`) | Window is 2026-10-30 to 2027-11-03, four timers, "0 completed trades" unsourced (`argon-review-verification.md:61`) | agree on action; Argon facts dropped |

## 3. Consensus

- The built design cannot reach 2x, and no post-1994 stock-picking result comes close
  (`senior-quant-direction-review.md:22-24`); Argon's matching claim that the replica was powered
  against 2x is verified (`argon-review-verification.md:64`).
- The only measured route to a much larger raw return is more market exposure, and that is a risk
  decision for the owner, not a research finding (`senior-quant-direction-review.md:31-33`;
  `argon-review-verification.md:103`).
- Paper-trader parity comes before alpha claims, and the trader stays frozen: Opus frames changes as
  proposals through the deploy checks (`senior-quant-direction-review.md:107-108`); the verification
  says measurement only (`argon-review-verification.md:124`).
- Phase 2 items that need no database come first; items that write to the price table wait for
  Decision 0 (`senior-quant-direction-review.md:113-123`; `argon-review-verification.md:121-122`).
- Crypto spot strategies are low priority (`senior-quant-direction-review.md:171-173`;
  `argon-review-verification.md:104`).

## 4. Disagreements and the evidence that settles them

**Stock-picking prospects.** Argon expects +1.5 to +4.0 pp/yr from a concentrated, low-turnover
sleeve. The verification drops the odds: with Argon's own MDE of 9.1 pp/yr, the chance of t ≥ 3 is
about 1% at +1.5 and 9% at +4.0 (`argon-review-verification.md:101`). The data plan also fails:
free stock-level PIT fundamentals start in 2009 and the membership file in 1996
(`argon-review-verification.md:100`). The momentum it leans on is +0.08 pp/yr post-1994 for big
stocks (`argon-review-verification.md:100`). Settled toward Opus: best realistic outcome about
+1 pp/yr, not 2x (`senior-quant-direction-review.md:137-140`).

**How much the gated WFO matters.** The 50 gated rows and 48 without an ungated arm are verified
(`argon-review-verification.md:48`). But the 87.5% halt rate comes from a synthetic model with
assumed inputs that "says nothing about any real strategy" (`argon-review-verification.md:47`), and
in the two checkable cases removing the gate changed no verdict (`argon-review-verification.md:48`).
Settled as a real limit the synthesis already states, not as a demonstrated cause; whether the
gates hid or manufactured any result stays unknown (`root-cause-challenge.md:266-267`). Re-running
without gates (P2-26) takes tens of hours and writes to the price table
(`argon-review-verification.md:79`).

**Paper-trader change versus measurement.** Argon proposes a patch with an expected payoff; the
payoff figure is wrong (it counts cumulative 4.4-month broker artifacts, and the MNST split gap is a
paper-broker quirk no trader patch recovers) and its dates are invented
(`argon-review-verification.md:99`). Settled toward Opus: measure, document and propose; payoff
"none in raw return" (`senior-quant-direction-review.md:106`).

## 5. Corrections

From the verification (Argon):
- Replica primary is 14.19% vs 14.83% and holdout t is 0.69, not 14.41% vs 15.05% and t 0.89
  (`argon-review-verification.md:51`).
- The DSR fix is commit 09f9f401; the two hashes Argon gives do not exist
  (`argon-review-verification.md:49`).
- The 2026-11-04 incident rewrote at most 27,579 rows for 19 symbols through the cached loader; the
  "166,431 rows" and `write_db` story is invented (`argon-review-verification.md:53`).
- TaxLotBook has 40 tests, not zero (`argon-review-verification.md:55`).
- No Phase 2 item was "halted"; none was approved or run (`argon-review-verification.md:84`).
- The replica sorts candidates alphabetically, not by 20-day return
  (`argon-review-verification.md:63`).
- "+2.3 pp/yr over 20 to 30 years" doubles wealth only over 32 years; 20 years needs 3.55, and the
  window's funding averaged 1.64%, not near zero (`argon-review-verification.md:39`).
- "50 negative WFO verdicts" includes GO rows (`argon-review-verification.md:48`).

From the synthesis (Opus):
- Opus's decision "accept and merge the synthesis" (`senior-quant-direction-review.md:219`) is
  partly overtaken: the synthesis records a note "added 2026-11-05 after the merge"
  (`root-cause-investigation.md:725-726`). Acceptance is still open (`root-cause-investigation.md:736`).
- Opus's run lists (`senior-quant-direction-review.md:190`) sit against the recorded decisions:
  no Phase 2 item is run and Decision 0 is no (`root-cause-investigation.md:729-730`).
  They are options for the owner, not approved work.

## 6. Options for the owner (ranked, at most three)

Standing rules: no Phase 2 item runs unless the owner revisits that decision; no writes to the price
table; the paper trader stays frozen; free data only; taxable account only. A NO-GO from any option
is a result, not a reason to stop the work.

**1. Paper-trader measurement and parity.**
- *Question:* do live decisions and values differ from the replica because of the partial bar, the
  dividend-inflated halt base or broker artifacts (`senior-quant-direction-review.md:94-98`)?
- *Cheapest admissible step:* write the parity proposal (archive, halt base, bar choice) and record
  MNST and AVB as paper-broker artifacts, from committed reports only
  (`senior-quant-direction-review.md:195`, `senior-quant-direction-review.md:102-103`).
- *Prerequisites:* to measure, the owner allows P2-14 and P2-05 (read-only, about 1 hour and 15
  minutes) (`senior-quant-direction-review.md:99-104`); any live change goes through the deploy checks.
- *Falsifier:* P2-14 finds partial-bar and final-bar decisions agree on all but a handful of 30
  sessions (Gate B, `senior-quant-direction-review.md:192-194`).
- *Stop gate:* "parity measured and archived" (`senior-quant-direction-review.md:108-109`).

**2. Funding-honest leverage figures for the 2027-09-30 A23 review.**
- *Question:* does 2.0x still reach the 2006-2026 gap at today-like funding
  (`senior-quant-direction-review.md:118-119`)?
- *Cheapest admissible step:* none beyond restating existing figures with their funding caveat;
  the Opus estimate (about +1.65 pp/yr) stays labelled unmeasured and not verified
  (`senior-quant-direction-review.md:67-70`).
- *Prerequisites:* the owner allows P2-04 (minutes, frozen inputs, no database)
  (`senior-quant-direction-review.md:114`); allocation stays the owner's call
  (`senior-quant-direction-review.md:160-161`).
- *Falsifier:* P2-04 shows 2.0x clearing the gap at +2.5 points of funding
  (`senior-quant-direction-review.md:231-232`).
- *Stop gate:* Gate A, record that no tested exposure reaches 2x at current rates
  (`senior-quant-direction-review.md:191-193`).

**3. One pre-registered long-history stock screen (accruals).**
- *Question:* can a single registered screen show about +1 pp/yr, and can the design resolve its bar
  at all (`senior-quant-direction-review.md:119`, `senior-quant-direction-review.md:137-140`)?
- *Cheapest admissible step:* write the brief and state the power limit before any run
  (`senior-quant-direction-review.md:139-140`).
- *Prerequisites:* a storage location for free downloads (`senior-quant-direction-review.md:224`);
  Part A on French portfolio proxies, since free stock-level fundamentals start in 2009
  (`argon-review-verification.md:125`); P2-02 needs the owner's allowance.
- *Falsifier:* P2-02 shows a French-screen MDE at or below +1.0 pp/yr, which would make Part A more
  decisive than Opus expects (`senior-quant-direction-review.md:233-234`).
- *Stop gate:* a Part A fail closes it; a B0 result below -1.0 pp/yr after tax is the kill
  (`senior-quant-direction-review.md:200-201`, `senior-quant-direction-review.md:144-146`).

## 7. Owner decisions (one list, with defaults)

| # | Decision | Default until the owner answers | Unlocks |
|---|---|---|---|
| 1 | Accept the synthesis (`senior-quant-direction-review.md:219`; `root-cause-investigation.md:736`) | Open; nothing changes | Item 2 |
| 2 | Changelog entry and research-snapshot update (`root-cause-investigation.md:733`, `root-cause-investigation.md:739-740`) | Wait for item 1 | Updated roster |
| 3 | Any Phase 2 item, including the seven no-database items (`senior-quant-direction-review.md:220`; `root-cause-investigation.md:743`) | None run (`root-cause-investigation.md:729`) | Options 1 to 3 beyond paperwork |
| 4 | A separate research copy of the price table, as an alternative to Decision 0 (`senior-quant-direction-review.md:221`) | Decision 0 stays no (`root-cause-investigation.md:730`) | Deferred price-table writers (`senior-quant-direction-review.md:120-122`) |
| 5 | Input-archive proposal for the frozen trader (`senior-quant-direction-review.md:222`) | No live change | A per-run input archive (`senior-quant-direction-review.md:101`) |
| 6 | Daily-loss halt base proposal (`senior-quant-direction-review.md:223`) | No live change | A halt computed from the broker value (`senior-quant-direction-review.md:102-103`) |
| 7 | Storage for new free downloads (`senior-quant-direction-review.md:224`) | None | P2-06, accruals |
| 8 | A yearly research budget (`senior-quant-direction-review.md:225`) | None set | Holm control across registered tests (`senior-quant-direction-review.md:180-181`) |
| 9 | Funding-honest leverage figures before the 2027-09-30 review, including A23 funding and tax records (P2-18) (`senior-quant-direction-review.md:226-227`; `root-cause-investigation.md:741-742`) | Review on its date with existing figures | Option 2 |
| 10 | Compute realized gain or loss per SPY sweep sale (read-only) (`argon-review-verification.md:123`) | Not done | A real answer on sweep tax churn |
| 11 | Whether closed momentum and factor NO-GOs may be reopened for a sleeve (`argon-review-verification.md:125`) | Stay closed | A wider option 3 |
| 12 | Deferred minor review notes in the ledger (`root-cause-investigation.md:744`) | Stay deferred | Nothing material |

## 8. What to ignore in the Argon review

All dropped or shown wrong by the verification (`argon-review-verification.md:117`):
- Its P2 item numbers and "skip the remaining 17 items" (`argon-review-verification.md:74-82`).
- The `write_db=False` guard, `fetch_tiingo_ohlcv`, `refresh_universe_history` and the 166,431-row
  incident (`argon-review-verification.md:53`, `argon-review-verification.md:63`).
- The 2027-05-02 freeze and the 2026-11-15 clock reset; the record says unresolved
  (`argon-review-verification.md:54`, `argon-review-verification.md:128`).
- "TaxLotBook has zero tests" (`argon-review-verification.md:55`).
- The NVDA/META story and "1.12 collapsing to -0.17" (`argon-review-verification.md:52`).
- The EDGAR plan, ids a05/a19 and `sp500_pit_constituents.csv` (`argon-review-verification.md:63`).
- "+1.5 to +4.0 pp/yr" and "15-25% to clear t ≥ 3" (`argon-review-verification.md:101`).
- "Recovers 0.85-1.00 pp/yr" and "1-2 pp/yr" of wash-sale leakage
  (`argon-review-verification.md:99`, `argon-review-verification.md:50`).
- "0 completed trades" and a single observer timer (`argon-review-verification.md:61`).
- The 30/90/180-day plan, built on the invented dates and data plan
  (`argon-review-verification.md:105`).
- The 19 cited paths that do not exist (`argon-review-verification.md:20-33`).

## Limits

Both reviews read the same synthesis, so their agreement is not independent evidence
(`argon-synthesis-review.md:12`). The Opus review is one reviewer with no second model family
(`senior-quant-direction-review.md:258-259`), and nobody verified it line by line the way the
verification checked Argon. This report adds no computation; every Opus estimate it repeats is
unmeasured.
