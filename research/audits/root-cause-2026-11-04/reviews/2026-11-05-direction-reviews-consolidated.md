# Two frontier-model reviews of the root-cause investigation: both say the built design cannot reach 2x

**Date:** 2026-11-05. **Type:** editorial comparison of two direction reviews. It adds no computation and no new evidence. **Status:** reading edition. The originals stay as written and are the sources of record.

This page reports what Claude Opus and a Gemini model (Argon) recommend after reading the investigation, and which parts of the weaker review survived checking. It makes no recommendation of its own. Where this page derived something itself, the text says "derived". Terms are defined at first use and in the reference sheet (`research/audits/root-cause-2026-11-04/reference-sheet.md`), which also maps each check and round name to its file. Citations are `file:line` with the date prefix dropped, and they point at the unchanged originals in this folder.

## 1. What each reviewer recommends (Opus is trusted, Argon is unverified)

Both reviews assess the synthesis (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md`) and its challenge (`research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-challenge.md`). The synthesis is the investigation's write-up. The challenge is the independent audit that argued the opposite case.

**Opus** (`research/audits/root-cause-2026-11-04/reviews/2026-11-05-senior-quant-direction-review.md`) is Claude Opus, one reviewer, read-only (`senior-quant-direction-review.md:258-264`). It is reliable as a single review. Its own estimates are labelled "(my estimate)" and are unmeasured (`senior-quant-direction-review.md:17-18`, `senior-quant-direction-review.md:259-261`, `senior-quant-direction-review.md:70`).

Opus agrees with the synthesis on the point that matters. The built design is about 85% SPY (a fund that tracks the S&P 500 and is the benchmark), so it cannot reach 2x (twice SPY's total return over the same period), and no stock-picking result after 1994 comes within reach of the gap (`senior-quant-direction-review.md:22-24`). Its ranked direction has three parts (`senior-quant-direction-review.md:27-31`):

1. Make the paper trader a trustworthy instrument: an input archive, the halt-base defect, the partial-bar measurement.
2. Run the handful of cheap re-run items that settle the leverage and power questions and write nothing to the price table.
3. Treat further stock picking as a small, pre-registered, long-history screen budget whose realistic best case is about +1 pp/yr, not 2x. A pp/yr is a percentage point a year.

Its fourth recommendation is that leverage is a risk decision for the owner, not a strategy. It makes no allocation recommendation (`senior-quant-direction-review.md:148-162`). It names the synthesis's most likely error: the leverage ceiling is read as more reachable than it is, because the one window that reaches the gap had 1.64% funding (`senior-quant-direction-review.md:25-27`, `senior-quant-direction-review.md:81-85`). At today-like funding its own estimate puts 2.0x at about +1.65 pp/yr against a gap of 3.55, unmeasured (`senior-quant-direction-review.md:61-70`). It also puts four items first that the synthesis lacks: one benchmark everywhere, after-tax as the primary metric for short-holding designs, a fixed yearly research budget, and a separate research copy of the price table (`senior-quant-direction-review.md:175-183`). These return as owner decisions in section 5.

**Argon** (`research/audits/root-cause-2026-11-04/reviews/2026-11-05-argon-synthesis-review.md`) is Gemini 4 Argon run through agy, the Antigravity agent harness (`argon-review-verification.md:3`). It did not read the Opus review (`argon-synthesis-review.md:165`). It is **unverified as written**: 43 of its 105 citations are broken or unsupported (`argon-review-verification.md:10`) and several specifics are invented (`argon-review-verification.md:13`). It ranks five directions (`argon-synthesis-review.md:64-113`):

1. Repair the evaluation machinery (an ungated fixed-rule replay and a tax engine) and run the safe re-run items.
2. Paper-trader reliability, with a staged parity proposal that respects the freeze.
3. Re-power stock picking on free data with a concentrated, low-turnover, long-hold sleeve.
4. Leverage framed strictly as the owner's risk decision.
5. Park small caps, long-short, crypto and options. Leave the intraday observer running.

**The verification** (`research/audits/root-cause-2026-11-04/reviews/2026-11-05-argon-review-verification.md`) is by Claude Sonnet, which did not read the Opus review (`argon-review-verification.md:3`). It states which Argon items survive (`argon-review-verification.md:90-105`):

- **Survives:** the weak spot in the gated walk-forward machinery (walk-forward optimization, WFO: pick settings on a past window, test on the next unseen one, repeat), though the synthesis already holds that position. The finding that +4.37 pp/yr is the plain 12-1 momentum anchor (12-1 momentum: buy the stocks that rose most over the past year, skipping the latest month). The long-hold tax direction. The leverage framing. "Park" for crypto, small caps, long-short and options, as Argon's judgement.
- **Corrected:** sweep tax churn, a bill-ETF or drift-band sweep (an untested option), partial-bar volume, the zero-database-first intent, the parity patch, the concentrated sleeve's data plan (which fails) and the 30, 90 and 180-day plan (built on invented dates).
- **Dropped:** the payoff odds for the concentrated sleeve and the claim that TaxLotBook has no tests.
- **Not verified:** anything else in the Argon file. Below, an Argon statement appears only where the verification marks it as surviving or corrects it, and the corrected form is the one shown.

Argon's numbers that trace to a check report hold. Anything that names a hash, a date, an item id or a file outside that list needs the verification beside it.

## 2. Where they agree: five points, none of them new

- The built design cannot reach 2x, and no post-1994 stock-picking result comes close (`senior-quant-direction-review.md:22-24`). Argon's matching claim, that the production replica (the live rules re-run on history) was powered against 2x, is verified (`argon-review-verification.md:64`).
- The only measured route to a much larger raw return is more market exposure. That is a risk decision for the owner, not a research finding (`senior-quant-direction-review.md:31-33`, `argon-review-verification.md:103`).
- Paper-trader parity comes before alpha claims, and the trader stays frozen. Opus frames changes as proposals through the deploy checks (`senior-quant-direction-review.md:107-108`). The verification records the trader as frozen and the clock-reset question as unresolved (`argon-review-verification.md:95`, `argon-review-verification.md:128`). It treats the partial-bar measurement (`P2-14`) as measurement only (`argon-review-verification.md:124`).
- Re-run items that need no database come first. Items that write to the price table stay deferred while the no-writes rule stands (`senior-quant-direction-review.md:113-123`, `argon-review-verification.md:121-122`).
- Crypto spot strategies are not worth time (`senior-quant-direction-review.md:214-215`, `argon-review-verification.md:104`).

Agreement between two readers of one document is a lower bar than it sounds. See Limits.

## 3. Where they differ and what settles it

**Stock-picking prospects.** Argon expects +1.5 to +4.0 pp/yr from a concentrated, low-turnover sleeve. The verification drops the odds. With Argon's own MDE (minimum detectable effect: the smallest true effect a test finds 80% of the time at its pass bar) of 9.1 pp/yr, the chance of a t of at least 3 is about 1% at +1.5 and 9% at +4.0 (`argon-review-verification.md:101`). A t-statistic is an effect divided by its standard error, and 3 is the usual bar for "probably not luck". The data plan also fails: free stock-level fundamentals that use only what was known on each date (point in time, PIT) start in 2009, and the membership file starts in 1996 (`argon-review-verification.md:100`). The momentum Argon leans on is +0.08 pp/yr after 1994 for big stocks (`argon-review-verification.md:100`). The verified evidence is consistent with Opus. The "about +1 pp/yr, not 2x" figure itself is Opus's unverified judgement (`senior-quant-direction-review.md:137-140`).

**How much the gated WFO matters.** Argon calls it the synthesis's biggest blind spot (`argon-synthesis-review.md:50-53`). The verification confirms 50 gated rows and 48 without an ungated arm (`argon-review-verification.md:48`). But the 87.5% halt rate comes from a synthetic model with assumed inputs that "says nothing about any real strategy" (`argon-review-verification.md:47`), and in the two checkable cases removing the gate changed no verdict (`argon-review-verification.md:48`). Opus sees a smaller risk: the Verdict Trust check's rules arrived in one commit, and other rules could create a few reliable NOs but not a GO (`senior-quant-direction-review.md:85-89`). Settled as a real limit that the synthesis already states, not a demonstrated cause. Whether the gates hid or manufactured any result stays unknown (`root-cause-challenge.md:266-267`). Re-running without gates (`P2-26`) takes tens of hours and writes to the price table (`argon-review-verification.md:79`).

**Paper-trader change versus measurement.** Argon proposes a patch with an expected payoff. The payoff figure is wrong: it counts cumulative 4.4-month broker artifacts, and the MNST split gap (MNST is Monster Beverage) is a paper-broker quirk that no trader patch recovers. Its dates are invented (`argon-review-verification.md:99`). Settled toward Opus: measure, document and propose, with payoff "none in raw return" (`senior-quant-direction-review.md:106`).

The remaining topics (partial-bar volume, tax, momentum, intraday) are in the side-by-side tables in section 6.

### Corrections

From the verification, about Argon:

- The replica's primary result is 14.19% against SPY's 14.83%, with a holdout t of 0.69, not 14.41% against 15.05% and t 0.89 (`argon-review-verification.md:51`).
- The fix for the deflated Sharpe ratio (DSR) bug is commit `09f9f401`. The two hashes Argon gives do not exist (`argon-review-verification.md:49`).
- The 2026-11-04 incident rewrote at most 27,579 rows for 19 symbols through the cached loader. The "166,431 rows" and `write_db` story is invented (`argon-review-verification.md:53`).
- TaxLotBook has 40 tests, not zero (`argon-review-verification.md:55`). A hand-calculable trace check is a fair new item, not a re-run id (`argon-review-verification.md:98`).
- No Optional Re-run List item was "halted". None was approved or run (`argon-review-verification.md:84`).
- The replica sorts candidates alphabetically, not by 20-day return (`argon-review-verification.md:63`).
- "+2.3 pp/yr over 20 to 30 years" doubles wealth only over 32 years. Twenty years needs 3.55 (`argon-review-verification.md:39`).
- "Near-zero interest rates" is wrong for the leverage window. Its funding averaged 1.64% (`argon-review-verification.md:39`).
- "50 negative WFO verdicts" includes GO rows (`argon-review-verification.md:48`).

From the synthesis, about Opus:

- Opus's decision "accept and merge the synthesis" (`senior-quant-direction-review.md:219`) is partly overtaken. The synthesis records a note "added 2026-11-05 after the merge" (`root-cause-investigation.md:725-726`), although its open list still says not merged (`root-cause-investigation.md:737`). The first version of the synthesis is on main as `e4362512` (`root-cause-investigation.md:808`). Acceptance is still open (`root-cause-investigation.md:736`).
- Opus's lists of items to run (`senior-quant-direction-review.md:190`) sit against the recorded decisions: no Optional Re-run List item is run and the no-writes rule is in force (`root-cause-investigation.md:729-730`). They are options for the owner, not approved work.

## 4. Three options, ranked in Opus's order

Standing rules: no Optional Re-run List item runs unless the owner revisits that decision. No writes to the price table. The paper trader stays frozen. Free data only. Taxable account only. A NO-GO from any option is a result, not a reason to stop the work.

The order follows Opus's ranked direction (`senior-quant-direction-review.md:28-31`). Option 1 is Opus's first recommendation. Option 2 carries the purpose of Opus's second recommendation (the leverage item `P2-04`) and its fourth, a risk decision and not a strategy (`senior-quant-direction-review.md:148`). Option 3 is Opus's third.

### Option 1: paper-trader measurement and parity

- *Question:* do live decisions and values differ from the replica because of the partial bar (today's unfinished price bar, which live signals use at 12:45 PT), the dividend-inflated halt base or broker artifacts (`senior-quant-direction-review.md:94-98`)?
- *Cheapest admissible step:* write the parity proposal covering the archive, the halt base and the bar choice (`senior-quant-direction-review.md:195`). Recording MNST and AVB (AvalonBay) as paper-broker artifacts from committed reports alone is derived by this page. Opus ties that step to the attribution read (`P2-05`, `senior-quant-direction-review.md:103-104`).
- *Prerequisites:* to measure, the owner allows the partial-bar measurement (`P2-14`) and the attribution read (`P2-05`). Both are read-only, about 1 hour for the first and 15 minutes for the second (`senior-quant-direction-review.md:99-104`). Any live change goes through the deploy checks.
- *Falsifier (derived by this page, not from Opus):* the partial-bar measurement finding that partial-bar and final-bar decisions agree on nearly all of the 30 sessions would make the bar choice immaterial. It would not falsify the halt-base or archive items, which stand on other evidence. Opus's partial-bar checkpoint says only that a difference on more than a handful of sessions means the proposal must pick one bar for both live and replica (`senior-quant-direction-review.md:192-194`). Opus's own change trigger runs the other way: a large partial-bar divergence moves parity ahead of everything (`senior-quant-direction-review.md:236`).
- *Stop gate:* "parity measured and archived" (`senior-quant-direction-review.md:108-109`).

### Option 2: funding-honest leverage figures for the 2027-09-30 A23 review

A23 is the owner-adopted 70% SPY / 30% SSO allocation. SSO is a ProShares fund that targets twice SPY's daily move. Funding is what borrowed exposure costs, here T-bill rates.

- *Question:* does 2.0x still reach the 2006-2026 gap at today-like funding (`senior-quant-direction-review.md:118-119`)?
- *Cheapest admissible step (derived by this page):* none beyond restating existing figures with their funding caveat. The Opus estimate of about +1.65 pp/yr stays labelled unmeasured and not verified (`senior-quant-direction-review.md:67-70`).
- *Prerequisites:* the owner allows the 2.0x funding re-run (`P2-04`: minutes, frozen inputs, no database) (`senior-quant-direction-review.md:114`). Opus asks first for the owner's drawdown tolerance in dollars (`senior-quant-direction-review.md:161-162`, `senior-quant-direction-review.md:252-253`). The caveat is ruin, not return: drawdowns of 66% to 85% in-window and over 90% in a 1929-style decline (`senior-quant-direction-review.md:156-157`). The 2.0x arm fell 84.8% (`senior-quant-direction-review.md:62`), and the century study's 30% arm fell 91.4% (`senior-quant-direction-review.md:84-85`). Allocation stays the owner's call (`senior-quant-direction-review.md:160-161`).
- *Falsifier:* the 2.0x funding re-run shows 2.0x clearing the gap at +2.5 points of funding. Even then "it does not make it safer" (`senior-quant-direction-review.md:231-232`).
- *Stop gate:* the leverage-reach checkpoint. Record that no tested exposure level reaches 2x at current rates (`senior-quant-direction-review.md:191-193`).

### Option 3: one pre-registered long-history stock screen (accruals)

The accruals screen tests a return pattern linked to how much of a firm's earnings are accounting accruals rather than cash. Pre-registered means the pass bar is written down before any result is seen.

- *Question:* can a single registered screen show about +1 pp/yr, and can the design resolve its bar at all (`senior-quant-direction-review.md:119`, `senior-quant-direction-review.md:137-140`)?
- *Cheapest admissible step:* write the brief and state the power limit before any run (`senior-quant-direction-review.md:139-140`).
- *Prerequisites:* a storage location for free downloads (`senior-quant-direction-review.md:224`). The long-history test (the first stage of the owner-approved forward-paper standard, `research/programme/evidence-standard/2026-10-23-forward-paper-standard.md`) runs on French portfolio proxies (free survivorship-free American portfolio returns back to 1926, published by Professor Kenneth French), since free stock-level fundamentals start in 2009 (`argon-review-verification.md:125`). The French-screen MDE item (`P2-02`) needs the owner's allowance.
- *Falsifier:* the French-screen MDE item shows an MDE at or below +1.0 pp/yr, which would make the long-history test more decisive than Opus expects (`senior-quant-direction-review.md:233-234`).
- *Stop gate:* a long-history-test fail closes it. A kill-only replay (one run on 2021-2026, where a pass proves nothing) that finds a result below -1.0 pp/yr after tax is the kill (`senior-quant-direction-review.md:200-201`, `senior-quant-direction-review.md:144-146`).

**Left out of the options.** Opus's "not worth time" list, apart from crypto spot, which sits in section 2 (`senior-quant-direction-review.md:211-215`). The concentration checkpoint: for the concentrated big-cap momentum item (`P2-06`), a result below +2 pp/yr after 1994 means concentration needs no further work (`senior-quant-direction-review.md:202-203`). The change triggers for that item and for small caps (`senior-quant-direction-review.md:235`, `senior-quant-direction-review.md:238`). Opus's concentration, small-cap and long-short notes (`senior-quant-direction-review.md:167-172`). The note that most of the 51 power failures mean power could not be computed (`senior-quant-direction-review.md:49-53`). The note that a 10 pp/yr sleeve edge takes 23 to 40 years to detect on paper (`senior-quant-direction-review.md:109-111`).

## 5. Owner decisions: 13 items, one list, each with a default

Defaults hold until the owner answers. The no-writes rule is the owner's answer of no to re-runs writing to the shared price cache.

| # | Decision | Default | Unlocks |
|---|---|---|---|
| 1 | Accept the synthesis (`senior-quant-direction-review.md:219`, `root-cause-investigation.md:736`) | Open, nothing changes | Decision 2 |
| 2 | Changelog entry and research-snapshot update (`root-cause-investigation.md:733`, `root-cause-investigation.md:739-740`) | Wait for decision 1 | Updated roster |
| 3 | Any Optional Re-run List item, including the seven that need no database (`senior-quant-direction-review.md:220`, `root-cause-investigation.md:743`) | None run (`root-cause-investigation.md:729`) | Options 1 to 3 beyond paperwork |
| 4 | A separate research copy of the price table, instead of the no-writes rule (`senior-quant-direction-review.md:221`) | The no-writes rule stays (`root-cause-investigation.md:730`) | The deferred price-table writers (`senior-quant-direction-review.md:120-122`) |
| 5 | Input-archive proposal for the frozen trader (`senior-quant-direction-review.md:222`) | No live change | A per-run input archive (`senior-quant-direction-review.md:101`) |
| 6 | Daily-loss halt base proposal (`senior-quant-direction-review.md:223`) | No live change | A halt computed from the broker value (`senior-quant-direction-review.md:102-103`) |
| 7 | Storage for new free downloads (`senior-quant-direction-review.md:224`) | None | Concentrated big-cap momentum item (`P2-06`), accruals |
| 8 | A yearly research budget (`senior-quant-direction-review.md:225`) | None set | Holm control (a correction for testing several ideas at once) across registered tests (`senior-quant-direction-review.md:180-181`) |
| 9 | Funding-honest leverage figures before the 2027-09-30 review, including A23 funding and tax records (`P2-18`) (`senior-quant-direction-review.md:226-227`, `root-cause-investigation.md:741-742`) | Review on its date with existing figures | Option 2 |
| 10 | One benchmark everywhere: SPY total return (price change plus reinvested dividends) against Alpaca's end-of-day equity, plus an exposure-matched SPY/cash book. Opus puts this first (`senior-quant-direction-review.md:175-178`) | Not adopted | The 5% cash reserve (about 0.9 pp/yr) is not read as stock-picking loss |
| 11 | Compute realized gain or loss per SPY sweep sale, read-only (`argon-review-verification.md:123`) | Not done | A real answer on sweep tax churn |
| 12 | Whether closed momentum and factor NO-GOs may be reopened for a sleeve (`argon-review-verification.md:125`) | Stay closed | A wider Option 3 |
| 13 | Deferred minor review notes in the ledger (`root-cause-investigation.md:744`) | Stay deferred | Nothing material |

## 6. Side by side: what each says, and what the verification did to Argon

The first table gives each reviewer's position by topic. The second gives the verification's disposition of the Argon position and whether the two reviewers agree. Dispositions are survives, corrected, dropped or not verified.

| Topic | Opus | Argon (unverified) |
|---|---|---|
| 2x with the built design (about 85% SPY) | Negative, high confidence. The replica could detect 4.24 pp/yr against gaps of 9.57 and 6.14 (`senior-quant-direction-review.md:37-45`) | Same: the replica was well powered against 2x (`argon-synthesis-review.md:32`) |
| Paper-trader reliability | First priority: measure the partial bar, propose an input archive and a broker-based halt base, document MNST and AVB as broker artifacts, leave the cap, sizing and parameters alone (`senior-quant-direction-review.md:93-111`) | A parity and tax-hygiene patch, a dated freeze and a clock reset (`argon-synthesis-review.md:76-86`) |
| The 29 unrun re-run items | No-database items first, defer price-table writers (`senior-quant-direction-review.md:113-123`) | Same intent, but wrong item numbers and a non-existent `write_db` guard (`argon-synthesis-review.md:67-68`) |
| Stock-picking power | "Undecided" at 1 to 4 pp/yr, not at 2x. Detecting +1.0 pp/yr at TE (tracking error: how far a book strays from its benchmark) 6.5% needs about 620 years (Opus estimate, not verified) (`senior-quant-direction-review.md:54-59`, `senior-quant-direction-review.md:126-136`) | Re-power with a 1994-2026 concentrated sleeve. +1.5 to +4.0 pp/yr, 15-25% chance of t at least 3 (`argon-synthesis-review.md:88-99`) |
| Momentum replay (+4.37 pp/yr) | Not addressed | Gates rejected 17 of 17 folds, so +4.37 pp/yr (t 0.76, 95% interval -7.07 to +16.05 on the arithmetic mean excess, not significant) is the plain 12-1 momentum anchor (`argon-synthesis-review.md:51`, `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens4-feasibility-and-ceiling.md:238`) |
| Gated-WFO machinery | The Verdict Trust check's rules came in one commit. Other rules could create a few reliable NOs, not a GO (`senior-quant-direction-review.md:85-89`) | The synthesis's "biggest blind spot" (`argon-synthesis-review.md:50-53`) |
| Leverage | Exposure, not skill. 2.0x at today-like funding about +1.65 pp/yr (Opus estimate, not verified). Drawdowns of 66% to 85% in-window and over 90% in a 1929-style decline. The 2.0x arm fell 84.8%. No allocation advice (`senior-quant-direction-review.md:61-70`, `senior-quant-direction-review.md:148-162`) | Same framing. 2.0x "+1.5 to +2.5" (`argon-synthesis-review.md:106-108`) |
| Tax | After-tax as the primary metric for short-holding designs (`senior-quant-direction-review.md:178-179`). Annual rebalancing realizes fewer short-term gains (inference, `senior-quant-direction-review.md:140-143`) | 52 SPY sweep trades churn a taxable account. Holds over 366 days cut the hurdle (`argon-synthesis-review.md:48`, `argon-synthesis-review.md:92`) |
| Partial-bar volume | Measure it (`P2-14`) (`senior-quant-direction-review.md:99-100`) | Partial volume suppresses entries of vbb (the volume-confirmed Bollinger voter, one of five in the live ensemble). Proposes a code change (`argon-synthesis-review.md:56`, `argon-synthesis-review.md:80`) |
| Crypto | Did not beat BTC (bitcoin) over three years (`senior-quant-direction-review.md:78-79`). Spot strategies "not worth time" (`senior-quant-direction-review.md:214-215`). Beyond spot and options, low (`senior-quant-direction-review.md:171-173`) | Park active crypto (`argon-synthesis-review.md:116`) |
| Intraday | Let the observer finish, do not read early (`senior-quant-direction-review.md:164-167`) | Leave it running. Dates, timer and trade count stated (`argon-synthesis-review.md:114`) |

| Topic | Verification disposition of Argon | Status |
|---|---|---|
| 2x with the built design | Survives. True, but already in the Test Power check and the synthesis (`argon-review-verification.md:64`) | Agree |
| Paper-trader reliability | Corrected. Divergences are real (Code Correctness check rows 1, 4, 8, 14). "Recovers 0.85-1.00 pp/yr" is wrong. Freeze and reset dates dropped as invented (`argon-review-verification.md:99`, `argon-review-verification.md:54`) | Differ: patch versus measure first |
| The 29 unrun re-run items | Corrected. Intent matches the no-writes rule. Ids mislabelled. No guard exists (`argon-review-verification.md:97`, `argon-review-verification.md:74-84`) | Agree on intent. Argon ids dropped |
| Stock-picking power | Square-root-of-time arithmetic survives. Payoff odds dropped (about 1% to 9%). The data plan is corrected, and it fails (`argon-review-verification.md:57`, `argon-review-verification.md:100-101`) | Differ |
| Momentum replay | Survives. The cited commit hashes are wrong (`argon-review-verification.md:49`, `argon-review-verification.md:93`) | Argon only |
| Gated-WFO machinery | Survives, but it is already the synthesis's position. Keep the synthetic-model caveat (`argon-review-verification.md:92`) | Differ in weight |
| Leverage | Survives. The 1.3x figures are verified: with A23 in force the funding-adjusted margin is +0.52 pp/yr. The 2.0x figure is inference, "about +1.7, unmeasured" (`argon-review-verification.md:58-59`, `argon-review-verification.md:127`) | Agree |
| Tax | Sweep churn corrected: trade count verified, wash-sale harm unestablished, "1-2 pp/yr" unsourced. Holding-period direction survives, with a saving of about 1.9 to 4.3 pp/yr. A separate bill-ETF or drift-band sweep is an untested option for the unfrozen shadow design (`argon-review-verification.md:50`, `argon-review-verification.md:102`, `argon-review-verification.md:95`, `argon-review-verification.md:123`) | Argon only for the sweep. Agree in kind on holding period |
| Partial-bar volume | Corrected. Direction plausible, 24-28% of volume missing in two samples, effect unmeasured (`argon-review-verification.md:62`) | Differ. Argon's direction plausible, unmeasured |
| Crypto | Survives as judgement. Figures verified (`argon-review-verification.md:60`) | Agree |
| Intraday | Action survives. Facts dropped: the window is 2026-10-30 to 2027-11-03, there are four timers, and "0 completed trades" is unsourced (`argon-review-verification.md:61`) | Agree on action |

**The two lists of re-run items differ.** Opus's seven no-database items are `P2-01` (regime split from committed output), `P2-02` (French-screen MDE), `P2-04` (2.0x at higher funding), `P2-05` (MNST and AVB attribution read), `P2-07` and `P2-08` (parity reads) and `P2-14` (partial-bar measurement) (`senior-quant-direction-review.md:114-116`). The verification's restatement of Argon's list is `P2-02`, `P2-04`, `P2-06`, `P2-09`, `P2-10` and `P2-14` (`argon-review-verification.md:121`). Opus also defers `P2-27` as a price-table writer (`senior-quant-direction-review.md:120`), which the verification's writer list does not include (`argon-review-verification.md:81`, `argon-review-verification.md:122`). Decision 3 in section 5 uses Opus's list.

## 7. What to ignore in the Argon review

Eleven items, all dropped or shown wrong by the verification (`argon-review-verification.md:117`). You may skip them with a clear conscience.

- Its re-run item numbers and "skip the remaining 17 items" (`argon-review-verification.md:74-82`).
- The `write_db=False` guard, `fetch_tiingo_ohlcv`, `refresh_universe_history` and the 166,431-row incident (`argon-review-verification.md:53`, `argon-review-verification.md:63`).
- The 2027-05-02 freeze and the 2026-11-15 clock reset. The record says unresolved (`argon-review-verification.md:54`, `argon-review-verification.md:128`).
- "TaxLotBook has zero tests" (`argon-review-verification.md:55`).
- The `NVDA` and `META` story and "1.12 collapsing to -0.17" (`argon-review-verification.md:52`).
- The EDGAR plan (EDGAR is the American Securities and Exchange Commission's public filings database), the ids `a05` and `a19`, and `sp500_pit_constituents.csv` (`argon-review-verification.md:63`).
- "+1.5 to +4.0 pp/yr" and "15-25% to clear t at least 3" (`argon-review-verification.md:101`).
- "Recovers 0.85-1.00 pp/yr" and "1-2 pp/yr" of wash-sale leakage (`argon-review-verification.md:99`, `argon-review-verification.md:50`).
- "0 completed trades" and a single observer timer (`argon-review-verification.md:61`).
- Argon's 30, 90 and 180-day plan, built on the invented dates and data plan (`argon-review-verification.md:105`).
- The 19 cited paths that do not exist (`argon-review-verification.md:20-33`).

## Limits

This page's own inference: both reviews assessed the same synthesis and challenge (`senior-quant-direction-review.md:5-8`, `argon-synthesis-review.md:12`), so their agreement is not independent evidence. It is one opinion in two hats.

The Opus review is one reviewer with no second model family (`senior-quant-direction-review.md:258-259`), and nobody verified it line by line the way the verification checked Argon. This page adds no computation, and every Opus estimate it repeats is unmeasured.

The verification's line numbers for the Argon file (the numbers it writes with an "L") predate the banner and the pointer line at the top of that file. Add 7 to get current line numbers.
