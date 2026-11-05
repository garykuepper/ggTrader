# Lens 1: Backtest validity (stocks): what contamination remains after the five known fixes

**Erratum 2026-11-04 (after review):** PARA price figures quoted in this report were measured
before an accidental database refresh at 23:48 PST; they are not reproducible from the database
now (see `docs/research/2026-11-04-phase1-checkpoint.md`, section on the incident).

**Date:** 2026-11-04 · **Branch:** `investigation/root-cause-2026-11-04` · **Type:** diagnosis
only (read-only; no re-runs of strategies) · **Spec:**
`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md` (section 5, Lens 1)
· **Per-row output:** `docs/research/_lens1_tape_state.json` (144 rows)

Abbreviations, expanded once: WFO = walk-forward optimization (train on a past window, test
on the next, repeat); OOS = out-of-sample; PIT = point-in-time (only using index
membership known on that day); DSR = deflated Sharpe ratio (a significance gate that
discounts for the number of parameter combinations tried); NDH = the lab's parameter
neighbourhood-density gate; CAGR = compound annual growth rate; tape = the `ohlcv` price
table in TimescaleDB that every stock backtest reads.

## Summary for the owner

1. **The five known bugs are fixed in the tape and code today.** Current-state checks on
   2026-11-04 found every stock bar stamped at midnight, Monday to Friday only, zero
   duplicate days for any symbol, and SPY with exactly one row per date (6,751 rows,
   6,751 dates). Recent SPY closes match yfinance exactly.
2. **Almost every number from before 2026-10-19 predates at least one fix.** Of the 144
   inventory rows, 55 are marked contaminated, 18 clean, and 71 unknown (most of the
   unknowns are design notes, operations changes or crypto, which have no stock backtest).
   Every WFO result run before 2026-10-19 predates the DSR units fix. That includes the
   "corrected" core figure (Sharpe 0.59 vs SPY 0.78, 2026-09-23). Phase 0 already said its
   "17/17 gates" count predates the fix (`docs/research/2026-11-04-phase0-trust-audit.md:30`).
   What Lens 1 adds is that the fix also changes which parameters each fold deploys: a
   failed DSR check makes the fold run on the anchor parameters (`gates_passed = ndh_passed
   and dsr_passed`, then `deploy_params = anchor.params`, lines 726-733 of
   `src/ggTrader/lab/wfo.py` at 09f9f40). So the fix changes the out-of-sample returns, not
   only the gate counts.
3. **Contamination not covered by the five fixes remains in the tape:**
   - **PARA is the wrong security** for its whole S&P 500 membership (2022-02-17 to
     2025-07-23). Over that span the tape's closes run from 72.20 to 113,900: near
     $100,000 through 2023 Q3, then falling from about 25,700 (2024 Q1) to about 42
     (2025 Q3). Median daily volume is 9 shares. The production replica, the most trusted stock result, traded it 3 times,
     and its taint detector did not flag it. Effect on the result: unmeasured.
   - **Stored prices mix dividend and split adjustment bases.** The cache never
     re-adjusts old bars, so where a later fetch overlaps an earlier one the tape shows a
     false one-day move. VZ shows a false -1.7% on 2026-07-20 (yfinance: -0.2%). SPY bars
     through at least 2026-07-24 sit 0.25% above today's yfinance series. APH and MNST
     were off by 2x before a manual repair on 2026-09-29.
   - **AVB still tracks EQR's price from 2026-07-20.**
   - **14 index members have no tape in the 2021-2026 window,** even after the lab's
     ticker normalisation (which maps BRK.B to BRK-B and BF.B to BF-B; both of those have
     full tape). The 14 include VIAC, ABC and ANTM (renamed tickers).
   - **The 2010-2020 tape has identity faults:** 5,595 daily moves of 40% or more across
     217 symbols, against 572 across 97 symbols from 2021.
4. **No current headline number rests on a drifting window.** The current headlines all
   pin their windows: the replica, the PIT core, A23, the momentum replay and the French
   screens. The window behind 1.12 cannot be recovered (`--eval-end` defaulted to "now").
   Project notes say the same of 1.14, but the roadmap records 1.14 as a pinned
   2021-01-31 to 2026-04-30 run (`docs/roadmap.md:158-170`). That points to code fixes,
   not window drift, as the reason it failed to reproduce. This rests on prose only, and
   data-side changes between 07-13 and 08-19 have not been ruled out: the SPY-duplicate
   collapse (1b9b7f5) and refetch adjustment seams. The July event-study NO-GOs used windows that
   ran "to present" and cannot be reproduced exactly.
5. **The backtests and the paper ledger do not measure returns the same way.** Backtests
   use adjusted (total-return) closes, apart from the adjustment seams above. The paper
   account fills at raw prices, Alpaca's paper account credits no dividends, and the code
   adds a dividend accrual ($320.24 so far) to the snapshot value only. Paper-vs-SPY
   comparisons have sometimes used SPY price return.

## Falsifiers

Written as overturn conditions for each Phase 0 hypothesis this lens touches
(`docs/research/2026-11-04-phase0-trust-audit.md`; spec section 4). They were drafted at
the start of the work, before the database queries were run; some prior reading of the
memory notes and changelog had already happened, so they are not fully blind.

- **H3 ("the early beat-SPY figures were produced by five bugs; corrected core 0.59 vs
  0.78; gate counts before 2026-10-19 void").**
  - Overturned if a headline beat-SPY figure (1.12, 0.99, 0.97, 1.14, 16% CAGR) turns out
    to have been run after the fix for each bug that affects its kind of result.
  - Overturned if the current tape still shows the bug signatures: weekend bars,
    non-midnight stamps, duplicate SPY dates.
  - Weakened if "0.59" itself turns out to predate a relevant fix, so that it is not the
    fully corrected figure.
  - The claim that the five bugs *alone* caused the drop from 1.12 is overturned if
    another documented change (window, voter set, leverage) moved the number as well.
- **H1, Lens 1 part ("sound tests mostly returned NOs"; replica rated reliable).**
  Overturned for the replica if its tape inputs were contaminated in a way the run did
  not detect, for example misidentified securities that it traded.
- **H4, Lens 1 part ("there is no clean holdout").** Overturned if the 2010-2020 tape
  passes an identity check: no implausible price levels, and moves of 40% or more limited
  to known corporate events.
- **Phase 0 trust-table claim ("July event studies: two known bugs favoured these
  strategies and they still lost").** Confirmed only if code shows the day-shift join gave
  those studies lookahead (an advantage) rather than delay.

## Steelman

The strongest evidence that the corrected pipeline's numbers are sound:

- **The bug signatures are gone.**
  - Weekday census: Monday 1,078,958 / Tuesday 1,182,620 / Wednesday 1,180,572 /
    Thursday 1,157,959 / Friday 1,152,640 bars, and no weekend bars.
  - Hour of day: all 5,752,749 yfinance daily bars are at 00:00.
  - Duplicates: zero duplicate (symbol, timestamp) keys and zero duplicate calendar days
    for any symbol.
  - SPY: 1,318 bars on the pinned window 2021-01-31 to 2026-04-30, one per date.
- **Benchmark agreement.**
  - The corrected SPY Sharpe stated in ef4e15f (0.914, 2021-2026) agrees with an
    independent plain-pandas check, 0.91 on 2021-02 to 2026-04
    (`docs/research/WEB_RESEARCH_CANDIDATES.md:1557-1567`).
  - DB SPY closes for 2026-10-29 to 11-03 equal yfinance's final closes exactly.
- **The fixes are traceable.** Each fix has a commit with regression tests that fail on
  the old code (ef4e15f, 1b9b7f5, 2eafab1, c9d1abe, 09f9f40). The tape migration kept
  its backup (table `ohlcv_yf_preshift_backup_20260820` exists).
- **Current headlines use pinned windows and run after the relevant fixes.**
  - The production replica (2026-10-19 19:06, after 09f9f40 at 16:04 and 7320c1d at
    17:22) uses PIT membership, next-bar fills, 5 bp costs and cost sensitivities.
  - The momentum replay (411fbdca, 21:25 the same day) reports 0/17 corrected gate passes.
- **Independent data agrees.** The factor, momentum and net-issuance screens use French
  portfolio data, not this tape, and reach the same "no large long-only edge" result. The
  tape faults below cannot explain them.

Does it hold up? For the bug signatures and the benchmark, yes. It does not cover
security identity (PARA), adjustment seams, or the DSR timing of every WFO figure before
2026-10-19 (Questions and answers below).

## Fix timeline

| # | Fix | What it affected | Fix commit(s) and date | Evidence |
|---|---|---|---|---|
| 1 | OHLCV day-shift (tz-aware write into the naive-UTC `timestamp` column; every equity bar stamped one calendar day early) | Bar labels only. Returns from one bar to the next were unchanged. Every join to a real-world date was off by one day: FOMC, earnings, Form 4, congress, index changes, ex-dates, short volume. SPY was the one symbol that mixed two alignments | Data migrated 2026-08-20 (5,678,783 rows, one transaction); writer fix ef4e15f, 2026-08-21 10:32 | Commit message ef4e15f; `src/ggTrader/data/live/cached_yfinance_loader.py:225-233`; memory note `project_ohlcv_day_shift_rootcause_2026-08-20`; backup table present |
| 2 | SPY duplicate rows (826 stray 04:00/05:00 SPY rows, 2022-12-27 to 2026-06-08) | Any lab run that loaded universe + SPY into one frame over 2023 onward: a NaN row for every real row. Both strategy and SPY Sharpe were deflated by about 1/√2 | Research-side collapse 1b9b7f5, 2026-07-25; database rows dropped in the 2026-08-20 migration (ef4e15f message) | Commit messages 1b9b7f5 and ef4e15f; memory note `project_spy_duplicate_rows_rootcause_2026-08-19`. Strays existed by about 2026-06-20: the cited SPY 0.58 matches the contaminated stored 0.571 |
| 3 | WFO anchor leak (the anchor fallback was fitted on the full evaluation span) | Every `run_wfo` result: the anchor and NDH/DSR gates were in the fold loop from 2026-06-17 (fa8ec4b5, be405aae). The same commit also added PIT `eligibility_mask` on `to_targets`, a 5-row purge for `train_gate.py`, and reversed the leveraged-rotation universe | 2eafab1, 2026-08-18 22:10 | `git show 2eafab18`; `docs/research/2026-08-18-wfo-anchor-leakage-fix.md` |
| 4 | PIT universe leak in signal-strategy sweeps (no membership mask; 14.1% of core entries fell on non-member days) | Signal strategies on index universes through `run_wfo`/`run_sweep`. Weight strategies were already PIT | c9d1abe, 2026-09-23 05:33 | `git show c9d1abe9`; `src/ggTrader/lab/sweep.py`; memory note `project_wfo_pit_membership_gap_2026-09-23` |
| 5 | DSR units (annualized Sharpe compared with per-day thresholds; "passed almost everything") | Every `run_wfo` fold decision: selected parameters versus anchor fallback. So OOS returns as well as gate counts | 09f9f40, 2026-10-19 16:04 | `git show 09f9f401` (changelog entry and the `observed_sr_annual` change in `src/ggTrader/lab/wfo.py`) |
| extra | Event studies counted pre-entry returns | FOMC and other event-study statistics | 7320c1d, 2026-10-19 17:22 | `git show 7320c1d6` |
| extra | APH/MNST mixed split adjustment (bars before 2026-07-20 were 2x too high) | Anything spanning 2026-07-20 that held APH or MNST, including the live warm-up study | Manual tape repair 2026-09-29 | `docs/changelog.md` 2026-09-29; backup `results/tape_repair_20260929` |
| extra | Benchmark tape keepalive (SPY had no bars after 2026-08-21; macro ETFs none after 07-20) | Any "vs SPY" run reaching past 2026-08-21 before the fix | 156d6174 / e1f45c85, 2026-09-16 | `src/ggTrader/paper/signal_runner.py:23-55` |

## Questions and answers

### (a) For every headline number, which fixes did it predate?

The full per-row answer is in `docs/research/_lens1_tape_state.json`. Rules:

- **contaminated:** the run date is earlier than a fix that affects its kind of result.
- **clean:** evidence shows it ran after every relevant fix.
- **unknown:** otherwise, with the reason given.

Counts: **55 contaminated, 18 clean, 71 unknown**. Of the unknowns, 25 are crypto rows
(Lens 8 owns them). Most of the rest have no tape backtest: design, feasibility, intake or
operations rows. How often each fix appears among the contaminated rows: DSR units 51,
anchor leak 38, SPY duplicates 38, PIT universe 29, day-shift 9, and 2eafab1's
`train_gate.py` fix 2 (the ML-gate rows).

By group:

| Group | Rows (examples) | Predates |
|---|---|---|
| June-July ensemble work: 1.12 baseline, 0.84 "validated", voter ablation, 3% sizing, ML gate, exit rules, IC/Kelly, MACD divergence, MTF, trailing stops, 1.14 blend, MidCap, Nasdaq core | `stocks-ensemble-ndh-baseline-1-12`, `stocks-three-sleeve-blend-1-14`, `stocks-ensemble-wfo-validated-2026-06-20` | SPY duplicates, anchor leak, PIT, DSR (all four that apply; day-shift does not change pure price returns). The ML-gate rows are the exception: they predate 2eafab1's `train_gate.py` fix |
| July event studies (PEAD, insider, congress, short interest, short volume, index deletion, headline sentiment) and FOMC WFO | `stocks-pead`, `cross-asset-fomc-drift-wfo` | Day-shift, SPY duplicates (all but headline sentiment ran 07-17..07-20; it ran 07-24, still before 1b9b7f5), anchor leak, DSR; FOMC also 7320c1d |
| July weight/ETF WFOs (idio vol, max effect, pairs, leveraged rotation and trend, commodity trend, FX hedge, Treasury curve) | `stocks-max-effect`, `cross-asset-treasury-curve` | SPY duplicates, anchor leak, DSR (leveraged rotation also the survivorship reversal in 2eafab1) |
| 2026-08-19 to 08-22 re-baselines (0.97 anchor-fix reproduction, 0.68 blend, 0.99 corrected tape, position sizing) | `stocks-sp500-core-corrected-tape-rebaseline` | PIT, DSR |
| 2026-09-23 to 09-24 PIT re-baseline (0.59 vs 0.78), IC/Kelly, core+SPY KEEP, month-end Treasury | `stocks-sp500-core-pit-rebaseline` | DSR only |
| 2026-10-19 onward replays, French-data screens, A23/A24, Shiller century study, 09-23 sanity checks, cross-asset holdout | `stocks-xs-momentum-corrected-replay`, `stocks-factor-screen-h1-h4` | None of the five (clean), except the replica rows, marked unknown for tape identity (part b). The cross-asset static 80/20 (2026-09-28) is also clean |

Two cross-asset rows share one results file,
`docs/research/_cross_asset_sleeve_results.json`, but are classified differently:

- `cross-asset-static-80-20-sleeve` is clean. Its headline is the pinned static figure
  (Sharpe 0.973 vs SPY 0.896), a fixed-weight backtest run on 2026-09-28 after the tape
  fixes, and needs no DSR. The WFO arm in the same file predates 09f9f40 but is not part
  of this headline.
- `cross-asset-sleeve-trend-arm` is unknown. Its headline combines a clean pinned
  comparison with the statement that the lookback WFO halted in folds 2-8, and that WFO
  predates 09f9f40.

The two ML-gate rows are tagged with 2eafab1's `train_gate.py` membership and purge-gap
fix, not the WFO anchor or PIT fixes.

### (b) What contamination remains today?

**Tape (SELECT-only checks, 2026-11-04):**

- **Day-shift residue: none.** All yfinance daily bars are at 00:00, Monday to Friday, as
  given in the Steelman.
- **Duplicates: none.** Zero duplicate keys and zero duplicate calendar days for every
  symbol, SPY included (6,751 rows = 6,751 dates).
- **Gaps:**
  - SPY since 2005 has a largest gap of 5 calendar days (2 cases, holiday weekends).
  - On 2021-2026, 33 symbols have 72 gaps longer than 5 days. The largest are SIVB
    (456 days, after its failure), DISCA (47), ISBC (37) and DISCK (35); the rest are
    mostly delisting tails.
- **Partial bars:**
  - Today's 2026-11-04 bar is partial. SPY shows volume 46.9M against 61.6M already
    reported by yfinance at the time of the check, and close 767.17. That is expected
    from the 12:45 PT run.
  - SPY self-heals: the benchmark refresh refetches the trailing 30 days
    (`src/ggTrader/paper/signal_runner.py:30-55`).
  - Stocks self-heal only because the one-start incremental fetch is pulled back to
    2026-08-11 by the stale symbol EA (last bar 2026-08-10, confirmed). See
    `src/ggTrader/data/live/cached_yfinance_loader.py:344-377` and the Lens 6 report.
  - Median volume ratio (day versus prior 20-day mean) for 2026-10-20 to 11-03 is
    0.89-1.22 across 502-503 symbols, with no day-wide collapse that would mark stored
    partial bars.
  - The VZ comparison for 2026-07-01 to 08-19 shows only a constant ratio, no isolated
    partial bars. This is not exhaustive (see Verification).
- **Identity problems (not covered by the five fixes):**
  - **PARA:** from 2021-02 the series is a different security, with closes 96,200 to
    113,900 through 2023 and median volume 9 on 2021-01-31 to 2026-04-30. PARA is an
    S&P 500 member 2022-02-17 to 2025-07-23 in
    `data/universe/sp500_constituents_history.csv.gz`, so PIT masking admits it.
  - **PARA in the replica:** the production replica's primary arm traded it 3 buys and
    3 sells (`docs/research/_production_replica_results.json`, key
    `windows.primary.arms.primary.trade_counts`), and `tainted_trades` is empty.
  - **SIVB:** shows garbage bars after its failure, for example $0.0013 to $0.07. Its
    membership ends before them (last listed 2023-01-04), so PIT runs after c9d1abe mask
    it. Before c9d1abe it produced Kelly's +170% day (memory note).
  - **AVB:** tracks EQR from 2026-07-20 (195.50 on 07-16, then 68.69 on 07-20 against
    EQR 68.80).
  - **Missing tape:** 14 of 611 symbols that were members on 2021-01-31 to 2026-04-30
    have no tape in that window, even after the lab's ticker normalisation
    (`normalize_yf_ticker`, `src/ggTrader/data/core/index_constituents.py:110-112`,
    applied at `src/ggTrader/lab/data.py:78`). The 14:
    - no rows at all: ABC, ANTM, BLL, COG, FBHS, FRC, GPS, HFC, MMC, PKI, RE, WLTW;
    - rows outside the window only: CDAY (3 rows, 2026-06-08 to 06-10) and VIAC (193
      rows, 2019-03-04 to 2019-12-04).

    Those member-days are absent from every backtest. BRK.B and BF.B are not missing:
    they map to BRK-B and BF-B, which have 1,318 bars each over the window.
  - **Old tickers reused:** FB (2025-06 onward, median volume 600), INFO (2024-10 onward)
    and DISCA (2022-04 onward) now carry other securities, but only outside their
    membership windows.
  - **Large moves:** 572 daily moves of 40% or more in 97 symbols from 2021 (the most are
    TNB, SIVB, CPWR, COMS and BMC, mostly reused tickers of non-members), against 5,595
    in 217 symbols before 2021.
  - **Other price faults:** 9,567 bars close below $0.05, 18 have high below low, and
    62,317 have zero volume.
- **Adjustment seams:**
  - The loader requests `auto_adjust=True`
    (`src/ggTrader/data/live/yfinance_loader.py:66`). The cache only upserts the window
    it refetches (`src/ggTrader/data/live/cached_yfinance_loader.py:256-265`) and never
    re-adjusts older bars.
  - **VZ:** the DB/yfinance ratio is about 1.0155 on 2026-07-14 to 07-17 and 0.9999 on
    07-20. The DB return on 07-20 is -1.73%, against -0.19% in yfinance. The ratio
    matches the adjustment for the dividend that went ex on 2026-10-09
    (1/(1 - 0.708/45.64) ≈ 1.0158), not the July dividend (≈ 1.0176). So bars up to
    07-17 were stored before the October adjustment, bars from 07-20 were refetched
    after it, and the October dividend's adjustment shows up as a misdated -1.5% move
    on 07-20.
  - **SPY:** DB closes on 06-12, 06-26, 07-02, 07-10, 07-17 and 07-24 are all 0.248%
    above yfinance's current series (one quarterly dividend), and equal from 10-29.
  - **APH/MNST:** the same mechanism at split scale was repaired by hand on 2026-09-29.
  - How many seams fall inside 2021-01-31 to 2026-04-30 was not measured (see
    Verification).

**Benchmark:**

- SPY's tape is now single-convention.
- Different reports quote different SPY Sharpe values because they measure different
  things: 0.78 over the WFO's out-of-sample span; 0.90/0.91 over the full pinned window
  with no risk-free rate (`docs/research/WEB_RESEARCH_CANDIDATES.md:1560-1567`); 0.914
  (ef4e15f). Lens 2 has to match the definitions before comparing results.
- Backtests use total-return SPY, while some paper comparisons used SPY price return
  (see d).

**Universe:**

- Membership comes from a static history file that ends 2026-06-02.
- The 14 missing tickers above make up a coverage gap; it removes member-days rather
  than adding survivors.
- Weight strategies were PIT from the start; signal strategies are PIT only from c9d1abe.
- There is no PIT MidCap 400 membership (inventory row `stocks-midcap400-reversion`).
- The 2010-2020 holdout tape has identity faults (TIE, MEE, CPWR), and the replica
  stopped itself on them (`docs/research/2026-10-19-production-replica-audit-nogo.md`,
  lines 91-106).

**Fills and costs:**

- WFO and vectorbt runs fill on the signal bar at its close plus 5 bp
  (Lens 6 fill check).
- The replica fills at the next close (lag 1, `src/ggTrader/lab/replica.py:178-179`).
- Live trades from a partial 12:45 PT bar.
- Costs are 5 bp per side with zero fees for stock WFO, from the
  `docs/research/_month_end_treasury_results.json` meta.
- Cross-asset and A23 use 1 bp per side; the replica tests 3 and 10 bp sensitivities.
- No commissions are modelled, matching Alpaca's commission-free stock trades. Lens 3
  owns the live versus same-close difference.

### (c) Does any current headline number rest on an unrecoverable or drifting window?

- **Current headlines: no.** These pin both ends of their windows:
  - the production replica (2021-01-31 to 2026-04-30);
  - the PIT core (same window, but it predates the DSR fix);
  - the xs/dual momentum replays;
  - A23/A24 (2006-06-21 to 2026-09-29);
  - the French screens (fixed historical months);
  - the cross-asset holdout (2007-06-01 to 2011-02-28).

  One discrepancy: the cross-asset 80/20 report table says the window starts 2021-02-01,
  while its raw meta says 2021-01-31.
- **Unrecoverable:** the 1.12 baseline (2026-06-26) used `--eval-end` = now (memory note
  `project_baseline_did_not_reproduce_2026-08-19`), as did the levers judged against it.
  The roadmap claims their drift was "negligible" (`docs/roadmap.md:170`).
- **Not unrecoverable, contrary to the project notes:** 1.14 is recorded as a pinned
  2021-01-31 to 2026-04-30 run at max-leverage 1.0 (`docs/roadmap.md:158-170`). Its
  re-run on the same window gave 0.68. That points to code fixes, not window drift. This
  comes from prose; the 2026-07-13 raw output was not found. Data-side changes between
  07-13 and 08-19 have not been ruled out either: the SPY-duplicate collapse (1b9b7f5)
  and refetch adjustment seams.
- **Drifting ("to present" or end not stated):** these July NO-GOs cannot be reproduced
  exactly: index deletion, max effect, pairs, short interest, short volume, leveraged
  rotation and trend, commodity trend, FOMC WFO, headline sentiment, and the 2026-08-18
  anchor A/B. All are also contaminated (part a).

### (d) yfinance adjustment and total return: are backtest and paper ledger consistent?

No.

- **Backtests read adjusted (total-return) closes.** These come from `auto_adjust=True`
  (`src/ggTrader/data/live/yfinance_loader.py:66`), with Tiingo `adjClose` as fallback
  (`docs/changelog.md`, dividend-accrual entry).
- **Adjustment is frozen at fetch time.** When a later fetch overlaps an earlier one, the
  tape gets a seam where a dividend or split shows up as a misdated price move (VZ, SPY,
  APH/MNST above). Inside a single fetch the series is a proper total-return series.
- **The paper account works in raw prices.** Fills are raw. Alpaca's paper account
  credits no dividends (`/v2/account/activities/DIV` returned `[]`, per the changelog).
  The trader adds a cumulative accrual to the snapshot value only: $320.24 over 19 rows,
  confirmed by Lens 6 at `src/ggTrader/paper/trader.py:1045-1056`, with sources from
  `src/ggTrader/paper/dividend_check.py`.
- **Live signals mix the two.** They are computed on the adjusted tape, seams included,
  while orders execute at raw prices. A seam can feed a false one-day move into a live
  indicator, such as VZ's -1.7% on 2026-07-20.
- **Paper-vs-SPY comparisons mix the two.** The Phase 0 table compares paper against
  "SPY price +4.63%", so the paper side includes dividends and the SPY side does not.
  Lens 3 owns the live ledger.

### (e) Crypto tape

Lens 1 ran no crypto checks; Lens 8 owns them.

- **What is known:** crypto bars come from a different writer (Kraken/Binance.US
  loaders). The changelog records crypto timezone bugs (`crypto-live-ohlcv-tz-bug`,
  `crypto-regime-filters` rows) and a Kraken partial-bar timestamp repeat
  (`docs/changelog.md`, LiveExchangeLoader dedupe entry).
- **Query cost:** one tape-wide query that grouped all venues (crypto included) ran for
  more than 11 minutes and was cancelled. Lens 8 should filter by venue and time.
- **What Lens 8 must check:**
  - the stamp hour per venue and interval;
  - duplicate (symbol, timestamp) rows;
  - gaps on the 24/7 calendar;
  - partial last bars;
  - `-USD` symbol normalization (the `SYMBOL_MAPPING` table);
  - whether any crypto row was written through the tz-aware path fixed in ef4e15f.

### Phase 0 hypotheses: confirmed or not

- **H3, partly confirmed:**
  - The June-July beat-SPY figures (0.84, 1.12, 1.14, about 16% CAGR) predate four of the
    five fixes. The August figures (0.97, 0.99) predate the PIT and DSR fixes.
  - The bug signatures are gone from today's tape.
  - Pre-2026-10-19 gate counts are void, and so are the fold-level parameter choices
    behind them.
  - The "corrected" 0.59 is not fully corrected: it predates 09f9f40. Phase 0 already
    noted that its "17/17 gates" count predates the fix
    (`docs/research/2026-11-04-phase0-trust-audit.md:30`). What Lens 1 adds is that a
    failed DSR check deploys the anchor parameters for that fold (lines 726-733 of
    `src/ggTrader/lab/wfo.py` at 09f9f40). So 0.59's out-of-sample returns would also
    change under the corrected gate, not only its gate count.
  - Not confirmed: that the five bugs *alone* caused the drop from 1.12. That run's window
    cannot be recovered, and it used the 3- and then 5-voter ensembles at different times.
- **H1 (Lens 1 part), not confirmed as "high trust":** the replica ran after all five
  fixes but traded a misidentified security (PARA) that its detector missed. Its tape
  state is unknown until that effect is measured.
- **H4 (Lens 1 part), confirmed:** the 2010-2020 tape has identity faults, and the
  replica's holdout stopped on tainted trades. There is no clean holdout on this tape.
- **"Two known bugs favoured the July event studies", could not confirm.** Commit ef4e15f
  says the day-shift offset pointed toward lookahead. But a bar labelled D held D+1's
  prices, so an entry at the event day's close is *delayed* a day, while a signal read
  from the event-day bar *sees* the next day. Which one applied depends on each study's
  join, and that was not checked here. The SPY-duplicate bug deflated both strategy and
  SPY Sharpe, so it did not plainly favour either side.

## Verification

All database access used a read-only psycopg2 session
(`conn.set_session(readonly=True)`, with a 300 s statement timeout added). The helper is
`/tmp/claude-1000/-home-flynn-ggTrader/47577f2e-9398-4897-a14d-c38613eb3a47/scratchpad/db_ro_t.py`,
copied from the Lens 6 helper. All runs went through
`scripts/run_capped.sh .venv/bin/python …` with a 6 GB cap; there were no 137 exits.

Commands and key outputs:

- **Tape checks.** Command: `lens1_tape.py` with selected query groups. Raw outputs are
  in `/tmp/claude-1000/-home-flynn-ggTrader/47577f2e-9398-4897-a14d-c38613eb3a47/scratchpad/lens1_tape_out.json`.
  - spy_summary `6751, 6751, 2000-01-03, 2026-11-04`; spy_hours `0: 6751`;
    spy_window_bars `1318`.
  - exact_dup_keys `0`; dup_calendar_days_all is empty; hour_of_day `00:00 5,752,749`;
    dow Monday-Friday as quoted above.
  - bad_prices `0 / 9,567 / 18 / 0 / 62,317` (close ≤ 0 / close < 0.05 / high < low /
    close null / volume 0).
  - spy_max_gap_days `5, 2`; gap_symbols_2021_2026 `33 symbols, 72 gaps`.
  - big_moves_by_era: `2021+ 572 moves / 97 symbols`, `pre2021 5,595 / 217`.
  - ea `last 2026-08-10`; avb_eqr as quoted; aph_mnst_moves: only MNST 2014-08-15
    +30.5% (a real move; no split-sized jumps remain).
  - stale_count `929` symbols with no bar since 2026-10-01 (non-current or delisted
    names; the 503 current symbols are fresh to 2026-11-04).
  - backup tables `ohlcv`, `ohlcv_no_data`, `ohlcv_yf_preshift_backup_20260820`.
- **Identity scan.** Command: `lens1_ident.py` over 2021-01-31 to 2026-04-30 members,
  output in `lens1_ident_out.json`. Results: 611 members; 595 had tape under the raw
  membership ticker and 16 did not. That scan did not apply the lab's normalisation, so I
  re-checked the 16 with `normalize_yf_ticker` (`l1_norm.py`, bounded SELECT on the
  normalised symbols): BF-B and BRK-B have 1,318 bars each in the window; CDAY has 3 rows
  (2026-06-08 to 06-10); VIAC has 193 rows (2019-03-04 to 2019-12-04); the other 12 have no
  rows. Corrected count: 597 members with tape, 14 without. Median volume under 20k: DISCA, FB, INFO, NVR (19,600, genuine), PARA, SW (pre-listing
  history only; 3-6M volume once listed); maximum/minimum close above 50x: APP, CVNA,
  PARA, SBNY, SIVB.
- **PARA quarterly levels and the replica's PARA trade count**, as quoted above.
- **yfinance MCP reads** (free, read-only): SPY 5-day and 6-month weekly; VZ 1-year
  monthly and 6-month daily. Compared by hand with DB closes for the same dates.
- **Fix dates and commit messages:** `git show` on ef4e15f8, 1b9b7f56, 2eafab18,
  c9d1abe9, 09f9f401, 7320c1d6, e1f45c85, 17b0971f, 064afdb7, fa8ec4b5, be405aae,
  411fbdca, 8bc365ee, 77014bb2. Artifact first-commit dates were taken from
  `git log --diff-filter=A` for every source report and raw artifact.
- **Haiku extraction spot-check.**
  - For the 77 rows whose source report carries a date, the extracted run_date matches
    that date in 72. The other 5 have explained differences: the extraction used the
    underlying run date for three rows cited by the 2026-09-10 summary, the FOMC
    artifact was committed 10-20 rather than 10-19, and one feasibility row has a null
    date.
  - Windows were checked against raw JSON meta for 9 rows: anchor-fix reproduction,
    corrected-tape re-baseline, PIT re-baseline, position sizing, core+SPY blend,
    month-end Treasury, the audit re-runs, FOMC clustered, and cross-asset (the
    2021-02-01 vs 2021-01-31 discrepancy).
  - Windows were checked against report text for 3 rows: PEAD (2015-06 long window;
    2021-01-31 to 2026-04-30 matched), index deletion (2015-01-01 start), and 1.14
    (`docs/roadmap.md:158`).
- **Builder for the JSON:** `lens1_state.py` (scratchpad); asserts 144 rows and no
  missing ids.
- **Database cancel.** The first tape-wide query grouped all venues and ran for more
  than 11 minutes. I cancelled my own backend with `pg_cancel_backend` (an admin function
  called in a SELECT; no data changed). The other session running then was TimescaleDB's
  own compression job, which I did not touch.
- **Gate:** `.venv/bin/python scripts/check_lens_report.py
  docs/research/2026-11-04-lens1-backtest-validity.md` exits 0.

**Not verified:**

- How many adjustment seams fall inside the pinned window, for all symbols. Cheapest
  test: compare DB closes with one fresh yfinance pull for the 597 members with tape and flag
  breaks in the ratio. Read-only, about 30 minutes.
- PARA's effect on the replica and on the PIT core. Cheapest test: a Phase 2 replica
  re-run with PARA excluded, or the PARA trade dates printed from a re-run. Roughly 1 hour,
  and it needs owner approval.
- Whether the SPY stray rows existed before 2026-06-20, which decides the 2026-06-12
  harness rows.
- The direction (lookahead or delay) of the day-shift for each July event study.
  Cheapest test: read each strategy's event join (about 10 minutes per strategy).
- Whether the code that generated the 2026-10-19 re-runs included 09f9f40. This is
  documented in prose; Git does not prove it, as
  `docs/research/_wfo_fold_attribution_results.json` notes.
- Partial bars written between 2026-06-20 and 2026-08-10 for symbols other than VZ/SPY.
- The 2026-07-13 raw output behind 1.14.

## Done checklist

- ✅ Fix timeline (five fixes plus three extras), with commits and dates.
- ✅ (a) Fixes each headline predates: all 144 rows in `docs/research/_lens1_tape_state.json`
  (55 contaminated / 18 clean / 71 unknown, each with a reason).
- ✅ (b) Tape: duplicates, stamp hour, weekdays, gaps, partial bars and identity checked
  today.
- ❓ (b) Adjustment-seam extent: unknown. Only VZ and SPY were measured. Cheapest test:
  the full-member ratio scan described in Verification.
- ✅ (b) Benchmark, universe, fills and costs answered. Live fill effects are left to
  Lens 3.
- ❓ (b) PARA's effect on the replica and core: unknown, because measuring it needs a
  re-run (Phase 2 list).
- ✅ (c) Window question: no current headline drifts. 1.12 is unrecoverable. The 1.14
  window is recorded as pinned, which contradicts the project notes. The July NO-GOs used
  windows that ran to present.
- ✅ (d) Adjustment and dividend handling: the backtests and the paper ledger are
  inconsistent, as described.
- ✅ (e) Crypto: known items and the Lens 8 checklist given; nothing duplicated.
- ✅ Haiku extraction spot-checked (77 dates, 12 windows).
- ❓ Day-shift direction per July event study: unknown, because the join code was not
  read. Cheapest test: about 10 minutes per strategy.
