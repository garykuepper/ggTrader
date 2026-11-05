# Lens 5: coverage gaps

**Date:** 2026-11-04. **Branch:** `investigation/root-cause-2026-11-04`.
**Scope:** read-only diagnosis for the root-cause investigation
(`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md`, section 5, Lens 5).
This lens lists what the project's record contains and lacks. It does not propose
strategies, rank options or say what to try.
**Raw results:** `docs/research/_lens5_coverage_matrix.json` (all 144 inventory ids, once each,
plus the gap table, hypothesis sourcing and the registry cross-check).
**Inputs:** inventory `docs/research/_investigation_inventory.json`; Lens 1
`docs/research/_lens1_tape_state.json`; Lens 2 `docs/research/_lens2_sensitivity.json` and
`docs/research/2026-11-04-lens2-verdict-audit.md`; Lens 6 `docs/research/_lens6_verdicts.json`;
Lens 9 `docs/research/_lens9_power.json`; an unreviewed Haiku extraction of coverage fields
(treated as leads, see Verification). Phase 0 (`docs/research/2026-11-04-phase0-trust-audit.md`)
is unreviewed and used only as a source of hypotheses. Lens 8 (crypto) had not been committed
when this was written; crypto rows here are pointers to it.

Abbreviations, expanded once: WFO = walk-forward optimization; OOS = out of sample; PIT = point
in time (index membership as it was on each date); MDE = minimum detectable effect (the
smallest true effect a test finds with 80% probability; Lens 9 uses t >= 3); NW = Newey-West
(a standard error allowing autocorrelated returns); pp = percentage points; ETF = exchange-traded
fund; LETF = leveraged or inverse ETF; SSO = 2x daily S&P 500 ETF; PEAD = post-earnings-
announcement drift; FOMC = Federal Open Market Committee; LLM = large language model; DSR =
deflated Sharpe ratio; GO / NO-GO = the project's pass / fail verdict labels; IBIT = iShares
Bitcoin Trust ETF; Holm = a multiple-testing correction; RSI, MACD, EMA = relative strength
index, moving-average convergence/divergence, exponential moving average (technical indicators).

## Summary

1. **The record is narrow in the dimensions a small account could use.** Of the 51 measured
   single-stock tests, 44 are large-cap only (S&P 500 or Nasdaq-100), and 65 of the 72 measured
   tests are long-only. There is no micro-cap universe, no options strategy, no futures
   simulation and no weekly-rebalanced cross-sectional strategy anywhere in the registry, code
   or git history. The single small-cap stock-selection arm (PEAD on the Russell 2000) ran on a
   200-name current-list snapshot that the project itself calls a partial universe.
2. **Gap table (26 areas):** 6 never tested, 3 tested badly (flawed test), 16 tested and
   inconclusive, 1 tested with evidence against at the size the test could detect (Treasury
   calendar effects, through the corrected FOMC event study). The six never-tested areas are
   micro caps, options, futures contracts, international equity exposure, weekly-rebalanced
   cross-sectional strategies and intraday (intraday is prospective evidence in progress, not
   evaluated here). The three tested-badly areas are small-cap stock selection, concentration
   (few names at large weights) and small-cap fundamentals.
3. **Hypothesis sources (72 measured tests):** 31 internal observations or construction levers,
   19 from LLM/web research batches (16 of them citing a published paper), 14 practitioner
   technical indicators (RSI, Bollinger, MACD, EMA cross and ensembles of them), 8 taken
   directly from published academic anomalies. 26 of 72 are tied to a published paper.
   33 of 72 are the RSI/Bollinger mean-reversion ensemble or levers on it (Lens 2 family count).
4. **Web-research backlog:** 56 trading entries plus 26 parked bullets (82 hypotheses) and 11
   method entries. 15 trading entries were run as measured tests, 1 was built without a result,
   3 entries and 1 parked bullet had only a one-off sanity check, and **62 of 82 were never run**.
5. **Published-anomaly decay:** McLean and Pontiff (2016) report that 97 published
   cross-sectional stock predictors earn 26% less out of sample and 58% less after publication.
   The project's own long-history screens show the same shape (momentum Small t 3.38 before
   1994, -0.77 after 2013). Which of the project's anomalies are among the 97 was not checked.
6. **The research prompt steered away from small-account advantages.** It asks for free data,
   one daily decision before the close, 1.0x, no futures, and to "avoid strategies that only work
   at institutional scale/capacity"; no line asks for capacity-limited, concentrated or
   micro-cap ideas (`docs/research/prompts/web-strategy-research-prompt.md:184-205`).

## Falsifiers

Written before the evidence was gathered. Each states what would show coverage is broader, or
narrower, than the inventory suggests, and the result.

- **F1 (broader):** the strategy registry or git history contains a small-cap, micro-cap,
  options, futures, concentrated-book or short-selling strategy with no inventory row. Checked:
  all 38 registry entries map to inventory rows (`src/ggTrader/lab/strategies/__init__.py:56-95`);
  git subject searches for micro, futures, options, shorting, international and weekly found no
  such strategy (Verification). Not met.
- **F2 (broader):** web-batch candidates marked untriaged were in fact run in the batch artifact
  folders. Checked: the four artifact folders hold sanity checks for A10, A12, A14, B5, the
  cross-asset sleeve, UUP/UDN buy-and-hold context, static SPY/SSO mixes and IBIT
  (`docs/research/artifacts-2026-09-28-web-batch/sanity_checks_output.txt`). Partly met: four
  entries gained a sanity check, none a measured test.
- **F3 (broader):** the Russell 2000 or MidCap 400 universes are full and point-in-time. Checked:
  they are 200- and 400-line current-list snapshots (`data/universe/` listing). Not met.
- **F4 (broader):** a concentrated stock arm (10 names or fewer at 10% or more each) has an OOS
  record. Checked: the only such arm (xs_momentum top_n 10) was never deployed OOS. Not met.
- **N1 (narrower):** more than 40% of measured tests share one signal family and universe. Met:
  33 of 72 are the ensemble/reversion family (46%).
- **N2 (narrower):** tests labelled small or mid cap did not use an investable PIT universe. Met
  for PEAD on the Russell 2000 and for the MidCap 400 rows.
- **Phase 0 hypotheses touched.** H1 (no stock-picking edge found) is weakened as a general
  statement if this lens finds untested areas bearing on stock selection; it does (small and
  micro caps, concentration). H7 (best free-data long-only large-cap effects +0.1 to +0.8 pp/yr)
  would be overturned as a statement about the project's reach if small-cap or concentrated
  tests with clean evidence existed; none exists, so H7 remains a statement about large-cap
  long-only effects only.

## Steelman

The strongest case that coverage was broad:

- **Volume and range.** 144 inventory rows, 72 measured tests (76 hypotheses counting registered
  multi-hypothesis screens), 38 registered strategies and 11 test families (Lens 2
  `multiple_testing`). Instruments span single stocks, equity index ETFs, leveraged and inverse
  ETFs, Treasury ETFs, commodity ETFs, gold, currency-hedged international ETFs, crypto spot,
  crypto perpetual funding and a spot-bitcoin ETF (coverage matrix below).
- **Long histories, not just 2021-2026.** The momentum, factor and net-issuance screens use 63 to
  100 years of survivorship-free French portfolios, including Small portfolios
  (`docs/research/RESEARCH_SNAPSHOT.md:163-167`); the century leverage study covers 1926-2026.
- **The project moved off its crowded core on purpose.** From July 2026 the research prompt
  asked for other asset classes, geographies and horizons, saying another U.S. large/mid-cap
  sort was unlikely to help (`docs/research/prompts/web-strategy-research-prompt.md:123-127`).
  Cross-asset, Treasury calendar, FX-hedge, commodity, leverage and crypto rows followed.
- **Hypotheses were screened, not just collected.** Later web batches checked every citation
  (none fabricated) and rejected 14 candidates on intake, several because the record already
  held a measured result or the effect had decayed
  (`docs/research/WEB_RESEARCH_CANDIDATES.md:1546-1556`, `docs/research/WEB_RESEARCH_CANDIDATES.md:1587-1598`).
- **Some short and market-neutral exposure was tested** (pairs, short-volume ratio, inverse-ETF
  rotation, crypto carry).

Does it hold? Partly. Range across asset classes is real. Range within stock selection is not:
the dimensions where a small account differs from a large one (micro and small caps,
concentration, options, short horizons other than the daily ensemble) are absent or tested on
flawed universes, and the long-history Small screens are value-weighted index portfolios with
assumed cost haircuts, not tradable small-cap books.

## Method

1. Took all 144 inventory ids. Read the Haiku extraction (`lens5-out-0..3.json`, 144 rows, many
   nulls) as leads only.
2. Normalized each row to fixed codes (instrument, market-cap segment, decision horizon,
   direction, signal family, idea source, published link, data, sizing) in one table, filling
   Haiku nulls from the inventory row's name and claim.
3. Re-read the source lines for 42 rows that the gap table relies on (listed per row in
   `rows[].source_for_fields`, `verified: true`).
4. Attached each row's Lens 1 tape state, Lens 2 class, Lens 6 code match and Lens 9 MDE.
5. Wrote the gap-table classification rules before classifying (JSON
   `classification_rules`), then classified each area from its cited rows, the registry and a git
   subject search.
6. Counted web-research entries with a fixed rule (Hypothesis sourcing).

## Coverage matrix

Counts from `coverage_counts` in the JSON. "Measured tests" = Lens 2 verdict type NO, GO or
TEST-INCONCLUSIVE (72 rows); the other rows are 42 status/feasibility labels and 30
operations/infrastructure rows with no Lens 2 row. "Decision horizon" is the decision or
rebalance cadence, not the holding period. Values were normalized by this lens; 102 of 144 rows
were not re-read against their sources (Verification).

**Instrument class** (Haiku null in 2 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| US single stocks | 78 | 51 |
| crypto spot | 21 | 3 |
| none (infrastructure or method) | 14 | 1 |
| US equity index ETF | 7 | 1 |
| leveraged/inverse ETFs | 6 | 5 |
| multi-asset ETF mix | 6 | 3 |
| Treasury ETFs | 4 | 4 |
| international equity / currency-hedged ETFs | 2 | 1 |
| crypto perpetual futures | 2 | 1 |
| spot-crypto ETF | 2 | 1 |
| options | 1 | 0 |
| commodity ETFs | 1 | 1 |

**Market-cap segment** (Haiku null in 57 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| na | 63 | 19 |
| large | 63 | 44 |
| mixed | 9 | 8 |
| unknown | 7 | 0 |
| mid | 2 | 1 |

**Decision horizon** (Haiku null in 95 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| daily | 56 | 35 |
| monthly | 24 | 20 |
| unknown | 22 | 4 |
| na | 21 | 2 |
| event | 14 | 7 |
| static | 5 | 4 |
| intraday | 2 | 0 |

**Direction** (Haiku null in 112 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| long | 106 | 65 |
| na | 24 | 2 |
| unknown | 8 | 0 |
| hedged | 3 | 2 |
| longshort | 2 | 2 |
| long_inverse | 1 | 1 |

**Signal family** (Haiku null in 13 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| infra | 39 | 3 |
| mean_reversion | 21 | 17 |
| event_calendar | 13 | 7 |
| sizing_exit | 12 | 10 |
| other | 12 | 4 |
| allocation_leverage | 10 | 7 |
| alt_data | 10 | 5 |
| momentum | 9 | 7 |
| trend | 6 | 4 |
| factor | 6 | 4 |
| carry | 4 | 3 |
| ml | 2 | 1 |

**Data source** (Haiku null in 53 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| ohlcv_daily | 69 | 56 |
| none | 17 | 2 |
| broker_live | 16 | 1 |
| crypto_ohlcv | 16 | 2 |
| alt_free | 15 | 6 |
| french | 5 | 4 |
| paid_unavailable | 3 | 0 |
| funding | 2 | 1 |
| intraday_bars | 1 | 0 |

**Sizing style** (Haiku null in 94 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| na | 47 | 3 |
| eq_slots | 33 | 23 |
| static_weights | 18 | 14 |
| unknown | 14 | 4 |
| vol_target | 7 | 5 |
| eq_quintile | 7 | 6 |
| conviction_kelly | 5 | 5 |
| eq_topn | 5 | 5 |
| vw_portfolio | 4 | 3 |
| eq_basket | 3 | 3 |
| pair_weights | 1 | 1 |

**Idea source** (Haiku null in 46 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| internal | 77 | 31 |
| web_batch | 35 | 19 |
| practitioner | 19 | 14 |
| published | 13 | 8 |

**Tied to a published paper** (Haiku null in 58 of 144 rows before Lens 5 filled them)

| value | all 144 rows | 72 measured tests |
|---|---|---|
| False | 94 | 44 |
| True | 48 | 26 |
| None | 2 | 2 |

What the matrix shows:

- **Market cap:** no row is small-cap only; small caps appear only inside "mixed" rows (PEAD's
  Russell 2000 arm, French Small portfolios, multi-sleeve blends). One measured test is mid-cap
  only (the 50-name sentiment pilot); the MidCap 400 reversion row is a status label.
- **Direction:** 65 of 72 measured tests are long-only; 2 are long/short (pairs, short-volume
  ratio), 2 hedged (FX hedge, crypto carry), 1 long/inverse ETF rotation.
- **Horizon:** daily decisions (35) and month-end rebalances (20) dominate; no weekly cadence;
  the only intraday rows are feasibility and source audits.
- **Data:** 56 of 72 measured tests use daily price bars only; 6 use free alternative data; 4 use
  French portfolios; no test uses stock-level point-in-time fundamentals (none are available,
  `docs/research/RESEARCH_SNAPSHOT.md:236-238`), options data or intraday bars.
- **Sizing:** the most common style is small fixed slots of about 3% per name (23 tests); the
  rest are quintile or top-N equal weights, static weights, volatility targets, and Kelly or
  conviction scaling within the same small-slot framework.

## Gap table

Classification rules (fixed before classifying; also in the JSON): **never tested** = no
measured inventory row exercised the area's defining feature and neither the strategy registry
nor a git subject search shows an implementation (feasibility rows may exist). **Tested badly
(flawed test)** = measured rows exist but each is Lens 2 "unreliable" or the record itself
documents that the test did not exercise the defining feature (partial or snapshot universe,
arm never deployed OOS, diagnostic outside the registered family). **Tested, inconclusive** =
measured rows exercised the feature, none is flawed in that way, and none meets the next rule.
**Tested, evidence against at the size the test could detect** = at least one measured NO row
whose Lens 9 MDE (t >= 3) is known and at or below the owner's 2x-SPY target gap over its
window, on a clean tape or one whose contamination favours the strategy (Lens 2 rules R2 and
R5); Lens 2's own class, which also needs the code check, is quoted beside it.

Per-id Lens 1/2/6/9 states are in `gap_table[].upstream_by_id` in the JSON.

| area | what the record contains (inventory ids) | quality of that testing | classification |
|---|---|---|---|
| Micro caps (single stocks below small-cap index size) | Nothing. No micro-cap universe file under data/universe/ (11 files: SP500, Nasdaq-100, MidCap 400, a 200-name Russell 2000 snapshot, ETF lists, coin listings); no registry entry; the only git subject containing 'micro' is a crypto tick-size fix (2ac0ff71). Ids: none. | Not applicable. Evidence: docs/research/2026-10-21-factor-screen-nogo.md:127 ('ggTrader has no point-in-time small-cap universe, no liquidity screening, and no small-cap execution infrastructure'); docs/research/WEB_RESEARCH_CANDIDATES.md:398-403 (PEAD found sensitive to microcap inclusion; not followed up); docs/research/WEB_RESEARCH_CANDIDATES.md:126-130 (short-interest effect concentrates in small/illiquid names); git log --all -i --grep=micro: 2ac0ff71 only. | **never tested** |
| Small caps: single-stock selection | One measured small-cap stock-selection arm: pead@russell2000 on 2015-06..2026-07 (40 folds), OOS Sharpe 0.77, tied with SPY, regime halt 60%, not re-run on the matched window. The leveraged-rotation Russell 2000 variant used constituent breadth only to time TNA/UWM index ETFs. Ids: `stocks-pead`, `stocks-leveraged-rotation`. | The record calls the Russell 2000 universe 'a partial ~200-symbol subset (not the full ~2000)' and 'a known ... data limitation'; the file is a current-list snapshot (survivorship), roadmap says ~29% loaded. Tape pre-fix (day shift, SPY duplicates, anchor leak, DSR units), Lens 6 partly, no raw artifact. Evidence: docs/research/2026-07-17-pead-nogo.md:19-20,47,72-75,100-105; docs/roadmap.md:154 ('Russell 2000 could not be included — its data is only ~29% loaded'); data/universe/russell2000_tickers_snapshot_2026-06-09.txt (200 lines). | **tested badly (flawed test)** |
| Small caps: portfolio-level size screens (French library) | Momentum Small HiPRIOR was a registered group (full t 3.11, fails Holm; post-1994 +0.20 pp/yr vs +1.0 bar; 1927-1993 t 3.38, 2013-2026 t -0.77). SMALL HiOP/HiBM/LoINV reported only as diagnostics outside the Holm family (t 0.47-0.90, post-2013 -6.3 to -6.8 pp/yr). Accruals: feasibility only. Ids: `stocks-momentum-french-screen`, `stocks-factor-screen-h1-h4`, `stocks-accruals-feasibility`. | Clean tape (Lens 1), no Lens 6 row, Lens 9 MDE not computable (no stored mean excess). Value-weighted portfolios with an assumed 3%/yr small-cap cost haircut, not a tradable small-cap book. Evidence: docs/research/2026-10-20-momentum-french-screen-nogo.md:18-31,71,83-90; docs/research/2026-10-21-factor-screen-nogo.md:125-135. | **tested, inconclusive** |
| Mid caps | MidCap 400 reversion (status 'Researching', anchor-driven 15/17 folds), the 3-sleeve SP500+MidCap+Nasdaq blend (GO, later reverted), sleeve correlations, and a 50-name MidCap 400 headline-sentiment pilot. Ids: `stocks-midcap400-reversion`, `stocks-three-sleeve-blend-1-14`, `stocks-three-sleeve-blend-reproduction`, `stocks-equity-diversification-sleeves`, `stocks-live-revert-blend-to-core`, `stocks-headline-sentiment`. | Every row carries pre-fix contamination (anchor leak, PIT universe, DSR units); MidCap 400 universe is a 2026-06-24 current-list snapshot; Lens 6 'no' for the MidCap and blend rows; blend GO is Lens 2 'unreliable GO'; the sentiment pilot held ~7 names from an alphabetical first-50 sample (record says unrepresentative). No raw artifacts. Evidence: docs/research/2026-07-24-headline-sentiment-nogo.md:15-21; data/universe/midcap400_tickers_snapshot_2026-06-24.txt (400 lines, current list); docs/research/_lens2_sensitivity.json classifications (stocks-three-sleeve-blend-1-14: unreliable GO). | **tested, inconclusive** |
| Concentration (few single stocks, large weights) | The only concentrated stock arm was xs_momentum top_n 10 (10% each) inside a 10/20/50 grid; the corrected replay deployed the anchor (top_n 50) in 17/17 folds, so no top_n 10 OOS return exists. Sizing study minimum was 8 slots per sleeve. The live/replica book holds at most 3 names at ~3% each (9.37% mean stock exposure), which is few names but small weights. The 7-name sentiment book was accidental. Ids: `stocks-xs-momentum-corrected-replay`, `stocks-position-sizing-regimes`, `stocks-headline-sentiment`, `stocks-production-replica`. | Replay tape clean, Lens 6 yes, MDE 22.66 pp/yr; but the concentrated arm itself was never deployed. Sizing study contaminated (PIT, DSR), single window, Lens 6 partly. Evidence: src/ggTrader/lab/strategies/momentum.py:25 (top_n grid [10, 20, 50]); docs/research/2026-11-02-wfo-fold-attribution.md:38-39,66-68 (17/17 anchor; winner OOS not recorded); docs/research/2026-10-19-production-replica-audit-nogo.md:259-263 (anchor top_n 50); docs/research/2026-08-19-position-sizing-regimes.md:24,96-101 (slot caps 8-30); docs/research/RESEARCH_SNAPSHOT.md:49-51 (9.37% mean stock exposure). | **tested badly (flawed test)** |
| Sizing and exit levers inside the ensemble | 3% position size (GO), sizing regimes (inconclusive), conviction and Kelly sizing (NO), exit-rule sweep, trailing stops, VIX throttling (NO). Ids: `stocks-three-pct-position-size`, `stocks-position-sizing-regimes`, `stocks-conviction-bb`, `stocks-ensemble-kelly-pit`, `stocks-ensemble-kelly-2026-07`, `stocks-exit-rule-sweep`, `stocks-trailing-stops`, `stocks-vix-entry-throttling`. | 3% size is 'unreliable GO'; the rest Lens 2 inconclusive; all but Kelly-PIT carry pre-fix contamination; no MDE. Evidence: docs/research/_lens2_sensitivity.json classifications for the listed ids. | **tested, inconclusive** |
| Long-short / shorting single stocks | Two measured long/short books on SP500: pairs (same-sector, monthly) and short-volume-ratio quintiles. Net issuance short side (BIG HiNI) is a French diagnostic, not a short test. Leveraged rotation shorted the index via inverse ETFs. Ids: `stocks-pairs-stat-arb`, `stocks-short-volume-ratio`, `stocks-net-issuance-screen`, `stocks-leveraged-rotation`. | Both books Lens 2 inconclusive, pre-fix tape (anchor leak, SPY duplicates, DSR; SVR also day shift), Lens 6 unverifiable/partly, no raw artifacts. Pairs modelled no borrow fee or margin interest; the record says the monthly cadence may be 'mistimed' and the cadence hypothesis was not tested. Evidence: docs/research/2026-07-17-pairs-stat-arb-nogo.md:20-25,98-105,140-160; docs/research/2026-07-19-short-volume-ratio-nogo.md:9-14; docs/research/2026-10-23-net-issuance-screen-nogo.md:20-25. | **tested, inconclusive** |
| Market-neutral | Same two equity books (pairs realized beta 0.03 to SPY); crypto delta-neutral carry on one year of Kraken funding (8 trades). Ids: `stocks-pairs-stat-arb`, `stocks-short-volume-ratio`, `crypto-funding-carry-phase4`. | As for long-short; carry row has no raw artifact (archived doc only). Crypto detail belongs to Lens 8. Evidence: docs/research/2026-07-17-pairs-stat-arb-nogo.md:98-99. | **tested, inconclusive** |
| Options (as instruments) | No options strategy in registry, code or git history. Option-implied skew as a stock signal was judged infeasible (no free historical chains). The research prompt lists listed options as available at the broker. Ids: `stocks-option-iv-skew`. | Not applicable. Evidence: docs/research/WEB_RESEARCH_CANDIDATES.md:446; docs/research/prompts/web-strategy-research-prompt.md:200-204; src/ggTrader/lab/strategies/__init__.py:56-95 (no options entry). | **never tested** |
| Futures contracts (commodity, index, rates) | None traded or simulated. Prompt states 'no futures'; commodity carry (A2), futures-basis reversal (A4), basis momentum (A20) left untriaged or blocked on paid curve history. Ids: `cross-asset-on-deck-candidate-feasibility`. | Not applicable. Evidence: docs/research/prompts/web-strategy-research-prompt.md:200-205; docs/research/WEB_RESEARCH_CANDIDATES.md:636,724,2245. | **never tested** |
| Commodities via ETFs | Commodity 12-1 trend over 14 commodity ETFs (NO); PDBC as one third of the 20% cross-asset sleeve (static GO shadow-only; trend arm NO). Ids: `cross-asset-commodity-trend`, `cross-asset-static-80-20-sleeve`, `cross-asset-sleeve-trend-arm`, `cross-asset-sleeve-holdout-2007-2011`. | Commodity trend pre-fix contaminated, Lens 6 unverifiable, no artifact. Sleeve rows clean tape but MDE not computable; Lens 2 inconclusive. Evidence: docs/research/2026-07-20-commodity-trend-nogo.md:22-39; docs/research/2026-09-28-cross-asset-sleeve.md:1,11,16. | **tested, inconclusive** |
| International equities (allocation or selection) | Only the FX-hedge overlay used international equity ETFs (EWJ/DXJ, EZU/HEZU), and it tested the currency-hedge ratio, not international equity exposure. Ex-US rotation, EM forecast combination (A19), dash-for-cash EFA and lagged US-to-international tilt are parked/untriaged. The Antonacci SPY/VEU dual momentum was folded into dual_momentum, whose lab version is SP500 single stocks identical to xs_momentum. Ids: `cross-asset-fx-hedge-overlay`. | FX overlay pre-fix contaminated, no artifact (not an international-exposure test in any case). Evidence: docs/research/2026-07-20-fx-hedge-overlay-nogo.md:16-21; docs/research/WEB_RESEARCH_CANDIDATES.md:1372-1379,2217,2048,2433 (parked/untriaged); docs/research/WEB_RESEARCH_CANDIDATES.md:1581-1585 (Antonacci folded into dual_momentum); docs/research/2026-10-19-production-replica-audit-nogo.md:259-263. | **never tested** |
| Intraday (prospective) | Feasibility and source audit only; no intraday bars or returns obtained in the repo. A separate owner-approved observer records one fixed SPY rule prospectively: long 15:30-15:59 ET when the prior-close-to-09:59 return is positive, else flat; $100,000 hypothetical, no leverage, 1/2 bp per side, 252 scheduled slots from 2026-10-30 to 2027-11-03; descriptive output only, no pass/fail. Ids: `stocks-intraday-feasibility`, `stocks-intraday-replication-source-audit`. | Not evaluated: results do not exist yet. Observer lives in a separate worktree and is off-limits; read only. Evidence: docs/research/2026-10-24-intraday-trading-feasibility.md:3-5,92; worktree /home/flynn/.t3/worktrees/ggTrader/research-intraday-stage0a: docs/research/2026-10-27-intraday-observer-activation.md (milestones, timers); docs/research/briefs/2026-10-25-intraday-prospective-observation.md:105-112 (fixed rule); docs/research/2026-10-25-intraday-prospective-protocol-review.md:23-50; commit 6f0362e0 (activation). | **never tested** (prospective) |
| Weekly-rebalanced cross-sectional strategies | Every weights strategy rebalances at month-end; the harness has no weekly path. Pairs and MAX reports name a weekly version as untested. Daily-signal multi-day holds (the ensemble family, overnight gap, event holds of 0-21 days) were tested. Ids: `stocks-pairs-stat-arb`, `stocks-max-effect`. | Not applicable for weekly; daily-signal rows as in the ensemble lines. Evidence: src/ggTrader/lab/data.py:266-271 (decision dates: window start plus month-ends); docs/research/2026-07-17-pairs-stat-arb-nogo.md:140-148; docs/research/2026-07-17-max-effect-nogo.md:105-112. | **never tested** |
| Daily-signal mean reversion on S&P 500 large caps (most-tested area) | 33 of the 72 measured tests are the ensemble/reversion family or its levers (Lens 2 family count). Fixed-default replica: excess -0.64 pp/yr, NW t -0.69. Ids: `stocks-production-replica`, `stocks-production-replica-holdout-2010-2020`, `stocks-sp500-core-pit-rebaseline`, `stocks-ensemble-ic-pit`, `stocks-voter-ablation-5-voter`, `stocks-overnight-gap`, `stocks-mtf-reversion`, `stocks-nasdaq100-single-core`. | Replica MDE 4.24 pp/yr at t>=3 is below the 9.57 pp/yr target gap, so the test could detect a target-sized book effect, but its tape is 'unknown' (PARA identity unfixed) and Lens 6 'partly', so R2 is not met. Older rows pre-fix contaminated; Nasdaq-100 core 'unreliable NO'. Evidence: docs/research/_lens2_sensitivity.json multiple_testing.families['ensemble/reversion and its levers'].n = 33; docs/research/RESEARCH_SNAPSHOT.md:44-51; docs/research/_lens9_power.json rows stocks-production-replica. | **tested, inconclusive** |
| Cross-sectional momentum (large caps) | 12-1 top-N replay (anchor 17/17), dual momentum identical to it, French Big/Small screen fails Holm and the +1.0 pp/yr post-1994 bar. Ids: `stocks-xs-momentum-2026-06`, `stocks-xs-momentum-corrected-replay`, `stocks-dual-momentum`, `stocks-momentum-french-screen`. | Replay clean, Lens 6 yes, but MDE 22.66 pp/yr exceeds the 9.54 pp/yr target gap; French screen MDE not computable. Evidence: docs/research/_lens9_power.json rows; docs/research/2026-10-20-momentum-french-screen-nogo.md:18-31. | **tested, inconclusive** |
| Fundamentals: value, quality, investment, issuance (large caps) | French BIG portfolios over 63 years: H1-H4 and BIG NegNI fail Holm and the +1.0 pp/yr bar (best +0.83). Alternate definitions left 'undetermined'. Ids: `stocks-factor-screen-h1-h4`, `stocks-net-issuance-screen`, `stocks-family-scope-review`. | Clean tape, no Lens 6 row, MDE not computable (no stored mean excess). Portfolio-level, value-weighted, not stock-level PIT fundamentals (none available). Evidence: docs/research/RESEARCH_SNAPSHOT.md:163-167,236-238,285-288. | **tested, inconclusive** |
| Fundamentals at small-cap scale | Only SMALL HiOP/HiBM/LoINV diagnostics outside the confirmatory family; no stock-level small-cap fundamentals data or universe. Ids: `stocks-factor-screen-h1-h4`, `stocks-accruals-feasibility`. | Diagnostic, not registered; clean French data; 3%/6% assumed haircuts. Evidence: docs/research/2026-10-21-factor-screen-nogo.md:125-135. | **tested badly (flawed test)** |
| Event-driven equity (PEAD, deletions, insider, congress) | Four measured NO-GOs (July 2026) on SP500 ever-members (PEAD also Russell 200-name). Event-timing re-audit parked UNRESOLVED (no paid data). Ids: `stocks-pead`, `stocks-index-deletion-fade`, `stocks-insider-cluster-buy`, `stocks-congress-trades`, `stocks-event-timing-audit-decision`. | All on the pre-fix tape with day-shift lookahead on event joins; Lens 6 partly; no raw artifacts; re-audit unresolved. Evidence: docs/research/RESEARCH_SNAPSHOT.md:30-40,236-240; docs/research/_lens1_tape_state.json (day_shift on all four). | **tested, inconclusive** |
| Alternative data (short interest, short volume, attention, sentiment) | Short interest and short volume ratio NO-GO (SP500), headline sentiment provisional NO-GO (50-name pilot), Google Trends built but blocked, analyst revisions and ADD infeasible. Ids: `stocks-short-interest`, `stocks-short-volume-ratio`, `stocks-retail-attention`, `stocks-headline-sentiment`, `stocks-analyst-revisions`, `stocks-anomaly-driven-demand`. | Measured rows pre-fix contaminated (day shift), Lens 6 partly, no artifacts; short-interest literature effect is in small/illiquid names outside the SP500 universe tested. Evidence: docs/research/2026-07-17-short-interest-nogo.md:53; docs/research/RESEARCH_SNAPSHOT.md:132,162. | **tested, inconclusive** |
| ML / LLM signals | LightGBM used only as an entry filter on the ensemble (found anti-predictive, disabled); LLM headline scoring as a primary signal on a 50-name pilot. No ML model as the primary stock-ranking signal. Ids: `stocks-ml-entry-gate-falsified`, `stocks-ml-feature-gate-deploy`, `stocks-headline-sentiment`. | Gate training leaked membership (favours strategy, so its loss stands as direction); Lens 6 unverifiable; no artifact. Sentiment pilot documented as unrepresentative. Evidence: docs/research/RESEARCH_SNAPSHOT.md:155. | **tested, inconclusive** |
| Treasury calendar effects (FOMC, month-end) | Pre-FOMC long-Treasury drift: day-clustered net t 0.981 over 113 dates. Month-end IEF: standalone premium real, SPY-funded overlay adds nothing. Ids: `cross-asset-fomc-drift-corrected-harness`, `cross-asset-fomc-drift-wfo`, `cross-asset-month-end-treasury`. | FOMC corrected study: clean tape, MDE 2.37 pp/yr <= target gap 5.11 (meets R2 and R5); Lens 6 unverifiable so Lens 2 class stays inconclusive. Month-end: MDE 4.94 <= 9.57 but DSR-units item (direction unknown). Evidence: docs/research/_lens9_power.json rows cross-asset-fomc-drift-corrected-harness, cross-asset-month-end-treasury; docs/research/RESEARCH_SNAPSHOT.md:137-138. | **tested, evidence against at the size the test could detect** |
| Cross-asset breadth (rates, commodities, gold, FX via ETFs) | Static 80/20 SPY+TLT/GLD/PDBC shadow GO (CAGR 13.61% vs SPY 14.70%); curve rotation, FX hedge, commodity trend NO-GO; 7 on-deck candidates infeasible or blocked. Ids: `cross-asset-static-80-20-sleeve`, `cross-asset-sleeve-holdout-2007-2011`, `cross-asset-sleeve-trend-arm`, `cross-asset-treasury-curve`, `cross-asset-fx-hedge-overlay`, `cross-asset-commodity-trend`, `cross-asset-on-deck-candidate-feasibility`, `cross-asset-etf-implied-fx-expectations`. | Sleeve rows clean but MDE not computable; others pre-fix contaminated, no artifacts. Untriaged: A2, A4, A6, A9, A11, A13, A17-A21, B1-B4, B6-B8, B12. Evidence: docs/research/RESEARCH_SNAPSHOT.md:75-81,135-141. | **tested, inconclusive** |
| Leverage and exposure (A23, LETF timing, century study, vol scaling) | 70/30 SPY/SSO owner allocation (after-tax +1.6 pp/yr, +0.77-0.80 at today's funding, below +1.0 bar); A24 rider NO-GO; century study keeps 30% (-91.4% drawdown); LETF rotation/trend NO-GO incl. Russell 2000 TNA/UWM; vol-managed SPY sanity check only; B13 drawdown control untested. Ids: `stocks-a23-static-leverage`, `stocks-a24-sleeve-trend-rider`, `stocks-century-leverage`, `stocks-leveraged-rotation`, `stocks-leveraged-trend`, `stocks-vol-managed-spy`, `stocks-two-symbol-sweep-design`. | A23/A24/century clean tape but no t-statistic (MDE not computable); LETF rows pre-fix contaminated, no artifacts. Evidence: docs/research/RESEARCH_SNAPSHOT.md:82-92,142-147,159-161; docs/research/WEB_RESEARCH_CANDIDATES.md:1573,2681. | **tested, inconclusive** |
| Crypto perpetuals / funding carry (Lens 8 owns detail) | One year of real Kraken funding, +0.45%, 8 trades; three-year basis proxy 'directionally credible'; lab candidate infeasible (one year of history). Legacy carry modules were deleted from the tree (git: src/ggTrader/strategies/carry/funding_carry.py). Ids: `crypto-funding-carry-phase4`, `crypto-perp-funding-carry-candidate`. | Archived doc, no raw artifact; Lens 1 unknown; Lens 2 inconclusive. See the Lens 8 report for venue, fees and benchmark. Evidence: docs/research/_investigation_inventory.json rows; docs/research/WEB_RESEARCH_CANDIDATES.md:136. | **tested, inconclusive** |
| Crypto spot (Lens 8 owns detail) | Kraken/Binance.US spot WFO runs (best +13.8% vs BTC +38.8%), IBIT session sanity check, BTC trend literature only. Ids: `crypto-edge-search-2026-06-08`, `crypto-wfo-textbook-reset`, `crypto-ibit-overnight-session`, `crypto-tsmom-btc`. | No raw artifacts; Lens 1 unknown; deferred to Lens 8. Evidence: docs/research/_investigation_inventory.json rows. | **tested, inconclusive** |

### Structural advantages of a small account: what the record contains

| advantage | record | classification above |
|---|---|---|
| No capacity limit: micro and small caps | no micro-cap universe; one small-cap stock arm on a 200-name snapshot; French Small portfolios as index-level screens | micro caps never tested; small-cap selection tested badly; French Small screens inconclusive |
| Concentration (few names, large weights) | only top_n 10 inside a WFO that deployed top_n 50 in 17/17 folds; live and replica books hold up to 3 names at about 3% each (9.37% mean stock exposure) | tested badly |
| Short horizons and fast turnover | daily-signal multi-day holds tested; weekly cross-sectional rebalancing never; intraday only prospective | weekly never tested; intraday prospective |
| Instruments a large fund may avoid (options, small illiquid names) | no options strategy; options-skew signal infeasible | never tested |
| Shorting single stocks | two S&P 500 long/short books with no borrow-fee model | tested, inconclusive |

The research prompt describes the account as unable to use futures, spot FX or leverage, with
one daily decision, and tells researchers to avoid capacity-dependent ideas, but it does not ask
for ideas that use the account's small size
(`docs/research/prompts/web-strategy-research-prompt.md:184-205`).

## Hypothesis sourcing

### Where the 72 measured tests came from

Counting rule: each measured inventory row gets one channel. "Published (direct)" = the study's
own brief anchors on a named paper and it did not arrive through a web batch; "LLM/web batch" =
it arrived through a batch in `docs/research/WEB_RESEARCH_CANDIDATES.md` (most cite papers);
"practitioner" = technical-analysis indicators; "internal" = the project's own observation or a
lever on an existing strategy.

| channel | measured tests | of which tied to a published paper |
|---|---|---|
| (a) published academic anomaly, direct (momentum screens and replays, factor and net-issuance screens, idio_vol, pairs) | 8 | 8 |
| (b) LLM/web research batches | 19 | 16 |
| (c) internal observation or construction lever | 31 | 2 |
| (d) practitioner lore (RSI, Bollinger, MACD, EMA cross ensembles) | 14 | 0 |
| total | 72 | 26 |

Every web batch was produced by running the project's prompt in an LLM research tool (Gemini,
ChatGPT deep research, Claude web research) and pasting the results back
(`docs/research/WEB_RESEARCH_CANDIDATES.md:9-25`, `docs/research/WEB_RESEARCH_CANDIDATES.md:1536-1545`, `docs/research/WEB_RESEARCH_CANDIDATES.md:2068-2075`, `docs/research/WEB_RESEARCH_CANDIDATES.md:2446-2456`, `docs/research/WEB_RESEARCH_CANDIDATES.md:2808-2815`).
The ensemble voters' origin is not stated in the record; "practitioner" is this lens's label for
standard technical indicators.

### Web-research backlog: how many were never run

Counting rule: an entry is a numbered heading (1-14, A1-A27, B1-B15, C1-C11) or a parked bullet
under a "D. Parked" or "Parked this batch" heading; C-items are methods, not trading hypotheses;
summary lines at `docs/research/WEB_RESEARCH_CANDIDATES.md:3173-3180` are not entries. "Run" = a
measured test exists (an inventory row with a measured verdict type, or a registry strategy with
a report). "Sanity check" = a one-off pandas check in a batch artifact folder.

| batch | trading entries | run | built, no result | sanity check only | never run | parked bullets (of which never run) |
|---|---|---|---|---|---|---|
| 2026-07-16 | 14 | 7 (#1, #3, #5, #9, #11, #12, #13) | 1 (#8) | 0 | 6 | 0 |
| 2026-07-19 | 13 | 5 (A1, A3, A5, A7, A8) | 0 | 0 | 8 | 11 (10; crypto session got a sanity check) |
| 2026-09-23 | 14 | 1 (A10) | 0 | 3 (A12, A14, B5) | 10 | 5 (5) |
| 2026-09-28 | 7 | 0 | 0 | 0 (A12 variant re-checked) | 7 | 5 (5) |
| 2026-09-30 | 3 | 2 (A23, A24) | 0 | 0 | 1 | 1 (1) |
| 2026-10-25 | 5 | 0 | 0 | 0 | 5 | 4 (4) |
| **total** | **56** | **15** | **1** | **3** | **37** | **26 (25)** |

So 62 of 82 trading hypotheses (37 + 25) were never run; 15 were. A further 14 candidates were
rejected on intake in the four later batches (2 + 4 + 6 + 2). The 2026-07-16 batch was itself
scoped away from technical signals because "the technical-signal category was already
exhausted" (`docs/research/WEB_RESEARCH_CANDIDATES.md:36-40`). The 2026-10-20 strategy triage
(32 flawed-test, 8 untested, 4 clean NO) is not in the repository (spec section 4), so it was not
used.

### Published anomalies and decay

- **External evidence:** McLean and Pontiff, "Does Academic Research Destroy Stock Return
  Predictability?", Journal of Finance 71(1), 5-32 (2016), doi 10.1111/jofi.12365, free text at
  https://gwern.net/doc/economics/2016-mclean.pdf. Abstract: 97 cross-sectional predictors;
  portfolio returns 26% lower out of sample and 58% lower post-publication; post-publication
  correlations with other published predictors rise; returns are higher in stocks with high
  idiosyncratic risk and low liquidity. Verified from the abstract as shown on several indexes;
  the full PDF was not read. It covers U.S. cross-sectional stock predictors only, not Treasury,
  commodity, FX or leverage rows.
- **How many published tests it bears on:** of the 26 measured tests tied to a paper, 16 are
  cross-sectional U.S. stock anomalies (PEAD; momentum in four rows; value/profitability/
  investment; net issuance; MAX; idiosyncratic volatility; short interest; short-volume ratio;
  insider buying; congressional trades; index deletion; pairs; headline sentiment). Whether each
  is among McLean and Pontiff's 97 is **unverified** (appendix not read). The other 10 are
  Treasury, commodity, FX, calendar or crypto effects.
- **The record's own decay measurements:** momentum Small t 3.38 (1927-1993), 1.11 (1994-2012),
  -0.77 (2013-2026) (`docs/research/2026-10-20-momentum-french-screen-nogo.md:83-90`); SMALL
  factor portfolios -6.3 to -6.8 pp/yr after 2013
  (`docs/research/2026-10-21-factor-screen-nogo.md:131-135`); PEAD reported gone in non-microcap
  stocks by 2006 and pre-FOMC equity drift gone after 2015, both as cited
  (`docs/research/WEB_RESEARCH_CANDIDATES.md:398-403`, `docs/research/WEB_RESEARCH_CANDIDATES.md:1593-1598`); fast trend Sharpe 0.84 to
  0.12 after 2009, as cited (`docs/research/WEB_RESEARCH_CANDIDATES.md:2165`); the IBIT
  overnight pattern reversed after a dedicated ETF launched
  (`docs/research/WEB_RESEARCH_CANDIDATES.md:1336-1345`).
- **Where the published effects are said to live:** the record notes that the cited effect
  concentrates in small or illiquid names outside the tested S&P 500 universe for short interest
  (`docs/research/WEB_RESEARCH_CANDIDATES.md:126-130`,
  `docs/research/2026-07-17-short-interest-nogo.md:53`) and for PEAD (lines 398-403). McLean and
  Pontiff report the same for predictor returns generally.

### Crowding and correlation

- **Within the project:** 33 of 72 measured tests are the ensemble/reversion family or its levers
  (Lens 2 `multiple_testing.families`); 44 of 51 single-stock tests are large-cap only. Nine
  consecutive equity diversification sleeves failed
  (`docs/research/briefs/2026-09-28-cross-asset-sleeve.md:308`); the PEAD and idio_vol return
  streams had 0.42 and 0.45 correlation to the core
  (`docs/research/2026-07-17-pead-nogo.md:62-65`).
- **Outside the project (crowding by other traders):** the record asserts but does not measure
  that technical signals on this universe are "close to fully arbitraged"
  (`docs/research/prompts/web-strategy-research-prompt.md:170-172`). No crowding measure was
  computed for any family; the one crowding-flow candidate (anomaly-driven demand) was
  infeasible. Family-level crowding is **unknown** beyond McLean and Pontiff's general finding.

## Registry versus inventory

All 38 registry entries have at least one inventory row; none is missing
(`registry_vs_inventory` in the JSON). Six per-universe leveraged entries share two combined rows,
so the Russell 2000 and Nasdaq-100 variants have no separate row. Several measured tests have no
registry entry (A23, A24, century leverage, the French screens, the ML gate, MidCap 400
reversion, crypto carry, IBIT). Git history also holds deleted legacy modules (crypto carry,
cross-sectional momentum, a hidden-Markov-model regime filter) that predate the lab; Lens 8 owns
the crypto ones.

## Questions and answers

**Q1. What was never tested, as opposed to tested badly?**
Answered. Never tested: micro caps; options as instruments; futures contracts; international
equity exposure (the FX-hedge overlay used international ETFs only to time a currency hedge);
weekly-rebalanced cross-sectional strategies (the harness decides at month-ends only,
`src/ggTrader/lab/data.py:266-271`); intraday (prospective, below). Tested badly: small-cap stock
selection (200-name snapshot), concentration (arm never deployed OOS), small-cap fundamentals
(diagnostics only). The spec's candidates: small and micro caps (above); concentration (tested
badly); long-short (tested, inconclusive: two pre-fix S&P 500 books without borrow costs);
cross-asset breadth (tested, inconclusive, except Treasury calendar effects, where the corrected
FOMC study had power for a target-sized effect and found none; Lens 2 still classes it
inconclusive because Lens 6 could not verify its code). Intraday is not a gap: an owner-approved
observer records one fixed SPY rule prospectively (long 15:30-15:59 ET when the
prior-close-to-09:59 return is positive, else flat; $100,000 hypothetical; no leverage; 1/2 bp per
side; 252 slots from 2026-10-30 to 2027-11-03; descriptive output, no pass mark), per
`/home/flynn/.t3/worktrees/ggTrader/research-intraday-stage0a/docs/research/briefs/2026-10-25-intraday-prospective-observation.md`
(lines 105-112) and the activation record in the same worktree (commit 6f0362e0). Its results
were not read or evaluated.

**Q2. Where did hypotheses come from, how correlated or crowded are they, and were most of them
published anomalies known to decay after publication?**
Answered in part. Sources: 31 internal, 19 LLM/web batches, 14 practitioner indicators, 8
published direct; 26 of 72 are tied to a paper, so most tested hypotheses were **not** published
anomalies. Most of the published ones (16 of 26) are U.S. cross-sectional stock anomalies of the
kind McLean and Pontiff find lose about half their return after publication, and the project's
own long-history screens show that decay. Correlation within the project is high (46% of tests
in one family; 86% of single-stock tests large-cap). External crowding per family is
**unknown**: no crowding measure exists in the record. Cheapest test: map each of the 16 stock
anomalies to McLean and Pontiff's appendix (free PDF) and record publication year against each
test window.

**Steelman question: what is the strongest evidence that coverage was broad, and does it hold?**
See Steelman: broad across asset classes, narrow within stock selection.

**Which structural advantages of a small account were never used?** Micro caps, options and a
weekly cadence were never used; small caps and concentration appear only in flawed tests (table
above). This lens does not say whether any of them would help.

## Verification

Commands run (all read-only):

- `python3 -c` loads of the inventory, Lens 1/2/6/9 JSON and the four Haiku files: 144 inventory
  ids, 144 Haiku ids (all matched), Lens 2 114 rows (the 30 ids without a row are operations or
  infrastructure), Lens 9 144 rows.
- Haiku null counts out of 144: universe 45, market-cap 57, horizon 95, holding period 119,
  direction 112, idea source 46, published flag 58, data 53, sizing 94.
- Source reads (grep and sed on the cited lines) for 42 rows: PEAD, short-volume ratio, pairs,
  short interest, MAX, index deletion, insider, congress, headline sentiment, commodity trend,
  leveraged trend and rotation, Treasury curve, FOMC WFO, month-end Treasury, cross-asset sleeve,
  century leverage, FX hedge, momentum French screen, factor screen, net issuance, xs_momentum
  replay and dual momentum, sizing regimes, diversification sleeves, intraday feasibility, A23,
  IBIT, production replica, equity pre-FOMC intake, turn-of-month, vol-managed SPY, rebalancing
  tilt and threshold tilt, option skew, analyst revisions, perp carry candidate, anomaly-driven
  demand, retail attention, DeFi, Form 144, MidCap 400 (inventory quotes only).
- Haiku check on those 42 rows (378 fields): 105 were null and were filled by this lens; among
  non-null fields one error was found and corrected (leveraged rotation horizon is monthly
  breadth, `docs/research/2026-07-16-leveraged-index-rotation-nogo.md:11`); no market-cap or
  direction value contradicted its source.
- Registry: parsed `src/ggTrader/lab/strategies/__init__.py:56-95` (38 names) and asserted each
  maps to an inventory id.
- `wc -l data/universe/*.txt`: Russell 2000 snapshot 200 lines, MidCap 400 400, Nasdaq-100 101,
  S&P 500 503; no small- or micro-cap PIT file.
- `git log --all -i --grep` for russell/small-cap/micro (12 subjects, none a micro-cap strategy;
  the only "micro" subject is the crypto tick-size fix 2ac0ff71), futures (4, none a futures
  strategy), shorting (0), options (none an options strategy), international/country (2, docs
  only), weekly (2: pairs report, MTF weekly RSI), intraday (16, the observer track);
  `git log --diff-filter=D` lists deleted legacy carry, momentum and regime-filter modules.
- `src/ggTrader/lab/data.py:266-271` and `src/ggTrader/lab/strategies/momentum.py:25` read.
- Web-research backlog headings, statuses and parked bullets listed by a scratch script; batch
  intake counts read from each batch header.
- McLean and Pontiff abstract confirmed via web search results (SSRN returned 403).
- Observer documents read in the worktree; timers, worktree and archive not touched.
- `.venv/bin/python scripts/check_lens_report.py docs/research/2026-11-04-lens5-coverage-gaps.md`:
  exit 0.

What remains unverified:

- 102 of 144 rows keep this lens's normalization of the Haiku lead without a source re-read;
  most are operations, status or legacy crypto rows not used in the gap table.
- Whether each published anomaly is in McLean and Pontiff's 97, and each paper's publication
  date relative to the test window.
- The 2026-10-20 private triage (not in the repository).
- Crypto coverage beyond pointers (Lens 8 not yet committed).
- Intraday observer results (not read by design).

## Done checklist

- ✅ Falsifiers written before evidence (F1-F4, N1-N2, Phase 0 H1 and H7).
- ✅ Steelman stated and tested.
- ✅ Coverage matrix: all 144 ids once, null counts reported, 42 rows verified.
- ✅ Gap table: 26 areas, each from cited rows with Lens 1/2/6/9 quality and a fixed-rule class.
- ✅ Q1 (never tested versus tested badly): answered.
- ✅ Q2 sources and decay: answered.
- ❓ External crowding per family: unknown (no measure in the record; cheapest test: map the 16
  stock anomalies to McLean and Pontiff's appendix and publication dates).
- ✅ Web-research backlog never-run count with a stated rule (62 of 82).
- ✅ Registry versus inventory (38 of 38 mapped).
- ❓ Crypto detail: unknown here; deferred to Lens 8 (not yet committed).
- ✅ No recommendations; intraday described, not evaluated.
