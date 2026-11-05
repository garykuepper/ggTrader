# Brief: a fair re-test of stock momentum for a taxable trading sleeve

**Status:** pre-registered 2026-10-20, before any result existed. The owner
handed the open decisions to the senior quant ("you are the senior quant, go with
your recommendations"), and §11 lists the calls made under that delegation. The
owner then approved this brief as written, on 2026-10-20. Its numbers are now the
bar, and changing any of them needs a new dated brief.

**Terms used below:**
- *Momentum* means buying the stocks with the biggest gains over the past year,
  leaving out the most recent month.
- A *t-stat* measures how sure we can be that an extra return isn't luck. Here
  t ≥ 3 means roughly a 1-in-740 chance that luck alone explains it.
- *Newey-West (NW)* is a t-stat method that allows for returns in one period
  being correlated with the next.
- *PIT* (point-in-time) means using index membership as it was on each date,
  not today's list.
- *Survivorship-free* data includes the companies that were later delisted or
  went bust.

## 1. The question, and the decision it feeds

- **Question.** Can a momentum stock sleeve beat SPY after tax, in a taxable
  account, by enough to justify real money? It has to earn that on evidence that
  isn't luck.
- **Decision.** Whether to stand up a momentum sleeve: first on paper at the
  parity-freeze lift (about 2026-12-29), then as a small real-money pilot.
- **Context.** The deployed ensemble stock picks are a NO-GO and retire at the
  freeze lift whatever this study finds
  (`docs/research/2026-10-19-production-replica-audit-nogo.md`).
- **Not affected.** The owner's A23 allocation (70/30 SPY/SSO) is a separate
  decision and does not change.

## 2. Why the existing evidence can't answer it

- **The only positive lead is too short to judge.** Momentum is the one strategy
  that came out ahead on the corrected harness.
  - It made +4.37 pt/yr CAGR over SPY.
  - But the NW t was only 0.76, with tracking error of 12.4%/yr over 1,064
    out-of-sample days, 2022 to 2026-04.
  - Source: `_audit_reruns_20261019.json`, `xs_momentum`.
  - `dual_momentum` produced identical results: its negative-momentum filter
    never bound at top-50.
- **Short windows can't reach the bar.** At an information ratio (IR, excess
  return ÷ tracking error) of about 0.36, reaching t = 3 takes about 70 years of
  data. Even 21 years would need an IR of 0.65 or more.
- **Our clean data is short.** The only clean point-in-time stock tape is the S&P
  500 from 2021 to 2026-04. "Clean" means it passed the artifact detector, not
  that it is independently certified. Before 2021, yfinance prices for delisted
  names are unusable: the 2010–2020 holdout had 72 contaminated tickers.
- **Our tax code is too simple.** The existing after-tax Book
  (`scripts/static_leverage_aftertax.py`) handles three ETFs, with approximate,
  backward-only wash sales. It is not a multi-stock lot engine.
- **Other indexes have no usable history.** Only the S&P 500 has a PIT membership
  history in the repo. The MidCap 400, Nasdaq-100 and Russell 2000 lists are
  today's members, which carries survivorship bias.

## 3. The standard: two parts, owner-delegated 2026-10-20

The original bar asked for a NW t ≥ 3 on our own implementation. That is
unreachable at any realistic IR on the data that exists. The standard therefore
has two parts:

- **Part A, the idea.** Momentum must reach t ≥ 3 on about 100 years of
  independent, survivorship-free data (Phase 1).
- **Part B, our version.** Our exact rules must beat SPY by at least
  +1.0 pt/yr *after tax*, on clean data not used to design them (Phase 3). Their
  t is reported but not required.

This makes our version partly a bet on a proven idea. Sizing and the stop rule in
§8 cap the cost of that bet being wrong.

## 4. Phase 0: fix the record (housekeeping, before Phase 1)

1. **dual_momentum.** Record it as identical to `xs_momentum` *on this replay*,
   because the filter never binds at top-50. It is not a separate test. Fix the
   report §2 table and the snapshot row.
2. **Corrected DSR gate.** With one-year training folds and the 12-setting grid,
   passing needs an in-sample Sharpe of roughly 2.5. Walk-forward output is
   therefore mostly the per-fold anchor. Note this in `docs/next_steps.md`, and
   state the gate's power limits without dismissing its rejections.
3. **FOMC.** Compute the event t-stat clustered by meeting date, since TLT, IEF
   and EDV share each FOMC day. Save it as an artifact and amend the snapshot and
   report rows. Do not cite the ~2.1 estimate.
4. **Research prompt templates.** Regenerate them with the `research-snapshot`
   skill. They still cite KEEP's 0.80 and "gates 17/17".

## 5. Phase 1: is momentum real, and in which company size? (free data)

- **Data.** Ken French Data Library, monthly, value-weighted:
  - *6 Portfolios Formed on Size and Momentum (2 x 3)*
    (`6_Portfolios_ME_Prior_12_2`). Use **Big HiPRIOR** (large-cap winners) and
    **Small HiPRIOR** (small-cap winners). Breakpoints are the NYSE median size
    and the 70th percentile of prior 12-2 month return. Big HiPRIOR is the
    nearest analogue to an S&P 500 momentum book, but it is not the S&P 500 and
    not top-decile momentum.
  - *Fama/French 3 Factors* (`F-F_Research_Data_Factors`). Market return is
    `Mkt-RF + RF`.
- **Freezing the data.** Commit the raw downloaded files with their download date
  and SHA-256 checksums before computing anything.
- **Hypotheses (two).** For each of *g* = Big and Small, the winners' return beats
  the market's.
- **Series.** Monthly excess return e_t = R_winners,t − R_mkt,t − c/12, where c is
  the annual cost haircut:
  - **base:** 1.0% for Big, 3.0% for Small;
  - **stress:** 2.0% and 6.0%.

  These are assumptions, not estimates: aggregate portfolio returns can't tell us
  turnover.
- **Test A1, significance.** Over 1927-01 → 2026-04 at base cost, compute the NW t
  with 12 lags.
  - Apply Holm across the two hypotheses at a family-wise one-sided level of
    p = 0.00135, which is t = 3.0 for a single test.
  - The stronger hypothesis needs p ≤ 0.000675 (about t ≥ 3.2); the other needs
    p ≤ 0.00135 (t ≥ 3.0).
- **Test A2, still works since publication.** Over 1994-01 → 2026-04, the winners'
  CAGR net of the base haircut minus the market's CAGR must be at least
  **+1.0 pt/yr**.
- **Pass rule.** A size group passes Phase 1 only if it passes both A1 and A2.
- **Diagnostics** (reported, not pass criteria):
  - subperiods 1927–1993, 1994–2012 and 2013–2026;
  - stress-cost results;
  - CAPM alpha and beta;
  - winners minus the same-size market segment;
  - the worst 12-month relative returns, i.e. crash episodes.
- **What happens next:**

| Phase 1 result | Next |
|---|---|
| Neither size passes | Momentum is closed and reported as NO-GO. No paid data. |
| Big passes | Phase 2 on the S&P 500. |
| Only Small passes | Skip Phase 2, since we have no small-cap PIT data. Phase 3 runs on paid small-cap data, and paper trading waits until Phase 3 passes. |
| Both pass | Phases 2–3 test the S&P 500 version as the primary, and only it can authorize paper or real money. Small caps run in Phase 3 as a diagnostic only; deploying them needs a brief of their own. |

- **Authority.** Passing Phase 1 authorizes more research only. It is not a
  deployment signal.

## 6. Phase 2: what do costs and taxes take from our version? (can only kill)

This window has already been seen, so this phase can rule momentum out but
cannot prove it.

- **Window.** 2021-01-31 → 2026-04-30, the replica audit's primary window, with
  420 calendar days of warm-up. This is a fixed-rule replay with no parameter
  selection.
- **Universe.** PIT S&P 500 members on each decision date (`sp500_members_asof`).
  A stock is eligible once it has at least 253 bars.
- **Ranking**, shared by both rules: P(t−21) / P(t−252) − 1, from adjusted
  closes on the decision date. Ties are broken by ticker.
- **Rule B, base** (diagnostic; continuity with the +4.37 run):
  - Decisions on the last trading day of each month.
  - Hold the top 50 at equal weight, rebalancing continuing names back to equal
    weight.
  - Fill at the next session's close, at 5 bp per side.
- **Rule T, tax-aware** (the **primary**):
  - **Timing.** Decisions on the last trading day of each quarter; fills at the
    next session's close, 5 bp per side.
  - **Locking.** Re-check the lock at every quarter-end. A held name outside the
    top 50 is *kept* only if at least one of its lots both shows an unrealized
    gain and has been held 365 days or less. Otherwise it is sold at that
    quarter-end. So a locked name whose gain turns into a loss is sold at the
    next quarter-end as a short-term loss, and one whose gaining lots have turned
    long-term is sold too.
  - **Slots.** Locked names take up some of the 50 slots. Fill the remaining
    slots with the highest-ranked names not already held, splitting available
    cash equally among the additions.
  - **Limits.** No trimming of held names, no borrowing, and no daily loss
    harvesting.
  - **Re-entry.** A name sold at a loss can't be bought again for 31 days. If the
    ranking says buy during that lock, take the next rank instead.
- **Cash.** Uninvested cash earns the 3-month T-bill rate (FRED `TB3MS`), taxed
  as ordinary income.
- **What the tax model covers.**
  - The sleeve starts with **$100,000** notional and is modelled standalone: its
    net losses offset only its own gains.
  - In the primary run, the $3,000 ordinary-income offset is **not** credited,
    and any carryforward left unused at terminal liquidation is valued at zero.
    This keeps the result independent of the owner's other income and of
    account size.
  - **Sensitivity run:** $3,000 a year is credited at 24% into sleeve cash, and
    any unused carryforward is valued at 15%.
- **Tax engine.** This must be built and unit-tested before any Phase 2 run.
  - **Lots.** Keep per-lot basis and acquisition date. Partial sales take the
    highest-basis lot first (HIFO).
  - **Wash sales** ([IRS Pub 550](https://www.irs.gov/publications/p550)):
    - A loss is disallowed when the same security is bought within 30 days
      before or after the sale.
    - The disallowed loss is added to the replacement lot's basis.
    - The replacement lot's holding period includes the original lot's.
  - **Netting.** Net short-term and long-term gains each year. A net loss offsets
    up to $3,000 of ordinary income; the rest carries forward with its
    short-term or long-term character kept.
  - **Dividends.** Qualified (taxed at the long-term rate) if the lot had been
    held at least 61 days by the ex-date; otherwise taxed at the ordinary rate.
    This simplifies the 121-day rule.
  - **Paying the tax.** Tax is paid at each year-end from sleeve cash, selling
    HIFO lots if needed.
  - **Rates.** 24% short-term and 15% long-term (primary), plus 40.8% / 23.8%
    as a sensitivity.
- **Endpoints.**
  - **Primary:** terminal liquidation at the window's end, with tax paid.
  - **Secondary:** no liquidation, valued net of the deferred tax liability.
- **Benchmarks.** Each is held as buy-and-hold under the same tax engine:
  - **SPY**, with dividends taxed as qualified and reinvested.
  - **MTUM** (iShares MSCI USA Momentum Factor ETF, launched April 2013), using
    its actual distributions.

  MTUM is the investable alternative, not a replication of our method.
- **Kill rule.**
  - **vs SPY:** if Rule T's after-tax CAGR minus SPY's is below **+1.0 pt/yr**
    (primary endpoint, 24/15), momentum is closed as NO-GO.
  - **vs MTUM:** if T beats SPY by that much but trails MTUM after tax, the
    recommendation becomes "hold MTUM instead of running it ourselves". The owner
    may still choose to run it themselves; that choice is recorded.
  - **Rule B** is reported but decides nothing.
  - **Significance:** NW t is reported, not required.

## 7. Phase 3: our exact rules on 2000–2020 (paid data; required before real money)

- **When to buy.** Buy the data only if Phase 1 passed and Phase 2, if run, did
  not kill. The vendor is chosen at that point.
- **Data requirements:**
  - daily survivorship-free US equity prices, including delisted names with their
    delisting returns or proceeds;
  - split and dividend events;
  - security identifiers that survive ticker changes;
  - historical index membership for the S&P 500, plus the S&P 600 if Phase 1
    favoured small caps;
  - coverage from 1998-06 (for warm-up) through 2020-12.
- **Run.** Rules B and T exactly as frozen in §6, from 2000-01-31 → 2020-12-31,
  using the same tax engine and benchmarks.
  - MTUM only counts over the common dates, 2013-04-30 → 2020-12-31.
  - For small caps, the benchmarks are SPY and IJR (the S&P 600 ETF).
- **Pass bar (Part B of the standard):**
  - Rule T's after-tax CAGR (primary endpoint, 24/15) minus SPY's is at least
    **+1.0 pt/yr** over the full window.
  - T does not trail MTUM after tax over the common dates. If it does, the
    recommendation is MTUM, as in §6.
  - For small caps, T must also not trail IJR after tax, so the size premium
    isn't mistaken for skill.
  - NW t is reported, not required.
- **Subperiods reported:** 2000–2009 (including the 2009 momentum crash) and
  2010–2020.

## 8. Phases 4–5: paper trading and real money (terms fixed now)

- **At the freeze lift (about 2026-12-29).** Retire the ensemble picks regardless
  of this study.
- **Phase 4, paper.** Starts at the freeze lift only if *Big* passed Phase 1 and
  Phase 2 did not kill. It does not wait for Phase 3. If only Small passed, see
  §5.
  - **What runs.** Rule T on the Alpaca paper account. This needs a new live
    path, built under its own plan and deployed through `ggtrader-deploy` after
    the freeze.
  - **Length.** 18 months, so lots actually reach long-term exits.
  - **What it checks: mechanics only.**
    - every order is explained by the rules (100% parity);
    - lots and basis match the broker;
    - splits and dividends are handled correctly;
    - no halts.
  - **Faults.** Any operational fault halts trading until it is fixed.
  - **Performance.** Recorded, not judged.
- **Phase 5, real money.** Only after Phase 3 passes *and* at least 6 months of
  clean paper trading.
  - **Size.** A pilot of **5%** of investable money, funded from the SPY/VOO
    core and never from A23's SSO leg.
  - **Growth.** At most **10%**, and only after 18 months of clean paper trading
    plus at least 12 months of the pilot without a stop.
  - **Reviews.** On fixed dates: the last trading day of each quarter.
  - **Stop.** If the sleeve's cumulative after-tax wealth since inception trails
    the same money held in SPY by **15% or more**.
    - There is no absolute-drawdown stop, because one would sell in every bear
      market.
    - Operational faults halt it immediately.
  - **Reporting.** Whole-account results are reported against the SPY/A23
    allocation the sleeve displaced.

## 9. Data-quality stop conditions

No verdict is issued while any of these is unresolved.

- **Phase 1:** a checksum mismatch, or missing months in either file.
- **Phases 2–3, price artifacts:**
  - Scan for daily moves of 40% or more and check them against an independent
    source.
  - If any trade falls within 5 bars of an unverified artifact, stop and
    investigate.
  - Same rule as `scripts/production_replica_audit.py`.
- **Phases 2–3, coverage:** below 95% of PIT member-days in Phase 2, or below 99%
  in Phase 3 (paid data should be complete; anything less is a vendor problem).
- **Tax engine:** wash-sale and carryforward unit tests, built from the
  Pub 550 examples, must pass before any Phase 2 run.
- **Identity:** a security-identity or delisting-proceeds audit failure on any
  name the rules trade.

## 10. Diagnostics, scope, and what's left to the builder

**Diagnostics** (reported in every phase where they apply, never pass criteria):
- sector concentration;
- beta to SPY;
- annual turnover;
- worst relative drawdown versus SPY, and time to recover;
- rebound stress (2009 in Phase 1; 2020 and 2022–23 in Phases 2–3);
- tax drag broken down into short-term, long-term, dividend and cash;
- for Rule T, how far the tax-aware holdings drift from pure momentum (the share
  of slots taken by locked names).

**Multiple testing.** Two hypotheses in Phase 1, Holm-adjusted. One primary rule
(T) in Phases 2–3; Rule B is diagnostic only.

**Out of scope:**
- any new strategy search, and any parameter tuning;
- volatility scaling, and sector caps (concentration is reported, not capped);
- the Nasdaq-100, since it overlaps the S&P 500 and amounts to a tech bet;
- daily loss harvesting;
- any live change before the freeze lift;
- IRA or retirement accounts;
- timing overlays;
- changes to A23.

**Anti-goals** (failure even if the work runs):
- a GO that rests on the 2021–2026 window;
- any change to the rules after seeing a result;
- tax-awareness quietly changing the momentum exposure without that drift being
  measured.

**Left to the builder:**
- code structure and file locations;
- download and parsing details;
- whether to extend `src/ggTrader/lab/replica.py` or write a weights engine;
- tax-engine internals beyond the rules in §6, and test design;
- which paid vendor to recommend at Phase 3 (the owner approves the purchase).

## 11. Calls made under the owner's delegation (check these)

1. **The two-part standard** (§3) replaces the implementation-level t ≥ 3.
2. **Phase 1 assumptions:**
   - cost haircuts of 1% / 3% a year, stressed at 2% / 6%;
   - windows 1927–2026 and 1994–2026;
   - Holm across the two sizes.
3. **Rule T is primary** and Rule B is diagnostic.
4. **MTUM:** if Rule T trails it after tax, the recommendation is MTUM.
5. **Dividend treatment:** the simplified 61-day rule, and HIFO lot selection.
6. **Paid data** is bought only after Phases 1–2, and only if they pass.
7. **Paper trading:** 18 months, starting at the freeze lift if Phases 1–2 pass.
8. **Real money:**
   - a 5% pilot after Phase 3 plus 6 months of clean paper;
   - a 10% cap;
   - quarterly reviews;
   - a stop at 15% behind SPY, with no absolute-drawdown stop.
9. **Tax model:** standalone, starting at $100,000, with no $3,000
   ordinary-income credit and unused carryforward valued at zero in the primary
   run. The sensitivity run credits both. This choice is conservative toward
   Rule T.
10. **Rule T's lock is re-checked every quarter.** A locked name whose gain turns
    into a loss is sold as a short-term loss.
11. **Small caps:** if only Small passes, paper trading waits for Phase 3. If
    both pass, the S&P 500 version is the only one that can authorize capital,
    and small caps are a diagnostic that would need their own brief.
