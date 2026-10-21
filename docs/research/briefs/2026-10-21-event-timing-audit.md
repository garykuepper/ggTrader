# Brief: event-timing and data-integrity audit for PEAD and insider cluster buying

**Status:** DRAFT 2026-10-21, awaiting owner approval. Nothing has been computed.

**Terms used below:**
- *PEAD (Post-Earnings-Announcement Drift)* means the tendency for a stock's price to continue drifting in the direction of an earnings surprise (actual reported earnings versus consensus expectation) for weeks or months after the announcement.
- *Earnings surprise* means the percentage difference between reported earnings per share (EPS) and the pre-announcement consensus estimate: `(Reported - Estimate) / |Estimate|`.
- *Insider cluster buying* means multiple corporate insiders (such as officers, directors, or major shareholders) purchasing shares of their own company on the open market within a short window of time.
- *SEC Form 4* is the mandatory disclosure filing submitted to the SEC when corporate insiders acquire or dispose of company securities.
- *Acceptance timestamp* means the exact date, hour, minute, and second when the SEC's EDGAR system officially received and accepted a filing (`acceptanceDateTime`), making it publicly accessible to the market.
- *10b5-1 plan* is a pre-arranged trading plan that automatically buys or sells shares on pre-set dates, indicating a scheduled/routine trade rather than a discretionary, opportunistic purchase.
- *Point-in-time (PIT)* means data reflects only information available at the exact historical decision moment, without revisionist adjustments, future restatements, or survivorship bias.
- *Survivorship-free* data includes companies that were later delisted, acquired, merged, or went bankrupt, avoiding the upward bias of testing only on companies that survived to the present day.
- *Cumulative Abnormal Return (CAR)* is the sum of a stock's returns over an event window minus a benchmark return, measuring excess movement around an announcement.
- *Lookahead bias* occurs when a strategy uses information that was not yet known or publicly tradeable at the simulated decision moment.

---

## 1. The question, and the decision it feeds

- **Question.** Can post-earnings-announcement drift (PEAD) and insider cluster buying be tested **FAIRLY** for the owner's taxable stock-trading path, and on what data?
- **Decision fed.** Whether to write a second, pre-registered test brief (with explicit pass bars) for either signal, and whether testing either signal requires purchasing commercial point-in-time paid data.
- **Context.** Both classic long-history equity routes are now closed as NO-GO:
  - Stock momentum failed its century-long screen ([`docs/research/2026-10-20-momentum-french-screen-nogo.md`](../2026-10-20-momentum-french-screen-nogo.md)).
  - Classic factors (robust profitability, book-to-market value, conservative investment, and the value + momentum combination) failed their 63-year to 99-year screens ([`docs/research/2026-10-21-factor-screen-nogo.md`](../2026-10-21-factor-screen-nogo.md)).
  These two event-driven signals — PEAD and insider cluster buying — represent the remaining active equity candidates from the 2026-10-20 triage.
- **Owner context.** Taxable account only. The owner wants an active stock-trading path with greater gains and accepts bigger swings. (The owner's core A23 allocation of 70/30 SPY/SSO is a separate decision and does not change.)

---

## 2. Why an audit first: repairing evidence before testing

Both signals carry historical NO-GO reports in the repository:
- PEAD was closed NO-GO on 2026-07-17 ([`docs/research/2026-07-17-pead-nogo.md`](../2026-07-17-pead-nogo.md)).
- Insider cluster buying was closed NO-GO on 2026-07-19 ([`docs/research/2026-07-19-insider-cluster-buy-nogo.md`](../2026-07-19-insider-cluster-buy-nogo.md)).

However, both tests were conducted on research infrastructure and underlying data feeds that were subsequently found to have severe defects:
1. **Day-shifted daily bars:** Database ingestion shifted historical timestamps by rebasing timezone-aware stamps into naive UTC, shifting bars by one full session before 2026-08-20 (`data/live/cached_yfinance_loader.py:215-226`).
2. **Survivorship bias:** Historical universes lacked proper point-in-time constituent membership handling prior to the point-in-time membership fix (`c9d1abe`, 2026-09-23).
3. **Insider transaction-date lookahead:** The original insider cluster strategy keyed entries off `transaction_date` (when the insider traded) rather than `filing_date` (when the trade became public), baking in a median 2-day (95th-percentile 5-day) lookahead bias, partially patched on 2026-07-25 (`1b9b7f5`).
4. **SPY benchmark duplicate rows:** Every research run covering 2023 onward suffered from duplicate SPY rows that artificially deflated reported Sharpe ratios by ~29% (`docs/research/2026-07-25-strategy-implementation-audit.md` §2.0).

A flawed or compromised backtest cannot be repaired simply by re-running the same strategy code. Before any new testing brief is drafted or pass bars are set, we must establish whether the underlying event timestamps, consensus estimates, and price histories are point-in-time, uncorrupted, and tradeable.

**Zero-returns rule:** This audit computes **NO returns, NO drift metrics, NO CAR, and NO strategy P&L**. By restricting outputs strictly to counts, timestamps, shares, and data-integrity flags, the audit cannot bias, curve-fit, or p-hack the pass bars of any subsequent test brief.

---

## 3. Audit questions (PEAD P1–P4, Insider I1–I4)

### Post-Earnings-Announcement Drift (PEAD)

- **P1. Announcement timing and intraday session alignment:**
  - Does stored `earnings_date` carry time-of-day (e.g. before-market-open vs after-market-close)?
  - If an announcement occurs after 16:00 ET, the first tradeable bar is the following session's open. Quantify what share of events would shift under a strict "next session open after the announcement timestamp" rule versus the current `REPORT_LAG_DAYS` gate.
  - *Code Complication & Reality Flag:* In `src/ggTrader/lab/earnings_surprise_data.py`, the database schema defines `earnings_date date NOT NULL`. In PostgreSQL, a `date` column strips time-of-day entirely; all stored rows in `earnings_surprise` record calendar dates only (`YYYY-MM-DD`). Furthermore, `src/ggTrader/lab/strategies/pead.py` implements a monthly-rebalanced portfolio selecting stocks at monthly `asof` and filling on `forward[0]`, rather than an event-driven strategy entering at the next session's open. A "next session open after announcement" rule represents a new daily event-trading execution model, not the existing monthly lab strategy.
- **P2. Estimate point-in-time validity:**
  - Is yfinance `eps_estimate` the true prevailing analyst consensus as of the announcement moment, or does it incorporate post-announcement revisions, subsequent accounting restatements, or split adjustments?
  - Evidence-based check: re-fetch a sample of historical announcements via `yf.Ticker.get_earnings_dates()`, compare against stored database values, and cross-reference a small random sample against a second public source (e.g. SEC 10-Q/8-K earnings releases or alternative free aggregators). Report what can and cannot be verified from free endpoints.
- **P3. Coverage and survivorship:**
  - Which exact symbol universe was backfilled into `earnings_surprise`? Are these S&P 500 ever-members (point-in-time) or today's surviving members?
  - Quantify annual coverage from 2002 to 2026: number of symbols with valid reports per year, total event count per year, and the share of events belonging to companies subsequently delisted or acquired.
  - *Code Complication & Reality Flag:* The 2026-07-17 PEAD report (`docs/research/2026-07-17-pead-nogo.md` §1) states the backfill covered "62,552 rows across 949 SP500 + Russell 2000 symbols" (with a partial ~200-symbol Russell 2000 subset). The audit must determine whether delisted historical constituents were included in this backfill or omitted.
- **P4. Price-tape usability:**
  - For each calendar year (2002–2026), what share of earnings surprise events have clean, point-in-time, non-contaminated price bars covering the full event window of $\pm 60$ trading days?
  - Must cite the documented 72-contaminated-ticker finding from the 2010–2020 holdout (`docs/research/briefs/2026-10-20-momentum-retest.md` §2) and evaluate against the repository's verified clean S&P 500 tape (2021-01 through 2026-04).

---

### Insider Cluster Buying (Form 4)

- **I1. Public availability and EDGAR acceptance timestamps:**
  - Form 4 `filing_date` is stored as a calendar date; does EDGAR expose an exact acceptance timestamp (specifically `acceptanceDateTime` in the SEC submissions API)?
  - On a representative sample of Form 4 filings, measure the share of filings accepted after 16:00 ET (where the first tradeable bar is the next trading day) versus during market hours.
  - *Code Complication & Reality Flag:* In `src/ggTrader/lab/form4_data.py`, `_filings_from_submissions_json()` extracted only `filingDate` (a calendar date string) and wrote to a database column `filing_date date`. The SEC submissions API exposes `acceptanceDateTime` (e.g. `2024-05-10T17:15:23.000Z`), but `form4_data.py` did not persist it. Thus, existing database rows in `form4_transactions` contain no intraday timestamps. Answering I1 requires querying the SEC EDGAR API directly.
- **I2. Amendments, duplicates, and 10b5-1 flag availability:**
  - *Amendments (Form 4/A):* How are amended filings handled?
    *Code Complication Flag:* In `form4_data.py:84`, the parser explicitly checks `if forms[i] == "4":`, discarding all `4/A` amended filings during enumeration. The audit must quantify the frequency of Form 4/A amendments and assess whether excluding them introduces erroneous or superseded transaction data.
  - *Duplicate handling:* In `form4_data.py:53-68`, the table schema primary key is `(accession_number, insider_cik, transaction_date, transaction_code, shares)`. The audit must verify whether multi-part transactions or same-day duplicate lots are correctly deduplicated or overwritten by `ON CONFLICT DO UPDATE`.
  - *10b5-1 plan flag availability by year:* In `form4_data.py:112`, the 10b5-1 flag is parsed via `_find(root, "aff10b5One")`.
    *Code Complication Flag:* The SEC adopted the mandatory `aff10b5One` XML checkbox on Form 4 in late 2022, effective April 1, 2023. Prior to 2023, Form 4 XML did not contain this field (pre-arranged plans were noted only in free-text footnotes). Consequently, in stored pre-2023 Form 4 data, `is_10b5_1_plan` is virtually always `false`, meaning routine scheduled trades could not be filtered out historically. The audit must quantify the distribution of `is_10b5_1_plan = true` across each calendar year (2015–2026).
- **I3. Coverage, survivorship, and fallback lookahead:**
  - Audit symbol coverage from 2015 to 2026 against point-in-time S&P 500 membership.
  - Quantify the share of cluster events belonging to delisted names.
  - *Code Complication & Reality Flag:* In `src/ggTrader/lab/strategies/insider_cluster.py:53-56`, `cluster_events()` handles missing `filing_date` by falling back to `transaction_date + 2 days`:
    ```python
    df["known_date"] = filed.fillna(df["transaction_date"] + pd.Timedelta(days=2))
    ```
    Furthermore, `load_form4_transactions()` in `form4_data.py:237` filters queries by `transaction_date BETWEEN :start AND :end`, not `filing_date`. The audit must quantify how many stored rows have null `filing_date`, how many filings occurred $> 2$ business days after the trade, and whether the fallback mechanism inadvertently creates lookahead.
- **I4. Price-tape usability:**
  - For each calendar year (2015–2026), what share of insider cluster events have clean, uncorrupted daily price bars covering the holding period ($\pm 60$ days to 252 trading days)?
  - Measure coverage specifically on the clean post-2021 tape versus the pre-2021 tape.

---

## 4. Method rules and audit protocol

To ensure the integrity of the audit and protect subsequent research phases from bias:

1. **SELECT-only database access:**
   - All queries against the local database (`earnings_surprise`, `form4_transactions`, `ohlcv`, `sp500_constituents`) must be read-only (`SELECT`).
   - Absolutely no `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, or migration commands may be executed.
2. **SEC EDGAR rate limiting and user-agent declaration:**
   - Any external requests to `data.sec.gov` or `sec.gov` must respect SEC fair-access guidelines: maximum **6 requests per second** (well below the SEC 10 req/s limit).
   - All HTTP requests must declare a descriptive User-Agent header (format: `User-Agent: ggTrader research contact@example.com`).
   - Fetching scripts must be fully resumable, caching raw JSON responses locally so that network interruptions or stops do not repeat requests.
3. **Pre-registered sampling rules:**
   - All audit sample sizes and sampling rules (e.g. stratified sampling across calendar years and market-cap tiers) must be fixed deterministically in the execution script before any external query is sent.
   - Fixed random seeds must be documented and reproducible.
4. **Strict prohibition on return calculations:**
   - **No returns may be computed or viewed.** The audit script and report are strictly barred from calculating forward returns, event-study drift, cumulative abnormal returns (CAR), post-filing price paths, Sharpe ratios, win rates, or portfolio P&L.
   - Permissible outputs are counts, percentages, timestamps, calendar day distributions, and symbol sets only.
5. **Artifacts produced:**
   - Structured JSON audit artifact: `docs/research/_event_timing_audit.json`.
   - Concise summary report: `docs/research/2026-10-21-event-timing-audit-report.md`.

---

## 5. What happens next (decision table)

Following completion of the audit, the senior quant and owner will review `docs/research/_event_timing_audit.json`. For each signal, the outcome will follow exactly one branch of the table below:

| Audit Finding | Status | Required Next Step |
|---|---|---|
| **(a) Testable on free data for years X–Y** | Data certified clean, point-in-time, and tradeable for window X–Y | Draft Phase 2 test brief specifying exact rules, universe, and pre-registered two-part pass bars (Part A statistical hurdle, Part B after-tax excess return vs SPY $\ge +1.0$ pt/yr). No testing occurs until the owner approves the Phase 2 brief. |
| **(b) Testable only with paid data** | Free data has unresolvable lookahead, survivorship, or timestamp corruption, but commercial vendor fixes it | Name the specific commercial vendor dataset (e.g. Compustat/CRSP point-in-time earnings dates, LSEG/I/B/E/S consensus history, or Thomson Reuters/Veracell Form 4 feed), document exactly what defect it resolves, estimate cost, and submit to the owner for purchase decision. |
| **(c) Not fairly testable** | Data cannot be verified point-in-time, or price tape is unusable, with no viable data path | Record findings plainly, close the candidate, and park the signal. Move to the next research candidate in `next_steps.md`. |

**Rule:** Under no circumstances will any backtest or return computation begin before the second, pre-registered test brief is written, reviewed, and approved by the owner.

---

## 6. Scope, boundaries, and untouchable files

- **Allowed code modifications:** New audit code and helper analysis scripts are strictly confined to `src/ggTrader/lab/` and `scripts/`.
- **Output files:** Audit data must write strictly to `docs/research/_event_timing_audit.json` and a report in `docs/research/`.
- **Untouchable paths:**
  - `src/ggTrader/paper/` (production paper-trading engine)
  - `src/ggTrader/data/live/` (live broker and execution data)
  - `.env` and environment configuration
  - `docker-compose.yml`, `Dockerfile`, or any container deployment configs
- **Database immutability:** Never write to, modify, or truncate any database table.

---

## 7. Calls made by the senior quant (owner to check)

1. **Audit-before-test instead of an immediate re-run:**
   The prior NO-GOs were evaluated on compromised infrastructure. However, jumping immediately into backtesting on unverified data risks producing another illusory result or false rejection. Auditing data feasibility first ensures that future test results are scientifically trustworthy.
2. **Strict prohibition on return computations during audit:**
   Computing returns while evaluating data properties introduces unconscious p-hacking and biases the selection of test windows and pass bars. Restricting the audit to timestamps, coverage, and missingness guarantees that Phase 2 pass bars remain uncorrupted.
3. **Joint audit of both event signals:**
   PEAD and insider cluster buying share the identical underlying price tape (daily S&P 500 member bars), the same historical survivorship questions, and analogous event-availability timestamp gates. Combining their data audit into a single protocol saves engineering effort while maintaining rigorous standards.
4. **Congressional trades (`congress_trades`) excluded from this audit:**
   Congressional trades share the filing-lag class of corporate insider filings under the STOCK Act, but suffer from substantially lower event frequency and disclosure delays extending up to 45 calendar days. Insider cluster buying represents the higher-density, cleaner corporate analogue. If Form 4 insider filings cannot clear the data-integrity and timing hurdle, congressional trades cannot either; if Form 4 clears and passes subsequent testing, congressional trades can follow under the same framework later.
