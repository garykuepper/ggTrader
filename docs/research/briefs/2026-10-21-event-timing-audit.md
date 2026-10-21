# Brief: event-timing and data-integrity audit for PEAD and insider cluster buying

**Status:** DRAFT 2026-10-21, awaiting owner approval. Nothing has been computed. Execution is blocked until §8 is approved with the brief; the builder must not choose or revise these values after inspecting audit results.

**Terms used below:**
- *PEAD (Post-Earnings-Announcement Drift)* means the tendency for a stock's price to continue drifting in the direction of an earnings surprise for weeks or months after the announcement.
- *Earnings surprise* means the percentage difference between reported earnings per share (EPS) and the pre-announcement consensus estimate; in the repository implementation, this value is taken directly from Yahoo Finance's supplied `surprise_pct` field (`earnings_surprise_data.py:59`) rather than calculated internally from a formula.
- *Insider cluster buying* means multiple corporate insiders (such as officers, directors, or major shareholders) purchasing shares of their own company on the open market within a short window of time.
- *SEC Form 4* is the mandatory disclosure filing submitted to the SEC when corporate insiders acquire or dispose of company securities.
- *Acceptance timestamp* means the exact date, hour, minute, and second when the SEC's EDGAR system officially received and accepted a filing (`acceptanceDateTime`), as distinguished from dissemination timing (when third-party feeds distribute it).
- *10b5-1 plan* is a pre-arranged trading plan that automatically buys or sells shares on pre-set dates, indicating a scheduled/routine trade rather than a discretionary, opportunistic purchase. The SEC compliance date requiring the `aff10b5One` checkbox on Form 4 was April 1, 2023.
- *Point-in-time (PIT)* means data reflects only information available at the exact historical decision moment, without revisionist adjustments, future restatements, or survivorship bias.
- *Survivorship-free* data includes companies that were later delisted, acquired, merged, or went bankrupt, avoiding the upward bias of testing only on companies that survived to the present day.
- *Right-censoring (administrative right-censoring)* occurs when an observation window ends simply because the study period or dataset ends (e.g. at the observation cutoff), rather than because a company failed, delisted, or suffered missing data.
- *Prefix invariance* means that at each sampled historical decision timestamp, reconstructed signals depend solely on filings and disclosures that were already public as of that exact moment. Adding later disclosures or amended filings must never alter signals formed at earlier timestamps.
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

However, both tests were conducted on research infrastructure and underlying data feeds that had known or suspected defects:
1. **Day-shifted daily bars:** Database ingestion shifted historical timestamps by rebasing timezone-aware stamps into naive UTC, stamping bars one calendar day early before 2026-08-20 (`data/live/cached_yfinance_loader.py:215-226`).
2. **Survivorship bias and constituent handling:** Commit `c9d1abe` (2026-09-23) fixed survivorship bias in SIGNAL sweeps and stated that weight strategies were already point-in-time correct. Because both candidate strategies here (`PeadStrategy` and `InsiderClusterBuyStrategy`) declare `target_kind = "weights"`, the demonstrated applicability of the signal-sweep defect to these specific implementations must be evaluated.
3. **Insider transaction-date lookahead:** The original insider cluster strategy keyed entries off `transaction_date` (when the insider traded) rather than `filing_date` (when the trade became public), baking in a median 2-day (95th-percentile 5-day) lookahead bias, partially patched on 2026-07-25 (`1b9b7f5`).
4. **SPY benchmark duplicate rows:** While duplicate SPY rows deflated reported Sharpe ratios in post-2023 WFO evaluations, historical SPY-contamination exposure for the original 2026-07-17 reports is unresolved (`docs/research/2026-07-25-strategy-implementation-audit.md:418`), as stray rows may have been backfilled after those runs executed against a clean frame.

Document each defect's demonstrated applicability to these implementations and run vintages; distinguish confirmed exposure from unresolved exposure. Existing NO-GOs remain historical findings. This audit neither overturns them nor establishes that correction improves either signal.

**Prohibition on return calculations and return-based detectors:** Follow the strict prohibition in §4.3: do not calculate or inspect new return series, performance summaries, post-event price paths, or return-based detectors. Existing documented outcome exposure must be recorded; existing integrity flags may be consumed without recomputing or inspecting their underlying return diagnostics. This audit computes **NO returns, NO drift metrics, NO CAR, and NO strategy P&L**, restricting outputs strictly to counts, timestamps, shares, and metadata integrity flags.

---

## 3. Audit questions (PEAD P1–P4, Insider I1–I4)

### Post-Earnings-Announcement Drift (PEAD)

- **P1. Announcement timing and session alignment:**
  - Normalize timestamps to America/New_York with DST and NYSE exchange calendar (holidays and early closes) per §8 A1.
  - Apply the tradeable event convention fixed in §8 A1 (first regular-session CLOSE at least 30 minutes after public availability, matching the close-fill simulator; report regular-session OPEN at least 30 minutes after availability as a secondary diagnostic only) and public availability source hierarchy in §8 A3 (8-K Item 2.02 acceptance timestamp matched to fiscal period, else yfinance intraday timestamp). Compare against both existing monthly-close implementations without implementing either replacement.
  - For date-only events, apply the fixed fallback in §8 A3 (close of the first trading session after the stated earnings date); missing intraday precision alone does not prove paid data is necessary.
  - *Code Complication & Reality Flag:* In `src/ggTrader/lab/earnings_surprise_data.py`, the schema defines `earnings_date date NOT NULL`. In PostgreSQL, a `date` column loses time-of-day entirely upon insert; all stored rows record calendar dates only (`YYYY-MM-DD`). Furthermore, `REPORT_LAG_DAYS = 1` in `available_as_of()` (`src/ggTrader/lab/strategies/pead.py:24,27`) is implemented as one calendar day (`pd.Timedelta(days=lag_days)`), not a trading session. In `src/ggTrader/lab/strategies/pead.py:38`, `PeadStrategy` implements a monthly-rebalanced portfolio selecting the top-fifth bucket of eligible surprises, placing targets on `forward[0]`, where the simulator fills at CLOSE prices. An event-driven next-session-open rule changes both the strategy cadence and the fill convention.
- **P2. Estimate point-in-time validity and provenance:**
  - Separately verify actual-EPS provenance and pre-announcement consensus provenance. Re-fetch agreement is a stability diagnostic, not PIT certification.
  - Audit GAAP versus adjusted EPS, fiscal-period matching, currency, split basis, percentage scaling, negative/zero estimates, duplicates, and missing values. Unsupported consensus provenance remains UNVERIFIED (per §8 A6).
  - *Code Complication & Reality Flag:* The strategy relies directly on Yahoo Finance's supplied `surprise_pct` (`earnings_surprise_data.py:59` in `fetch_earnings_surprises()`), not an internally computed formula. The audit must inspect how this field is formed in the upstream feed and verify whether `eps_estimate` reflects true pre-announcement consensus or post-announcement revisions, restatements, or split adjustments. Report what can and cannot be verified from free endpoints (§8 A10).
- **P3. Coverage, survivorship, and universe construction:**
  - Build coverage denominators from dated PIT membership independently of stored events, using S&P 500 PIT membership from `data/universe/sp500_constituents_history.csv.gz` (record SHA-256) for 2002-01 → 2026-04 (member-quarters per year) as primary denominator (§8 A5). Audit delisted and acquired issuers with zero records, current-endpoint failures, historical ticker/CIK mappings, ticker reuse, share classes, and corporate actions linking events to the correct price security.
  - Report attempted, resolved, fetched, and stored coverage separately. Never use subsequent survival as trading eligibility. The static 2026-06-09 Russell 2000 snapshot is reported separately as a survivorship-biased diagnostic and can never support FREE—VERIFIED (§8 A5).
  - Quantify annual coverage from 2002 to 2026: number of symbols with valid reports per year, total event count per year, and the share of events belonging to companies subsequently delisted or acquired.
  - *Code Complication & Reality Flag:* The historical 2026-07-17 PEAD report (`docs/research/2026-07-17-pead-nogo.md` §1) claimed backfill coverage across "949 SP500 + Russell 2000 symbols". This is a historical claim, not verified coverage. In reality, the PEAD backfill requested S&P 500 ever-members (2015 to run time) plus a static 2026-06-09 Russell 2000 snapshot (not historical membership). Current database counts for `earnings_surprise` (round-1 snapshot observations (2026-10-21), to be remeasured under the audit's read-only protocol, with query time, SQL, date column, numerator/denominator and future-date cutoff recorded in the evidence manifest) stand at 62,373 rows across 762 symbols, with a latest date of 2026-07-17. Delisted historical members lacking records must be audited against independent dated membership.
- **P4. Price-tape usability and missingness accounting:**
  - Report missing coverage separately for administrative right-censoring, listing inception, delisting/merger, halts, and unexplained missingness. Do not discard events because their subsequent trading history ends; administrative right-censoring is reported, never a reason to drop an event (§8 A4). Apply the fixed observation cutoff (event availability on or before 2026-04-30) and the evaluation window of 63 trading days (§8 A4).
  - For each calendar year (2002–2026), measure the share of earnings surprise events with a price on the tradeable date (§8 A1, A6). Completeness over the 63-trading-day window is reported as a diagnostic only, split by the missing-coverage causes above; it is never an eligibility condition.
  - *Tape Integrity Context:* Cite the documented 72-contaminated-ticker finding from the 2010–2020 holdout ([`docs/research/briefs/2026-10-20-momentum-retest.md`](2026-10-20-momentum-retest.md) §2). Describe the repository's 2021-01 through 2026-04 S&P 500 tape as detector-screened, not certified. No newly calculated return detectors may be used (follow §4.3).

---

### Insider Cluster Buying (Form 4)

- **I1. Public availability, acceptance timestamps, and dissemination:**
  - Form 4 `filing_date` is stored as a calendar date; determine whether EDGAR exposes exact acceptance timestamps (`acceptanceDateTime` in the SEC submissions API).
  - Normalize timestamps to America/New_York with DST and NYSE exchange calendar (holidays and early closes) per §8 A1. Cross-check SEC API acceptance timestamps against filing headers/index records on the deterministic sample (§8 A9); distinguish acceptance from dissemination (§8 A2).
  - Apply the tradeable event convention fixed in §8 A1 (first regular-session CLOSE at least 30 minutes after public availability, matching the close-fill simulator; report regular-session OPEN at least 30 minutes after availability as a secondary diagnostic only). Compare against both existing monthly-close implementations without implementing either replacement.
  - For date-only events, apply the fixed fallback in §8 A2 (close of the first trading session after filing_date); missing intraday precision alone does not prove paid data is necessary.
  - *Code Complication & Reality Flag:* In `_filings_from_submissions_json()` (`src/ggTrader/lab/form4_data.py:93`), the parser extracted only `filingDate` (a calendar date string) and did not store `acceptanceDateTime`. Furthermore, the table schema (`form4_data.py:53`) has no acceptance timestamp column; all stored rows record `filing_date date` only. Answering I1 requires querying the SEC EDGAR API directly to evaluate acceptance timestamps against dissemination and session open.
- **I2. Amendments, reporting owners, duplicate handling, and 10b5-1 plan fields:**
  - *Amendments (Form 4/A):* In `_filings_from_submissions_json()` (`src/ggTrader/lab/form4_data.py:106`), the parser checks `if forms[i] == "4":`, discarding all `4/A` amended filings. The audit must quantify the frequency of Form 4/A amendments and determine whether excluding them leaves superseded transaction data in the historical record.
  - *Reporting owner inventory & private transactions:* In `parse_form4_transactions()` (`form4_data.py:121`), the XML parser extracts only the first reporting owner (`reportingOwner/reportingOwnerId/rptOwnerCik`), discarding any co-reporting owners. Transaction code `P` includes private purchases under SEC Form 4 instructions. Inventory all reporting owners, roles, related entities, direct/indirect holdings, security classes, acquisition codes, and availability of transaction-size fields. Quantify whether private purchases and multiple related entities are being treated as independent insider conviction. Do not tune role or size filters during this audit.
  - *Duplicate handling & primary key collisions:* Reconcile sampled raw transaction rows to stored rows. Distinguish exact duplicate ingestion from legitimate lots or security classes that collide under the current primary key (`form4_data.py:66`; upsert at :205 in `_store_filings()` overwrites price and other fields). Quantify losses/overwrites and their effect on event reconstruction; make no DB corrections.
  - *10b5-1 plan field audit:* Audit PRESENT-TRUE, PRESENT-FALSE, and ABSENT plan fields, including supported XML encodings.
    *Code Complication & Regulatory Reality:* The SEC compliance date (not "effective date") for mandatory inclusion of the `aff10b5One` checkbox was April 1, 2023. In `parse_form4_transactions()` (`form4_data.py:124`), an absent field is parsed as `False` via `(_find(root, "aff10b5One") or "false").strip().lower() == "true"`, meaning an absent tag is stored as `is_10b5_1_plan = False`. Thus, `FALSE ≠ verified non-plan`. Stored database TRUE shares (stored transaction-row shares grouped by filing year; round-1 snapshot observations (2026-10-21), to be remeasured under the audit's read-only protocol, with query time, SQL, date column, numerator/denominator and future-date cutoff recorded in the evidence manifest) demonstrate this structural break: 0% for 2015–2022; 1.10% for 2023; 5.70% for 2024; 5.67% for 2025; and 5.44% for 2026 (through 2026-07-17). Prior to the 2023 compliance date, routine scheduled trades cannot be distinguished from discretionary trades via this field.
- **I3. Coverage denominators, prefix invariance, and data anomalies:**
  - *Coverage denominators & ticker mapping:* Build coverage denominators from dated PIT membership independently of stored events, using S&P 500 PIT membership from `data/universe/sp500_constituents_history.csv.gz` (record SHA-256) for 2015-01 → 2026-04 (member-months per year) as primary denominator (§8 A5). Audit delisted and acquired issuers with zero records, current-endpoint failures, historical ticker/CIK mappings, ticker reuse, share classes, and corporate actions linking events to the correct price security. Report attempted, resolved, fetched, and stored coverage separately. Never use subsequent survival as trading eligibility. Note that `scripts/form4_backfill.py:74` resolves tickers via today's SEC `company_tickers.json` and skips unresolved names. Current database counts for `form4_transactions` (round-1 snapshot observations (2026-10-21), to be remeasured under the audit's read-only protocol, with query time, SQL, date column, numerator/denominator and future-date cutoff recorded in the evidence manifest) stand at 783,982 rows across 1,309 symbols and 1,218 issuer CIKs (latest date 2026-07-17); the historical report's citation of 764 symbols is an unverified historical claim.
  - *Prefix invariance verification:* Verify prefix invariance: at each sampled decision timestamp, reconstruct clusters solely from filings already public. Adding later disclosures must not alter earlier signals. Audit amendment corrections only from their own public timestamps; never retroactively replace the historical information set. In `src/ggTrader/lab/strategies/insider_cluster.py:46` (`cluster_events()`, sequential consumption loop at :65) and in `InsiderClusterBuyStrategy.select()` (clustering at :126, active event filtering at :137 after clustering), future events are filtered only after clustering. The audit must test whether adding later filings changes historical cluster formation.
  - *Transaction-date extraction and filing-lag distribution:* Audit transaction-date extraction against public-availability-date inclusion (the loader filters by transaction_date, `load_form4_transactions()` in `form4_data.py:230`), including late filings whose transactions precede the load boundary. After anomaly quarantine (§8 A8), report filing-lag distributions and the share exceeding two business days, using the §8 exchange calendar (§8 A1). Report omitted disclosures separately from the null-date fallback.
  - *Malformed records and anomaly quarantine:* Count malformed, implausible, future-dated, and transaction-after-filing records before measuring lags or clustering. Reconcile deterministic samples (§8 A9) to original filings; quarantine unresolved records under the frozen rule (§8 A8: `transaction_date < 1990-01-01`, `transaction_date > filing_date`, or any date after query run date quarantined and counted, never corrected or used). Stored database records exhibit notable anomalies (round-1 snapshot observations (2026-10-21), to be remeasured under the audit's read-only protocol, with query time, SQL, date column, numerator/denominator and future-date cutoff recorded in the evidence manifest): transaction dates span from `0015-11-11` to `2031-01-29`, with 78 rows having transaction date after filing date, and 4 future-dated transactions.
  - *Fallback lookahead evaluation:* In `src/ggTrader/lab/strategies/insider_cluster.py:60` (within `cluster_events()`), the fallback expression for missing `filing_date` evaluates `filed.fillna(df["transaction_date"] + pd.Timedelta(days=2))` (two calendar days, not trading sessions). However, empirical database verification shows zero rows with null `filing_date` (0 / 783,982 null; round-1 snapshot observations (2026-10-21), to be remeasured under the audit's read-only protocol, with query time, SQL, date column, numerator/denominator and future-date cutoff recorded in the evidence manifest), meaning this specific fallback currently has zero DB exposure.
- **I4. Price-tape usability and right-censoring:**
  - Report missing coverage separately for administrative right-censoring, listing inception, delisting/merger, halts, and unexplained missingness. Do not discard events because their subsequent trading history ends; administrative right-censoring is reported, never a reason to drop an event (§8 A4). Apply the fixed observation cutoff (event availability on or before 2026-04-30, §8 A4) and specify calendar versus trading days for every window.
  - Note that in `InsiderClusterBuyStrategy.select()` (`src/ggTrader/lab/strategies/insider_cluster.py:136`), `hold_days` is evaluated in calendar days (`(asof_naive - event_date).dt.days`), not trading sessions.
  - For each calendar year (2015–2026), measure the share of insider cluster events with a price on the tradeable date (§8 A1, A6). Completeness over the 126, 189 and 252 calendar-day holding windows is reported as a diagnostic only, split by the missing-coverage causes above; it is never an eligibility condition.
  - Evaluate coverage on the detector-screened 2021–2026-04 tape versus the earlier unverified tape without computing returns or running return-based detectors (follow §4.3).

---

## 4. Method rules and audit protocol

To ensure the integrity of the audit and protect subsequent research phases:

1. **Read-only DB connection and explicit queries:**
   - Use a read-only DB connection and explicit SELECTs; never invoke cache, schema-initialization, or schema-ensuring loaders. Both existing functions `load_earnings_surprises()` (`earnings_surprise_data.py:127`) and `load_form4_transactions()` (`form4_data.py:221`) call `ensure_schema()`, which executes DDL (`CREATE TABLE IF NOT EXISTS`).
   - Freeze and identify the historical constituent membership file: `data/universe/sp500_constituents_history.csv.gz` (record SHA-256; there is no `sp500_constituents` database table, §8 A5).
   - Fail closed on attempted writes. Absolutely no `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, or migration commands may be executed.
2. **SEC EDGAR rate limiting, caching, and execution controls:**
   - Before external requests, freeze sample sizes, strata, selection algorithm, seed, request/runtime limits, retry policy, and stopping rules as defined in §8 A9–A12.
   - Any external requests to `data.sec.gov` or `sec.gov` must respect SEC fair-access guidelines: maximum **5 requests per second** (well below the SEC 10 req/s limit; §8 A11). Total external request limits: ≤ 20,000 SEC requests and ≤ 2,000 yfinance requests; 3 retries with exponential backoff; stop immediately on HTTP 403/429 bursts and record the request as UNRESOLVED evidence (§8 A11). Runtime cap is 6 hours; script must checkpoint and resume (§8 A11).
   - All HTTP requests must declare a descriptive User-Agent header (a project name plus a real contact address read from an environment variable at run time, never committed to the repo; §8 A11).
   - Raw responses must be cached under `data/external/event_timing_audit/` with a SHA-256 `MANIFEST.json` and never deleted (§8 A12). Focused tests belong under `tests/lab/`; follow `scripts/run_tests.sh` and ruff gates. CLI scripts must retain checkpoints and support `--help`.
3. **Strict prohibition on return calculations:**
   - Do not calculate or inspect new return series, performance summaries, post-event price paths, or return-based detectors. Existing documented outcome exposure must be recorded; existing integrity flags may be consumed without recomputing or inspecting their underlying return diagnostics.
   - Permissible outputs are counts, percentages, timestamps, calendar day distributions, and symbol sets only. Describe 2021–2026-04 as detector-screened, not certified. Defer all performance thresholds to the separately approved test brief.
4. **Artifacts produced:**
   - Structured JSON audit artifact: `docs/research/_event_timing_audit.json`.
   - Concise summary report: `docs/research/2026-10-21-event-timing-audit-report.md`.

---

## 5. What happens next (decision table)

Execution is blocked until §8 is approved with the brief; the builder must not choose or revise these values after inspecting audit results. Apply these branches in order: FREE—VERIFIED when every registered free-data requirement (§8 A6) passes for an eligible window (§8 A7); otherwise PAID—DOCUMENTED PATH when a documented paid or mixed-source route addresses every blocking requirement; otherwise NO VERIFIED PATH only when the registered source search (§8 A10) is complete and documented evidence rules out every assessed route; otherwise UNRESOLVED. Request failures, exhausted budgets, and missing documentation are unresolved evidence, not proof of infeasibility. Output must preserve every failed or unknown requirement per signal (requirement matrix). Vendor feasibility requires documented field coverage and historical availability (§8 A10); marketing descriptions are insufficient. These statuses establish data feasibility only.

Following completion of the audit, the senior quant and owner will review `docs/research/_event_timing_audit.json`. Each candidate signal will be assigned exactly one of the four statuses:

| Status | Audit Finding | Required Next Step |
|---|---|---|
| **FREE—VERIFIED** | Free sources meet every registered provenance, availability, identity, coverage, and reconstruction requirement for the eligible window (§8 A6–A7). Price history has the specified metadata checks and existing detector screening; remaining limitations are disclosed. This is data feasibility, not independent certification of an uncorrupted tape. | Draft Phase 2 test brief specifying exact rules, universe, and pre-registered pass bars. Phase 2 will separately preregister statistical and after-tax SPY comparison criteria. The owner's existing two-part standard (+1.0 pt/yr after tax for our version) is cited as standing policy; this audit neither selects nor calibrates performance thresholds. No testing occurs until the owner approves the Phase 2 brief. |
| **PAID—DOCUMENTED PATH** | At least one §8 A6 requirement fails or is unverified on free data, and a documented paid or mixed-source route addresses every blocking requirement (§8 A10). | Document the dataset found, its exact field specifications, historical point-in-time availability and cost, citing vendor documentation. Submit the purchase case to the owner for decision. |
| **UNRESOLVED** | Provenance or timing cannot be verified from free sources, and commercial vendor feasibility remains unproven or undocumented. | Record findings plainly, document the unverified properties, and park the signal. Move to the next research candidate. |
| **NO VERIFIED PATH** | Data cannot be verified point-in-time, prefix invariance is broken, or price tape is unusable, with no documented or viable data path. | Record findings plainly, close the candidate, and park the signal. Move to the next research candidate. |

**Rule:** Under no circumstances will any backtest or return computation begin before the second, pre-registered test brief is written, reviewed, and approved by the owner.

---

## 6. Scope, boundaries, and untouchable files

- **Allowed code modifications:** New audit code and helper analysis scripts are strictly confined to `src/ggTrader/lab/` and `scripts/`. Focused unit/audit tests may be added under `tests/lab/`. All code must adhere to `scripts/run_tests.sh` and ruff gates. CLI scripts must retain checkpoints and support `--help`.
- **Allowed output files and caches:**
  - Audit data writes strictly to `docs/research/_event_timing_audit.json` and `docs/research/2026-10-21-event-timing-audit-report.md`.
  - External raw API responses must be cached under `data/external/event_timing_audit/` with a SHA-256 `MANIFEST.json` (§8 A12) to ensure deterministic, resumable execution without redundant network calls, and never deleted.
- **Universe and database handling:**
  - Freeze and identify the historical constituent membership file (`data/universe/sp500_constituents_history.csv.gz`, record SHA-256; §8 A5); there is no `sp500_constituents` database table.
  - Use a read-only DB connection with explicit SELECT statements. Fail closed on attempted writes.
  - Never invoke existing cache, schema-initialization, or schema-ensuring loaders (`load_earnings_surprises`, `load_form4_transactions`, or `cache_symbol_surprises`), as they call `ensure_schema()` and execute DDL.
- **Untouchable paths:**
  - `src/ggTrader/paper/` (production paper-trading engine)
  - `src/ggTrader/data/live/` (live broker and execution data)
  - `.env` and environment configuration
  - `docker-compose.yml`, `Dockerfile`, or any container deployment configs
- **Database immutability:** Never write to, modify, alter, or truncate any database table.

---

## 7. Calls made by the senior quant (owner to check)

1. **Audit-before-test instead of an immediate re-run:**
   The prior NO-GOs were evaluated on research infrastructure with known defects. However, jumping immediately into backtesting on unverified data risks producing illusory results or false rejections. Document each defect's demonstrated applicability to these implementations and run vintages; distinguish confirmed exposure from unresolved exposure. Existing NO-GOs remain historical findings. This audit neither overturns them nor establishes that correction improves either signal.
2. **Strict prohibition on return computations during audit:**
   Follow §4.3: do not calculate or inspect new return series, performance summaries, post-event price paths, or return-based detectors. Existing documented outcome exposure must be recorded; existing integrity flags may be consumed without recomputing or inspecting their underlying return diagnostics. Restricting the audit to timestamps, coverage, and missingness keeps data-quality verification strictly separated from strategy evaluation.
3. **Joint audit of both event signals:**
   Both audits share daily-price infrastructure and overlapping S&P 500 coverage. PEAD also includes a static Russell snapshot; universe-specific provenance and coverage conclusions must remain separate. Combining their data audit into a single protocol saves engineering effort while maintaining rigorous standards.
4. **Congressional trades (`congress_trades`) excluded from this audit:**
   Congressional trades are excluded from the scope of this audit to maintain focus on corporate event-driven signals. If congressional trading is evaluated in the future, it will be scoped under its own independent data audit.

---

## 8. Protocol annex (fixed by the senior quant; the builder may not choose or revise these after seeing audit results)

Execution is blocked until §8 is approved with the brief; the builder must not choose or revise these values after inspecting audit results.

- **A1. Clock:** All timestamps normalized to `America/New_York` with DST, using the NYSE exchange calendar (holidays and early closes) via `pandas_market_calendars` if installed, else `exchange_calendars` (record which; add to `pyproject.toml` if new). Processing buffer = 30 minutes. An event is tradeable at the first regular-session CLOSE at least 30 minutes after public availability (matches the close-fill simulator). Also report the first regular-session OPEN at least 30 minutes after availability, as a secondary diagnostic only.
- **A2. Public availability — insider:** The SEC EDGAR acceptance timestamp (`acceptanceDateTime`), cross-checked against the filing index/header on the sample; date-only fallback = close of the first trading session after `filing_date`.
- **A3. Public availability — PEAD:** The acceptance timestamp of the company's SEC Form 8-K with Item 2.02 (Results of Operations) matched to the fiscal period, where found; else the yfinance timestamp if a fresh fetch provides time-of-day; date-only fallback = close of the first trading session after the stated earnings date.
- **A4. Observation cutoff and windows:** Event availability on or before 2026-04-30 (end of the detector-screened tape). Windows: PEAD 63 trading days; insider the existing `hold_days` settings 126, 189, 252 CALENDAR days. Right-censoring is reported, never a reason to drop an event.
- **A5. Universes:** S&P 500 point-in-time membership from `data/universe/sp500_constituents_history.csv.gz` (record SHA-256) is the primary denominator for both signals, 2015-01 → 2026-04 for insider and 2002-01 → 2026-04 for PEAD (per year, member-quarters for PEAD, member-months for insider). The static 2026-06-09 Russell 2000 snapshot is reported separately as a survivorship-biased diagnostic and can never support FREE—VERIFIED.
- **A6. Coverage acceptance (per calendar year):** ≥ 95% of denominator units resolved to an identity (ticker↔CIK) and fetched; ≥ 95% of stored events map to a price security with a price on the tradeable date (excluding administrative right-censoring); unexplained missingness ≤ 2%; and, for PEAD, consensus-estimate provenance documented as pre-announcement (else that requirement is UNVERIFIED).
- **A7. Eligible window:** The longest contiguous run of calendar years in which every A6 requirement passes; minimum 5 years; ties → the most recent run.
- **A8. Anomaly quarantine (frozen):** `transaction_date < 1990-01-01`, `transaction_date > filing_date`, or any date after the date the audit query runs → quarantined and counted, never corrected or used.
- **A9. Sampling:** Deterministic, seed `20261021`. Insider: 300 filings stratified evenly by filing year 2015–2026 for timestamp/header/raw-XML reconciliation. PEAD: 300 events stratified evenly by year 2002–2026 for 8-K matching and EPS checks; plus a re-fetch of 100 symbols, of which ≥ 30 are delisted or acquired S&P 500 ever-members.
- **A10. Sources:** Free — SEC EDGAR (submissions API, filing indexes, 8-K documents), yfinance. Paid routes may be documented only from vendor documentation that states field coverage and historical point-in-time availability; name none in this brief.
- **A11. Limits:** SEC ≤ 5 requests/second with a declared User-Agent, ≤ 20,000 SEC requests and ≤ 2,000 yfinance requests in total; 3 retries with exponential backoff; stop immediately on HTTP 403/429 bursts and record the request as UNRESOLVED evidence. Runtime cap 6 hours; checkpoint and resume.
- **A12. Cache:** Raw responses under `data/external/event_timing_audit/` with a SHA-256 `MANIFEST.json`; never deleted.
