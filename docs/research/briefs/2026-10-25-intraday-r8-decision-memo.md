# Decision memo: proposed r8 protocol for the intraday SPY screen

**Status: DRAFT, UNAPPROVED.** This memo proposes one protocol for r8 of the
[intraday brief](2026-10-24-intraday-feasibility.md). Until the owner approves it, r7, the
[approved forward-paper standard](2026-10-23-forward-paper-standard.md), the
[Stage 0A contract](2026-10-25-intraday-stage0a-contract.md) and the worklist are unchanged.
Stage 0A stays closed **unresolved**. No network request, price inspection, pull, signal,
return, shadow or order is authorized by this draft. The recommendations are not accepted
until the owner marks the approval block.

The owner constraints are fixed and not reopened: one long-only SPY rule, no borrowing or
shorting, free data only, a taxable account and the production freeze.

Evidence: [Opus assessment](../2026-10-25-intraday-opus-review-assessment.md),
[proposal verification](../2026-10-25-intraday-opus-proposal-verification.md),
[estimates verification](../2026-10-25-intraday-opus-estimates-verification.md),
[Argon assessment](../2026-10-25-intraday-argon-review-assessment.md),
[Stage 0A coverage](../2026-10-25-intraday-stage0a-coverage.md),
[checked r7 fields](../_intraday_r8_field_check_2026-10-25.json).

## Carried forward from r7 (unchanged)

- **Rule.** On each scored session, at 15:30 ET go long SPY if `r_first > 0`, otherwise stay
  flat. Exit at the close. There are no filters and no parameter search.
- **Returns.** Prices are `adjustment=raw`. `r_first = P_0959 / P_close(t−1) − 1` and
  `r_last = P_exit / P_1530 − 1`. The daily series is
  `s_t = 1[r_first,t > 0] × r_last,t − cost_t`, where `cost_t` is the round-trip cost on
  traded days and zero on flat days. Every scored session stays in the series, and flat days
  are zeros.
- **Costs.** 1 bp/side base (2 bp round trip) and 2 bp/side stress. The verdict uses base
  cost. Stress is reported, and it cannot rescue a failure or fail a pass by itself.
- **A1 (the only statistical gate).** This is a one-sided Newey-West (NW) t on the mean of
  `s_t`, from an intercept-only OLS with Bartlett weights and **exactly 5 lags**. It passes at
  unrounded **t ≥ 3.0**. A zero or non-finite variance is statistically unresolved.
  - **Recommend:** use statsmodels `cov_type="HAC"` with `maxlags=5` and
    `use_correction=True`.
  - The Python, statsmodels, numpy and pandas versions are recorded in r8 before any pull.
  - NW lags 0, 10 and 20 are diagnostics only.
- **A2.** The sleeve's after-tax-policy CAGR must be **≥ +3.0%/yr**. The ledger is defined as
  follows:
  - **Wealth.** Wealth `W` starts at the 2016-01-04 anchor close, and
    `W_t = W_{t−1}(1 + s_t)`. The sleeve holds 100% of then-current equity in the window and
    earns zero on idle cash.
  - **Annual tax debit.** On each year's last session, debit **24%** of that year's positive
    net realized P&L, `Σ W_{t−1}·s_t`. Every trade closes the same day, so all P&L is
    realized. A zero or negative year gets no tax benefit, and no loss carries forward.
  - **Terminal partial year.** The final partial year (2026-01 → 2026-09-30) accrues its
    liability at the terminal valuation.
  - **Proposed explicit valuation convention (not previously fixed in r7).**
    `CAGR = (W_end/W_0)^(1/y) − 1`, with
    `y = days(2016-01-04 → 2026-09-30)/365.25`.
- **Verdict.** H1 passes only if A1 **and** A2 both pass.
- **Diagnostics (non-gating).** Subperiods 2016–19, 2020–23 and 2024–26-09. Break-even cost.
  Worst month, losing streak and maximum drawdown. Negative-signal and zero-signal counts. The
  ex-dividend-flat sensitivity.
- **Displacement diagnostic.** The SPY total-return comparison (distributions reinvested, no
  interim or terminal tax, timestamps aligned) is a mandatory, non-gating diagnostic. Its tax
  basis is unchanged from r7.
- **Label.** *Post-original-sample and single-shot*, not an untouched holdout. One candidate,
  zero completed return tests.

## Planning power (conditional arithmetic, not evidence)

The figures below assume N ≈ 2,670 sessions, a whole-calendar daily SD of 25 bp, IID normal
returns and a 50% trade frequency. **Every one of those inputs is assumed and unmeasured.**
Gao's published scale gives a rough 27.7 bp proxy only under additional conditional-
variance assumptions. Under the stated 25 bp planning assumptions, A1
needs a net mean of:

| Target | bp/session | bp gross per trade at base cost |
|---|---:|---:|
| Expected t = 3 (≈50% power) | 1.451 | 4.90 |
| 80% power | 1.859 | 5.72 |
| 90% power | 2.072 | 6.14 |

Power under the actual HAC gate, and for A1 and A2 together, is **unknown**. A2 has no single
equivalent daily mean. A modest real edge can fail; that is not a reason to change a gate.

## Decisions

### D1 — Research role and route after a pass

**Problem.** r7 §8 routes a standalone A2 pass "under the approved forward-paper standard",
which does not fit a same-day 15:30 rule. That standard uses pre-open decisions, next-close
fills, T-bill cash and SPY-relative hurdles. Its B0 replay has its own 2021-01-31 → 2026-04-30
window and benchmark treatment, so it is not a whole-window hurdle that could be reused here.
A sleeve can pass A2 and still trail SPY.

**Recommend.**
- **Scope of the verdict.** The research verdict answers one question: *is there a
  standalone, after-tax-policy edge worth further research?* A1 and A2 are kept exactly as
  in r7, and no SPY-relative research gate is added.
- **SPY comparison.** SPY displacement stays a mandatory, non-gating diagnostic. The report
  must state the mismatch explicitly: a standalone pass says nothing about beating SPY going
  forward.
- **Route after a pass.** r8 replaces the r7 §8 pass route. A pass leads only to *drafting a
  new intraday forward proposal*. It does not exempt anything from the approved standard and
  does not pre-approve any amendment to it. That proposal must set its funding source,
  benchmark, hurdle and timing in a new dated brief, which the owner approves separately.
- **Capital.** None follows from either verdict. This memo makes no personal capital-allocation
  recommendation.

**Flag only (not this file).** The forward standard says "approved" at line 3 and "draft" at
line 40. The owner may want a separate erratum.

### D2 — Window and anchor

**Recommend.**
- **Scored window.** **2016-01-05 → 2026-09-30.** The 2016-01-04 full-session close is a
  signal input only and is not scored. This applies the r7 line 69 fallback once, because the
  2015-12-31 input is not in the documented history.
- **Fixed dates.** The start and the anchor are fixed. If the 2016-01-04 anchor fails D4
  validation, the study is **UNRESOLVED (data)**. It is not zeroed, and the start does not roll
  forward.
- **2014 option.** The 2014 option is closed. The Stage 0A verdict is not relabeled.
- **Session list.** The scored-session list comes from Alpaca's official calendar, frozen at
  collection time. The offline count of 2,679 is not eligibility.

### D3 — Price convention and the future execution gate

**Recommend:** one global, explicitly **idealized** bar convention. The study is labeled a
**proxy feasibility screen**. Bar prices are trade prints, not guaranteed fills or observed
auction results.

| Price | Convention |
|---|---|
| Signal | Close of the 09:59 bar |
| Entry | Open of the 15:30 bar |
| Exit (scored full session, official 09:30–16:00) | Close of the 15:59 bar |
| Prior close | Close of the last minute bar before the official close of the **immediately preceding official session** per the calendar: the 15:59 bar for a 16:00 close, and the 12:59 bar for a 13:00 early close |

- **One rule for both closes.** The exit and the prior close use the same
  "last minute before the official close" rule.
- **No variants or fallbacks.** The anchor is never the 15:59 bar on a half-day. It never
  skips back to an earlier full session. There is no fallback to another minute, to the daily
  bar or to the auction.
- **Half-days.** Half-days are not scored, but their anchor bar must pass validation when they
  precede a scored session. This follows r7 line 69.
- **Auction data.** Auction data is dropped from collection, scoring and diagnostics. No
  auction-record precedence (`6`/`M`/`9`) is adopted.
- **Sizing.** r7's fractional 100%-of-equity sizing is kept as an idealized ceiling.
- **Diagnostic table.** The 15:59-bar-close versus daily-close table is a non-gating
  diagnostic, computed only at scoring.

**Owner choice (not recommended):** make the auction close primary. This needs a frozen
record-selection rule, and current evidence cannot supply one, because later samples show two
`P` records per session.

**Future execution gate.** This applies before any order-based shadow and is not part of this
study. A separate brief must fix the exit order type and its partial-fill, rejection and
missed-exit rules.
- **If that brief picks a market-on-close (`CLS`) exit:** it must use whole-share sizing with
  residual cash, because fractional `CLS` is unsupported. It must handle Alpaca's 15:50 ET
  `CLS` cutoff. It must confirm that the account supports whole-share `CLS`, which the order
  matrix marks `Yes*`, "contact sales", so this is **unverified**.
- **Other order types.** These requirements do not automatically apply to other designs, such
  as a fractional DAY market order. Any market-order exit near 16:00 is a new holding
  interval and needs its own registration. It is not equivalent to the 15:59 bar close.

### D4 — Data validation (signal-blind)

**Problem.** Zeroing a day for an archive defect is unsound for two reasons. A defect in
today's archive is not evidence of what a live trader saw, and selective missingness can bias
the result. Validating only intended trades would also compute the signal before signals are
permitted.

**Recommend.**
1. **Signal-blind validator.** Validation is mechanical and never computes `r_first`, a side
   or any return. It outputs counts and defect lists only, and it ends **PASS** or
   **UNRESOLVED (data)**.
2. **All four endpoints are required on every scored session.** Every scored full-calendar
   session must have all four endpoint valuations valid: the prior close (per D3, including
   half-day anchors), 09:59 close, 15:30 open and 15:59 close.
   - A required endpoint that is absent, duplicated, non-positive, non-finite or misaligned
     makes the study **UNRESOLVED (data)**. Misaligned means its New York timestamp does not
     match the calendar minute or session.
   - There is no data-based zeroing, no day deletion, no gap quota, no rolling start and no
     price fallback.
3. **Study-level stops** also end the study UNRESOLVED (data):
   - a calendar mismatch;
   - an unflagged half-day;
   - an incomplete pagination or provenance record (r7 §3);
   - an SPY split or a documented vendor adjustment change inside the window
     (a proposed explicit automatic stop, rather than an undocumented repair).
   Ordinary ex-dividend repricing is not a defect (r7 §9).
4. **Interior-minute completeness and close reconciliation become non-gating diagnostics.**
   This loosens r7 §9 and is labeled as such. The quality gate becomes endpoint-only, which is
   proportional and has no lookahead.
5. **Zero days.** The only zero days in a valid score come from the registered
   `r_first ≤ 0` flat rule, never from an archive failure.
6. **Live missing-data handling.** A truly observable missing-data or no-order policy belongs
   only in a separately approved prospective shadow. No claim is made that a missing archive
   bar would have been visible to a live trader.
7. **Valuation fallbacks are not recommended now.** Any valuation fallback, or sensitivity
   test that drops missing trades, would need separate registration before prices are seen.

### D5 — Meaning of A1; attribution

**Recommend.**
- **A1 is the one profitability gate.** No predictive co-gate and no beta co-gate are added.
- **Attribution.** Report the following as continuous, non-gating numbers:
  - the identity `mean(d·R) = cov(d,R)·(n−1)/n + mean(d)·mean(R)` in bp/session;
  - the slope `β₁` from `r_last = α + β₁·1[r_first > 0] + ε` with its NW-5 t;
  - the net always-long last-half-hour comparator.
  Under D4 every scored session has all four endpoints, so the comparator values every
  session.
- **Wording.** There is no categorical label, no t-threshold label on `β₁` and no "similar
  returns" cutoff. A1 shows only whether the fixed rule's net mean passed the registered
  profitability test. It does not test whether the morning sign adds information beyond
  afternoon drift. No result is described as proof of an effect, and a failed test is not
  proof that there is no effect.
- **Predictive tests.** Testing prediction formally would be a different hypothesis. It would
  need a new protocol fixed before any price is seen.

### D6 — Scope cuts, isolation and approvals

**Recommend.**
- **Scope cuts.** Drop IEX from collection and diagnostics, because delayed SIP is sufficient
  and IEX had gaps in Stage 0A. Drop the standalone 20-date Stage 0B; the whole-window
  validator replaces it.
- **No production-account orders at any step.** The production trader runs at 15:45 ET with
  a SPY sweep, so it would overlap the rule's holding window.
- **Separate account for any order-based shadow.** It needs a separate isolated account,
  whose availability is **unverified**.

**Approvals: two owner decisions, no others.**

1. **Protocol approval (this memo's defaults).** This authorizes:
   - **Finalizing r8.** Transcribe these decisions and the r7 formulas above into a dated
     brief revision, with no new choices. Any needed deviation goes back to the owner.
     Record r8's hash.
   - **Building the code.** Implement the collector, signal-blind validator and scorer, tested
     only on mocked or synthetic data, then record the code hash. The scorer refuses to run
     without a PASS validation artifact whose r8, code and data-manifest hashes match.
   - **No market-data access.** This approval allows no broker or market-data request
     and no price inspection. Routine development dependency installation from official
     package sources is allowed; new project dependencies must be recorded in
     `pyproject.toml`. No new service or infrastructure is authorized.
2. **Data-and-scoring approval (one explicit decision, after the code is reviewed).** This
   covers exactly:
   - one frozen whole-window SIP pull, resumable with r7 §3 provenance;
   - one validation of that pull;
   - **one automatic score, only on a registered PASS with matching hashes**, writing raw
     JSON and a GO/NO-GO report.

   UNRESOLVED (data) closes the study. No step allows a discretionary stop, a re-score or
   choosing among results.

Shadows, broker orders, live capital and any change to `src/ggTrader/paper/`, cron, Docker or
the production freeze always need their own separate approval.

## Approval block

Mark each item **default / alternative / defer**. Approving all defaults is approval 1 above.

| Item | Default | Alternative |
|---|---|---|
| D1 role and route | Standalone A1+A2 screen. SPY comparison mandatory but non-gating. A pass leads only to drafting a new intraday forward proposal | — |
| D2 window | Score 2016-01-05 → 2026-09-30; 01-04 anchor fixed; invalid anchor = UNRESOLVED | — |
| D3 prices | Idealized proxy: 09:59 close, 15:30 open, 15:59 exit, prior close = last minute before the preceding session's official close; auction dropped | Auction primary (not recommended) |
| D4 validation | Signal-blind; all four endpoints required on every scored session or UNRESOLVED; interior minutes diagnostic | — |
| D5 A1 | One profitability gate; continuous non-gating attribution | — |
| D6 scope/approvals | Drop IEX and 0B; two approvals as above | — |
| A1 HAC setting | `use_correction=True`, 5 lags, versions recorded before pull | — |

## Draft checks (not protocol approval)

Opus drafted and revised this memo after the parent reviewed decision-time
missingness, half-day anchors, diagnostic gates and approval scope. Flash checked
the carried-forward numerical fields and relative links; the parent checked those
against r7 independently. Parent clarifications label the new valuation convention,
qualify the volatility proxy and distinguish dependency setup from prohibited
market-data access. This remains a proposal until the owner approves it.
