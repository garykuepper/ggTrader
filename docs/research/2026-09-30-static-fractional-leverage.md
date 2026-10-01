# Static Fractional Leverage (SPY + SSO, ~1.3x): after-tax vs SPY buy-and-hold — GO for owner decision, A24 sleeve rider NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-09-30
**Audience:** Owner (real-money allocation decision) & Quantitative Research Collaborators
**Owner decision (2026-09-30 21:04 PT): ADOPT 1.3x**, with the funding-cost result known; terms in `docs/next_steps.md` REAL-MONEY PLAN.
**Brief:** `docs/research/briefs/2026-09-30-static-fractional-leverage.md` (backlog A23, batch 2026-09-30)
**Driver / raw results:** `scripts/static_leverage_aftertax.py` → `docs/research/_static_leverage_aftertax_results.json`

## 1. Executive Summary & Core Engine Audit

This is not a strategy test. It answers one allocation question the owner asked
after the 2026-09-30 leveraged-ETF research batch: *is holding SPY plus a fixed
slice of SSO (ProShares 2x S&P 500) likely to beat plain SPY after tax in a
taxable account, and what does it cost in drawdown?* No signal, no timing, no
Sharpe gate — the pre-registered bar (brief, written before the run) was
**after-tax CAGR ≥ SPY + 1.0 pt/yr on 2006-06 → 2026-09 (real SSO) and not below
SPY on a labelled synthetic 1993–2006 holdout.** Sharpe below SPY was declared
expected and not a fail.

**Result: the 70/30 SPY/SSO book (~1.3x) clears both bars on the historical
window — but the §3 Rank 2 funding-cost check, run the same day, shows the
margin falling to +0.80 pt/yr at today's ~4% bills, below the +1.0 bar.** The
GO is therefore conditional on funding costs, not unconditional (see the
funding-cost paragraph at the end of §2). On 20.3 years of real
SSO it earned **12.00% after tax vs SPY's 10.40%** at a 24%/15% bracket (+1.60
pt/yr; +1.55 pt at 40.8%/23.8%), turning $100k into **$992k vs $741k** (+34%
terminal wealth). On the synthetic 1993–2006 holdout it earned 9.53% vs 8.89%
(+0.64 pt) — above SPY, below the main-window margin, because the holdout
contains 2000–02. Pre-tax Sharpe is **0.61 vs 0.64** (worse, as predicted): this
is compensation for holding more equity risk, not an edge. The lot-level tax
engine confirms *why* it works after tax: the band-rebalanced book realized
under $1k of short-term gains in 20 years; everything else deferred to terminal
liquidation at long-term rates.

**What it costs.** Max drawdown **−66% vs −55%** (2008–09), **−42% vs −34%**
(2020), **−32% vs −25%** (2022); 995 vs 884 trading days to recover the 2007
peak; and it **trailed SPY in 24% of all rolling 10-year windows since 1993**,
worst by −3.3 pt/yr for the decade ending March 2009. Leverage also delivered
**1.2 pt/yr less than beta alone predicts** (realized 12.71% pre-tax vs a
beta-matched 13.94%): financing at T-bill + spread, SSO's 0.88% fee, and the
daily-reset covariance drag are real and measured here, not assumed away.

**The A24 rider (60% SPY core + 40% sleeve switched SSO↔SPY on a 10-month SMA,
±2% buffer, monthly) is NO-GO as a return improvement.** Against its
pre-registered comparator — the static book, after tax, on the real-SSO window —
it lost (11.73% vs 12.00% at 24/15; 10.80% vs 11.33% at the top bracket) while
realizing **$87k of short-term gains** vs under $1k for the static book and
taking **195 vs 110 days** to recover from 2020. It bought ~8 pt of shallower
drawdown (−58% vs −66%) and did better in the synthetic 2000–02 era, which is
the era slow-trend rules were built around. That is drawdown insurance paid
for in taxes and whipsaw, not alpha; the closed `leveraged_trend_*` arc
(2026-07-16) and the 2026-09-28 cross-asset trend result said the same.

**Already closed and not re-tread here:** whole-book 200-day SSO↔T-bill
rotation (duplicate of `leveraged_trend_*`, NO-GO), VIX/VIX3M gates (B6, low),
RSI(2) on LETFs (rejected on intake). See §4.

## 2. Quantitative Performance Context

Main window 2006-06-21 → 2026-09-29 (real SSO, 20.3 years), $100k start,
1 bp/side, distributions reinvested, month-end band check traded next bar,
tax paid the following April from the portfolio, terminal liquidation.

| Book | Pre-tax CAGR | Pre-tax Sharpe | MaxDD | After-tax CAGR 24/15 | After-tax CAGR 40.8/23.8 | Terminal $ (24/15) | ST gains realized |
|---|---|---|---|---|---|---|---|
| **SPY buy-and-hold** | 11.10% | **0.64** | −55.3% | 10.40% | 9.78% | 741k | $0.7k |
| Static 1.25x (75/25, ±8pp) | 12.48% | 0.61 | −64.7% | 11.80% | 11.11% | 908k | $0.9k |
| **Static 1.3x (70/30, ±8pp)** | 12.71% | 0.61 | −66.4% | **12.00%** | **11.33%** | **992k** | $0.9k |
| Static 1.5x (50/50, ±10pp) | 13.76% | 0.59 | −72.6% | 12.98% | 12.30% | 1,182k | $9.0k |
| Static 2.0x (SSO, no rebalance) | 15.54% | 0.57 | −84.8% | 14.69% | 14.04% | 1,694k | — |
| Idealized 1.3x daily rebalance | 12.24% | 0.59 | −67.4% | 11.75% | 10.89% | 953k | — |
| A24 rider (60 core + 40 sleeve, 10m SMA ±2%) | 12.09% | 0.60 | −58.0% | 11.73% | 10.80% | 944k | **$87.2k** |

Beta-matched pre-tax expectation (rf 1.64% + β·(SPY − rf)): 1.3x → 13.94%,
1.5x → 15.83%. Realized: 12.71% and 13.76%. **Shortfall 1.2 and 2.1 pt/yr** =
the all-in cost of getting leverage through SSO.

Holdout 1993-02-01 → 2006-06-20 (**synthetic SSO**, 13.4 years; see §6 for how
it was built and calibrated):

| Book | Pre-tax CAGR | Sharpe | MaxDD | After-tax 24/15 | After-tax 40.8/23.8 |
|---|---|---|---|---|---|
| SPY | 9.64% | 0.62 | −47.8% | 8.89% | 8.26% |
| Static 1.3x | 10.33% | 0.55 | −60.1% | 9.53% | 8.97% |
| Static 1.5x | 10.64% | 0.52 | −66.5% | 9.81% | 9.17% |
| A24 rider | 11.04% | 0.58 | −53.4% | 10.70% | 9.95% |

Full stitched 1993 → 2026 (33.7 years): SPY 10.08% / 1.3x **11.38%** / 1.5x
12.05% / A24 11.40% after tax at 24/15.

**Stress panel (pre-tax, peak-to-trough, full stitched series):**

| Episode | SPY | 1.3x | 1.5x | SSO | A24 |
|---|---|---|---|---|---|
| 2000-03 → 2002-10 (synthetic) | −47.8% | −60.1% | −66.5% | −79.2% | −53.4% |
| 2007-10 → 2009-03 (real) | −55.3% | −66.2% | −73.3% | −84.8% | −57.9% |
| 2011 | −18.6% | −24.6% | −27.4% | −36.2% | −23.5% |
| 2015-16 | −12.7% | −17.3% | −20.8% | −26.2% | −16.8% |
| 2020-02 → 2020-03 | −33.7% | −42.4% | −48.6% | −59.3% | −45.1% |
| 2022 | −24.7% | −32.3% | −37.3% | −46.8% | −29.4% |

Recovery to prior peak (trading days): after 2009-03-09 — SPY 884, 1.3x 995,
1.5x 1,096, SSO 1,441, A24 889. After 2020-03-23 — SPY 100, 1.3x 110, 1.5x 112,
A24 **195**.

**Rolling 10-year CAGR vs SPY (pre-tax, all windows 1993–2026):** 1.3x behind SPY
in **23.8%** of windows, worst **−3.3 pt/yr** (decade ending 2009-03); 1.5x
26.5% / −6.0 pt; SSO 37.5% / −12.6 pt; A24 10.1% / −0.9 pt.

**Wash-sale sensitivity:** immaterial for the static books (<0.01 pt/yr; under
$7k disallowed over 33 years) because they almost never sell at a loss. It
costs A24 0.05–0.16 pt/yr (~$80k disallowed on the main window from
SSO→SPY→SSO round trips).

**Funding-cost sensitivity (run 2026-09-30, `--rf-offset`, applied as a daily
drag on real SSO since it finances ~1x NAV; raw JSON
`_static_leverage_aftertax_rf_plus0.015.json` / `_rf_plus0.025.json`).** The
main window's T-bills averaged 1.64%; bills are ~4.1% today. Holding
everything else at history:

| Bills vs sample avg | 70/30 after-tax CAGR 24/15 | margin vs SPY (10.40%) | top bracket margin |
|---|---|---|---|
| as history (+0.0) | 12.00% | **+1.60 pt** | +1.55 pt |
| +1.5 pt (~3.1% bills) | 11.53% | **+1.13 pt** | +1.10 pt |
| +2.5 pt (~4.1% bills, today) | 11.20% | **+0.80 pt** | +0.79 pt |

At today's funding cost the 1.3x book is **below the pre-registered +1.0 pt
bar**, with the same −66% drawdown. 1.25x is worse still (+0.70 pt). The
edge is real but thin, and it is a function of the bill rate: roughly
0.3 pt of after-tax margin per 1 pt of bills on a 30% SSO slice. This is
the number that should drive the decision, not the historical +1.6.

**Band vs daily rebalancing:** the ±8pp band beat idealized daily rebalancing by
0.47 pt pre-tax and 0.25–0.44 pt after tax (momentum drift plus far fewer
taxable sells). Banding is the right policy, and it is not a tuning knob here:
one width per ratio, fixed in the brief.

## 3. Actionable Research Directions

This report's job was to produce a number for a decision, so the "directions"
are what would change that number, ranked by decision value per effort.

### Rank 1: Decide the ratio and the exit rule — owner action, not research

**Mechanism.** 70/30 SPY/SSO, checked at month-end, traded the next day only if
SSO's weight has left 22–38%; restore with new cash first, otherwise sell the
highest-basis long-term lots. No timing overlay. Distributions reinvested.
**Why 1.3x and not 1.5x:** 1.5x adds +1.0 pt/yr after tax on the main window but
takes the 2008 drawdown from −66% to −73%, recovery from 995 to 1,096 days, and
the worst rolling decade from −3.3 to −6.0 pt/yr behind SPY. The Kelly-optimal
range in the literature spans 1.17x (Thorp, 1926–84) to 2.4x (Smirnov,
1996–2024); 1.3x sits inside the conservative half of every estimate.
**Failure mode to pre-commit against:** abandoning at −50%. The historical
record for 1.3x says: a decade behind SPY has happened (ending 2009), and
recovery from 2008 took four years. Write the exit rule now — the only
defensible ones are "never" or a pre-set rebalance to 100% SPY at a *date*, not
at a drawdown level, because selling at the trough is what converts a paper
loss into a permanent one plus a tax bill.
**Payoff / Effort / Failure.** +1.6 pt/yr after tax, measured on 2006–2026;
+0.6 pt on a synthetic 1993–2006. Effort: none in the lab. Risk: a 2000–2012
style decade during which leverage costs money and returns nothing.

### Rank 2: Funding-cost regime check — DONE 2026-09-30 (see §2 table)

**Mechanism.** The 1.2 pt/yr beta shortfall was measured with T-bills averaging
1.64% over 2006–2026. At today's ~4.1% bills the financing leg of SSO costs
roughly 2.5 pt/yr more than the sample average on the 2x leg (≈0.75 pt on the
30% slice). Re-run the main window with `--spread` and a rate shift to see
whether the +1.6 pt margin survives a decade of 4–5% bills. **Measured:** +0.80 pt/yr at +2.5 pt bills (today), +1.13 at +1.5 pt — the
prior guess of +0.8–1.0 was right, and it lands *below* the bar today.
**Why it differs from rejected work:** nothing closed looks at LETF financing
cost regimes. **Effort:** S — the driver already takes `--spread`; add a
rate-offset flag.

### Rank 3: Fill the 1.3x book from the sweep, not from the stock sleeve

If adopted live, the natural implementation is to make SSO the cash-sweep
target for a 30% share rather than touching the frozen stock-picking sleeve
(`cash_sweep.py` sweeps idle cash into one symbol; a 70/30 split needs a
two-symbol sweep with the band rule). This is **out of scope until the parity
window ends (~2026-12-29)** and needs its own deploy brief. Nothing in this
report touches live code.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

- **A24 sleeve trend rider — NO-GO as a return improvement (this report).**
  Loses to the static book after tax on the real-SSO window at both brackets,
  realizes ~$87k ST gains, and roughly doubles the 2020 recovery time. Its
  drawdown benefit is real (−58% vs −66%) and its 2000–02 record is better, but
  that is the regime slow-trend rules were fitted to. Anyone re-proposing a
  leverage-sleeve trend switch must beat **this report's static 1.3x after-tax
  numbers**, not SPY.
- **Whole-book 200-day SSO↔T-bill rotation (Gayed–Bilello LRS)** — duplicate of
  `leveraged_trend_*` (`docs/research/2026-07-16-leveraged-trend-following-nogo.md`,
  NO-GO vs own buy-and-hold); rejected on intake in the 2026-09-30 batch.
- **Breadth-driven leveraged rotation** —
  `docs/research/2026-07-16-leveraged-index-rotation-nogo.md`, NO-GO.
- **VIX-level entry gate** — NO-GO (2026-06-28); **VIX/VIX3M gate** stays low
  (B6): 2022 failure mode, ~10 switches/yr, unsourced performance claims.
- **RSI(2) / short-hold reversion on LETFs** — rejected on intake: every gain
  short-term by construction.
- **Idealized daily-rebalanced fractional leverage** — measured here as strictly
  worse than banding; don't build it.

## 5. Operational Roadmap: Recommended First Action

1. **Owner decides** whether −66%-class drawdowns and a possible decade behind
   SPY are acceptable for +1.6 pt/yr after tax. If yes: 70/30, band 22–38%,
   month-end check, exit rule written down in `docs/next_steps.md` before any
   money moves. If no: SPY buy-and-hold stands, and this arc is closed GO-but-
   declined.
2. Run the Rank 2 funding-cost check (one afternoon) before committing real
   money — it is the one input that could pull the margin below the bar.
3. Live implementation, if any, waits for the parity window (~2026-12-29) and
   goes through the `ggtrader-deploy` skill as a two-symbol sweep change.
4. `research-snapshot` skill: move A23 and A24 from `WEB_RESEARCH_CANDIDATES.md`
   into the roster.

## 6. Contrarian Evaluation & Parked Research

**Method limits — read before quoting a number.**
- **Synthetic SSO (1993–2006).** Built as `2·r_SPY(total return) − (T-bill +
  0.75%)/252 − 0.88%/252` from FRED `DTB3`. Calibrated on the real overlap
  2006-06 → 2026-09: correlation with real SSO **0.9956**, mean residual
  **−0.15%/yr** (applied as a haircut pre-2006), residual vol 3.6%/yr. The
  synthetic has **zero distributions**, so all its return is deferred price
  gain — a small tax-deferral advantage vs real SSO (which paid ~1%/yr of
  distributions in 2024). Holdout margins (+0.6 pt) are therefore, if anything,
  slightly generous.
- **Tax engine simplifications.** Federal only (no state); NIIT included only in
  the 40.8/23.8 bracket; SSO distributions taxed at the LT/qualified rate (some
  may be ordinary income in reality); capital losses carried forward against
  future gains with no $3k ordinary offset; wash sales approximated as "loss
  sale within 30 days *after* a buy of the same ticker" (no forward look);
  taxes paid from the portfolio each April by pro-rata sells. Terminal
  liquidation taxes everything at the end — the after-tax CAGRs are
  *liquidation* CAGRs, the strict form.
- **Not a WFO run.** There are no folds or gates: the object under test is an
  allocation, not a parameterized strategy, and lot-level accounting needs a
  stateful loop that `simulate_weights` can't do. The lab's Sharpe-vs-SPY gate
  was declared inapplicable in the brief before the run; the honest reading is
  that this book **fails** that gate (0.61 vs 0.64) and passes only the
  after-tax-wealth bar.
- **Survivorship of the instrument.** SSO has existed 20 years and ProShares'
  fee waiver expired 2026-09-30; a fund closure or fee change is a live risk
  the backtest cannot price. UPRO (3x) was not tested as a component on
  purpose.
- **Regime.** 2006–2026 contains one great crash and two long bull runs. The
  1993–2006 holdout adds 2000–02 and shows the margin shrinking to +0.6 pt.
  A 1966–1982-style stretch is absent from all windows; the rolling-10-year
  statistic (behind SPY 24% of the time) is the closest available proxy.

**Parked.**
- **UPRO-based fractional mixes (e.g. 85/15 SPY/UPRO ≈ 1.3x).** Same exposure
  with a smaller, more fee-efficient slice, but 3x daily reset has a worse
  covariance drag (Bianchi & Goldberg: −5.6 vs −2.2 pt/yr over 2022–23) and
  only 17 years of history. Cheap to add to the driver; not decision-relevant
  today.
- **B13 drawdown-controlled de-leveraging (Grossman–Zhou)** and a **conditional
  vol cut of the sleeve (B5-sleeve)** — overlays on top of this book. A24's
  result sets their bar: any overlay must beat static 1.3x after tax on
  2006–2026, and pay for its short-term realizations. Low prior.
- **Financing-aware ratio** (lower the SSO share when bills are above ~4%) —
  a timing rule in disguise; park unless Rank 2 shows the margin dies at
  current rates.
