# Brief: classic long-only factor screen for a taxable trading sleeve

**Status:** pre-registered; approved by the owner 2026-10-21 before anything was computed. Its numbers are now the bar; changing any of them needs a new dated brief.

**Terms used below:**
- *Factor* means a quantifiable firm characteristic (such as value, profitability, or investment rate) that academic research associates with systematic differences in expected stock returns.
- *Value (HiBM)* means buying stocks with high book equity relative to market equity (cheap on book value).
- *Robust Profitability (HiOP)* means buying stocks with high operating profitability relative to book equity.
- *Conservative Investment (LoINV)* means buying stocks of firms that grow their total assets conservatively (low annual asset growth rate).
- *Momentum (HiPRIOR)* means buying stocks with high prior returns over the preceding 12 to 2 months (skipping the most recent month).
- A *t-stat* measures the signal-to-noise ratio of excess return: a larger $t$ means the extra return is harder to explain as noise under the null model. Thresholds around $t \approx 3.00$ to $\approx 3.40$ correspond to small one-sided asymptotic normal-tail probabilities ($p \approx 0.00135$ down to $p \approx 0.00034$), conditional on model assumptions, rather than the probability that the strategy has no edge.
- *Newey-West (NW)* is a t-stat method that adjusts standard errors for serial correlation (returns in one period correlated with subsequent periods) and heteroskedasticity.
- *Holm correction* is a step-down procedure that controls the family-wise error rate across multiple simultaneous hypothesis tests without being as overly conservative as a simple Bonferroni cutoff.
- *CAGR* (Compound Annual Growth Rate) is the annualized geometric rate of return.
- *Survivorship-free* data includes all historical companies, including those that were subsequently delisted, acquired, or went bankrupt.
- *PIT* (point-in-time) means using index constituents and fundamental accounting data as known on each historical decision date, without lookahead bias.

## 1. The question, and the decision it feeds

- **Question.** Do any classic long-only stock factors (other than momentum, closed NO-GO in [`docs/research/2026-10-20-momentum-french-screen-nogo.md`](../2026-10-20-momentum-french-screen-nogo.md)) still beat the market after costs since 1994, by enough to justify a paid-data test of an S&P 500 version for the owner's taxable stock-trading path?
- **Decision fed.** Whether to write a Phase 2 brief (paid survivorship-free data, exact rules, after-tax) for a passing factor.
- **Owner context.** Taxable account only. The owner wants an active stock-trading path with greater gains and accepts bigger swings.
- **Not affected.** The owner's core A23 allocation (70/30 SPY/SSO) is a separate decision and does not change.

## 2. Why: two-part standard and data limits

- **The two-part standard.** Evaluating active factor strategies requires a two-part standard (modeled on the momentum brief §3):
  - **Part A, the idea.** The factor must demonstrate statistical significance (NW t ≥ 3-level significance under family-wise control) on decades of survivorship-free academic data.
  - **Part B, our version.** Our exact implementation must beat SPY by at least +1.0 pt/yr *after tax*, on clean point-in-time data not used to design the rules (evaluated in a subsequent phase).
- **Why existing internal data cannot answer Part A.** Under a stable-return approximation ($t \approx \text{IR} \cdot \sqrt{\text{years}}$), IRs of 0.35–0.40 need roughly 56–73 years for an expected $t$ of 3, and 72–94 years for the first Holm hurdle ($\approx 3.40$). These are signal-to-noise illustrations, not detection guarantees or power calculations. Our own clean point-in-time equity tape covers the S&P 500 from 2021 to 2026-04 only (~5 years). Evaluating multi-decade factor persistence is impossible on internal data alone.
- **French data is external and ready.** The Ken French Data Library provides over 60 years of free, value-weighted, survivorship-free US equity portfolio returns. French data is external to ggTrader, but published factor definitions were developed using historical returns in this sample. A2 is not uniformly post-publication. The 2013–2026 diagnostic measures recent performance and includes some pre-publication observations. A pass supports further research under the registered assumptions; it does not establish an independent post-publication edge. The data downloading, verification, and screening harness already exists in the repo ([`src/ggTrader/lab/french_data.py`](../../../src/ggTrader/lab/french_data.py), [`src/ggTrader/lab/momentum_screen.py`](../../../src/ggTrader/lab/momentum_screen.py)).

## 3. Data (Ken French Data Library)

Monthly, value-weighted portfolios from the Ken French Data Library. All raw source archives must be frozen with download timestamp and SHA-256 checksums recorded in `data/external/french/MANIFEST.json` before any computation begins, identical to the momentum study protocol.

- **Operating Profitability:** `6_Portfolios_ME_OP_2x3_CSV.zip` → **BIG HiOP** (large-cap robust profitability).
- **Book-to-Market:** `6_Portfolios_2x3_CSV.zip` (size × book-to-market) → **BIG HiBM** (large-cap value).
- **Investment:** `6_Portfolios_ME_INV_2x3_CSV.zip` → **BIG LoINV** (large-cap conservative investment).
- **Prior Return (Momentum):** already-frozen `6_Portfolios_ME_Prior_12_2` → **BIG HiPRIOR** (large-cap winners, used only inside the combo hypothesis H4).
- **Market Benchmark:** already-frozen `F-F_Research_Data_Factors` → Market total return = `Mkt-RF + RF`.
- **Section and columns.** All three new files use the section `Average Value Weighted Returns -- Monthly` and the exact column labels `BIG HiOP`, `BIG HiBM`, `BIG LoINV` (and `SMALL HiOP`, `SMALL HiBM`, `SMALL LoINV`), checked against the live files on 2026-10-20 without computing any returns.
- **Not the S&P 500.** French's "BIG" means above the NYSE median market cap, roughly 1,000+ stocks including mid-caps, not the ~500 S&P 500 names. Value here is pure book-to-market, which misses intangible capital. A pass shows the factor is real in broad large caps; it does not show an S&P 500 version will clear Part B.

## 4. Hypotheses: a family of FOUR (Big only)

The confirmatory test family comprises exactly four primary hypotheses, restricted to large-cap (Big) portfolios:

- **H1:** BIG HiOP > market (robust profitability premium).
- **H2:** BIG HiBM > market (value premium).
- **H3:** BIG LoINV > market (conservative investment premium).
- **H4:** Combo = 50% BIG HiBM + 50% BIG HiPRIOR (rebalanced monthly) > market (value + momentum combination).

Small-cap portfolios are explicitly excluded from the confirmatory family (see §7 and §11).

## 5. Series and cost haircuts

Monthly excess return series are defined as:

$$e_t = R_{\text{portfolio},t} - R_{\text{mkt},t} - \frac{c}{12}$$

where $c$ is the fixed annual cost haircut subtracted monthly before compounding. These are policy haircuts, not calibrated turnover, execution-cost, or tax estimates. H4's higher allowance is an assumption reflecting its momentum exposure. Passing establishes robustness only to the stated haircut.

- **H1–H3 (HiOP, HiBM, LoINV):**
  - **Base haircut:** $c = 1.0\%$/yr.
  - **Stress haircut:** $c = 2.0\%$/yr.
- **H4 (Value + Momentum Combo):** R_combo,t = 0.5 · R_BIG HiBM,t + 0.5 · R_BIG HiPRIOR,t (gross, rebalanced every month), then the haircut below is subtracted once from the combo.
  - **Base haircut:** $c = 1.5\%$/yr.
  - **Stress haircut:** $c = 3.0\%$/yr.
  *(Higher allowance is an assumption reflecting its momentum exposure).*

## 6. Tests: significance (A1) and Since-1994 economic persistence (A2)

- **Common Window:** 1963-07 → 2026-04. The end month is fixed at 2026-04, the last month of the already-frozen momentum and factor files; newly frozen files are truncated to it. Operating profitability (`6_Portfolios_ME_OP_2x3`) and investment (`6_Portfolios_ME_INV_2x3`) start in 1963-07, establishing a single common sample window across all four hypotheses.
- **Test A1 (Statistical Significance):**
  - Evaluated over the full common window (1963-07 to 2026-04) at the base cost haircut.
  - A1 tests $H_0: E[e_t] \le 0$ against $E[e_t] > 0$, using the registered Bartlett NW(12) standard error and $p = \Phi(-t)$. These are asymptotic normal-tail p-values conditional on the model assumptions; they do not measure the probability that the strategy has no edge. Family-wise control is correspondingly nominal/asymptotic.
  - Test statistic: mean monthly excess return $e_t$ divided by its Bartlett Newey-West standard error with 12 lags ($\text{NW } t$).
  - One-sided p-values: $p = \Phi(-t)$.
  - Multiple testing adjustment: Holm step-down procedure across the family of four hypotheses at an overall family-wise error rate of $\alpha = 0.00135$ (equivalent to a one-sided $t \ge 3.00$ for a single hypothesis).
  - Procedure: Apply Holm to all four base-cost A1 p-values before filtering on A2. Sort unrounded p-values, compare sequentially with the listed thresholds, and stop at the first failure; that hypothesis and all later ones fail A1. Use exact p thresholds, not rounded t equivalents. Reuse the existing `holm` and NW implementations. Holm is valid under arbitrary dependence, so H2 and H4 sharing BIG HiBM costs power, not validity.
  - Ordered Holm significance thresholds (from smallest $p$ / largest $t$ to largest $p$):
    1. Rank 1: $p \le 0.0003375$ (approximate $t \approx 3.40$).
    2. Rank 2: $p \le 0.00045$ (approximate $t \approx 3.32$).
    3. Rank 3: $p \le 0.000675$ (approximate $t \approx 3.21$).
    4. Rank 4: $p \le 0.00135$ (approximate $t \approx 3.00$).
    *(All $t$-values marked as approximate normal critical equivalents).*
- **Test A2 (Since-1994 economic persistence):**
  - Evaluated over 1994-01 → 2026-04 (388 monthly observations).
  - Executable formula (using decimal returns):
    $$\Delta\text{pp} = 100 \cdot \left[ \left(\prod_{t=1}^{388} (1 + R_{p,t} - c/12)\right)^{12/388} - \left(\prod_{t=1}^{388} (1 + R_{m,t})\right)^{12/388} \right]$$
  - A2 passes when the unrounded result is $\ge 1.0$ (i.e. at least **+1.0 pt/yr** / +100 bps/yr).
  - Market CAGR is gross of costs and tax.
  - For H4, form the monthly weighted return first, then subtract its haircut once.
  - Note this calculation matches `momentum_screen.cagr_diff_pp`.
- **Pass Rule:** A factor passes Phase 1 if and only if it satisfies **both A1 and A2**.
- **Taxable caution.** A2 is net of the cost haircut only, not tax. A thin A2 pass leaves limited margin for incremental tax drag versus the benchmark; whether Part B fails requires the registered stock-level tax replay.

## 7. Diagnostics

Diagnostics are reported to characterize behavior, risk, and subperiod stability; they are never pass criteria and cannot override failed gates.

- **Century-long history where data allow:** Value (BIG HiBM) and Combo (H4) evaluated over 1927-01 → 2026-04 (the momentum file starts 1927-01; raw value data starts 1926-07).
- **Subperiods:** 1963-07 → 1993-12, 1994-01 → 2012-12, and 2013-01 → 2026-04.
- **Post-2013 net CAGR difference:** Reported plainly for all candidates. Robust profitability and investment were formally published circa 2013–2015 (e.g., Novy-Marx 2013; Fama and French 2015). The 2013–2026 diagnostic measures recent performance and includes some pre-publication observations.
- **Stress-cost evaluation:** Full-window NW $t$, $p$, and post-1994 net CAGR difference evaluated under the stress cost haircuts (2.0% for H1–H3; 3.0% for H4).
- **CAPM regression:** Annualized arithmetic alpha and market beta only. No alpha $t$-statistic is reported: the momentum study removed its version as mis-specified, and a valid one is out of scope for a diagnostic. Following the momentum study correction, alpha is estimated by regressing gross portfolio excess returns over the risk-free rate on market excess returns over the risk-free rate ($R_{i,t} - R_{f,t} = \alpha + \beta (R_{m,t} - R_{f,t}) + \epsilon_t$).
- **Worst 12-month relative return:** Compounded rolling 12-month winners-to-market wealth ratio $\prod (1 + R_{p}) / \prod (1 + R_{m}) - 1$.
- **Relative drawdown and recovery:** Maximum peak-to-trough decline in net relative wealth $\prod (1 + R_p - c/12) / \prod (1 + R_m)$ from initial value 1.0, peak and trough dates, and calendar months to recovery (or null if unrecovered).
- **Small-cap analogues:** SMALL HiOP, SMALL HiBM, and SMALL LoINV reported as diagnostics only, strictly outside the confirmatory Holm family, because we have no point-in-time small-cap universe or execution capability to trade them. They use a 3.0%/yr base and 6.0%/yr stress haircut, as in the momentum brief.

## 8. What happens next

| Phase 1 Result | Next Step |
|---|---|
| None pass | Factor screen closed NO-GO, no paid data. Report the closure plainly and propose the next research step. |
| One or more pass | Only the single passing factor with the smallest $p$-value (tie: larger post-1994 net CAGR difference) advances to a separate Phase 2 brief (paid 2000–2020 survivorship-free data, exact S&P 500 rules, after-tax vs SPY ≥ +1.0 pt/yr = Part B). |

**Authority:** Passing Phase 1 authorizes further research only. It is not an authorization for deployment or real capital. Advancing one candidate limits research scope; it does not remove winner's-curse bias or establish error control across successive studies. Holm covers this registered four-hypothesis family only. Phase 2 must freeze implementation rules before inspecting stock-level outcomes and distinguish implementation validation from a fresh temporal holdout.

## 9. Data-quality stops

Computation must stop immediately, and no verdict may be rendered, if any of the following occur:
- SHA-256 checksum mismatch against `MANIFEST.json`.
- Missing calendar months within the registered analysis window (1963-07 through 2026-04).
- Stop on duplicate months, non-finite required returns, or non-positive monthly wealth factors after costs. No dropping, filling, interpolation, or shortened analysis windows is permitted.
- File layout or header changes that fail strict schema parsing.
- Hand-patching or manual data edits are strictly prohibited; any defect must be reported.

## 10. Scope and boundaries

- **Code modifications:** New code is restricted strictly to `src/ggTrader/lab/` and `scripts/` (generalizing the existing momentum screen rather than duplicating it).
- **Output artifact:** Results must be written to `docs/research/_factor_screen_results.json`.
- **Untouchable paths:** Never touch `src/ggTrader/paper/`, `src/ggTrader/data/live/`, `.env`, or Docker files.
- **Research boundaries:**
  - No parameter search, tuning, or factor definition modifications after viewing data.
  - No inclusion of short-term reversal or small-cap execution sleeves.
  - Pre-registration principle: no data download or metric calculation may occur until the owner explicitly approves this brief.

## 11. Calls made by the senior quant (owner to check)

1. **Big-only Holm family of four:** Testing only large-cap portfolios controls multiple testing burden while matching the investable large-cap universe. Small caps are reported as diagnostics only.
2. **Short-term reversal excluded:** Short-term (1-month) reversal requires near-100% monthly portfolio turnover, creating prohibitive transaction friction in taxable execution. Furthermore, ggTrader's mean-reversion daily ensemble has already failed in live replica audits.
3. **Common 1963-07 window:** Restricting the primary statistical window to 1963-07 aligns all four factors to identical macro regimes, since OP and INV data begin in July 1963.
4. **Combo uses existing frozen momentum file:** H4 combines the new Value series with the already-frozen, checksummed `6_Portfolios_ME_Prior_12_2` Big HiPRIOR series.
5. **H4 cost haircut of 1.5% base / 3.0% stress:** These are policy haircuts, not calibrated turnover, execution-cost, or tax estimates. H4's higher allowance is an assumption reflecting its momentum exposure. Passing establishes robustness only to the stated haircut.
6. **Only the strongest passer advances:** If multiple factors clear A1 and A2, only the factor with the lowest $p$-value (tie: larger post-1994 net CAGR difference) advances to Phase 2 paid data testing. Advancing one candidate limits research scope; it does not remove winner's-curse bias or establish error control across successive studies. Holm covers this registered four-hypothesis family only. Phase 2 must freeze implementation rules before inspecting stock-level outcomes and distinguish implementation validation from a fresh temporal holdout.
