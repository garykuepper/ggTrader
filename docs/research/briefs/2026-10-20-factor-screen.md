# Brief: classic long-only factor screen for a taxable trading sleeve

**Status:** DRAFT 2026-10-20, awaiting owner approval. Nothing has been computed.

**Terms used below:**
- *Factor* means a quantifiable firm characteristic (such as value, profitability, or investment rate) that academic research associates with systematic differences in expected stock returns.
- *Value (HiBM)* means buying stocks with high book equity relative to market equity (cheap on book value).
- *Robust Profitability (HiOP)* means buying stocks with high operating profitability relative to book equity.
- *Conservative Investment (LoINV)* means buying stocks of firms that grow their total assets conservatively (low annual asset growth rate).
- *Momentum (HiPRIOR)* means buying stocks with high prior returns over the preceding 12 to 2 months (skipping the most recent month).
- A *t-stat* measures how sure we can be that an extra return isn't luck. Here thresholds around t ≈ 3.00 to ~3.40 correspond to extreme one-sided normal-tail probabilities (from 1-in-740 down to roughly 1-in-3,000 against luck alone).
- *Newey-West (NW)* is a t-stat method that adjusts standard errors for serial correlation (returns in one period correlated with subsequent periods) and heteroskedasticity.
- *Holm correction* is a step-down procedure that controls the family-wise error rate across multiple simultaneous hypothesis tests without being as overly conservative as a simple Bonferroni cutoff.
- *CAGR* (Compound Annual Growth Rate) is the annualized geometric rate of return.
- *Survivorship-free* data includes all historical companies, including those that were subsequently delisted, acquired, or went bankrupt.
- *PIT* (point-in-time) means using index constituents and fundamental accounting data as known on each historical decision date, without lookahead bias.

## 1. The question, and the decision it feeds

- **Question.** Do any classic long-only stock factors (other than momentum, closed NO-GO in [`docs/research/2026-10-20-momentum-french-screen-nogo.md`](../2026-10-20-momentum-french-screen-nogo.md)) still beat the market after costs since publication, by enough to justify a paid-data test of an S&P 500 version for the owner's taxable stock-trading path?
- **Decision fed.** Whether to write a Phase 2 brief (paid survivorship-free data, exact rules, after-tax) for a passing factor.
- **Owner context.** Taxable account only. The owner wants an active stock-trading path with greater gains and accepts bigger swings.
- **Not affected.** The owner's core A23 allocation (70/30 SPY/SSO) is a separate decision and does not change.

## 2. Why: two-part standard and data limits

- **The two-part standard.** Evaluating active factor strategies requires a two-part standard (modeled on the momentum brief §3):
  - **Part A, the idea.** The factor must demonstrate statistical significance (NW t ≥ 3-level significance under family-wise control) on decades of independent, survivorship-free academic data.
  - **Part B, our version.** Our exact implementation must beat SPY by at least +1.0 pt/yr *after tax*, on clean point-in-time data not used to design the rules (evaluated in a subsequent phase).
- **Why existing internal data cannot answer Part A.** Detecting an excess return edge over the market at a realistic information ratio (e.g., IR ≈ 0.35 to 0.40) at t ≥ 3 requires ~60 to 70 years of independent data. Our own clean point-in-time equity tape covers the S&P 500 from 2021 to 2026-04 only (~5 years). Evaluating multi-decade factor persistence is impossible on internal data alone.
- **French data is independent and ready.** The Ken French Data Library provides over 60 years of free, value-weighted, survivorship-free US equity portfolio returns. The data downloading, verification, and screening harness already exists in the repo ([`src/ggTrader/lab/french_data.py`](file:///home/flynn/ggTrader/src/ggTrader/lab/french_data.py), [`src/ggTrader/lab/momentum_screen.py`](file:///home/flynn/ggTrader/src/ggTrader/lab/momentum_screen.py)).

## 3. Data (Ken French Data Library)

Monthly, value-weighted portfolios from the Ken French Data Library. All raw source archives must be frozen with download timestamp and SHA-256 checksums recorded in `data/external/french/MANIFEST.json` before any computation begins, identical to the momentum study protocol.

- **Operating Profitability:** `6_Portfolios_ME_OP_2x3` → **BIG HiOP** (large-cap robust profitability).
- **Book-to-Market:** `6_Portfolios_2x3` (size × book-to-market) → **BIG HiBM** (large-cap value).
- **Investment:** `6_Portfolios_ME_INV_2x3` → **BIG LoINV** (large-cap conservative investment).
- **Prior Return (Momentum):** already-frozen `6_Portfolios_ME_Prior_12_2` → **BIG HiPRIOR** (large-cap winners, used only inside the combo hypothesis H4).
- **Market Benchmark:** already-frozen `F-F_Research_Data_Factors` → Market total return = `Mkt-RF + RF`.

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

where $c$ is the fixed annual cost haircut subtracted monthly before compounding. These haircuts are pre-registered structural assumptions, not empirical turnover estimates (aggregate French portfolios do not provide individual transaction records).

- **H1–H3 (HiOP, HiBM, LoINV):**
  - **Base haircut:** $c = 1.0\%$/yr.
  - **Stress haircut:** $c = 2.0\%$/yr.
- **H4 (Value + Momentum Combo):**
  - **Base haircut:** $c = 1.5\%$/yr.
  - **Stress haircut:** $c = 3.0\%$/yr.
  *(Higher haircut reflects the higher portfolio turnover inherent in the 12-2 momentum leg).*

## 6. Tests: significance (A1) and modern persistence (A2)

- **Common Window:** 1963-07 → last month common to all files. Operating profitability (`6_Portfolios_ME_OP_2x3`) and investment (`6_Portfolios_ME_INV_2x3`) start in 1963-07, establishing a single common sample window across all four hypotheses.
- **Test A1 (Statistical Significance):**
  - Evaluated over the full common window (1963-07 to common end month) at the base cost haircut.
  - Test statistic: mean monthly excess return $e_t$ divided by its Bartlett Newey-West standard error with 12 lags ($\text{NW } t$).
  - One-sided p-values: $p = \Phi(-t)$.
  - Multiple testing adjustment: Holm step-down procedure across the family of four hypotheses at an overall family-wise error rate of $\alpha = 0.00135$ (equivalent to a one-sided $t \ge 3.00$ for a single hypothesis).
  - Ordered Holm significance thresholds (from smallest $p$ / largest $t$ to largest $p$):
    1. Rank 1: $p \le 0.0003375$ (approximate $t \approx 3.40$).
    2. Rank 2: $p \le 0.00045$ (approximate $t \approx 3.32$).
    3. Rank 3: $p \le 0.000675$ (approximate $t \approx 3.21$).
    4. Rank 4: $p \le 0.00135$ (approximate $t \approx 3.00$).
    *(All $t$-values marked as approximate normal critical equivalents).*
- **Test A2 (Modern Post-Publication Persistence):**
  - Evaluated over 1994-01 → same common end month.
  - Portfolio CAGR net of base haircut minus market CAGR must be at least **+1.0 pt/yr** (+100 bps/yr).
- **Pass Rule:** A factor passes Phase 1 if and only if it satisfies **both A1 and A2**.

## 7. Diagnostics

Diagnostics are reported to characterize behavior, risk, and subperiod stability; they are never pass criteria and cannot override failed gates.

- **Century-long history where data allow:** Value (BIG HiBM) and Combo (H4) evaluated back to 1927-01 over their full available data span.
- **Subperiods:** 1963–1993, 1994–2012, and 2013–end (inclusive).
- **Post-2013 net CAGR difference:** Reported plainly for all candidates. Robust profitability and investment were formally published circa 2013–2015 (e.g., Novy-Marx 2013; Fama and French 2015). Therefore, the post-1994 window includes significant pre-publication / in-sample history for OP and INV; the post-2013 subperiod serves as their true out-of-sample modern check.
- **Stress-cost evaluation:** Full-window NW $t$, $p$, and post-1994 net CAGR difference evaluated under the stress cost haircuts (2.0% for H1–H3; 3.0% for H4).
- **CAPM regression:** Annualized arithmetic alpha, alpha $t$-statistic, and market beta. Following the momentum study correction, alpha is estimated by regressing gross portfolio excess returns over the risk-free rate on market excess returns over the risk-free rate ($R_{i,t} - R_{f,t} = \alpha + \beta (R_{m,t} - R_{f,t}) + \epsilon_t$).
- **Worst 12-month relative return:** Compounded rolling 12-month winners-to-market wealth ratio $\prod (1 + R_{p}) / \prod (1 + R_{m}) - 1$.
- **Relative drawdown and recovery:** Maximum peak-to-trough decline in net relative wealth $\prod (1 + R_p - c/12) / \prod (1 + R_m)$ from initial value 1.0, peak and trough dates, and calendar months to recovery (or null if unrecovered).
- **Small-cap analogues:** SMALL HiOP, SMALL HiBM, and SMALL LoINV reported as diagnostics only, strictly outside the confirmatory Holm family, because we have no point-in-time small-cap universe or execution capability to trade them.

## 8. What happens next

| Phase 1 Result | Next Step |
|---|---|
| None pass | Factor screen closed NO-GO, no paid data. Report the closure plainly and propose the next research step. |
| One or more pass | Only the single passing factor with the smallest $p$-value advances to a separate Phase 2 brief (paid 2000–2020 survivorship-free data, exact S&P 500 rules, after-tax vs SPY ≥ +1.0 pt/yr = Part B). |

**Authority:** Passing Phase 1 authorizes further research only. It is not an authorization for deployment or real capital.

## 9. Data-quality stops

Computation must stop immediately, and no verdict may be rendered, if any of the following occur:
- SHA-256 checksum mismatch against `MANIFEST.json`.
- Missing calendar months within the registered analysis window (1963-07 through common end month).
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
5. **H4 cost haircut of 1.5% base / 3.0% stress:** Reflects the blended turnover of an annual/semi-annual value rebalance combined with a higher-turnover 12-2 momentum leg.
6. **Only the strongest passer advances:** If multiple factors clear A1 and A2, only the factor with the lowest $p$-value advances to Phase 2 paid data testing, preventing multiple testing leakage in subsequent phases.
