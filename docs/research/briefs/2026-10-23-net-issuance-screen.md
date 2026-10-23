# Brief: long-only large-size net-issuance screen for a taxable trading sleeve

**Status:** DRAFT 2026-10-23, awaiting owner approval. Nothing has been computed.

**Terms used below:**

- *Net share issues (NI)* is the change in the natural log of split-adjusted shares outstanding between fiscal year-ends, as defined by the [Ken French size × NI library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/Data_Library/tw_5_ports_me_NI.html). Negative NI identifies net repurchasers; zero NI has its own bucket.
- *BIG* is the largest NYSE market-equity quintile in this file. It is a broad US-stock sort, not an S&P 500 portfolio or a point-in-time implementation of our stock rules.
- *LoNI* is the source's lowest named NI quintile. The file also names `NegNI` and `ZeroNI` buckets outside the five named NI quintile cells; §11 flags the resulting interpretation issue.
- *Newey-West (NW) t-stat* measures mean monthly excess return relative to a standard error allowing serial correlation and changing variance. A one-sided asymptotic-normal p-value is not the probability that the idea is true.
- *CAGR* is compound annual growth rate. *pt/yr* is the percentage-point difference between two CAGRs.
- *PIT* (point-in-time) means using membership and information available at each historical decision date. *Survivorship-free* includes firms that later disappeared.
- *Part A* screens the idea on long academic portfolio history. Under the [approved forward-paper standard](2026-10-23-forward-paper-standard.md), *Part B* is B0 (a free, kill-only replay) followed by B1 (a free forward shadow study), subject to a separate approved candidate brief.

## 1. The question, and the decision it feeds

- **Question.** Does the biggest-size-quintile, lowest-net-issuance-quintile French portfolio beat the French market after a fixed cost haircut, with both long-history statistical significance and at least +1.0 pt/yr net CAGR advantage since 1994?
- **Decision fed.** Whether this single candidate may proceed to a separately registered B0 and then B1 under the [forward-paper standard](2026-10-23-forward-paper-standard.md). A Part A pass authorizes research only, not deployment or capital.
- **Owner context.** The account is taxable only. The owner wants a stock-picking path with greater gains and accepts larger swings. Existing production paper trading and its freeze continue under their own rules.

## 2. Why: prior evidence and data limits

- **Skeptical prior.** Published replication work finds post-publication decay and weak replication across stock anomalies: [McLean and Pontiff (2016, *Journal of Finance*)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12365) and [Hou, Xue, and Zhang (2020, *Review of Financial Studies*)](https://academic.oup.com/rfs/article-abstract/33/5/2019/5236964). These studies mostly examine predictive spreads, not this particular long-only BIG cell versus the market. **Inference:** long-only large-cap anomaly legs rarely clear this higher market-beating hurdle after publication; a null is the expected outcome. This is a prior, not a computed result for NI.
- **Publication diagnostic.** [Pontiff and Woodgate (2008, *Journal of Finance*)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2008.01335.x) documented the share-issuance relation. A post-2008 diagnostic will expose whether the registered leg persists after that publication; it cannot rescue a failed gate.
- **External history, limited inference.** French's frozen, value-weighted portfolios provide long, survivorship-free academic return history. The anomaly's definition was developed using historical returns, so the full-window A1 and much of A2 are retrospective, not an independent temporal holdout. French BIG is not the S&P 500, and these files do not establish a tradable, after-tax stock strategy. The existing [`french_data.py`](../../../src/ggTrader/lab/french_data.py) and [`factor_screen.py`](../../../src/ggTrader/lab/factor_screen.py) supply the freeze/parse and calculation patterns for later implementation.

## 3. Data (Ken French Data Library)

Use monthly **value-weighted** returns. Before any return calculation, download and freeze the two new raw archives with download time, exact URL, and SHA-256 in `data/external/french/MANIFEST.json`; verify the checksums on every later read. The already-frozen `F-F_Research_Data_Factors` supplies market total return `Mkt-RF + RF` and `RF` for CAPM. The two archives were read in `/tmp` on 2026-10-23 **only for section titles, column headers, and first/last month labels**; no returns from them were inspected or computed.

| Source | Exact value-weighted monthly section | Registered labels | First–last month in section |
|---|---|---|---|
| [25 size × NI portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_ME_NI_5x5_CSV.zip) | `Average Value Weighted Returns -- Monthly` | H1: `BIG NegNI` (large-cap net repurchasers); diagnostics: `BIG ZeroNI` and the five BIG issuer quintiles `BIG LoNI`, `ME5 NI2`, `ME5 NI3`, `ME5 NI4`, `BIG HiNI` | 1963-07–2026-08 |
| [Portfolios formed on NI](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_NI_CSV.zip) | `Value Weighted Returns -- Monthly` | Net-repurchaser CSV column: `< 0 ` (trailing space in raw header; `< 0` after stripping) | 1963-07–2026-08 |

The 5 × 5 archive contains **seven** named NI buckets per size quintile: negative, zero, and five named quintile cells. The diagnostic univariate `< 0 ` portfolio includes small caps; it is not a BIG-only test. Freeze the current archives only after approval, then truncate all analysis to the registered **2026-04** endpoint despite the archives' later month labels. Verify that the already-frozen market file covers every required month through that endpoint.

## 4. Hypothesis: a family of ONE (BIG only)

**H1:** the `BIG NegNI` value-weighted portfolio (large-cap net repurchasers) beats the French market after the base haircut. This is the sole confirmatory hypothesis and the only portfolio whose A1/A2 values determine the Part A verdict. `BIG NegNI`, `BIG ZeroNI`, the other four BIG quintile cells, and the univariate `< 0 ` portfolio cannot be substituted for H1 after seeing returns. No cross-study error control is claimed for choosing this new candidate after earlier screens.

## 5. Series and cost haircuts

For monthly decimal returns, define the registered excess series

$$e_t = R_{\text{BIG NegNI},t} - R_{\text{mkt},t} - c/12.$$

The portfolio is formed at the source's annual June rebalance. **Base:** $c=1.0\%$/yr. **Stress:** $c=2.0\%$/yr. Subtract $c/12$ each month before compounding, as in the factor brief. These are policy assumptions for annual-rebalance friction, not measured turnover, execution cost, or tax. The market is gross of those costs and tax. A pass establishes robustness only to the stated base haircut.

## 6. Tests: significance (A1) and since-1994 economic persistence (A2)

- **A1 window:** first month in the 5 × 5 monthly value-weighted section, **1963-07**, through **2026-04** inclusive. No later source month enters any test. Use the base-cost $e_t$ above; test $H_0:E[e_t]\leq0$ against $E[e_t]>0$. Compute the mean divided by its Bartlett NW standard error with **12 monthly lags**, using the existing `nw_t` convention. The one-sided p-value is $\Phi(-t)$. **A1 passes only if the unrounded NW $t\geq3.0$ and one-sided $p\leq0.00135$.** There is one test, so **no Holm correction**.
- **A2 window:** **1994-01 → 2026-04**, exactly **388 monthly observations**, at the base haircut. Use the factor brief's executable decimal-return formula:

  $$\Delta\mathrm{pp}=100\left[\left(\prod_{t=1}^{388}(1+R_{p,t}-c/12)\right)^{12/388}-\left(\prod_{t=1}^{388}(1+R_{m,t})\right)^{12/388}\right].$$

  **A2 passes if the unrounded CAGR difference is $\geq+1.0$ pt/yr.** This is net of the haircut only, not after tax. The market is the frozen French total-return market, not SPY.
- **Part A pass rule:** H1 passes **if and only if A1 and A2 both pass**. No diagnostic can change either gate. The long history and since-1994 window are not uniformly post-publication; a pass is not proof of an independent modern edge.

## 7. Diagnostics

Report these for interpretation, **never as pass criteria**. A weak post-publication or stress result is reported plainly, without selecting a different window or NI cell.

- **Post-2008:** net CAGR difference versus market from **2009-01 → 2026-04**, after the 2008 Pontiff–Woodgate publication.
- **Subperiods:** 1963-07 → 1993-12, 1994-01 → 2008-12, and 2009-01 → 2026-04; report base-cost NW $t$, one-sided $p$, and net CAGR difference in each.
- **Stress cost:** full-window NW $t$ and $p$, plus 1994-01 → 2026-04 net CAGR difference at 2.0%/yr.
- **CAPM:** annualized arithmetic alpha and market beta from gross $R_p-R_f$ regressed on $R_m-R_f$ over the A1 window. No alpha $t$ is reported.
- **Relative risk:** worst compounded rolling 12-month gross portfolio-to-market wealth ratio minus one; maximum peak-to-trough decline in base-cost relative wealth $\prod(1+R_p-c/12)/\prod(1+R_m)$ from initial value 1.0, peak/trough dates, and calendar months to recovery (null if unrecovered).
- **Univariate repurchasers:** the value-weighted `< 0 ` column from the second archive at **the same 1.0%/yr base and 2.0%/yr stress haircuts**. It **includes small caps** and cannot replace the BIG-only H1.
- **Issuance shape:** report `BIG ZeroNI` and the five BIG issuer quintiles (`BIG LoNI`, `ME5 NI2`, `ME5 NI3`, `ME5 NI4`, `BIG HiNI`) beside `BIG NegNI` to show monotonicity or its absence. For each of these six cells and the univariate repurchaser column, report full-window base and stress NW $t$/$p$, plus base and stress net CAGR differences for 1994-01 → 2026-04 and 2009-01 → 2026-04, whether favorable or unfavorable. Never merge buckets.

## 8. What happens next

| Part A result | Next step |
|---|---|
| H1 fails A1 or A2 | Close this net-issuance screen **NO-GO**. Publish the failed gate(s) and diagnostics, use **free data only**, and propose the next research step. |
| H1 passes both | Write a separate fixed-rules candidate brief, then **B0** (free PIT 2021-01-31 → 2026-04-30 kill-only after-tax replay) and, only if B0 does not kill and evidence is complete, **B1** (at least 18 months of free forward shadow evidence) under the [approved standard](2026-10-23-forward-paper-standard.md). **No paid data ever.** |

Passing Part A permits further research only. It does not authorize a production change, a broker order, or real capital. B0/B1 require their own frozen implementation, cost, tax, and evidence rules before any run.

## 9. Data-quality stops

Stop before any verdict if an archive is absent from `MANIFEST.json`, a SHA-256 differs, a required section or exact header is missing or changed, or the market factor is not verified. Stop on duplicate or missing calendar months in either registered window, non-finite required returns, or any non-positive monthly portfolio-after-cost or market wealth factor, including diagnostic inputs. No dropping, filling, interpolation, hand-patching, or shortened window is allowed. Report the defect and leave the verdict unresolved until a new approved plan resolves it.

## 10. Scope and boundaries

- **This task:** this brief only. No return calculation, result JSON, source archive freeze, test run, or production change before owner approval.
- **Later approved implementation:** new code only under `src/ggTrader/lab/` and `scripts/`, reusing `factor_screen` and `french_data`; raw JSON results to `docs/research/_net_issuance_screen_results.json` next to the report.
- **Untouchable paths:** never touch `src/ggTrader/paper/`, `src/ggTrader/data/live/`, `.env`, or Docker files. No paid dataset or subscription at any phase. No parameter search, post-result hypothesis swap, or retrofit of the haircut or window.

## 11. Calls made by the senior quant (owner to check)

1. **Single-hypothesis family:** H1 alone faces A1/A2. No Holm is needed within this family, but the previous factor screen and momentum NO-GOs remain closed; serial candidate selection is not covered by this one-test error rate.
2. **Low volatility excluded:** it is primarily a risk-reduction effect, unlikely to meet A2's absolute return bar and contrary to the owner's greater-gains goal.
3. **Industry momentum excluded:** the momentum family is closed NO-GO, and industry momentum would require constructing a new signal rather than selecting a ready long-only French cell.
4. **Other literature candidates excluded:** accruals, E/P, CF/P, D/P, long-term and short-term reversal, and BAB are outside this brief because the reviewed evidence points to short-leg dependence, overlap with the closed value family, high turnover, or leverage. Their exclusion is a scope choice, not a new measured NO-GO for each.
5. **Source-label choice, resolved by the senior quant:** H1 is `BIG NegNI`. French ranks positive net issuers into quintiles and keeps net repurchasers (`NegNI`) and zero issuers (`ZeroNI`) in separate buckets, so `BIG LoNI` firms are still net issuers. The published long-side idea is that firms shrinking their share count outperform, which is `NegNI`. Both buckets have complete data for all 754 months, 1963-07 → 2026-04; this was checked from the missing-value codes only, with no returns viewed. `BIG LoNI` is a diagnostic. No post-result substitution.
6. **Forward-standard text conflict, unresolved in its source:** its status and [`docs/next_steps.md`](../../next_steps.md) say owner-approved on 2026-10-23, but its §§1, 8, and 9 still call the replacement a draft or proposed. This brief follows the approved B0/B1 path as instructed and does not silently edit that source or treat its stale draft wording as authority for paid data.
7. **B1 stop-boundary conflict, unresolved in its source:** the forward standard's §5 writes the quarterly breach condition as relative wealth **`< -0.15`**, which excludes equality, while the same section and §8 resolution R5 say **15% or more behind** breaches, which includes equality. This does not affect Part A. A later B1 candidate brief must reconcile the operator before its rules are frozen; no B1 verdict should rest on that ambiguity.
