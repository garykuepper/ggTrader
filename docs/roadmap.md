# Project Roadmap

This document outlines the goals, ongoing work, and research direction for ggTrader. 

- For a history of project changes, see the [Changelog](changelog.md).
- For a full engineering overview, see the [Project Snapshot (June 2026)](project_snapshot_2026-06-23.md).
- For how the codebase works, see the [Architecture Guide](architecture.md).

---

## Technical Terms & Jargon Buster

If you are new to the terminology, here is a quick guide to terms used in this roadmap:
- **Indicator / Voter**: A mathematical rule used to analyze charts (like price averages or volume surges) that "votes" on whether to buy or sell a stock.
- **Ensemble (Voting)**: Combining multiple indicators to vote on trades. A "5-voter" ensemble means 5 indicators must vote.
- **Position Sizing / Exposure**: The percentage of your total account value allocated to a single trade (e.g. 3% per trade).
- **In-Sample (IS) / Training**: Testing a strategy on past data to find the best settings.
- **Out-of-Sample (OOS) / Testing**: Testing the strategy on new data that it has never seen before to prove it works in the "real world."
- **Overfitting**: A strategy that is too perfectly tuned to past data, causing it to perform well in training but fail in real testing.
- **Sharpe Ratio**: A risk-adjusted return score. It answers: "Was the profit worth the volatility/risk?" Higher is better; above 1.0 is considered strong.
- **Drawdown (DD)**: The maximum peak-to-trough percentage drop in account value (e.g., losing 11% before recovering).
- **Walk-Forward Efficiency (WFE)**: A metric comparing performance during testing (OOS) versus training (IS). WFE near or above 1.0 indicates that training performance carried over well to real-world testing.
- **Safety Gates**: Automatic checks that reject trading rules if they appear too risky, unstable, or overfitted.
- **Paper Trading**: Virtual trading with live prices and fake money.

---

## Roadmap at a Glance

This table shows the current status of the main project components.

| Component | Status | Description (Plain English) |
|---|---|---|
| **Simulation Lab Shipped** | ✅ Done | Built the core research harness (June 15, 2026) containing 10 strategies, parameter sweeps, and rolling test windows. |
| **5-Indicator Voting Default** | ✅ Done | Deployed the 5-indicator voting system (`bb+rsi+ema+macd+vbb`) as the default. It uses a majority vote of 5 indicators and filters out a harmful weekly indicator (MTF). |
| **3% Trade Size (Position Sizing)** | ✅ Done | Increased trade size per signal from 2% to 3% of cash. This utilizes idle cash and allows us to beat the market index (SPY) on a risk-adjusted basis. |
| **Gate Safety Adjustment** | ✅ Done | Modified our safety gates to prevent them from rejecting strategy settings that are consistently profitable. This improved overall test outcomes. |
| **Stable Settings Selection** | ✅ Done | Made the system select parameters that have a history of working consistently over many past months rather than just the most recent month. |
| **Machine Learning Filter** | ❌ Rejected | The LightGBM entry filter is off and stays off. A June-28 bake-off falsified it three ways: the win-probability model is anti-predictive, an expected-value redesign is *worse*, and the only robust-looking axis (volatility) is a 2020 artifact that adds nothing out-of-sample. These 10 daily features hold no stable entry-selection signal. |
| **Live Paper Trading** | 🟢 Live | Running our 5-indicator strategy with 3% trade size on virtual money with automated safety limits. Logs only real, completed fills to avoid accounting errors. (June 27: verified live fills reconcile exactly against the broker; redeployed the trader so the honest-fill-logging code is actually running; moved the daily run 30 min earlier so orders fill before the close instead of queuing overnight.) |
| **Next Steps** | 🔵 Next | Monitor virtual trading for 5–10 days → fund a live $1,000 account → go live. Start research on blending S&P 500 and MidCap 400 stocks. |
| **Kelly-Criterion Sizing** | ❌ Rejected | Sizing trades by a pooled, causal Kelly fraction (instead of flat 3%) was tested July 6 in honest walk-forward across 3 multipliers × 17 SP500 folds. OOS Sharpe 0.98 fell short of the 1.12 baseline, drawdown widened to -17.0% (vs -11%), and no multiplier won a majority of folds (best was 7/17). Position sizing is now closed as a lever; remaining directions are weighted voting (already closed) and genuinely orthogonal ones (new asset class). |
| **Future Research** | 🧪 Research | Eight distinct equity-side hypotheses now tested and rejected in honest walk-forward on this daily-bar SP500 universe (ML gate, exit rules, weighted voting, Kelly sizing, cross/dual momentum, overnight-gap reversion, idiosyncratic volatility). This isn't bad luck — it's evidence the daily-bar equity research book is close to exhausted for this system. New equity-family variants (sector-neutral reversion, calendar effects) need a stronger prior before funding more research time. The parked crypto-carry sleeve remains the clearest structurally-different lever (gated on >$10k capital + funding-data backfill). |
| **3-Sleeve Blend (leverage-realistic)** | 🟡 Conditional GO | July 12: capped at 1.0x leverage (what the live account can actually use), the SP500+MidCap400+Nasdaq100 blend hits Sharpe 1.14 / MaxDD −5.39% — slightly beats the 1.12 core Sharpe and roughly halves drawdown, at the cost of lower CAGR (9.93% vs 16.3%). Promising, but the SP500-core baseline itself needs re-measuring under the same point-in-time universe construction (`eligible_at`) before this is apples-to-apples. See §3. |

> **Status Legend:** ✅ Done · 🟢 Live · 🔵 Next · 🧪 Research · ❌ Rejected · ⏸ Deferred

---

## 1. Our "North Star" Goal

Our primary objective is to build a **flexible multi-strategy research lab** that performs honest, realistic walk-forward testing. We will only invest real capital in trading systems that survive strict out-of-sample tests. 

**Current Thesis (June 2026):** Combining multiple trading indicators (Bollinger Bands, RSI, EMA, MACD, and Volume Bollinger Bands) works because their failures are independent. When one indicator is wrong, the others usually don't vote, preventing bad trades. We closed our performance gap against the S&P 500 index not by trying to find a "magic" new signal, but by **adjusting our trade sizes** to 3% so our money doesn't sit idle in cash. This is the core lesson: our edge comes from **trade sizing and the structure of the ensemble itself, not from second-guessing individual trades** — both entry-side machine-learning filtering and exit-rule changes (take-profit / time-stop) were tested exhaustively in honest walk-forward and **rejected** (June 28). The current RSI-based exit and flat 3% sizing are, as far as we can measure, already the right choices. The nearest goal is going live with this validated config. Weighted voting was the final equity lever and was **tested & rejected (June 28)** — IC-weighting the voters lowered risk-adjusted return (Sharpe 1.01 < 1.12), closing the equity selection book. **State of the research program (July 7):** two more independently-motivated hypotheses — overnight-gap reversion (a session-structure anomaly) and cross-sectional idiosyncratic volatility (a genuinely different, defensive-premium risk factor, not another price-derivative signal) — were built, gated, and honestly WFO'd; both closed NO-GO against the 1.12 baseline. That brings the count to eight distinct equity-side hypotheses tested and rejected. The consistent shape across all eight — plausible economic premise, correctly caught by the NDH/DSR gates once tested at full scale, not a bug in the gates — is itself the finding: **daily-bar, large-cap equity research is close to arbitraged out for this system.** Further equity-family proposals (sector-neutral reversion, calendar effects) should clear a higher bar before consuming research time; the two directions with a genuinely different premise are a new asset class (the parked delta-neutral crypto-carry sleeve, gated on >$10k capital + a funding-data backfill) and, if its own correlation-vs-core study supports it, idio_vol as a blended diversification sleeve rather than a standalone strategy.

---

## 2. In-Flight Tasks (~4 Weeks)

### Phase 1: Core System Improvements (Completed)
These are changes made directly to the trading logic to optimize performance.

- **Separate Buying & Selling Rules** (`ensemble.py`): We decoupled buying and selling. The RSI indicator can now exit a trade independently, rather than requiring a majority vote from all indicators to sell.
- **True Volatility (ATR) Fix** (`feature_gate.py`): Fixed our volatility indicator to use high and low prices rather than just close prices, giving the machine learning model better data.
- **Realistic Testing Check** (`wfo.py`): Verified our walk-forward test pipeline to make sure future data wasn't leaking into the past.
- **Safety Gate Fix** (`gates.py`): Fixed a math bug where our safety gates were over-rejecting profitable parameters.
- **3% Position Sizing** (`data.py`): Increased standard trade size from 2% to 3%, boosting returns and beating the S&P 500.
- **Stable Parameters** (`wfo.py`): Programmed the live trader to choose settings with a track record of stability across past folds.

### Phase 2: Voting & Filter Upgrades (Up Next)
- **~~Exit Strategy Sweeps~~** (`ensemble.py`) — ❌ **Tested & Rejected (June 28).** Swept take-profit and time-stop ("max-hold") exits, additive and as a full replacement, against the current RSI exit in honest SP500 walk-forward. The tournament recommended the **unmodified baseline**; no exit variant earned a stable place (the winning exit param flipped fold-to-fold — fitting noise), and the replacement arm (drop the indicator exit, rely on a fixed take-profit) produced a −39.7% drawdown. The current RSI exit stands. Mechanism is built and tested (kept for future exit research), but no live change.
- **~~Weighted Indicator Voting~~** (`ensemble_ic.py`) — ❌ **Tested & Rejected (June 28).** Built `ensemble_ic`: weights each voter by its trailing cross-sectional Spearman Information Coefficient (how well that indicator ranked next-3-day returns), recomputed quarterly — leak-safe by construction, 291 tests, whole-branch review clean. Honest SP500 walk-forward (same harness/window as the 1.12 baseline; SPY 0.58 confirms apples-to-apples) gave OOS **Sharpe 1.01 < 1.12**, CAGR 19.6% (higher) but **MaxDD −17.3% vs −11%** (worse), gates **14/17 vs 16/17**. The winning weight config was picked in only **3/17 folds** (flips fold-to-fold — fitting noise, same tell as the exit sweep). IC weighting buys raw return by concentrating into higher-conviction entries at the cost of drawdown, so risk-adjusted return *falls*. Kept as a registered **non-default** research strategy; **not deployed** (live stays on the equal-weight `ensemble` baseline). This was the final equity-book lever — the **equity selection book is now closed**.
- **~~Dynamic Machine Learning Filter~~** (`feature_gate.py`) — ❌ **Rejected (June 28).** A "blocks more trades when volatile, loosens when calm" filter is exactly the volatility filter the June-28 bake-off proved is a 2020 artifact with no out-of-sample value. Entry-level ML/feature gating is closed; do not re-attempt on these features.

### Phase 3: Transitioning from Virtual to Live Trading
- [x] Alpaca paper trading adapter, signal generator, risk checks, and Telegram alerts set up.
- [x] Machine learning filters and risk safety guardrails deployed.
- [x] Virtual trading accounts set up and baseline reset.
- [ ] Monitor virtual trading fills for 5–10 consecutive days.
- [ ] Fund the live Alpaca account with $1,000.
- [ ] Swap API keys from paper to live.
- [ ] Confirm order sizes and fills match expected behavior.

### Phase 2b: Completed & Rejected Ideas
- **MACD Divergence Signal** (✅ Shipped): Built and verified. It improved our Sharpe ratio from 0.68 to 0.89.
- **Volume Bollinger Band Reversion** (✅ Shipped): Built and verified. It is now part of our default 5-indicator strategy.
- **Multi-Timeframe (MTF) Reversion** (❌ Rejected): Built and tested, but it degraded performance (Sharpe dropped from 0.68 to 0.49). It was removed.
- **6-Indicator Ensemble** (❌ Rejected): We rejected a 6-indicator model because it included the harmful MTF signal. We kept the 5-indicator model instead.
- **Conviction Position Sizing** (✅ Shipped but unused): Sizes trades based on indicator strength. It reduced risk but did not generate extra profit, so it remains inactive.
- **Trailing Stops** (❌ Rejected): Testing proved that trailing stops exit trades during normal price fluctuations, destroying profitability.
- **Momentum Strategies** (❌ Rejected): Ranks stocks by momentum. Testing showed negative performance, as large-cap momentum is highly competitive and well-arbitraged.

---

## 3. Future Research Directions

These are open research paths that are not bound to a specific deadline, ordered by expected payoff:

* **~~Exit-Rule Optimization~~** (❌ Rejected, June 28): Swept take-profit and time-based ("max-hold") exits against the current RSI-cross exit in honest walk-forward (31 exit combos × 17 SP500 folds). **Result: no improvement.** The WFO's recommended live config was the unmodified baseline (RSI exit, no take-profit, no time-stop); exit-param winners flipped fold-to-fold (noise, correctly caught by the stability/deflated-Sharpe gates); and the replacement arm (no indicator exit) drew down −39.7%. Combined with the entry-gate rejection, this closes the "second-guess individual trades" family of levers — our edge lives in sizing and ensemble structure. *(Confound noted: the run pinned signal params to isolate exits, so its aggregate isn't comparable to the 1.12 full-grid baseline; the valid within-grid comparison still favors the no-exit baseline.)*
* **~~Dynamic Trade Sizing (Kelly Criterion)~~** (❌ Rejected, July 6): Sized trades by a pooled, causal Kelly fraction estimated from the strategy's own closed-trade history (instead of flat 3%), swept across quarter/half/full-Kelly multipliers in honest walk-forward (3 combos × 17 SP500 folds). **Result: no improvement.** OOS Sharpe 0.98 fell short of the 1.12 baseline, drawdown widened to −17.0% (vs −11%), and no multiplier won a majority of folds (best was 7/17 — a fold-count fluke on the same order as the earlier IC-weighted-voting rejection). Closes the position-sizing lever; the flat 3% baseline remains the right choice.
* **Diversification Ranking** (⚪ Planned): If the system generates more buy signals than we have cash for, rank them by indicator strength and sector diversity to avoid buying too many stocks in the same industry.
* **~~Overnight Gap Reversion~~** (❌ Rejected, July 7): New orthogonal-to-price-momentum candidate — fades extreme overnight gap-down Z-scores (open vs. prior close), filled at the same bar's close per the existing voter convention. Full honest WFO (604-symbol SP500 universe, 17 folds, 27 param combos): **OOS Sharpe -0.01 vs. SPY 0.58 baseline** (CAGR -0.2% vs 13.0%, MaxDD -9.1% vs -22.1%), aggregate WFE sitting exactly at the 0.50 floor, and the winning param combo selected in only 3/17 folds — the same fold-instability signature (noise, not edge) that sank IC-weighted voting and Kelly sizing. Most folds ran on the regime-halt/anchor fallback rather than the strategy's own signal. Closes candidate A from the July-6 strategy-recommendations review; code (`overnight_gap` strategy, registry, tests) kept as a non-default, gate-validated-NO-GO reference implementation.
* **Cross-Sectional Idiosyncratic Volatility** (🧪 Diversification-only, July 7): Candidate B from the July-6 strategy review — a weight-based, long-only sleeve equal-weighting the eligible universe's lowest-idiosyncratic-variance quintile each month (residual variance vs. the universe's own equal-weighted return, the same market-factor convention `compute_vol_scalar` already uses). Required first generalizing the gated `--wfo` harness to support weight-based strategies at all (`wfo.py`'s fold-sweep now dispatches on `target_kind`, validated against the existing `xs_momentum` strategy before this one was tested). Full honest WFO (SP500, 17 folds, 6 combos): **OOS Sharpe 0.57 vs. SPY 0.58** (CAGR 10.2% vs 13.0%), aggregate WFE 0.51 (barely clears the 0.50 floor), winner selected in only 4/17 folds. **Rejected as a standalone strategy** — doesn't approach the deployed core's Sharpe. Notably different shape than the other rejections though: MaxDD -17.2% vs. SPY's -22.1%, a genuinely shallower drawdown consistent with the defensive-premium premise. **Follow-up correlation study (July 7):** measured the actual return correlation between `idio_vol`'s OOS stream and the deployed 5-voter ensemble core's OOS stream (1,863 overlapping trading days, same July-7 WFO methodology, both against the current SP500 universe) — **0.447**, moderate: lower than SP500-vs-MidCap's 0.70, a bit above MidCap-vs-Nasdaq's 0.35. Combined with its shallower drawdown and near-SPY Sharpe, this is a real (if modest) diversification case — not dead, but not a priority either. **Verdict: kept as a documented, non-default diversification-sleeve candidate for a future `--blend` study, not pursued further as a standalone allocation.**
* **~~Macro Machine Learning Features~~** (❌ Rejected, June 28): Feeding broader market metrics (VIX, rates) into a machine-learning entry filter. Closed alongside the rest of entry-side ML — the June-28 bake-off showed these daily features carry no stable out-of-sample entry-selection signal, and a volatility-based filter is a 2020 artifact. Macro data may still have value for *sizing/regime* (see Exposure Scaling), not for entry gating.
* **Pairs Trading (Statistical Arbitrage)** (⚪ Planned): Identify pairs of stocks that historically move together, buying one and selling the other when they drift apart, expecting them to converge.
* **Large + Mid-Cap Portfolio Blend** (🧪 Researching): Tested a 50/50 and 70/30 blend of S&P 500 (Large Cap) and MidCap 400 (Mid Cap) stocks. Mid-caps show promising reversion characteristics (beats MDY: 15.0% CAGR / 1.08 Sharpe after survivorship haircut vs 9.1% / 0.40). **Gate investigation (June 27) settled the "anchor-driven" question:** the safety gates are *not* miscalibrated — the Deflated Sharpe gate passes every fold, and the rejections come from the Neighborhood Density gate correctly flagging that mid-cap optimal settings sit on genuinely noisier, less-robust plateaus. We improved the circuit-breaker's recovery rule (now "2 of the last 3 clean windows" instead of "2 consecutive," which cut defensive-anchor folds from 15 to 10), but the residual caution is real, not a bug. **Decision:** rather than weakening overfit protection to force mid-caps through, deploy mid-cap only as a diversification sleeve *behind* the deploy-clean large-cap core — judged on blend diversification, not as a standalone gated strategy.

**Diversification measurement (June 27).** We measured the actual return correlations between our reversion sleeves (the number that decides whether more universes are worth it). Result overturned the "the easy diversification is used up" assumption: S&P 500 vs MidCap 400 is only **0.70** (moderate — real benefit), and **MidCap 400 vs Nasdaq-100 is just 0.35** (genuinely independent). Nasdaq-100 is also the single best standalone sleeve (Sharpe 1.24, shallow drawdown). Blending the low-correlation MidCap+Nasdaq pair gives the best risk-adjusted result of anything tested (Sharpe **1.39**, drawdown **−4.0%**, volatility 4.2% vs the S&P 500's 6.8%). Because that blend is so low-volatility, it can be *scaled up* (our position-size lever) to roughly match the S&P 500's growth while keeping a shallower drawdown — the diversification and exposure-scaling levers compound. Russell 2000 could not be included — its data is only ~29% loaded (a backfill prerequisite). *Caveat: these blend figures are idealized (no safety-gate halts); absolute numbers will come down when run through the live gates, but the correlation-driven risk reduction holds.*

**3-sleeve vs. 4-sleeve blend study (July 8, ❌ Rejected — idio_vol adds nothing to the blend).** With `--blend` now able to include weight-based strategies (`universe_fn` threaded into `run_blend()`, commit `39d18dd`), ran the first honest, gate-validated comparison of the SP500+MidCap400+Nasdaq100 core blend with and without `idio_vol@sp500` layered in as a fourth sleeve (both via the inverse-vol/target-vol overlay in `blend_curves`, `--eval-start 2021-01-31 --eval-end 2026-04-30`). **3-sleeve baseline:** CAGR 13.89%, Sharpe **1.03**, MaxDD −10.59%. **4-sleeve (+idio_vol):** CAGR 13.97%, Sharpe **1.01**, MaxDD −12.08%. Adding `idio_vol` moved CAGR by +0.08pp (noise) while *worsening* both Sharpe (−0.02) and drawdown (−1.49pp) — the opposite of a diversification benefit, despite `idio_vol`'s standalone 0.447 core-correlation and shallower solo drawdown (see above). Likely explanation: `idio_vol`'s own Sharpe (0.57) is low enough that its inverse-vol weight in the blend just dilutes the three higher-Sharpe sleeves without buying enough independent risk reduction to compensate. **Verdict: do not add `idio_vol` to the deployed blend mix.** This closes the idio_vol research arc opened July 7 — no further follow-up planned unless a future candidate specifically targets low-Sharpe/low-correlation trade-offs differently.

**Leverage-realistic 3-sleeve blend verdict (July 12, 🟡 Conditional GO — supersedes the June-27 and July-8 blend numbers).** Both prior blend figures were not directly comparable: June 27's "gate-honest 1.05" came from `multi_sleeve_research.py`/`portfolio_blend.py`, retired June 29 when `blend.py` replaced them; July 8's "idealized 1.03" came from the current tool but at its `--max-leverage 2.0` default — leverage the live paper account, which trades unlevered flat-3% cash sizing, cannot actually take. Re-ran the identical 3-sleeve mix (`ensemble@sp500,ensemble@midcap400,ensemble@nasdaq100`, same `--eval-start 2021-01-31 --eval-end 2026-04-30` window) at `--max-leverage 1.0`, and pulled the persisted leverage diagnostics for both: **2.0x-cap run:** Sharpe 1.03, CAGR 13.89%, MaxDD −10.59%, avg realized leverage **1.63x** (most days actually levered up near the cap). **1.0x-cap (deployable) run:** Sharpe **1.14**, CAGR 9.93%, MaxDD **−5.39%**, avg leverage 0.97x (essentially unlevered, as expected). Capped at the leverage the account can actually use, the blend **slightly beats the SP500-core 1.12 Sharpe baseline and roughly halves its drawdown** (−5.39% vs −11%), at the cost of substantially lower raw CAGR (9.93% vs 16.3%) — a legitimate risk/return trade rather than the fold-flip noise signature that killed every prior weighting/sizing lever. **Open caveat before treating this as a clean apples-to-apples result:** the blend's own SP500 sleeve inside this run scored Sharpe 0.97 (CAGR 13.41%, MaxDD −11.01%), short of the cited 1.12 baseline. This gap is at least partly mechanical, not a leverage effect — `blend.py`'s sleeves use `eligible_at()`, true per-fold point-in-time index membership, while the standalone `--wfo` baseline uses `equity_universe_between()`, a static union of every symbol that was ever a member across the full 2021–2026 window. The static-universe baseline can trade names in periods before/after their actual index membership; the blend's sleeve cannot. This is a known, real difference in universe construction, not a bug, but it means the 1.14-vs-1.12 comparison isn't single-methodology. **Verdict:** promising enough to keep the diversification arc open (contra June 27's closure, which predates both the leverage fix and the retired-tool ambiguity) — recommend funding a follow-up that re-measures the SP500-core baseline itself under `eligible_at()` PIT construction before any deploy decision, so the comparison is apples-to-apples on both leverage *and* universe mechanics. Not yet a basis for live capital allocation.

---

## 4. System Infrastructure

This table tracks the technical foundation that supports our research and trading.

| Feature | Status | Description (Plain English) |
|---|---|---|
| **Database Storage** | ✅ Done | Saves simulation logs to `lab_runs` and `lab_periods` tables in our TimescaleDB database. |
| **Price Data Backfill** | ✅ Done | Loaded 4 million rows of S&P 500 history into the local database so simulations run offline without downloading files. |
| **Tiingo Data Backup** | ✅ Done | Connects to Tiingo API to download data for companies that went bankrupt or got acquired, which Yahoo Finance misses. |
| **Multi-Index Support** | ✅ Done | Supports running tests on different stock lists (S&P 500, Nasdaq 100, Russell 2000) using simple flags. |
| **Parameter Sweeps** | ✅ Done | Command-line tool (`--sweep`) that automatically tests a grid of strategy settings. |
| **Walk-Forward Engine** | ✅ Done | Runs rolling tests (`--wfo`) that simulate periodic strategy adjustments. |
| **Circuit Breakers** | ✅ Done | Automatically stops simulation tests if the strategy's testing efficiency falls below safety levels. |
| **Speed Optimization** | ✅ Done | Improved simulation speed by 2.07x through code profiling, reducing runtimes. |
| **AI Assistant Database Link** | ✅ Done | Created a database link (MCP Server) that lets AI assistants query simulation metrics directly. |
| **Portfolio Blending Tool** | ✅ Done | Script to analyze performance and correlation when mixing different index universes. |
| **Database Walk-Forward Logs** | ⚪ Planned | Storing detailed rolling test steps in the database (currently kept in files). |
| **Run Comparator CLI** | ⚪ Planned | A command (`ggt compare`) to display side-by-side metric tables of two different simulation runs. |

---

## 5. Summary of Research Findings

### What Worked
* **5-Indicator Voting (BB+RSI+EMA+MACD+VolBB)**: Our best strategy config. Combining these 5 indicators beats the S&P 500 index on a risk-adjusted basis (Sharpe ratio of 1.12 vs 0.58).
* **Negatively Correlated Indicators**: Combining trend and reversion indicators cancels out false alarms.
* **Point-in-Time Universes**: Using actual historical index lists prevents our backtests from inflating returns by 1% to 3% per year.
* **Robustness Gates**: Safely filtered out overfitted parameters during testing.

### What Failed
* **Trailing Stops**: Exited trades too early during normal fluctuations, ruining reversion strategies.
* **Momentum Strategies**: Ranks stocks by momentum. Performed poorly over the last 5 years in large-cap stocks.
* **Weekly Indicators (MTF)**: Harmed overall performance because weekly signals were too slow for daily trading.
* **Conviction-Based Position Sizing**: Sizing trades based on indicator strength did not add extra profits compared to flat trade sizing.
* **ML / Feature Entry Gate (falsified three ways, June 28)**: We tried to pick *better* entries with a machine-learning filter on 10 daily features. (1) The original win-probability model is anti-predictive — the entries it blocks out-earn the ones it keeps. (2) Rebuilding it to predict expected return (magnitude) instead of direction is *worse*, not better — its scores are even more inversely correlated with realized return out-of-sample. (3) The one robust-looking axis, a volatility-expansion filter ("skip falling-knife dips"), turned out to be entirely a March-2020 effect: in 2024–2025 the high-volatility entries actually did *better*, and an honest split (set the threshold on the first half of history, apply to the second) showed zero benefit. Conclusion: these features carry no stable cross-sectional entry-selection signal. Our edge has always come from exposure/sizing and exits, never from filtering which name to buy.
* **Exit-Rule Changes — Take-Profit & Time-Stop (rejected, June 28)**: We swept fixed profit targets and time-based ("max-hold") exits against the current RSI exit, both layered on top and as a full replacement, in honest SP500 walk-forward. None improved on the baseline: the walk-forward recommended the *unmodified* config, the best exit setting changed every fold (a tell-tale sign of fitting noise, which the robustness gates correctly rejected), and dropping the indicator exit to rely on a profit target produced a −39.7% drawdown. The current RSI exit is, as far as we can measure, already the right rule. Together with the entry-gate result, this closes the entire "second-guess each trade" family — what's left to improve is sizing and the ensemble's structure, not per-trade entry/exit selection.
* **Weighted Voting — IC-Weighted Ensemble (`ensemble_ic`, rejected, June 28)**: We built a strategy that weights each indicator's vote by its trailing cross-sectional Information Coefficient (how well the indicator ranked the next 3 days' returns), recomputed quarterly and proven leak-free. In honest SP500 walk-forward it scored OOS **Sharpe 1.01 vs the 1.12 equal-weight baseline**, with a deeper drawdown (−17.3% vs −11%) and a worse gate pass rate (14/17 vs 16/17). The best weight configuration was chosen in only 3 of 17 folds — it changed fold-to-fold, the same noise signature as the exit sweep. Re-weighting the voters bought more raw return by leaning into higher-conviction entries, but the extra drawdown meant *risk-adjusted* return fell. This was the last untested equity lever; with it rejected, the **equity selection book is closed** and research turns to a different asset class. (Code kept as a non-default research strategy; not deployed.)

---

## 6. Project Timeline

* **June 15, 2026**: Shipped the core lab simulation engine and momentum strategy tests.
* **June 16, 2026**: Integrated signal-based strategies and loaded 4 million rows of S&P 500 data into the database.
* **June 17, 2026**: Shipped the parameter sweep tool for grid testing.
* **June 18, 2026**: Built Bollinger Band and RSI mean reversion strategies and the walk-forward testing framework.
* **June 19, 2026**: Tested trailing stops and proved they are harmful for reversion strategies.
* **June 20, 2026**: Verified the Voting Ensemble strategy and deployed paper trading on Alpaca.
* **June 22, 2026**: Added the LightGBM machine learning filter and risk guardrails.
* **June 24, 2026**: Ran a comprehensive voting ablation test, proving that a 5-indicator majority vote is our most robust trading signal.
* **June 25, 2026**: Raised trade sizes to 3% to utilize cash, successfully beating the S&P 500 index on a risk-adjusted basis. Shipped this config as the default.
* **June 26, 2026**: Adjusted our safety gates to prevent over-rejection of good rules and added stable settings selection to the live trader.
* **June 27, 2026**: Reconciled live paper fills against the broker (clean); found and fixed a stale deployment so the honest-fill-logging code actually runs, and moved the daily run before the close. Improved the circuit-breaker recovery rule (2-of-3 clean windows). Settled the mid-cap question: the gates are working as designed (mid-cap settings are genuinely noisier), so mid-cap will be a diversification sleeve, not a standalone strategy.
* **June 28, 2026**: Ran a gate-objective bake-off to decide the best way forward on the ML filter. Falsified entry-level ML/feature gating three ways (anti-predictive classifier; even-worse EV regressor; volatility filter is a 2020 artifact). Verified the gate is *not* live on the paper trader (disabled by default, no env enables it). Then built and ran the exit-rule sweep (take-profit + time-stop, 31 combos × 17 SP500 folds) — also rejected: the walk-forward recommends the unmodified RSI exit, exit params fit noise, and the no-indicator-exit replacement drew down −39.7%. Fixed a latent walk-forward bug (crash on any "off"/None sweep axis) found en route. Then built `ensemble_ic` (Spearman-IC-weighted voting, leak-safe, 291 tests, reviewed clean) and ran it as the final equity-book experiment — also rejected: OOS Sharpe 1.01 < 1.12 baseline, drawdown −17.3% vs −11%, winning weights chosen in only 3/17 folds (noise). Net: entry-filtering, exit-tuning, and weighted-voting levers are all now closed — the **equity selection book is closed**; next research turns to a new asset class (parked crypto-carry sleeve, gated on >$10k capital + funding-data backfill).
* **June 29, 2026**: Infrastructure, not research — made it easy to *try combinations* of strategies. (1) Collapsed the triplicated strategy registry to one source of truth (`STRATEGY_REGISTRY`); adding a strategy is now one line. (2) Promoted the portfolio-of-sleeves blend from throwaway research scripts into a first-class, persisted `ggt lab --blend "<strategy>@<universe>,..."` command (validated inverse-vol/target-vol overlay; retired `multi_sleeve_research.py` + `portfolio_blend.py`). This is the tooling the parked crypto-carry sleeve will use to blend with the equity book — equity-only diversification stays a closed NO-GO. 305 lab tests green.
* **July 6, 2026**: Ran the last untested equity-side lever — Kelly-criterion position sizing. Built `ensemble_kelly` (pooled, causal, expanding-window Kelly fraction sized off the strategy's own closed-trade history, falling back to the flat-3% baseline whenever there's no measurable edge yet, hard-capped at the live risk guard's 5% ceiling; 24 new tests, TDD, no look-ahead — the causal property was independently verified end-to-end). WFO'd 3 multipliers (0.25/0.5/1.0) × 17 SP500 folds — **rejected**: OOS Sharpe 0.98 < 1.12 baseline, drawdown widened to −17.0% vs −11%, and no multiplier won a majority of folds (best was 7/17, a fold-count fluke on the same order as `ensemble_ic`'s 3/17). Net: **position sizing is now closed** alongside entry-filtering, exit-tuning, and weighted-voting — every lever on the deployed equity book has been tried and rejected in honest walk-forward. The only remaining research direction is a genuinely orthogonal one: the parked crypto-carry sleeve (gated on >$10k capital + funding-data backfill).

---

*Back to [README.md](../README.md).*
