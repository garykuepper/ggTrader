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
| **3-Sleeve Blend (leverage-realistic)** | 🟢 Live | July 13: capped at 1.0x leverage (what the live account can actually use), the SP500+MidCap400+Nasdaq100 blend hits Sharpe 1.14 / MaxDD −5.39% vs. the SP500-core's own matched-window number of Sharpe 0.97 / MaxDD −11.0% — a full 0.17 Sharpe gain and roughly half the drawdown, for a CAGR cost of 9.93% vs 13.4%. (The often-cited "1.12" baseline turned out to be from a different eval window, not this one.) July 14: wired into the live `PaperTrader` (sleeve-aware sizing, margin pre-flight check, `--live` flag) — this blend, not the single-sleeve core, is now the deployed config. See §3. |
| **Leveraged/Inverse ETF Rotation & Trend Following** | ❌ Rejected | July 16: two independent attempts to time 2x/3x leveraged ETFs (UPRO/TQQQ/TNA and inverse pairs) both closed NO-GO. Breadth-driven long/inverse/cash rotation: Sharpe −0.14 to −0.37 vs SPY, drawdowns −80% to −89%, defensive-halt active almost every fold. A simpler long-only trend filter + volatility-targeting overlay was structurally healthier (no persistent halt) but still lost to just buying and holding the same ETF in every universe tested. See §3 and `docs/research/2026-07-16-leveraged-index-rotation-nogo.md` / `docs/research/2026-07-16-leveraged-trend-following-nogo.md`. |

> **Status Legend:** ✅ Done · 🟢 Live · 🔵 Next · 🧪 Research · ❌ Rejected · ⏸ Deferred

---

## 1. Our "North Star" Goal

Our primary objective is to build a **flexible multi-strategy research lab** that performs honest, realistic walk-forward testing. We will only invest real capital in trading systems that survive strict out-of-sample tests. 

**Current Thesis (June 2026):** Combining multiple trading indicators (Bollinger Bands, RSI, EMA, MACD, and Volume Bollinger Bands) works because their failures are independent. When one indicator is wrong, the others usually don't vote, preventing bad trades. We closed our performance gap against the S&P 500 index not by trying to find a "magic" new signal, but by **adjusting our trade sizes** to 3% so our money doesn't sit idle in cash. This is the core lesson: our edge comes from **trade sizing and the structure of the ensemble itself, not from second-guessing individual trades** — both entry-side machine-learning filtering and exit-rule changes (take-profit / time-stop) were tested exhaustively in honest walk-forward and **rejected** (June 28). The current RSI-based exit and flat 3% sizing are, as far as we can measure, already the right choices. The nearest goal is going live with this validated config. Weighted voting was the final equity lever and was **tested & rejected (June 28)** — IC-weighting the voters lowered risk-adjusted return (Sharpe 1.01 < 1.12), closing the equity selection book. **State of the research program (July 7):** two more independently-motivated hypotheses — overnight-gap reversion (a session-structure anomaly) and cross-sectional idiosyncratic volatility (a genuinely different, defensive-premium risk factor, not another price-derivative signal) — were built, gated, and honestly WFO'd; both closed NO-GO against the 1.12 baseline. That brings the count to eight distinct equity-side hypotheses tested and rejected. The consistent shape across all eight — plausible economic premise, correctly caught by the NDH/DSR gates once tested at full scale, not a bug in the gates — is itself the finding: **daily-bar, large-cap equity research is close to arbitraged out for this system.** Further equity-family proposals (sector-neutral reversion, calendar effects) should clear a higher bar before consuming research time; the two directions with a genuinely different premise are a new asset class (the parked delta-neutral crypto-carry sleeve, gated on >$10k capital + a funding-data backfill) and, if its own correlation-vs-core study supports it, idio_vol as a blended diversification sleeve rather than a standalone strategy.

**State of the research program (July 16):** the leverage-realistic 3-sleeve blend (§below) was wired into live paper trading, and two independent attempts to find a genuinely orthogonal lever in **leveraged/inverse ETFs** — a new instrument class, not another equity signal — were built, gated, and honestly WFO'd against all three universes; both closed NO-GO (see the Glance table above). That brings the count to ten distinct hypotheses tested and rejected across two asset-class attempts. Going forward, `docs/research/RESEARCH_SNAPSHOT.md` and `docs/research/prompts/edge-research-agent-prompt.md` are the living, auto-regenerated (via the `research-snapshot` skill) source of truth for the full roster and ranked next-edge candidates — check there first rather than relying on this section, which is hand-maintained and can drift between updates.

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
* **~~Pairs Trading (Statistical Arbitrage)~~** (❌ Rejected, July 17): See below.
* **Large + Mid-Cap Portfolio Blend** (🧪 Researching): Tested a 50/50 and 70/30 blend of S&P 500 (Large Cap) and MidCap 400 (Mid Cap) stocks. Mid-caps show promising reversion characteristics (beats MDY: 15.0% CAGR / 1.08 Sharpe after survivorship haircut vs 9.1% / 0.40). **Gate investigation (June 27) settled the "anchor-driven" question:** the safety gates are *not* miscalibrated — the Deflated Sharpe gate passes every fold, and the rejections come from the Neighborhood Density gate correctly flagging that mid-cap optimal settings sit on genuinely noisier, less-robust plateaus. We improved the circuit-breaker's recovery rule (now "2 of the last 3 clean windows" instead of "2 consecutive," which cut defensive-anchor folds from 15 to 10), but the residual caution is real, not a bug. **Decision:** rather than weakening overfit protection to force mid-caps through, deploy mid-cap only as a diversification sleeve *behind* the deploy-clean large-cap core — judged on blend diversification, not as a standalone gated strategy.

**Diversification measurement (June 27).** We measured the actual return correlations between our reversion sleeves (the number that decides whether more universes are worth it). Result overturned the "the easy diversification is used up" assumption: S&P 500 vs MidCap 400 is only **0.70** (moderate — real benefit), and **MidCap 400 vs Nasdaq-100 is just 0.35** (genuinely independent). Nasdaq-100 is also the single best standalone sleeve (Sharpe 1.24, shallow drawdown). Blending the low-correlation MidCap+Nasdaq pair gives the best risk-adjusted result of anything tested (Sharpe **1.39**, drawdown **−4.0%**, volatility 4.2% vs the S&P 500's 6.8%). Because that blend is so low-volatility, it can be *scaled up* (our position-size lever) to roughly match the S&P 500's growth while keeping a shallower drawdown — the diversification and exposure-scaling levers compound. Russell 2000 could not be included — its data is only ~29% loaded (a backfill prerequisite). *Caveat: these blend figures are idealized (no safety-gate halts); absolute numbers will come down when run through the live gates, but the correlation-driven risk reduction holds.*

**3-sleeve vs. 4-sleeve blend study (July 8, ❌ Rejected — idio_vol adds nothing to the blend).** With `--blend` now able to include weight-based strategies (`universe_fn` threaded into `run_blend()`, commit `39d18dd`), ran the first honest, gate-validated comparison of the SP500+MidCap400+Nasdaq100 core blend with and without `idio_vol@sp500` layered in as a fourth sleeve (both via the inverse-vol/target-vol overlay in `blend_curves`, `--eval-start 2021-01-31 --eval-end 2026-04-30`). **3-sleeve baseline:** CAGR 13.89%, Sharpe **1.03**, MaxDD −10.59%. **4-sleeve (+idio_vol):** CAGR 13.97%, Sharpe **1.01**, MaxDD −12.08%. Adding `idio_vol` moved CAGR by +0.08pp (noise) while *worsening* both Sharpe (−0.02) and drawdown (−1.49pp) — the opposite of a diversification benefit, despite `idio_vol`'s standalone 0.447 core-correlation and shallower solo drawdown (see above). Likely explanation: `idio_vol`'s own Sharpe (0.57) is low enough that its inverse-vol weight in the blend just dilutes the three higher-Sharpe sleeves without buying enough independent risk reduction to compensate. **Verdict: do not add `idio_vol` to the deployed blend mix.** This closes the idio_vol research arc opened July 7 — no further follow-up planned unless a future candidate specifically targets low-Sharpe/low-correlation trade-offs differently.

**Leverage-realistic 3-sleeve blend verdict (July 13, 🟢 GO — supersedes the June-27 and July-8 blend numbers).** Both prior blend figures were not directly comparable: June 27's "gate-honest 1.05" came from `multi_sleeve_research.py`/`portfolio_blend.py`, retired June 29 when `blend.py` replaced them; July 8's "idealized 1.03" came from the current tool but at its `--max-leverage 2.0` default — leverage the live paper account, which trades unlevered flat-3% cash sizing, cannot actually take. Re-ran the identical 3-sleeve mix (`ensemble@sp500,ensemble@midcap400,ensemble@nasdaq100`, same `--eval-start 2021-01-31 --eval-end 2026-04-30` window) at `--max-leverage 1.0`, and pulled the persisted leverage diagnostics for both: **2.0x-cap run:** Sharpe 1.03, CAGR 13.89%, MaxDD −10.59%, avg realized leverage **1.63x** (most days actually levered up near the cap). **1.0x-cap (deployable) run:** Sharpe **1.14**, CAGR 9.93%, MaxDD **−5.39%**, avg leverage 0.97x (essentially unlevered, as expected).

The blend's own SP500 sleeve inside this run scored Sharpe 0.97 (CAGR 13.41%, MaxDD −11.01%), short of the commonly-cited "1.12" headline baseline — first suspected as a universe-construction mismatch (`eligible_at()` vs `equity_universe_between()`), but that theory was checked and **retracted**: `ensemble` is a `target_kind="signals"` strategy, and `_sweep_fold_dispatch` (`wfo.py:378`) only routes through `universe_fn`/`eligible_at()` for `target_kind="weights"` strategies — the blend's SP500 sleeve uses the exact same static-universe mechanism as the standalone path. Re-running the standalone `--wfo` baseline under the identical window (`--eval-start 2021-01-31 --eval-end 2026-04-30`) reproduced the blend sleeve number almost exactly (Sharpe 0.97, CAGR 13.4%, MaxDD −11.0%) — confirming there's no methodology divergence between the two code paths. **The actual explanation is simpler: the "1.12" headline figure was measured on a different eval window** (the CLI's `--eval-end` defaults to "now," which drifts every time it's re-run without an explicit end date) — not the window this blend study uses. It is not the correct comparator here.

**The valid, single-window comparison is therefore:** SP500-core alone (this window) Sharpe **0.97** / CAGR 13.4% / MaxDD −11.0%, vs. the leverage-realistic 3-sleeve blend Sharpe **1.14** / CAGR 9.93% / MaxDD **−5.39%**. On a matched window, the blend beats the core by a full 0.17 Sharpe and roughly halves the drawdown, for a smaller (not larger) CAGR concession than the mismatched-window comparison implied. **Verdict: GO — reopen the diversification arc**, contra June 27's closure (which predates both the leverage fix and the retired-tool ambiguity). This is now a live-capital decision: wired into `PaperTrader` July 14 (sleeve-aware sizing via `RiskGuard.sleeve_slot_caps`/`sleeve_position_notional`, a margin pre-flight check enforcing the unlevered assumption, `--live` CLI flag) — it is the deployed config, not just a research number. *Side finding, resolved July 13:* the "1.12" figure is used as a fixed reference point across several other roadmap entries (Kelly sizing, IC-weighted voting, exit-rule sweep); an audit confirmed those all used `--eval-end="now"` within ~10 days of each other and of the 1.12 baseline's own run — negligible drift (Sharpe moves ~0.15 over a 2.5-month window shift), no re-runs needed.

**Market-neutral pairs / stat-arb mean reversion (July 17, ❌ Rejected).** The
lab's first market-neutral construction — every prior strategy is long-only
directional. Same-sector, correlation-filtered (≥0.7 trailing 126-day),
naive 1:1 unhedged log-spread z-score, monthly rebalance (`pairs_stat_arb.py`,
`STRATEGY_REGISTRY["pairs_stat_arb"]`). Full honest WFO, SP500 universe,
2015–present, 42 folds, 54 combos: **OOS Sharpe -0.42 vs SPY 0.88**, CAGR
-2.6% vs 15.2%, aggregate WFE **-0.16** (below the 0.50 floor and negative —
overfitting, not a modest-but-real edge), regime-halt active 28/42 folds,
live-recommended params selected in only 1/42 folds (unstable). The
market-neutrality design goal *was* achieved mechanically — OOS return
correlation to SPY 0.092, beta 0.030 — but near-zero correlation to a
benchmark that returned +15%/yr doesn't help when the strategy's own return
is negative. Open question flagged for a future decisive test (not
resolved by this report): the lab's `weights`-strategy harness only
supports monthly rebalancing (`rebalance_dates`), while pairs/stat-arb
mean-reversion conventionally needs daily/weekly monitoring — the
regime-halt rate and negative WFE are at least consistent with "mistimed
signal" rather than "no edge exists," but confirming that needs a real
infrastructure change (faster rebalance cadence for `weights` strategies),
not a parameter re-sweep. Full report:
`docs/research/2026-07-17-pairs-stat-arb-nogo.md`.

**Leveraged/inverse ETF rotation & trend following (July 16, ❌ Rejected — both attempts).** The first genuinely new *instrument class* tried (not another equity signal): 2x/3x leveraged ETFs (UPRO/SSO, TQQQ/QLD, TNA/UWM) and their inverse pairs, across SP500/Nasdaq100/Russell2000. **Attempt 1 — breadth-driven rotation** (`leveraged_rotation.py`): rotates long/inverse/cash driven by the existing validated `EnsembleSignal`'s breadth across each universe's constituent stocks. Full honest WFO, all three universes: OOS Sharpe **−0.14 to −0.37** vs SPY 0.65–0.77, MaxDD **−80% to −89%**, defensive-halt active on 22–23 of 26 folds in every universe — the strategy spent almost the entire backtest on its fallback anchor params, not its own selected signal. An ablation confirmed a concurrent survivorship-bias fix (point-in-time universe membership) wasn't the operative variable; the mechanism itself doesn't survive leveraged-ETF decay/whipsaw. **Attempt 2 — long-only trend filter** (`leveraged_trend.py`): simpler design, hold the leveraged ETF only when its underlying unleveraged index is above its trailing SMA, else cash, layered with the lab's existing realized-vol-targeting overlay to reduce whipsaw exposure. Structurally much healthier (no persistent halt, gate pass 14–18/26 vs 3–4/26) and did cut drawdown sharply (−10% to −17% vs. buy-and-hold's −59% to −85%), but **still lost to simply buying and holding the same ETF** in every universe — OOS Sharpe 0.40–0.62 vs. buy-and-hold's 0.48–0.90, giving up 20–40 points of CAGR that a plain SMA filter can't recover once it misses the biggest rallies. **Verdict: closed, both mechanisms.** Full reports: `docs/research/2026-07-16-leveraged-index-rotation-nogo.md`, `docs/research/2026-07-16-leveraged-trend-following-nogo.md`.

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
| **Research Snapshot Tooling** | ✅ Done | A local Claude Code skill (`research-snapshot`, July 16) that regenerates `docs/research/RESEARCH_SNAPSHOT.md` (full strategy roster + verdicts) and a self-contained research-agent prompt from source of truth (git log, `docs/research/`, the strategy registry) on demand — built specifically so these living docs don't hand-drift stale the way this roadmap's own timeline did. |

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
* **Leveraged/Inverse ETF Timing — both mechanisms (rejected, July 16)**: The first attempt at a genuinely new instrument class (2x/3x leveraged ETFs) rather than another equity signal. A breadth-driven long/inverse/cash rotation collapsed to −80% to −89% drawdowns and spent almost the whole backtest on its defensive fallback. A simpler long-only trend filter with a volatility-targeting overlay avoided that collapse (no persistent halt) but still lost badly to just buying and holding the same leveraged ETF — the timing signal gave up far more upside than it saved in downside. Both closed; see §3.

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
* **July 13, 2026**: Resolved the leverage-realism ambiguity in the 3-sleeve blend arc — re-ran the SP500+MidCap400+Nasdaq100 blend at a deployable `--max-leverage 1.0` instead of the earlier research default of 2.0x. Result: Sharpe **1.14**, MaxDD **−5.39%**, a full 0.17 Sharpe gain and roughly half the drawdown of the SP500 core alone on the same matched window. Also closed the "1.12 baseline" eval-window-drift question raised June 27 — audited and confirmed negligible (~10 days apart across all the levers that cite it). **Verdict: GO**, reopening the diversification arc closed June 27.
* **July 14, 2026**: Wired the leverage-realistic blend into the live `PaperTrader` — sleeve-aware position sizing (`RiskGuard.sleeve_slot_caps`/`sleeve_position_notional`), a margin pre-flight check enforcing the unlevered assumption before any real order, and a `--live` CLI flag. This blend, not the single-sleeve core, is now the deployed live-paper configuration. Also began building the first attempt at a genuinely new instrument class: `leveraged_rotation.py`, a breadth-driven 2x/3x leveraged-ETF long/inverse/cash rotation strategy across SP500/Nasdaq100/Russell2000 (not yet WFO-validated at this point — see July 16).
* **July 16, 2026**: Ran the first honest WFO of the leveraged-ETF rotation strategy across all three universes — **rejected**: OOS Sharpe −0.14 to −0.37 vs SPY, MaxDD −80% to −89%, defensive-halt active on 22–23 of 26 folds everywhere. Built and tested a second, simpler design — `leveraged_trend.py`, a long-only SMA trend filter with a volatility-targeting overlay — as a genuinely different mechanism, not a re-sweep of the rejected one. Structurally much healthier (no persistent halt) and cut drawdown sharply, but **also rejected**: loses to naive buy-and-hold of the same ETF in every universe tested. Closes the leveraged-ETF instrument-class attempt, two mechanisms tried. Built the `research-snapshot` skill (regenerates `docs/research/RESEARCH_SNAPSHOT.md` and a self-contained edge-research agent prompt from source of truth) specifically to stop roadmap/next-steps drift like the one this update fixes; seeded 4 ranked candidates for the next research push (market-neutral pairs/stat-arb, post-earnings-announcement drift, options-derived signals, and revisiting the parked crypto-carry gate).
* **July 17, 2026**: Built and WFO-tested Rank 1 of those 4 candidates — `pairs_stat_arb.py`, same-sector correlation-filtered spread mean reversion, the lab's first market-neutral (long+short) construction. **Rejected**: OOS Sharpe −0.42 vs SPY 0.88, aggregate WFE −0.16 (overfitting signature, not a modest edge), regime-halt active 28/42 folds, live-recommended params stable in only 1/42 folds. Market-neutrality itself was achieved (OOS correlation to SPY 0.092, beta 0.030) — the long/short mechanics work, the entry/exit signal doesn't carry an edge at this monthly cadence. Flagged an untested, decisive follow-up rather than closing the mechanism class outright: the `weights`-strategy harness only supports monthly rebalancing, while pairs/stat-arb conventionally needs daily/weekly monitoring — a real infrastructure change, not queued speculatively. Also merged a 14-candidate externally-sourced strategy backlog into `docs/research/WEB_RESEARCH_CANDIDATES.md` (first web-discovery-pipeline run) and parallelized the WFO harness's per-combo grid sweep across CPU cores (`wfo.py`'s `_sweep_fold_weights`, via `joblib.Parallel` — combos are independent, so this is a safe win on the 4-core research box; verified no test regressions, 552 passing). Full report: `docs/research/2026-07-17-pairs-stat-arb-nogo.md`. Same day,
also cleared the lowest-effort candidate from the new web backlog —
`max_effect` (MAX/lottery-demand quintile filter, candidate #11, zero new
data infrastructure needed). **Also rejected**, but a genuinely different
character: OOS Sharpe 0.39-0.45 vs SPY 0.76-0.88 with *healthy* WFO
diagnostics (WFE 0.84-0.97, gate pass 31/42) — a real, correctly-measured
effect that's simply too weak to beat SPY, not an overfitting artifact. A
diversification follow-up (same method as `idio_vol`'s July 7 check) found
0.692 OOS return correlation to the deployed core — higher than
`idio_vol`'s already-insufficient 0.447 — closing the diversification angle
too. Full report: `docs/research/2026-07-17-max-effect-nogo.md`. Continuing
effort-first through the backlog, also built and tested the free-data cut
of #3 (short-interest/cost-to-borrow): `short_interest` strategy on a new
FINRA consolidated-short-interest data pipeline
(`short_interest_data.py`, 86,793-row backfill covering 150 real settlement
dates 2020-04–present — a real correctness bug was found and fixed here,
`discover_settlement_dates()` replacing a naive calendar-guess that
silently missed 47% of actual cycles due to weekend/holiday date shifts).
**Rejected**: OOS Sharpe 0.27 vs SPY 0.61, WFE 0.11 (below the 0.50 floor),
regime-halt 16/20 folds — a noise/overfitting rejection this time (unlike
`max_effect`'s healthy-but-weak profile), consistent with the literature's
own value-weighted-insignificance caveat for large-cap short-interest
effects. Full report: `docs/research/2026-07-17-short-interest-nogo.md`.
Next, built and tested #12 (PEAD, lower-coverage names) on a new yfinance
earnings-surprise pipeline (`earnings_surprise_data.py`, 62,552 rows). This
one is worth reading closely as a process lesson, not just a rejection: an
initial long-window (2015-2026) SP500 test looked like the strongest
standalone result of the whole session — beat SPY (Sharpe 0.93 vs
0.76-0.90), healthy WFE (1.01), low regime-halt (20%), moderate 0.422
correlation to the deployed core. Per the established playbook (the
`idio_vol` precedent), this triggered an actual 4-sleeve blend test rather
than being reported as a standalone win — and that test **overturned the
finding**. Re-run on the exact window the deployed 3-sleeve blend was
validated on (2021-2026), the edge evaporated (Sharpe 0.58, tied with
SPY), and adding it as a 4th sleeve made the deployed blend measurably
worse (Sharpe 1.14→1.06, MaxDD -5.39%→-6.51%). The 3-sleeve baseline
reproduced exactly (Sharpe 1.14, MaxDD -5.39%) confirming no tooling
drift. Russell 2000 (the candidate's own "lower-coverage" test) also only
tied SPY with a much higher regime-halt rate (60% vs SP500's 20%),
contrary to the literature's expectation of a stronger effect there.
**Rejected, both standalone (matched window) and as a blend sleeve.** Full
report: `docs/research/2026-07-17-pead-nogo.md`. Next, tested #13 (S&P 500
index-deletion overshoot fade) — the fastest build of the whole session
(zero new data infrastructure, built directly on the already-maintained
point-in-time SP500 membership history). **Rejected, and not a close
call**: OOS Sharpe 0.30 vs SPY 0.76, MaxDD **-68.7%** (worst drawdown of
any candidate closed this session, nearly double SPY's own -33.7%), gate
pass 17/42 (40%), regime halt 32/42 folds (76%). Likely cause: many real
S&P 500 deletions reflect genuine fundamental deterioration (bankruptcy
risk, earnings collapse), not just mechanical index-committee timing —
buying the deletion buys falling knives as often as oversold-but-fine
names. Also found and fixed a real infrastructure bug: `simulate_weights`
crashed (`IndexError`, deep in vectorbt) on a fold where every grid combo
picked zero symbols across the whole window — a scenario only a
sparse-event strategy like this one can trigger; fixed with a flat-equity
short-circuit for the all-empty case, two regression tests added,
benefits any future sparse-event strategy in this lab. Full report:
`docs/research/2026-07-17-index-deletion-fade-nogo.md`.
* **July 18, 2026**: Checked #7 (Anomaly-Driven Demand) for feasibility
  before building anything — **infeasible**, same class of blocker as #2.
  The Chen & Zimmermann firm-level characteristics dataset (verified
  against the `openassetpricing` package's actual source code) is keyed
  purely by CRSP `permno`, has no ticker column, and the package's own
  pipeline calls a WRDS connection directly for some signals. No free
  permno-to-ticker crosswalk exists — confirmed via search, CRSP's
  ticker-history identity mapping is a WRDS subscription product. A
  name-matching heuristic crosswalk was considered and rejected (ticker
  reuse over the dataset's multi-decade span makes it a real silent-data-
  corruption risk). Deprioritized pending a WRDS-access decision, not
  built. See `docs/research/WEB_RESEARCH_CANDIDATES.md` candidate #7 for
  detail.
* **July 18-19, 2026**: Built and tested #1 (insider cluster-buying, SEC
  Form 4) — the highest-effort candidate in the backlog, explicitly
  scoped to full SP500 despite an estimated multi-hour build (user
  decision). Built a full SEC EDGAR Form 4 pipeline
  (`src/ggTrader/lab/form4_data.py`: ticker→CIK resolution, per-issuer
  filing enumeration with pagination, ownership-XML parsing via
  `defusedxml`), verified against real SEC endpoints before committing to
  the build. Backfill: 833,158 transaction rows across 764 SP500 symbols
  since 2015, ~24 hours (rate-limited to respect SEC's ~10 req/sec
  guidance), essentially zero fetch errors throughout. Initial long-window
  WFO looked distinctive: OOS Sharpe 0.76 (near-tied SPY 0.77), shallowest
  drawdown and lowest deployed-core correlation (0.382) of any candidate
  this session. Following the process lesson from PEAD's closure, a
  matched-window retest (deployed blend's own 2021-2026 validation window)
  and a real 4-sleeve blend test both overturned the initial impression:
  standalone Sharpe fell to 0.39 vs SPY 0.58, and the blend showed no
  improvement (Sharpe 1.14→1.12, MaxDD -5.39%→-5.45%). **Rejected,
  standalone and as a blend sleeve** — the third diversification-sleeve
  candidate closed this way this session (after `idio_vol` and `pead`).
  `form4_data.py` and the backfill remain reusable infrastructure. Full
  report: `docs/research/2026-07-19-insider-cluster-buy-nogo.md`.
* **July 19, 2026**: Built and tested #9 (Congressional/House STOCK Act
  trade mirroring), House-only scope. New data pipeline
  (`src/ggTrader/lab/house_ptr_data.py`: House Clerk's bulk annual index +
  text-extractable PTR PDF parsing, verified no OCR needed even for
  2015-era filings), 41,443 rows across 7,578 filings 2015-2026,
  essentially clean backfill. **This was the strongest long-window
  standalone result of the entire session**: OOS Sharpe 0.89 vs SPY 0.77,
  gate pass 34/40 (85%), stability 13/40 (33%) — all the highest of any
  candidate tested. Applied the now-established matched-window + blend-test
  discipline immediately rather than reporting it as promising, and it
  overturned a third time this session: standalone Sharpe fell to 0.36 vs
  SPY 0.58 on the deployed blend's 2021-2026 validation window, and the
  4-sleeve blend test showed the worst degradation of any diversification
  candidate tested (Sharpe 1.14→1.04, MaxDD -5.39%→-5.43%, worse than
  `pead`'s 1.06 and `insider_cluster_buy`'s 1.12). **Rejected, standalone
  and as a blend sleeve.** Three consecutive candidates (`pead`,
  `insider_cluster_buy`, `congress_trades`) have now failed this exact
  way — flagged in the closure report as a standing pattern worth
  tracking, not three independent coincidences. `house_ptr_data.py` and
  the backfill remain reusable infrastructure. Full report:
  `docs/research/2026-07-19-congress-trades-nogo.md`.
* **July 19, 2026**: Built #8 (retail-attention-conditioned factors) as
  `retail_attention`, testing Da/Engelberg & Gao's actual validated core
  finding (search-volume spikes → short-term buying pressure) via a new
  Google Trends pipeline (`google_trends_data.py`, pytrends), 19 tests
  passing against injected fakes. **Not resolved** — an initial feasibility
  spot-check (39/40 rapid queries succeeded) under-sampled Google's real
  rate limiting; starting the actual ~750-symbol backfill triggered a 429
  lockout that didn't clear on retry. Paused (not closed, not
  infeasible) pending a later retry with more conservative pacing — no
  IP-rotation or other circumvention attempted. See
  `docs/research/WEB_RESEARCH_CANDIDATES.md` candidate #8.
* **July 19, 2026**: Built and tested #5 (stealthy shorts, free-data-only
  cut) as `short_volume_ratio` — market-neutral, long lowest-quintile /
  short highest-quintile by trailing short-volume ratio, the second
  market-neutral construction in this lab after `pairs_stat_arb`. New
  FINRA daily short-volume pipeline (`short_volume_data.py`, a different
  dataset than `short_interest_data.py`'s bi-monthly consolidated short
  interest; CDN retention verified via bisection to start 2018-08-01),
  1,189,652 rows backfilled. WFO (SP500, 27 folds): OOS Sharpe 0.21 vs SPY
  0.72, gate pass 16/27 (59%), regime halt active 20/27 folds (74%,
  persistent) — a market-neutral book with a shallow drawdown (-9.2%) but
  near-zero return, reading as "mechanically correct construction, no
  differentiating signal at this fidelity" rather than an overfitting or
  eval-window-drift rejection. **Rejected.** No matched-window follow-up
  needed — never showed standalone promise on the long window to begin
  with. `short_volume_data.py` and the backfill remain reusable
  infrastructure. Full report:
  `docs/research/2026-07-19-short-volume-ratio-nogo.md`.
* **July 20, 2026**: Checked #4 (crypto perpetual-futures funding-rate
  carry) for feasibility before building anything — **infeasible for an
  honest WFO**, closing the long-parked crypto-carry line (§6 internal
  Rank 4) on the same evidentiary basis, not just deprioritization.
  Binance.US doesn't offer perpetual futures at all (spot-only, per
  `ccxt`), so Kraken Futures was always the only real venue. Verified via
  both `ccxt` and Kraken's own native historical-funding-rates API
  directly: only a rolling ~1 year of hourly funding data is retained
  (earliest available 2025-07-13 as of this check) — nowhere near the
  12mo-train + 3mo-test-per-fold, many-folds depth this project's WFO
  methodology requires. Same class of blocker as #2 and #7 (infeasible
  for the required methodology, not an engineering-effort problem). See
  `docs/research/WEB_RESEARCH_CANDIDATES.md` candidate #4.
* **July 19, 2026**: Checked #14 (option-implied volatility skew) for
  feasibility before building anything — **infeasible for an honest WFO**,
  same class of blocker as #2/#7/#4. `yfinance`'s `option_chain()` takes
  only a future-expiry selector, no historical/as-of parameter
  (current-snapshot-only). Alpaca's option-bars API (already integrated
  here) returned zero bars for a real contract in both 2022 and 2024
  windows — its options market data only starts in 2024, with no
  historical listed-contracts feed to reconstruct past chains from. Deep
  historical options-chain data remains a paid-vendor problem
  (OptionMetrics IvyDB), exactly as the original candidate write-up
  flagged, and the signal may just be re-deriving #3's already-NO-GO'd
  borrow-cost signal through a noisier instrument anyway. No code built.
  **This closes out the effort-ordered candidate queue entirely** — all 14
  `WEB_RESEARCH_CANDIDATES.md` candidates and all 4 `RESEARCH_SNAPSHOT.md`
  §6 internal candidates are now resolved (10 NO-GO, 4 infeasible, 1
  paused pending a transient rate-limit retry). The deployed 3-sleeve
  blend (Sharpe 1.14, MaxDD -5.39%) survived every diversification
  attempt this session. See `docs/research/WEB_RESEARCH_CANDIDATES.md`
  candidate #14.
* **July 20, 2026**: Built and tested candidate A1 (dynamic FX hedge
  overlay) from the cross-asset register's home-lab priority queue —
  **REJECTED**. New `fred_data.py` module (free FRED CSV data, no API
  key, point-in-time correct) + `fx_hedge_overlay` strategy (carry+PPP-
  value+trend on EWJ/DXJ and EZU/HEZU hedged/unhedged pairs, the only
  currently-active such pairs). Standalone WFO Sharpe 0.41 vs SPY 0.74
  looked like a routine NO-GO, but SPY is the wrong benchmark for this
  candidate — the paper's actual claim is dynamic-vs-static hedging, not
  beating equities. The decisive test compared the dynamic strategy
  against static 100%-unhedged/100%-hedged/50-50 baselines on the same
  instruments over the same OOS window: **the dynamic strategy
  underperformed every static alternative** (0.41 vs 0.53/0.66/0.73
  Sharpe), a clean "the signal construction adds no value" rejection
  (WFE 1.10, not overfitting). Found and fixed two real bugs along the
  way: a `DataFrame.iterrows()` pandas gotcha that silently corrupted
  cached FRED values to `NaT`, and a mis-verified FRED series ID (an
  `curl | wc -l` check had miscounted an HTML 404 page's lines as real
  CSV data). Full report:
  `docs/research/2026-07-20-fx-hedge-overlay-nogo.md`. Next up: candidate
  A7 (pre-FOMC long-Treasury drift).
* **July 20, 2026**: Built and tested candidate A7 (pre-FOMC long-Treasury
  drift) — **REJECTED**, despite a strong, current, directly-on-point
  citation (Pan & Peng, June 2026). New `fomc_calendar.py` module (free
  Federal Reserve calendar scraper, two source-page formats stitched
  together, 122 scheduled FOMC dates 2011-2026, cross-checked against
  known dates like the December 2015 "liftoff" meeting) + `fomc_drift`
  strategy (long TLT/IEF/EDV the day before a scheduled announcement,
  exit at/after, using the `"signals"` protocol for exact-bar timing
  rather than this lab's monthly-rebalance `"weights"` protocol). Found
  and fixed a real bug on first run: this lab's OHLCV bars carry a
  non-midnight time-of-day (`16:00:00+00:00`) while FOMC dates are
  midnight-normalized, so an exact-timestamp equality check silently
  matched zero events (an all-zero-trades result was the tell that
  something was structurally broken, not that there was no signal) —
  fixed by comparing on calendar date, regression-tested before the fix.
  Full 54-fold WFO: **OOS total return 0.45% over 13.5 years**
  (essentially flat), Sharpe 0.10, WFE 0.44 (below the 0.50 overfitting
  floor), regime halt active (winning combo selected in only 14/54
  folds) — a clean "no exploitable drift in this implementation" result.
  Full report: `docs/research/2026-07-20-fomc-drift-nogo.md`. Next up:
  candidate A3 (commodity medium-term trend).
* **July 20, 2026**: Audited the project's high strategy-rejection rate
  (~30 tried, ~6 deployed) at the user's request. Found and fixed one
  real gap: `gates.py`'s NDH expectancy calc used raw `total_return_pct`
  as a stand-in for per-trade expectancy because trade count was never
  tracked (`simulate_weights`/`simulate_signals` now expose real
  `n_trades` via vbt's `pf.trades.count()`) — traced through the gate
  math and confirmed this doesn't retroactively flip any past verdict
  (NDH only tests expectancy's sign, which was already correct). Built a
  new supplementary test, `src/ggTrader/lab/event_study.py` (Welch's
  t-test, event-day vs ordinary-day mean return), specifically for
  sparse/event-driven candidates where annualized Sharpe is a poor fit
  (fomc_drift's per-fold OOS Sharpe swung -3.59 to +3.35). Re-checked
  both sparse closed candidates through this lens: **fomc_drift's effect
  is real and significant gross (p=0.007, ~1%/year) but is entirely
  consumed by realistic round-trip trading costs (p=0.76 net)** — a
  precise, well-understood reason for the NO-GO, not noisy gate math (see
  addendum in `docs/research/2026-07-20-fomc-drift-nogo.md`).
  **index_deletion_fade shows no detectable gross effect at all**
  (p=0.72), confirming its NO-GO independently (addendum in
  `docs/research/2026-07-17-index-deletion-fade-nogo.md`). Conclusion: the
  high rejection rate is mostly real and expected (decades-old published
  anomalies subject to documented decay, retail-implementation gaps for
  institutional-edge candidates) — not evidence of broken tooling, though
  the event-study addition is a genuine, lasting improvement for future
  sparse candidates. 15 new tests, full suite still passing (same 2
  pre-existing unrelated failures). No mechanical re-run of all closed
  strategies — the n_trades fix doesn't change their math and no other
  concrete bugs were found on inspection.
* **July 20, 2026**: Built and tested candidate A3 (commodity medium-term
  trend) — **REJECTED**. New 14-ETF single-commodity universe (metals/
  energy/ags; BAL and JJC checked and found delisted 2023, excluded) +
  `commodity_trend` strategy: 12-1 cross-sectional momentum (reusing this
  lab's established `xs_momentum` pattern) plus a market-wide realized-
  volatility regime filter distinct from the already-rejected equity
  VIX-regime-throttling idea, tested on its own merits in a different
  asset class. Full 50-fold WFO: OOS Sharpe 0.13 vs SPY 0.74, **MaxDD
  -37.0% — worse than SPY's -33.7%** despite the vol-regime filter being
  specifically designed to avoid crash exposure, aggregate WFE undefined,
  regime halt active despite 82% fold-stability for the recommended
  combo. Full report: `docs/research/2026-07-20-commodity-trend-nogo.md`.
  Next up: candidate A5 (Treasury term-structure factors, ETF-
  approximation version).
* **July 20, 2026**: Built and tested candidate A5 (Treasury term-
  structure factors, ETF-approximation version) — **REJECTED**. New
  `TreasuryCurveStrategy`: a curve steepener/flattener regime signal using
  real FRED constant-maturity Treasury yields (`DGS2`/`DGS10`, reusing
  `fred_data.py`), one-hot rotating across SHY/IEF/TLT. Standalone WFO
  Sharpe 0.19 looked routinely weak, but per A1's lesson the decisive test
  compared the dynamic strategy against static duration baselines on the
  same OOS window: **static 100% SHY beats it on every metric** (Sharpe
  0.90 vs 0.19, CAGR 1.5% vs 0.8%, MaxDD -5.7% vs -8.2%) — the "dynamic
  loses to static" pattern from A1, now confirmed twice. Full report:
  `docs/research/2026-07-20-treasury-curve-nogo.md`. This closes only the
  3-ETF approximation — the paper's actual 4-factor model needs
  instruments (cash Treasuries/STRIPS/futures) this project doesn't have,
  and remains untested. Four candidates now tested from the cross-asset
  register (A1/A7/A3/A5), all REJECTED. Next up: candidate A8
  (headline/LLM sentiment), pending a point-in-time data feasibility
  check.
* **July 24, 2026**: Built and tested candidate A8 (headline/LLM
  sentiment on small/mid-cap equities) — **REJECTED**. New
  `HeadlineSentimentStrategy` + `headline_sentiment_data.py`: LLM-scored
  (`deepseek-flash` via local LiteLLM) sentiment on Alpaca `/news`
  headlines, point-in-time correct (0-day publish lag by design), a
  50-symbol/2-year midcap400 pilot (7,975 unique headlines scored, real
  score variance). WFO (3 folds, rolling 12mo/3mo): OOS Sharpe 0.25 vs
  SPY 1.14, CAGR 4.8% vs 22.9%, aggregate WFE 0.02 (target ≥ 0.50) — the
  in-sample edge (train Sharpe ~0.80-1.01) essentially evaporates OOS,
  and one fold fails the NDH gate outright on parameter instability. Not
  an infrastructure failure — three real bugs (reasoning-model token
  truncation, a `None`-response crash, a DST timezone bug in
  `pd.to_datetime`) were found and fixed via TDD along the way, and the
  sentiment pipeline itself scores correctly. Full report:
  `docs/research/2026-07-24-headline-sentiment-nogo.md`. Five candidates
  now tested from the cross-asset register (A1/A7/A3/A5/A8), all
  REJECTED, closing the register's home-lab priority queue. Next up per
  the register's ordering: candidate B1 (stablecoin stress signal) — a
  risk-overlay/leverage-reduction trigger, not a standalone alpha sleeve,
  a different shape of candidate than everything tested so far.
* **July 24, 2026 (feasibility-only pass, B1/B2/B3, no builds)**: Checked
  the next three B-series candidates for feasibility before committing
  build time; each hit a real gap, none started. **B1 (stablecoin
  stress):** data is actually excellent and already in the DB (`BTC-USD`
  + `USDT-USD` at 1-minute granularity, Kraken spot, no new integration
  needed) but (a) minute bars are stale (stop 2025-12-31) and (b) crypto
  trading itself is dormant (`kraken_ledger` last entry 2026-06-06, no
  active cron) — B1's design is a risk overlay on live crypto exposure,
  and there's currently no live crypto exposure to overlay. **B2 (FOMC
  country-ETF price discovery):** blocked by a project-wide gap, not a
  missing backfill — every non-crypto symbol in the DB (1,415 of them) is
  daily-bar-only (yfinance); the mechanism needs the ETF's first 15-30
  minute reaction to an FOMC announcement, which daily bars structurally
  cannot capture, and free intraday-equity sources don't have the history
  for a multi-year backtest. **B3 (bond ETF stale-NAV dislocations):**
  partially feasible (Treasury yields already on hand from A5, credit-
  spread FRED series and `HYG`/`JNK`/`LQD` prices are easy adds) but the
  actual signal input — historical NAV/premium-discount data — has no
  confirmed free source yet, and the register's own recommended redesign
  (a fair-value-residual estimate from bond trades + Treasury + credit-
  index moves, not the raw discount) is a meaningfully heavier build than
  any candidate tested so far. Paused here by request rather than pushing
  into B3's open-ended design work; next session should either resolve
  B3's NAV-data question or check B4 (crypto volatility-premium regime
  indicator) fresh.
* **July 25, 2026 (implementation audit — no code changes)**: Audited all 17
  registered strategies to separate "genuinely failing" from "failed by our
  own bugs," per an explicit ask. Full report:
  `docs/research/2026-07-25-strategy-implementation-audit.md`. **Headline
  finding — a severe data-integrity bug:** `SPY` is the only yfinance symbol
  with bars outside the 16:00/17:00 close convention (826 rows at 04:00/05:00
  from 2022-12-27). Since `lab/cli.py:152` and `lab/blend.py:86` load
  `universe + ["SPY"]` into one frame, the union index carries **two rows per
  trading day from 2023** (462-504/yr vs the correct ~250), which understates
  every equity Sharpe covering 2023+ by **~29%** (measured: AAPL buy-and-hold
  0.848 contaminated vs 1.187 clean; CAGR/MaxDD unaffected). Corroborated by
  A8's cited SPY 1.14 vs a true 1.61 — ratio 0.708 ≈ 1/√2. Rankings survive
  (the benchmark deflates identically) but absolute values and DSR gate
  outcomes do not. **`pairs_stat_arb`'s NO-GO is invalidated for 2023+** —
  the same bug made its coverage guard reject every pair, so it selected
  nothing for ~a third of its window; needs a re-run. Also found: a ~2-day
  `transaction_date`-vs-`filing_date` lookahead in `insider_cluster` (inflates
  results, so its NO-GO stands), a display-only bug making the `OOS` column
  constant `0.00` in every WFO table ever printed, and `commodity_trend`
  reaching a wrong conclusion about its vol filter by benchmarking against SPY
  instead of commodities (NO-GO survives on Sharpe 0.13 vs DBC 0.24, but the
  filter *did* cut drawdown: -37.0% vs -59.9%). `headline_sentiment` (A8)
  downgraded to **provisional** (3 folds vs 20-54 elsewhere; ~2×
  Industrials-skewed sample; 7-8 stock portfolio). Verified sound: the harness
  produces GO for the deployed baseline (positive control), survivorship
  handling (universe contains SIVB/FRC/TWTR; 1.7% missing, mostly renames),
  point-in-time discipline and next-bar entry across all strategies,
  `ensemble_ic`'s leak guard, and costs (optimistic, so rejections are
  conservative). `retail_attention` confirmed correctly labeled PAUSED (its
  Google Trends table has 0 rows). On-deck feasibility consolidated: A4/A6/A9/
  B2 infeasible, B3 blocked, A2 thin (~3 usable pairs), B1/B4 data-feasible
  but gated on crypto trading being dormant.

---

*Back to [README.md](../README.md).*
