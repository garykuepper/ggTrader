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
| **Machine Learning Filter** | ✅ Done | Retrained our LightGBM machine learning filter, which blocks trades that have a low probability of success. |
| **Live Paper Trading** | 🟢 Live | Running our 5-indicator strategy with 3% trade size on virtual money with automated safety limits. Logs only real, completed fills to avoid accounting errors. (June 27: verified live fills reconcile exactly against the broker; redeployed the trader so the honest-fill-logging code is actually running; moved the daily run 30 min earlier so orders fill before the close instead of queuing overnight.) |
| **Next Steps** | 🔵 Next | Monitor virtual trading for 5–10 days → fund a live $1,000 account → go live. Start research on blending S&P 500 and MidCap 400 stocks. |
| **Future Research** | 🧪 Research | Exploring weighted voting, macro market data filters, and Kelly sizing (smart trade sizing). |

> **Status Legend:** ✅ Done · 🟢 Live · 🔵 Next · 🧪 Research · ❌ Rejected · ⏸ Deferred

---

## 1. Our "North Star" Goal

Our primary objective is to build a **flexible multi-strategy research lab** that performs honest, realistic walk-forward testing. We will only invest real capital in trading systems that survive strict out-of-sample tests. 

**Current Thesis (June 2026):** Combining multiple trading indicators (Bollinger Bands, RSI, EMA, MACD, and Volume Bollinger Bands) works because their failures are independent. When one indicator is wrong, the others usually don't vote, preventing bad trades. We closed our performance gap against the S&P 500 index not by trying to find a "magic" new signal, but by **adjusting our trade sizes** to 3% so our money doesn't sit idle in cash. Our main areas for improvement are diversification (blending large-cap and mid-cap stocks), weighted voting, and machine learning filter upgrades.

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
- **Weighted Indicator Voting** (`ensemble.py`): Give more voting weight to indicators that have performed better in past training, rather than treating all votes equally.
- **Dynamic Machine Learning Filter** (`feature_gate.py`): Make the machine learning filter adjust dynamically—blocking more trades when the market is highly volatile, and loosening up when the market is calm.
- **Exit Strategy Sweeps** (`ensemble.py`): Run simulation sweeps over different exit rules (like fixed profit targets or time-based limits) to find the best way to close trades.

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

* **Exposure Scaling & Regime Mapping** (✅ Shipped & 🧪 Ongoing): We resolved under-deployment by raising trade sizes to 3%. Open research: Can we use a market indicator (like volatility or trend direction) to dynamically change trade sizes?
* **Dynamic Trade Sizing (Kelly Criterion)** (⚪ Planned): Use machine learning probabilities to calculate exactly how much money to risk on each trade (a math-based system to maximize growth while avoiding ruin).
* **Diversification Ranking** (⚪ Planned): If the system generates more buy signals than we have cash for, rank them by indicator strength and sector diversity to avoid buying too many stocks in the same industry.
* **Macro Machine Learning Features** (⚪ Planned): Feed broader market metrics (like the VIX fear index or interest rates) into the machine learning filter.
* **Pairs Trading (Statistical Arbitrage)** (⚪ Planned): Identify pairs of stocks that historically move together, buying one and selling the other when they drift apart, expecting them to converge.
* **Large + Mid-Cap Portfolio Blend** (🧪 Researching): Tested a 50/50 and 70/30 blend of S&P 500 (Large Cap) and MidCap 400 (Mid Cap) stocks. Mid-caps show promising reversion characteristics (beats MDY: 15.0% CAGR / 1.08 Sharpe after survivorship haircut vs 9.1% / 0.40). **Gate investigation (June 27) settled the "anchor-driven" question:** the safety gates are *not* miscalibrated — the Deflated Sharpe gate passes every fold, and the rejections come from the Neighborhood Density gate correctly flagging that mid-cap optimal settings sit on genuinely noisier, less-robust plateaus. We improved the circuit-breaker's recovery rule (now "2 of the last 3 clean windows" instead of "2 consecutive," which cut defensive-anchor folds from 15 to 10), but the residual caution is real, not a bug. **Decision:** rather than weakening overfit protection to force mid-caps through, deploy mid-cap only as a diversification sleeve *behind* the deploy-clean large-cap core — judged on blend diversification, not as a standalone gated strategy.

**Diversification measurement (June 27).** We measured the actual return correlations between our reversion sleeves (the number that decides whether more universes are worth it). Result overturned the "the easy diversification is used up" assumption: S&P 500 vs MidCap 400 is only **0.70** (moderate — real benefit), and **MidCap 400 vs Nasdaq-100 is just 0.35** (genuinely independent). Nasdaq-100 is also the single best standalone sleeve (Sharpe 1.24, shallow drawdown). Blending the low-correlation MidCap+Nasdaq pair gives the best risk-adjusted result of anything tested (Sharpe **1.39**, drawdown **−4.0%**, volatility 4.2% vs the S&P 500's 6.8%). Because that blend is so low-volatility, it can be *scaled up* (our position-size lever) to roughly match the S&P 500's growth while keeping a shallower drawdown — the diversification and exposure-scaling levers compound. Russell 2000 could not be included — its data is only ~29% loaded (a backfill prerequisite). *Caveat: these blend figures are idealized (no safety-gate halts); absolute numbers will come down when run through the live gates, but the correlation-driven risk reduction holds.*

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

---

*Back to [README.md](../README.md).*
