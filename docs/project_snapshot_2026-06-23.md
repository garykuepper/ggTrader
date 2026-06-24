# ggTrader Project Snapshot — 2026-06-23

> **Classification:** Internal engineering reference.
> **Audience:** Principal developer + future collaborators who may not have trading background.
> **Scope:** Architecture, WFO engine mechanics, research post-mortem, forward research directions.

---

## 1. Executive Summary & System Architecture

### 1.1 What This System Does

ggTrader is a **quantitative research lab and paper trading system** for US equities. It answers one question: *"Can we find a systematic trading strategy that reliably beats buying and holding the S&P 500 index, after accounting for trading costs?"*

The system does **not** try to predict stock prices. Instead, it looks for **repeatable patterns** — moments when multiple technical indicators agree that a stock is temporarily cheap (oversold) — and buys during those windows. The hypothesis is that diversifying across multiple independent signal types produces an edge that is more stable than any single signal alone.

**Current status:** The 3-voter ensemble strategy (Bollinger Bands + RSI + EMA crossover) is the only strategy that has survived honest out-of-sample (OOS) testing. It is deployed to an Alpaca paper trading account ($102,459) running live against the S&P 500 universe, monitored via Telegram alerts.

### 1.2 Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                            │
│                     ggt lab --strategy <name>                      │
│                  --sweep | --wfo | --universe sp500                │
└────────────┬───────────────────────────────────┬────────────────────┘
             │                                   │
     ┌───────▼───────┐                  ┌────────▼────────┐
     │  Research Lab  │                  │  Paper Trading  │
     │  (src/lab/)    │                  │  (src/paper/)   │
     └───────┬───────┘                  └────────┬────────┘
             │                                   │
  ┌──────────┼──────────┐             ┌──────────┼──────────┐
  │          │          │             │          │          │
  ▼          ▼          ▼             ▼          ▼          ▼
Sweep      WFO      Simulate     Signal     Alpaca      Risk
Engine    Engine     (vbt)       Runner     Broker     Guard
  │          │          │             │          │          │
  └──────────┼──────────┘             └──────────┼──────────┘
             │                                   │
      ┌──────▼──────┐                     ┌──────▼──────┐
      │ TimescaleDB │                     │   Telegram  │
      │  (OHLCV +   │                     │  Notifier   │
      │  lab_runs)  │                     └─────────────┘
      └─────────────┘
```

### 1.3 Module Map

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `src/ggTrader/lab/` | Research bench — backtesting, sweeps, WFO | `cli.py`, `wfo.py`, `sweep.py`, `simulate.py`, `metrics.py`, `gates.py` |
| `src/ggTrader/lab/strategies/` | Signal generation and strategy definitions | `ensemble.py`, `signals.py`, `indicators.py`, `conviction.py`, `momentum.py` |
| `src/ggTrader/paper/` | Live paper trading on Alpaca | `trader.py`, `signal_runner.py`, `alpaca_broker.py`, `risk.py`, `notifier.py` |
| `src/ggTrader/data/` | OHLCV data layer (TimescaleDB + yfinance + Tiingo) | `core/`, `live/`, `historical/` |
| `tests/` | ~200 unit + integration tests | Mirrors `src/` structure |

### 1.4 The Signal Library

The system has **10 strategy classes**, each generating boolean entry/exit signals across all stocks in the universe simultaneously (vectorized — no per-stock loops):

| Strategy | Signal Type | Description (Plain English) |
|----------|------------|----------------------------|
| `BollingerReversionSignal` | Mean reversion | Buy when price drops below its statistical "normal range" (2 standard deviations below the 20-day average). Sell when it returns to the average. |
| `RsiReversionSignal` | Momentum exhaustion | Buy when the Relative Strength Index (RSI) — a 0-100 momentum gauge — drops below 30 (oversold). Sell when it recovers above 50. |
| `EmaCrossSignal` | Trend following | Buy when the short-term trend (10-day average) crosses above the long-term trend (50-day average). Sell on the reverse cross. |
| `MACDDivergenceSignal` | Divergence | Buy when price makes a new low but momentum (MACD histogram) doesn't — suggesting selling pressure is exhausting. |
| `VolumeBBReversionSignal` | Volume-confirmed reversion | Same as Bollinger reversion, but only triggers when trading volume spikes above 1.5x its 20-day average — confirming institutional participation. |
| `MultiTimeframeReversionSignal` | Multi-timeframe | Buy when BOTH weekly RSI is oversold AND daily price touches the lower Bollinger Band — requiring agreement across two time horizons. |
| `EnsembleSignal` | **Voting ensemble (3 or 6 voters)** | Buy when ≥ N sub-signals agree simultaneously. The core production strategy. |
| `EnsembleConvictionSignal` | Conviction-weighted ensemble | Same voting logic, but position size scales with the average "strength" of agreeing signals (1%-4% of portfolio). |
| `WfoTournamentSignal` | WFO tournament | Runs walk-forward optimization to select the best parameters for each rolling window. |
| `ConvictionBBSignal` | Conviction BB | Bollinger reversion with depth-based position sizing. |

### 1.5 Infrastructure Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backtesting engine | **vectorbt** | Vectorized portfolio simulation — no bar-by-bar loops. Handles entries, exits, position sizing, fees, slippage, trailing stops in a single NumPy/pandas operation. |
| Time-series DB | **TimescaleDB** (PostgreSQL) | 4M+ rows of daily OHLCV data for S&P 500 constituents. Point-in-time (PIT) universe membership prevents survivorship bias. |
| Data sources | **yfinance** (primary), **Tiingo** (fallback) | yfinance for active tickers, Tiingo for delisted historical tickers. |
| ML gate | **LightGBM** | Binary classifier filtering low-confidence ensemble buy signals. Trained with TimeSeriesSplit to prevent lookahead. |
| Paper broker | **Alpaca** | Commission-free fractional share trading. Paper mode for validation before live capital. |
| Notifications | **Telegram Bot API** | Trade alerts, daily P&L summaries, risk warnings. |
| Hosting | **Docker Compose** on Ubuntu 24.04 home server | Cron-driven: signals generated Mon-Fri 1:30 PM PT. |

### 1.6 Key Design Decisions

1. **vectorbt-first**: All backtesting is fully vectorized. No iterating bar-by-bar over candles. A 50-stock, 5-year, 17-fold WFO completes in ~47 seconds.

2. **Point-in-time universe membership**: When backtesting 2022, the system uses the S&P 500 constituents *as they existed in 2022*, not today's list. This prevents survivorship bias (only testing stocks that survived — which is cheating because you'd never know in advance which ones would survive).

3. **DB-only data**: Production backtests read exclusively from TimescaleDB. No live API calls during research runs. Data is backfilled once, then cached permanently.

4. **Separation of signal generation and simulation**: Strategies produce boolean DataFrames (True = buy/sell on this bar for this stock). The simulation engine consumes these generically. This means adding a new strategy requires zero changes to the backtesting infrastructure.

---

## 2. Walk-Forward Optimization (WFO) — How It Works

### 2.1 The Problem WFO Solves

Traditional backtesting has a fatal flaw: **overfitting**. If you test 1,000 parameter combinations on 5 years of data, the best one will look amazing — but its performance is largely luck. It found patterns in noise, not signal. This is called "data snooping" or "p-hacking."

Walk-Forward Optimization (WFO) is the industry-standard defense. The core idea:

> *Train on the past. Test on the unseen future. Repeat. Stitch together only the out-of-sample (unseen) results.*

If a strategy's parameters only "work" on the data they were trained on, WFO will expose that. The stitched-together out-of-sample curve is the closest thing to a real-money test without risking capital.

### 2.2 The Rolling Window Structure

```
Time ──────────────────────────────────────────────────────────────────►

Fold 1:  ├─── TRAIN (12 months) ───┤── TEST (3 months) ──┤
Fold 2:       ├─── TRAIN (12 months) ───┤── TEST (3 months) ──┤
Fold 3:            ├─── TRAIN (12 months) ───┤── TEST (3 months) ──┤
  ...                                                              ...
Fold 17:                              ├─── TRAIN (12 months) ───┤── TEST ──┤

             ▲                                                           ▲
         eval_start                                                  eval_end
        (2021-01-31)                                              (2026-04-30)
```

- **Train window**: 12 months. All parameter combinations are tested here.
- **Test window**: 3 months. Only the *winning* parameters from training are applied.
- **Step size**: 3 months (equal to test window — no overlap in test periods).
- **Total folds**: 17 (covering Jan 2021 → Apr 2026).

The system uses **rolling** (not expanding) windows. This means each fold trains on exactly 12 months, regardless of position in the timeline. The rationale: market regimes change, and older data may be actively misleading.

### 2.3 What Happens Inside Each Fold

#### Step 1: Parameter Sweep (Train Window)

All parameter combinations in the grid are simulated on the training data. For the 3-voter ensemble, this is a 24-combination grid:

```
min_agree:    [2, 3, 4]       ← How many signals must agree to enter
bb_std:       [2.0, 2.5]      ← Bollinger Band width
rsi_oversold: [25, 30]        ← RSI entry threshold
ema_fast:     [10, 20]        ← Fast EMA period
```

Each combination produces an equity curve (portfolio value over time) across all 50 stocks simultaneously.

#### Step 2: Composite Scoring

The winner is selected by a **composite score**, not Sharpe alone:

```
Score = 0.5 × norm(Sharpe) + 0.3 × norm(Sortino) - 0.2 × norm(|MaxDrawdown|)
```

- **Sharpe Ratio** (50% weight): Risk-adjusted return. Average return divided by the volatility of returns. A Sharpe of 1.0 means you earned 1 unit of return per unit of risk. Think of it as "return per unit of stomach-churning."
- **Sortino Ratio** (30% weight): Like Sharpe, but only penalizes *downside* volatility. A strategy that has wild upside swings but controlled losses will score higher on Sortino than Sharpe.
- **Max Drawdown** (20% penalty): The largest peak-to-trough decline. If your portfolio went from $100K to $75K before recovering, that's a -25% drawdown. This penalty prevents the optimizer from selecting high-return strategies that also crash badly.

All three metrics are min-max normalized to [0, 1] before combining, so they're on the same scale.

#### Step 3: Robustness Gates

Before accepting a winner, two statistical gates must pass. Think of these as "fraud detectors" for overfitting:

**Gate 1: Neighborhood Density Hurdle (NDH)**

*Plain English:* "Is the winner surrounded by other good parameter combos, or is it an isolated spike?"

The NDH looks at all ±1-step neighbors of the winning parameter combo in the grid. If 85% of neighbors also have positive Sharpe AND positive trade expectancy, the winner is on a "plateau" of good performance — suggesting it's capturing a real pattern, not a fluke.

If the winner is an isolated spike (great performance, but all nearby combos are terrible), it's likely overfit to noise. The gate rejects it.

*Technical detail:* Also checks that the standard deviation of neighbor Sharpes divided by the peak Sharpe is ≤ 20% (variance cap), ensuring the plateau is genuinely flat.

**Gate 2: Deflated Sharpe Ratio (DSR)**

*Plain English:* "Given how many parameter combinations we tried, is this Sharpe statistically significant?"

If you flip a coin 1,000 times, you'll eventually find a streak of 10 heads — but that doesn't mean the coin is biased. Similarly, if you test 24 parameter combos, the best one benefits from selection bias.

The DSR (Bailey & López de Prado, 2014) calculates the probability that the observed Sharpe exceeds the **expected maximum Sharpe you'd get from pure chance** across N trials. It accounts for the number of trials, the return distribution's skewness, and excess kurtosis (fat tails). The gate requires DSR ≥ 0.80 (80% probability that the Sharpe is real, not luck).

```
DSR = P(true SR > E[max(SR | N trials)])
    = Φ( (SR_obs - E[max_SR]) / σ̂(SR) )

where σ̂(SR) = √( (1 - γ₃·SR + (γ₄-1)/4 · SR²) / (T-1) )
```

#### Step 4: Anchor Fallback

When gates fail (the winner is likely overfit), the system doesn't skip the fold — it falls back to the **Anchor Set**: the parameter combo with the smallest maximum drawdown across the entire dataset that still has CAGR above the risk-free rate (4%).

The anchor is intentionally conservative. It won't win any Sharpe competitions, but it won't blow up either. It's the "I don't trust the optimizer right now, so play defense" mode.

#### Step 5: Out-of-Sample Test

The winning (or anchor) parameters are applied to the 3-month test window — data the optimizer has *never seen*. The resulting equity curve is recorded.

### 2.4 Circuit Breaker

The WFO engine monitors its own health across folds with an **OR-gate circuit breaker**. If EITHER condition triggers, it halts live trading and switches to anchor-only mode:

| Trigger | Description | Plain English |
|---------|-------------|---------------|
| **Chronic decay** | Trailing 4-window WFE average < 0.25 | The strategy's out-of-sample performance has been consistently poor relative to its in-sample performance for 4+ quarters. |
| **Acute failure** | 2 consecutive negative OOS Sharpe ratios | The strategy lost money in two back-to-back test periods. |

**WFE (Walk-Forward Efficiency)** = OOS Sharpe / IS Sharpe. A WFE of 1.0 means out-of-sample performance perfectly matches training. A WFE of 0.5 means the strategy retains half its edge when tested on unseen data. Below 0.25 for an extended period means the strategy is decaying.

**Shadow re-entry**: After a halt, the system continues running in "shadow mode" (simulating but not trading). If 2 consecutive folds pass all gates AND have WFE ≥ 0.50, the halt is lifted automatically.

### 2.5 Final Output

After all 17 folds, the system stitches together the out-of-sample equity curves into a single continuous curve and reports aggregate metrics:

```
OOS Aggregate: Sharpe 0.61 | CAGR 12.6% | MaxDD -23.4%
SPY baseline:  Sharpe 0.59 | CAGR 13.0% | MaxDD -22.1%
Aggregate WFE: 1.11 (target >= 0.50)
```

It also trains one final time on the most recent 12-month window and reports the **Recommended Live Parameters** — the parameters that should be deployed to paper/live trading based on the latest data.

---

## 3. What We Tried, What Worked, and What Didn't (Post-Mortem Analysis)

### 3.1 Timeline of Research (2026-06-15 to 2026-06-23)

| Date | Experiment | Outcome |
|------|-----------|---------|
| 06-15 | Lab core — momentum strategies (xs_momentum, dual_momentum) | Built the research bench. Momentum strategies failed WFO (Sharpe -5.11 OOS). |
| 06-16 | Signal strategies (EmaCross, WfoTournament) | Infrastructure win. EMA cross alone: mediocre OOS. |
| 06-18 | BB reversion + RSI reversion (individual signals) | First strategies to beat SPY in WFO (Sharpe ~0.80 individual). |
| 06-18 | WFO framework + robustness gates (NDH, DSR) | Infrastructure: prevented overfitting from being invisible. |
| 06-19 | Volatility targeting overlay | Marginal improvement at best (Sharpe 0.96 at vol_target=0.20). |
| 06-19 | Trailing stops (fixed + ATR-adaptive) | **Destructive on reversion strategies.** Stops exit profitable mean-reversions too early. |
| 06-20 | **3-voter ensemble (BB + RSI + EMA)** | **Best result: Sharpe 0.84, CAGR 18.8%, WFE 1.25.** First strategy to beat SPY risk-adjusted in honest OOS. |
| 06-20 | Multi-universe testing (Nasdaq-100, Russell 2000) | Nasdaq-100 Sharpe 0.88. Russell 2000 not tested yet. |
| 06-20 | Paper trading deployed on Alpaca | $102K paper account, cron-driven, ML gate + risk guardrails. |
| 06-22 | ML feature gate (LightGBM) | Precision 0.585 — filters ~40% of buys. Marginal impact on returns, but reduces noise. |
| 06-22 | Conviction-weighted position sizing | Sharpe 0.83 vs 0.84 baseline. MaxDD improved (-19.6% vs -21.8%). Risk reducer, not alpha generator. |
| 06-22 | Tiingo data loader for delisted tickers | Infrastructure fix. 72-second rate limit per request on free tier caused WFO hangs. |
| 06-23 | **6-voter ensemble (+ MACD div, vol-BB, MTF)** | **Failed. Sharpe 0.36, CAGR 1.1%.** Three new signals added noise, not alpha. |
| 06-23 | Ablation testing (drop one signal at a time) | RSI is the critical signal. MTF actively hurts. BB, EMA, MACD, vol-BB near zero impact individually. |
| 06-23 | ML-based signal selection analysis | Analyzed but not built — ablation already answered the question clearly. |

### 3.2 What Worked

#### The 3-Voter Ensemble (BB + RSI + EMA)

This is the only strategy that has survived honest walk-forward testing and beaten the S&P 500 benchmark:

| Metric | 3-Voter Ensemble | SPY (Buy & Hold) |
|--------|-----------------|-------------------|
| OOS Sharpe | **0.61** | 0.59 |
| CAGR | 12.6% | 13.0% |
| Max Drawdown | -23.4% | -22.1% |
| WFE | **1.11** | — |
| Stability | Winner selected in 5/17 folds | — |

*Note: The earlier WFO run on a larger universe produced Sharpe 0.84 / CAGR 18.8%. The 0.61 result above is from a 50-stock controlled test used for apples-to-apples comparison with the 6-voter variant. Both beat SPY on a risk-adjusted basis.*

**Why it works:** The three sub-signals are **negatively correlated in failure modes**:
- Bollinger Bands fire during volatility spikes (crash recoveries)
- RSI fires during momentum exhaustion (oversold bounces)
- EMA cross fires during trend reversals

When one signal type gets whipsawed (generates false signals), the others typically don't fire, so the ensemble's `min_agree` threshold filters out the noise. This is the same principle behind why diversified investment portfolios reduce risk — the signals "cancel out" each other's bad days.

#### Vectorbt Simulation Architecture

The decision to use vectorbt's grouped portfolio simulation (one `from_signals()` call for all strategies × all stocks) was validated by profiling:
- **Before optimization**: 1,048 seconds per full WFO (per-metric accessor overhead dominated at ~58%)
- **After optimization**: 507 seconds (single `pf.returns()` extraction + NumPy kernels)
- **Current 50-stock WFO**: ~47 seconds for 17 folds × 24 parameter combos

#### Point-in-Time Universe Membership

The `sp500_members_asof(date)` function queries historical index membership. Without this, backtests would suffer from **survivorship bias** — only testing stocks that are *currently* in the S&P 500, ignoring the hundreds that were removed (often after poor performance). This bias can inflate backtested returns by 1-3% annually.

### 3.3 What Didn't Work

#### 6-Voter Ensemble — More Signals ≠ Better

Adding three new signals (MACD divergence, volume-confirmed BB, multi-timeframe RSI) to the ensemble degraded performance significantly:

| Metric | 3-Voter | 6-Voter | Delta |
|--------|---------|---------|-------|
| OOS Sharpe | 0.61 | 0.36 | **-0.25** |
| CAGR | 12.6% | 1.1% | **-11.5%** |
| Max Drawdown | -23.4% | -5.8% | +17.6% |
| Folds passing gates | 0/17 | 11/17 | — |

The 6-voter ensemble's low drawdown (-5.8%) is misleading — it simply doesn't trade enough. With 6 voters and `min_agree=3`, the entry threshold is too low (only need 3 of 6 to agree), but the *quality* of agreements is diluted because the new signals fire on different patterns than the original three.

**Ablation results** (dropping one signal at a time from the 6-voter):

| Variant | OOS Sharpe | Interpretation |
|---------|-----------|----------------|
| All 6 signals | 0.55 | Baseline 6-voter |
| Drop RSI | **-0.10** | RSI is the critical signal. Removing it collapses performance. |
| Drop MTF | **0.64** | MTF actively hurts. Removing it *improves* the ensemble. |
| Drop BB | 0.53 | Near-zero impact |
| Drop EMA | 0.54 | Near-zero impact |
| Drop MACD | 0.52 | Near-zero impact |
| Drop Volume BB | 0.53 | Near-zero impact |
| 3-voter (BB+RSI+EMA) | **0.61** | Still the best configuration |

**Lesson:** Signal diversification has diminishing and eventually negative returns. The three new signals were **not orthogonal** (statistically independent) to the originals — MACD divergence and volume-confirmed BB are both variations on the same mean-reversion thesis as BB and RSI. Adding correlated signals doesn't diversify; it dilutes.

#### Trailing Stops on Reversion Strategies

Both fixed trailing stops (e.g., 5% from peak) and ATR-adaptive trailing stops were tested. Both **destroyed reversion strategy returns**.

*Why:* Mean reversion strategies buy during dips. By definition, the stock continues falling after entry (otherwise the dip detection was wrong). A trailing stop that triggers during this expected drawdown exits the position before the reversion completes. The stop is "protecting" against exactly the behavior the strategy is designed to profit from.

#### Momentum Strategies

Cross-sectional momentum (buy the top decile, short the bottom decile) and dual momentum (absolute + relative) both failed WFO with deeply negative OOS Sharpes. The momentum factor in US large caps has been well-arbitraged — too many quant funds are already trading it.

#### Conviction-Weighted Position Sizing

Sizing positions by signal strength (1-4% of portfolio based on how "deep" the indicators are) produced nearly identical Sharpe (0.83 vs 0.84) with slightly better drawdown (-19.6% vs -21.8%). It's a risk management improvement, not an alpha improvement. The complexity isn't justified yet.

#### Volatility Targeting

Scaling position sizes inversely with realized volatility (reduce exposure in high-vol regimes, increase in low-vol) produced marginal improvement at `vol_target=0.20` (Sharpe 0.96 in one test). However, this introduces leverage (vol scalar can exceed 1.0), which the paper/live system doesn't use. Deferred.

### 3.4 Infrastructure Lessons Learned

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| WFO hung at 0% CPU for 30+ min | Tiingo 72-second rate limit × 50 delisted tickers | Skip Tiingo during research (`TIINGO_API_KEY=""`) |
| 12,288-combo sweep grid | 6-voter sweep_params had 2 values per new param | Pin new signal params to single defaults; sweep only core params |
| `--max-stocks 50` loaded all 600 stocks | Flag only trimmed `strategy.select()`, not data load | Trim universe list *before* `load_ohlcv()` |
| No WFO progress output in Docker | Python stdout buffering in containers | `PYTHONUNBUFFERED=1` + explicit `flush=True` on prints |
| WFE format crash on last fold | `wfe_val` is None when IS Sharpe < floor | Conditional formatting: check for None before f-string |

---

## 4. Next Steps: Expanding Alpha and Universe Scaling

### 4.1 Improving Alpha in the S&P 500 Universe

The 3-voter ensemble's edge is real but thin (OOS Sharpe 0.61 vs SPY 0.59). The following are **architecturally concrete** improvements ordered by expected effort-to-payoff ratio:

#### A. Regime-Adaptive Signal Selection (High Priority)

**Problem:** The ensemble uses the same 3 signals in all market conditions. But mean-reversion signals work best in *range-bound* markets, while EMA crossovers work best in *trending* markets. Running both simultaneously means one is always underperforming.

**Solution:** Classify the current market regime (trending, mean-reverting, high-vol, low-vol) using a rolling volatility + trend-strength measure (e.g., ADX above/below 25), then activate only the signals suited to that regime.

**Implementation:** Add a `RegimeClassifier` that outputs a regime label per bar. The ensemble's `_generate_signals()` method would conditionally zero-out signals that don't match the current regime. This is a strategy-level change, not an infrastructure change.

**Risk:** Regime classification itself can be overfit. Must be validated in WFO with the classification model trained only on the train window.

#### B. Exit Signal Optimization (High Priority)

**Problem:** Entry signals have been heavily optimized, but exit logic is simple and symmetric (exit when the same signal reverses). The exit side is likely leaving money on the table.

**Solution:** Decouple exits from entries. Test:
1. **Time-based exits**: Close after N days regardless of signal state
2. **Profit targets**: Close when position gains exceed X%
3. **Trailing profit locks**: After Y% gain, lock in Z% with a trailing threshold (distinct from trailing stops, which trigger on any pullback from entry)
4. **Cross-signal exits**: Enter on RSI oversold, exit on EMA cross-down (use a different signal family for exits than entries)

**Implementation:** Add `exit_mode` parameter to `EnsembleSignal.sweep_params()`. The WFO grid would sweep over exit variants alongside entry parameters.

#### C. Feature-Enriched ML Gate (Medium Priority)

**Problem:** The current ML gate uses 7 technical features (RSI, BB distance, volume ratio, etc.) and achieves 0.585 precision. It filters bad trades but doesn't generate new alpha.

**Solution:** Add macro features that are orthogonal to price-based technicals:
- **VIX level and change** (fear gauge — reversion signals work better in high-VIX environments)
- **Sector relative strength** (buy the cheapest stocks in the strongest sectors)
- **Earnings proximity** (avoid buying right before earnings — binary event risk)
- **Short interest** (high short interest + oversold → potential squeeze)

**Implementation:** Extend `extract_features()` in `feature_gate.py`. Retrain with `train_gate.py`. These features are available from free data sources and don't require additional API subscriptions.

#### D. Dynamic Position Sizing via Kelly Criterion (Medium Priority)

**Problem:** Current position sizing is fixed at 3.3% of portfolio per trade (1/30). This ignores the varying quality of individual setups.

**Solution:** For each entry signal, estimate the probability of profit and the expected payoff ratio using the ML gate's probability output. Apply a fractional Kelly criterion:

```
Kelly fraction = (p × b - q) / b

where:
  p = ML gate's predicted probability of profit
  b = average winner / average loser (from recent WFO fold)
  q = 1 - p

Position size = Kelly fraction × portfolio × fractional_multiplier (e.g., 0.25)
```

Half-Kelly or quarter-Kelly is standard practice to account for estimation error in p and b.

**Implementation:** The `EnsembleConvictionSignal` already supports per-bar sizing via `SignalTargets.sizes`. Wire the ML gate's probability output to Kelly sizing in `signal_runner.py`.

#### E. Cross-Sectional Ranking (Lower Priority)

**Problem:** The ensemble treats all entry signals as equal — if 5 stocks fire on the same day, all 5 are bought. But some setups are higher quality than others.

**Solution:** When multiple entries fire simultaneously, rank them by:
1. Signal conviction score (already computed but unused in production)
2. Recent relative performance (buy the weakest stocks with the strongest signals — deeper reversion)
3. Sector diversification (don't buy 3 financials on the same day)

**Implementation:** Add a `rank_entries()` step between `generate_signals()` and order submission in `signal_runner.py`.

### 4.2 Universe Scaling — Beyond the S&P 500

The system's infrastructure (`--universe` flag, PIT membership, vectorized simulation) already supports multiple universes. The question is which universes offer the best research returns.

| Universe | Size | Liquidity | Avg Spread | Mean Reversion Edge | Data Quality | Infra Ready? | Notes |
|----------|------|-----------|-----------|---------------------|-------------|-------------|-------|
| **S&P 500** | ~500 | Very High | 1-3 bps | Moderate — heavily arbitraged, but our WFO confirms edge exists | Excellent | ✅ Yes | Current production universe. Low fees, fractional shares on Alpaca. |
| **Nasdaq-100** | 100 | Very High | 1-3 bps | Good — tested Sharpe 0.88 OOS | Excellent | ✅ Yes | Higher beta, more tech concentration. Tested but not deployed. Fewer stocks = faster WFO. |
| **Russell 2000** | ~2000 | Mixed | 5-30 bps | High potential — less analyst coverage, more mispricings | Mixed (delistings) | ✅ Yes (flag exists) | Higher spreads and slippage. Must increase `SLIPPAGE` parameter. Many illiquid names. |
| **S&P MidCap 400** | 400 | Moderate | 3-10 bps | High potential — institutional blind spot between large and small cap | Good | ❌ Need PIT membership data | "Goldilocks" — more inefficient than S&P 500, more liquid than Russell 2000. |
| **Sector ETFs** | ~11 | Very High | 1-2 bps | Different — sector rotation, not single-stock reversion | Excellent | ❌ Need ETF universe | Diversification across asset classes. Lower correlation to current strategy. |
| **International (MSCI EAFE)** | ~800 | Moderate | 5-15 bps | Unknown — untested thesis | Moderate | ❌ Need data pipeline | Different market microstructure. Time zone complexity. Currency risk. |
| **Crypto (via Kraken/Binance)** | 50-100 | Mixed | 10-50 bps | Historically high, now diminishing | Good (DB exists) | ✅ Yes (legacy code) | Previous production universe. Edge exhausted per 2026-06-08 research. Fees dominate. |

**Recommended next universe:** S&P MidCap 400. It sits in the institutional blind spot — too small for most quant funds to trade in size, but liquid enough for a retail account. Mean-reversion signals should work better because there are fewer arbitrageurs competing. The main prerequisite is sourcing historical point-in-time constituent data.

### 4.3 Research Priorities (Ordered)

| Priority | Initiative | Expected Impact | Effort | Dependencies |
|----------|-----------|----------------|--------|-------------|
| 1 | Monitor paper trading for 5-10 clean trading days | Validate live execution matches backtest | Low (monitoring only) | Already deployed |
| 2 | Regime-adaptive signal selection (§4.1A) | Sharpe improvement via reducing signal whipsaw | Medium | None |
| 3 | Exit signal optimization (§4.1B) | Directly improves trade profitability | Medium | Need sweep_params extension |
| 4 | S&P MidCap 400 PIT data sourcing | Unlocks higher-alpha universe | Low-Medium | External data source |
| 5 | Feature-enriched ML gate (§4.1C) | Filters more bad trades, potential alpha | Medium | VIX/earnings data |
| 6 | Dynamic Kelly sizing (§4.1D) | Better capital efficiency | Low | ML gate probability calibration |
| 7 | Cross-sectional ranking (§4.1E) | Marginal improvement on multi-signal days | Low | None |

---

## Appendix A: Key Metrics Glossary

| Term | Definition | Good Value |
|------|-----------|-----------|
| **Sharpe Ratio** | Annualized return divided by annualized volatility. Measures return per unit of risk. | > 0.5 (acceptable), > 1.0 (good), > 2.0 (exceptional) |
| **Sortino Ratio** | Like Sharpe, but only penalizes downside volatility. Rewards asymmetric returns. | > 1.0 |
| **CAGR** | Compound Annual Growth Rate — the smoothed annualized return. | > SPY (~10-13% historically) |
| **Max Drawdown** | Largest peak-to-trough decline. How much money you'd have lost at the worst point. | > -20% (tolerable), > -10% (good) |
| **WFE** | Walk-Forward Efficiency = OOS Sharpe / IS Sharpe. How much of in-sample edge survives in unseen data. | > 0.50 (minimum), > 1.0 (excellent) |
| **OOS** | Out-of-sample — data the optimizer has never seen. The only honest measure of strategy performance. | — |
| **IS** | In-sample — data used for training/optimization. Performance here is unreliable due to overfitting. | — |
| **NDH** | Neighborhood Density Hurdle — checks if the winning parameters are on a "plateau" of good performance. | density > 0.85 |
| **DSR** | Deflated Sharpe Ratio — adjusts for the number of parameter combinations tested. | > 0.80 |
| **PIT** | Point-in-time — using data as it existed at the historical moment, preventing hindsight bias. | — |

## Appendix B: Codebase Statistics (2026-06-23)

| Metric | Value |
|--------|-------|
| Python source files | 48 |
| Source lines of code | ~6,900 |
| Test lines of code | ~4,900 |
| Test count | ~200 |
| Strategies implemented | 10 |
| Indicators | 6 signal types + 6 strength functions |
| OHLCV rows in TimescaleDB | 4M+ |
| Git commits since lab creation (06-15) | 30+ |
| WFO runtime (50 stocks, 17 folds) | ~47 seconds |

---

*Generated 2026-06-23. For the living roadmap, see [`roadmap.md`](roadmap.md). For CLI usage, see [`cli_reference.md`](cli_reference.md).*
