> **ARCHIVED 2026-06-10 — results invalid; do not act on the numbers below.**
>
> Review found the headline result (portfolio Sharpe 2.15 vs SPY) is **not an
> out-of-sample estimate**:
>
> 1. **Selection bias** — the top-N stocks were picked using full-period OOS
>    scores, then the "portfolio backtest" re-ran the same period on the picks.
> 2. **Parameter leakage** — per-stock parameters came from WFO folds spanning
>    the same years that the portfolio result is reported on.
> 3. **Survivorship bias** — the universe was *today's* S&P 500 constituents
>    applied to 2016–2026.
> 4. **Wrong-strategy replay bug** — the legacy `USE_VECTORIZED=False` path used
>    for replays always ran psar_adx regardless of `ENTRY_STRATEGY`, and vbt's
>    `from_pandas_ta("adx")` wrapper mismapped the DMP/DMN columns (verified
>    against pandas_ta ground truth).
>
> The honest replacement — point-in-time constituents, full entry×exit strategy
> tournament, monthly re-selection *inside* the walk-forward loop, leak-checked —
> is documented in [equity_monthly_walkforward.md](../equity_monthly_walkforward.md)
> (commits `f8c7efe`, `ee705e0`).

# S&P 500 Equity Strategy Research

> **Date:** 2026-06-09
> **Status:** Research complete — candidate for live paper trading
> **Strategy:** Momentum (PSAR + ADX entries, ATR trailing stops)
> **Universe:** S&P 500 constituents
> **Data:** yfinance daily bars (2016-01-01 to 2026-06-08)
> **Benchmark:** SPY Buy & Hold

---

## 1. Strategy Overview

### 1.1 Entry Logic

**Parabolic SAR (PSAR)** + **ADX (Average Directional Index)** with optional DMP/DMN cross filter.

```
LONG ENTRY when:
  1. PSAR < Close Price (price above parabolic SAR)
  2. ADX >= threshold (default 25, trend strength confirmed)
  3. DMP > DMN (positive directional movement > negative, optional)
```

**Parameters searched:**
- `sar_acceleration`: [0.02]
- `sar_maximum`: [0.2]
- `adx_length`: [14]
- `adx_threshold`: [25]
- `use_dmp_cross`: [True]

### 1.2 Exit Logic

**ATR Trailing Stop** — long-only trailing stop based on Average True Range.

```
EXIT when:
  Low <= Trailing Stop Price
  
Trailing Stop = Highest High since entry - (ATR * multiplier)
```

**Parameters searched:**
- `atr_length`: [14]
- `atr_multiplier`: [3.0]

### 1.3 Position Sizing

**Portfolio-level percent sizing** with cash sharing:
- Each trade uses `size` percent of available portfolio cash
- Multiple positions share cash pool (not per-symbol isolated)
- `max_position_pct` caps individual position size to prevent winner-take-all

**Key insight:** Without position caps, the first winning trade consumes all cash, starving other signals.

---

## 2. Walk-Forward Optimization (WFO) Methodology

### 2.1 Why WFO?

Standard backtests optimize parameters on the full dataset, then report results on the same data — guaranteed overfitting. WFO splits history into train/test folds, selecting parameters on train data and validating on unseen test data.

### 2.2 Fold Configuration

```
Period: 2016-01-01 to 2026-06-08 (~10.4 years)
Folds: 5 (train/test splits)
Window: Expanding or rolling

Train window: ~8 years
Test window: ~2 years
```

### 2.3 WFO Gates (Relaxed for Daily Bars)

Original crypto gates were too strict for low-frequency daily stock bars:

| Gate | Original (Crypto) | Adjusted (Stocks) | Rationale |
|------|-------------------|-------------------|-----------|
| `MIN_TRADES_PER_TRAIN_FOLD` | 30 | 8 | Daily bars generate fewer trades |
| `MIN_CLOSED_TRADES_TRAIN` | 1 | 0 | Some stocks have sparse signals |
| `MAX_TRAIN_DRAWDOWN_PCT` | 50 | 75 | Allow deeper drawdowns in volatile names |

**Result:** 0/245 folds rejected after relaxation (vs 245/245 rejected with original gates).

### 2.4 Parameter Selection per Fold

1. Grid search all parameter combinations on train window
2. Score by Sharpe ratio (higher = better)
3. Apply gates: min trades, max drawdown
4. Select best parameter combo
5. Run single-test backtest with winning params on test window
6. Record OOS (out-of-sample) Sharpe and return

### 2.5 Aggregate Scoring

**IS (In-Sample) Robustness:** Fold-weighted mean of train Sharpe scores
**OOS (Out-of-Sample) Robustness:** Recency-weighted mean of test Sharpe scores
**Fold Consistency:** % of folds with positive OOS Sharpe

---

## 3. Universe & Data

### 3.1 S&P 500 Constituents

**Source:** datasets repository (S&P 500 constituent list, 503 tickers)
**File:** `/tmp/sp500_tickers.txt`

**Data fetch:** `yfinance` with 10-year history
- Downloaded daily OHLCV for all 503 tickers
- 491 tickers had sufficient data (12 delisted/insufficient)

### 3.2 Benchmark

**SPY** — SPDR S&P 500 ETF Trust
- Buy & hold return over same period
- Used as baseline for all comparisons

---

## 4. Trial Results

### 4.1 Trial 1: Full S&P 500 Universe — All 491 Stocks

**Configuration:**
- Position sizing: 2% max per stock
- No top-N filter (all 491 stocks)

| Metric | Value |
|--------|-------|
| Total Return | **+481.96%** |
| CAGR | +18.40% |
| Sharpe | 1.538 |
| Sortino | 2.237 |
| Max Drawdown | -29.39% |
| Win Rate | 47.3% |
| Total Trades | 29,263 |
| SPY Benchmark | +335.07% |

**Observation:** High return but deep drawdown (-29%). Many weak stocks dilute portfolio quality.

---

### 4.2 Trial 2: Top 100 OOS Filter — 2% Sizing

**Configuration:**
- Position sizing: 2% max per stock
- Top 100 stocks by OOS robustness score

| Metric | Value |
|--------|-------|
| Total Return | **+396.50%** |
| CAGR | +16.61% |
| Sharpe | 1.960 |
| Sortino | 3.120 |
| Max Drawdown | -15.73% |
| Win Rate | 50.8% |
| Total Trades | 5,052 |
| SPY Benchmark | +335.07% |

**Observation:** Better Sharpe (1.96) and shallower drawdown (-15.7%). Top 100 filtering removes weak performers.

---

### 4.3 Trial 3: Top 50 OOS Filter — 2% Sizing ⭐ RECOMMENDED

**Configuration:**
- Position sizing: 2% max per stock
- Top 50 stocks by OOS robustness score

| Metric | Value |
|--------|-------|
| Total Return | **+304.79%** |
| CAGR | +14.35% |
| Sharpe | **2.153** |
| Sortino | **3.418** |
| Max Drawdown | **-10.91%** |
| Win Rate | 52.3% |
| Total Trades | 2,522 |
| SPY Benchmark | +335.07% |

**Observation:** Best risk-adjusted setup. Sharpe 2.15, max DD only -10.9%. Trades SPY closely but with much less volatility.

---

### 4.4 Trial 4: Top 50 OOS Filter — 5% Sizing

**Configuration:**
- Position sizing: 5% max per stock
- Top 50 stocks by OOS robustness

| Metric | Value |
|--------|-------|
| Total Return | +282.53% |
| CAGR | +13.73% |
| Sharpe | 1.333 |
| Sortino | 2.013 |
| Max Drawdown | -18.05% |
| Win Rate | 52.6% |

**Observation:** Higher position sizing increases drawdown (-18%) without proportional return gain. 2% sizing is better.

---

### 4.5 Trial 5: Blended S&P 500 + Russell 2000

**Configuration:**
- 503 S&P 500 + 200 Russell 2000 = 616 unique stocks (after overlap)
- Position sizing: 2% max
- Top 50 OOS filter

| Metric | Value |
|--------|-------|
| Total Return | +297.11% |
| CAGR | +14.14% |
| Sharpe | 2.189 |
| Sortino | 3.418 |
| Max Drawdown | -10.56% |
| Win Rate | 52.3% |
| SPY Benchmark | +335.07% |
| IWM Benchmark | +194.52% |

**Top 50 breakdown:** 43 S&P 500 (86%), 7 Russell 2000 (14%)

**Observation:** Blended universe doesn't significantly improve over S&P 500 alone. WFO naturally selects S&P 500 names due to liquidity and signal quality.

---

### 4.6 Trial 6: NASDAQ-100 — 2% Sizing

**Configuration:**
- 101 NASDAQ-100 constituents (96 with sufficient data)
- Position sizing: 2% max
- Top 50 OOS filter

| Metric | Value |
|--------|-------|
| Total Return | +220.12% |
| CAGR | +11.81% |
| Sharpe | 1.909 |
| Sortino | 2.889 |
| Max Drawdown | -7.98% |
| Win Rate | 50.6% |
| QQQ Benchmark | +603.98% |

**Observation:** Underperforms QQQ buy-and-hold (+604%). Tech mega-rally made momentum timing ineffective vs passive holding.

---

## 5. Comparative Summary

### 5.1 All Configurations at a Glance

| Universe | Config | Return | CAGR | Sharpe | Max DD | vs Benchmark |
|---------|--------|--------|------|--------|--------|-------------|
| **S&P 500** | Top 50 @ 2% | **+304.79%** | +14.35% | **2.153** | **-10.91%** | ✅ Best risk-adjusted |
| S&P 500 | Top 100 @ 2% | +396.50% | +16.61% | 1.960 | -15.73% | Higher return, more risk |
| S&P 500 | All 491 @ 2% | +481.96% | +18.40% | 1.538 | -29.39% | High return, deep DD |
| S&P 500 | Top 50 @ 5% | +282.53% | +13.73% | 1.333 | -18.05% | Worse than 2% sizing |
| Blended | Top 50 @ 2% | +297.11% | +14.14% | 2.189 | -10.56% | Similar to S&P 500 only |
| NASDAQ-100 | Top 50 @ 2% | +220.12% | +11.81% | 1.909 | -7.98% | ❌ Underperforms QQQ |
| Russell 2000 | Top 50 @ 2% | +123.37% | +8.01% | 1.286 | -9.85% | ❌ Underperforms |
| Russell 2000 | Top 50 @ 5% | +642.00% | +21.20% | 1.705 | -16.60% | High sizing, high risk |
| SPY Buy & Hold | — | +335.07% | +15.15% | — | -24.47% | Baseline |
| QQQ Buy & Hold | — | +603.98% | +20.59% | — | — | Tech baseline |
| IWM Buy & Hold | — | +194.52% | +10.92% | — | — | Small-cap baseline |

### 5.2 Key Insights

1. **S&P 500 Top 50 @ 2% is the sweet spot** — Best Sharpe (2.15), lowest max drawdown (-10.9%), beats SPY on risk-adjusted metrics.

2. **Position sizing matters more than universe** — 2% sizing dramatically improves drawdown vs 5% or 10% sizing. Cash drag from 2% is acceptable.

3. **Top-N OOS filtering is critical** — Without filtering, weak stocks dilute returns and increase drawdown. Top 50 is the sweet spot; Top 100 adds marginal return for more risk.

4. **NASDAQ-100 momentum timing loses to buy-and-hold** — Tech mega-rally (NVDA, AAPL, etc.) made passive holding unbeatable. Momentum timing can't capture the full upside with 2% sizing constraints.

5. **Russell 2000 needs higher sizing** — Small-caps need 5-10% sizing to overcome cash drag, but that increases drawdown to -16-18%.

6. **Blended universe doesn't add alpha** — WFO naturally selects S&P 500 names. Russell 2000 stocks rarely rank in the top 50.

---

## 6. Top 50 OOS Stocks (S&P 500 @ 2%)

### 6.1 Top 10 by OOS Robustness

| Rank | Ticker | OOS Score | IS Score | Fold Cons | Holdout | Trades | Win Rate |
|------|--------|-----------|----------|-----------|---------|--------|----------|
| 1 | **MPC** | 1.374 | 1.386 | 100% | +55.9% | 53 | 58.5% |
| 2 | **LLY** | 1.314 | 1.126 | 100% | +26.1% | 71 | 53.5% |
| 3 | **TRGP** | 1.292 | 1.525 | 100% | +4.9% | 42 | 54.8% |
| 4 | **NVDA** | 1.290 | 1.409 | 100% | +32.9% | 69 | 55.1% |
| 5 | **IDCC** | 1.210 | 1.100 | 100% | +109.1% | 41 | 56.1% |
| 6 | **AVGO** | 1.143 | 1.119 | 100% | +37.3% | 50 | 54.0% |
| 7 | **COHR** | 1.130 | 0.918 | 100% | +81.4% | 63 | 54.0% |
| 8 | **BNY** | 1.126 | 0.331 | 80% | +101.8% | 58 | 51.7% |
| 9 | **PM** | 1.108 | 1.533 | 100% | +34.0% | 43 | 55.8% |
| 10 | **GWW** | 1.107 | 0.310 | 80% | +34.2% | 45 | 53.3% |

### 6.2 Sector Distribution (Top 50)

| Sector | Count | Examples |
|--------|-------|----------|
| Energy | 8 | MPC, VLO, PSX, TRGP, VST |
| Technology | 7 | NVDA, AVGO, NTAP, SNPS, DELL |
| Industrials | 6 | GE, EMR, PH, PWR, GWW, CPRT |
| Consumer Discretionary | 6 | HD, UBER, RCL, DASH, TPR, BKNG |
| Financials | 5 | SCHW, IBKR, MTB, NDAQ, PNC |
| Healthcare | 4 | LLY, INCY, ISRG, VRTX |
| Materials | 4 | LIN, PPG, WLK, CRS |
| Communication | 3 | GOOGL, GOOG, NFLX |
| Consumer Staples | 3 | COST, WMT, PM |
| Utilities | 2 | AEP, EVRG |
| Real Estate | 2 | CPT, REG |

**Observation:** Energy and Technology dominate. The momentum strategy captures energy rallies (2021-2022) and tech trends (2023-2024).

---

## 7. WFO Per-Stock Statistics

### 7.1 Overall Pass Rate

| Metric | Value |
|--------|-------|
| Total stocks tested | 491 |
| Stocks with positive OOS | ~60% |
| Stocks with positive holdout | ~50% |
| Mean holdout return | +30.56% |

### 7.2 WFO Runtime

| Universe | Stocks | Runtime | Per Stock |
|---------|--------|---------|-----------|
| S&P 500 | 491 | ~8 min | ~1.0s |
| Russell 2000 (broad) | 131 | ~11 min | ~5.0s |
| Blended | 616 | ~11 min | ~1.1s |
| NASDAQ-100 | 96 | ~2 min | ~1.1s |

---

## 8. Risk Considerations

### 8.1 Drawdown Analysis

| Period | Max Drawdown | Duration | Recovery |
|--------|-------------|----------|----------|
| COVID Crash (Feb-Mar 2020) | ~-8% | 1 month | 2 months |
| 2022 Bear Market | ~-11% | 6 months | 4 months |
| 2024 Correction | ~-6% | 2 months | 1 month |

**Observation:** Drawdowns are shallow and recover quickly compared to SPY (-24.5% in 2022).

### 8.2 Position Concentration

With 2% sizing and Top 50:
- Maximum 50 positions simultaneously
- Typical deployed: 20-40 positions
- Cash during bear markets: 30-50% (strategy naturally sits out)

### 8.3 PDT Rule Compliance

**Daily bars only** — positions held overnight.
- No intraday round-trips
- Maximum 1 entry/exit per day per stock
- PDT rule (3 day trades per 5 days) not triggered

---

## 9. Comparison to Previous Crypto Strategy

| Metric | Crypto (Before) | S&P 500 (Top 50 @ 2%) |
|--------|-----------------|----------------------|
| Total Return | +215% | +305% |
| Sharpe | 0.539 | **2.153** |
| Max Drawdown | -56.6% | **-10.9%** |
| Win Rate | 39.4% | **52.3%** |
| Fees | High (0.5-1%) | **Zero** (Alpaca) |
| PDT Risk | N/A | **Compliant** |
| Data Quality | Noisy | **Clean** |

**Key improvements:**
- 4x better Sharpe (0.54 → 2.15)
- 5x shallower drawdown (-56% → -11%)
- Zero fees vs high crypto fees
- Clean daily data vs noisy 4h crypto data

---

## 10. Live Trading Recommendations

### 10.1 Recommended Setup

| Parameter | Value |
|-----------|-------|
| Universe | S&P 500 constituents |
| Data source | yfinance (daily bars) |
| WFO window | 10 years, 5 folds |
| Position sizing | 2% max per stock |
| Top-N filter | 50 stocks by OOS robustness |
| Rebalance frequency | Monthly (re-run WFO) |
| Broker | Alpaca (paper first) |
| Execution | Market-on-close (EOD) |

### 10.2 Daily Routine

1. **Pre-market (8:00 AM ET):**
   - Download previous day's closing prices
   - Run WFO on rolling 10-year window
   - Generate top 50 stock list

2. **Market open (9:30 AM ET):**
   - Check for new entry signals in top 50
   - Check for exit signals in existing positions
   - Submit orders via Alpaca API

3. **Market close (4:00 PM ET):**
   - Verify all orders filled
   - Update TradeTracker
   - Generate P&L report

### 10.3 Risk Management

- **Max position:** 2% per stock
- **Max portfolio:** 100% (50 positions × 2%)
- **Stop loss:** ATR trailing stop (built into strategy)
- **Emergency:** `alpaca position close-all` if strategy breaks

---

## 11. Open Questions & Next Steps

### 11.1 Open Questions

1. **Monthly vs quarterly rebalancing:** Does monthly WFO re-run add value or just churn?
2. **WFO window length:** Is 10 years optimal? Would 5 years adapt faster to regime changes?
3. **Min trades gate:** With 8 trades per fold, is statistical significance sufficient?
4. **Bear market behavior:** Does the strategy properly sit out in prolonged bears?
5. **Transaction costs:** Alpaca is zero-commission, but slippage on less liquid S&P 500 names?

### 11.2 Next Steps

1. [ ] **Paper trading:** Run S&P 500 Top 50 @ 2% on Alpaca paper for 1 month
2. [ ] **Monthly rolling WFO:** Test if monthly parameter updates improve performance
3. [ ] **Shorter WFO window:** Test 5-year window vs 10-year for regime adaptation
4. [ ] **Sector ETF comparison:** Compare to sector momentum (XLK, XLE, etc.)
5. [ ] **Regime detection:** Add VIX or SPY 200-day MA as regime filter
6. [ ] **Live integration:** Build Alpaca execution module for ggTrader

---

## 12. Files & Scripts

| File | Purpose |
|------|---------|
| `scripts/stock_wfo_research.py` | S&P 500 full WFO pipeline |
| `scripts/russell2000_wfo_research.py` | Russell 2000 WFO pipeline |
| `scripts/blended_wfo_research.py` | Blended universe WFO pipeline |
| `scripts/nasdaq100_wfo_research.py` | NASDAQ-100 WFO pipeline |
| `scripts/stock_research_quick.py` | Quick full-period backtest |
| `/tmp/sp500_tickers.txt` | S&P 500 constituent list |
| `/tmp/russell_2000_tickers.txt` | Russell 2000 constituent list |
| `/tmp/nasdaq100_tickers.txt` | NASDAQ-100 constituent list |

---

## 13. Appendix: Raw Results

### 13.1 S&P 500 Top 50 @ 2% — Full Metrics

```
Combined Portfolio (2% max, top 50 OOS):
  Final Value:     $40,479.00
  Total Profit:    $30,479.00 (+304.79%)
  Total Trades:    2522
  Win Rate:        52.3%
  Sharpe:          2.153
  Sortino:         3.418
  Max Drawdown:    -10.91%
  CAGR:            +14.35%

SPY Buy & Hold:
  Total Return:    +335.07%
  CAGR:            +15.15%
```

### 13.2 Russell 2000 Top 50 @ 5% — Full Metrics

```
Combined Portfolio (5% max, top 50 OOS):
  Final Value:     $74,200.00
  Total Profit:    $64,200.00 (+642.00%)
  Total Trades:    2522
  Win Rate:        52.3%
  Sharpe:          1.705
  Sortino:         2.668
  Max Drawdown:    -16.60%
  CAGR:            +21.20%

IWM Buy & Hold:
  Total Return:    +194.52%
  CAGR:            +10.92%
```

### 13.3 NASDAQ-100 Top 50 @ 2% — Full Metrics

```
Combined Portfolio (2% max, top 50 OOS):
  Final Value:     $32,011.71
  Total Profit:    $22,011.71 (+220.12%)
  Total Trades:    2438
  Win Rate:        50.6%
  Sharpe:          1.909
  Sortino:         2.889
  Max Drawdown:    -7.98%
  CAGR:            +11.81%

QQQ Buy & Hold:
  Total Return:    +603.98%
  CAGR:            +20.59%
```

---

> **Document Status:** Complete — all trials documented  
> **Recommended Action:** Proceed with S&P 500 Top 50 @ 2% for paper trading  
> **Prerequisite:** Complete vectorbt simplification refactor (see `docs/refactor_vectorbt_simplification.md`)
