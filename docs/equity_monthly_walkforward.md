# Equity Research: Rolling Monthly Walk-Forward (S&P 500)

> **Date:** 2026-06-10
> **Status:** Simulation setup complete and checked for errors; first full simulation run is complete.
> **Replaces:** `docs/archive/sp500_equity_strategy_research.md` (invalid results due to selection bias, parameter leakage, and survivorship bias) and `docs/archive/refactor_vectorbt_simplification.md` (which used incorrect code APIs).
> **Commits:** `4e7af64`, `6a0acc6`, `63714e0` (code cleanups) · `f8c7efe`, `ee705e0` (infrastructure setup)

---

## 1. What This Is (Plain English)

This document records an honest, realistic simulation (backtest) of our trading strategies on S&P 500 stocks. 

To ensure the simulation is realistic, we enforce strict rules so the system never "cheats" by using information it wouldn't have had in the real world:

- **Point-in-Time Universe**: We use the actual historical members of the S&P 500 on each specific trading day (using database records from 1996 to mid-2026). We do not apply today's stock list to past years. This avoids **survivorship bias** (ignoring companies that went bankrupt).
- **Strategy Tournament**: Instead of choosing one trading strategy beforehand, we test 33 different strategy variations (combinations of buying and selling rules) on past data and let the system choose the best-performing rules for each individual stock.
- **Selection Inside the Loop**: At the end of each month, the system looks at the past 2 years of price history for each stock to find the best-performing strategy settings. It then freezes those settings and uses them to trade that stock during the next month. It repeats this process month-by-month, stitching the results into one final performance curve (the "equity curve").
- **Leak-Checked**: We run a validation test (`--leak-check`) that deletes future data and confirms the system still makes the exact same choices. This ensures no future information is "leaking" into past decisions (**PASSES**).

### Real-World Limitations
- **Data Gaps (Survivorship Bias):** Yahoo Finance does not keep data for stocks that have been delisted (e.g. companies that went bankrupt or were bought, like FRC or VIAC). Out of 668 historical S&P 500 companies, 52 could not be downloaded.
- **Monthly Boundary Reset:** The simulation acts as if all trades are closed at the end of each calendar month. This is slightly different from real life, where a trade might be held across the end of a month.
- **Fees & Slippage:** We assume zero commission fees (`FEES=0.0`) and apply a small penalty for market price differences when buying/selling (`SLIPPAGE=0.0005` or 0.05% per trade).

---

## 2. How to Run the Research

Make sure your virtual environment is active before running these terminal commands:

```bash
source .venv/bin/activate

# 1. Smoke Test (takes a few minutes): 
# Run a quick check using only 20 stocks over 6 months with basic strategies.
python -u scripts/sp500_monthly_walkforward.py --quick

# 2. Leak Check: 
# Verify that the strategy does not accidentally look ahead into future data.
python -u scripts/sp500_monthly_walkforward.py --quick --leak-check

# 3. Full Simulation Run:
# Runs the full test in the background (can take a long time, but can be paused and resumed).
nohup python -u scripts/sp500_monthly_walkforward.py --jobs 4 --run-id sp500_monthly_v1 \
    > results/monthly_wf/full_run_v1.log 2>&1 &
```

Simulation checkpoints are saved in `results/monthly_wf/<run-id>/month=YYYY-MM/`. If the run is stopped, it will automatically resume from the last saved month. The final results will be saved in `summary.json` and `equity_curve.parquet`.

---

## 3. The Code Cleanups (Refactoring)

We removed the legacy (older) code paths to ensure all simulations use our fast, vectorized calculation engine. 

We verified this cleanup by comparing test runs:
- **Vectorized Grid Run**: Produced identical mathematical results, showing that our cleanups did not break the math.
- **Legacy Bug Fixes**: Identified and fixed two major bugs:
  1. **Wrong-Strategy Replay**: A bug was causing the system to run default strategies instead of the custom ones selected.
  2. **Indicator Mismapping**: A calculation bug in the legacy code was mapping price indicators incorrectly (specifically for the ADX trend indicator).

---

## 4. Simulation Results (Run `sp500_monthly_v1`)

> **Status: COMPLETE** (took about 26 minutes per simulated month using 4 CPU cores, spanning 64 months).
> **Verdict: NO-GO** — The strategy was nearly flat (gained almost nothing) over 5+ years, while a simple index fund (SPY) doubled. 

### Key Performance Metrics (2021 to 2026)

Here are the definitions of the metrics we use:
- **Total Return**: The total percentage gained or lost.
- **CAGR (Compound Annual Growth Rate)**: The average annual growth rate.
- **Sharpe Ratio**: A measure of risk-adjusted return (how much return you got per unit of volatility). A higher number is better; under 1.0 is considered weak.
- **Sortino Ratio**: Similar to the Sharpe ratio, but only penalizes downward price swings (bad risk).
- **Ann. Volatility**: How wildly the portfolio value fluctuates in a year.
- **Max Drawdown**: The largest peak-to-trough drop in portfolio value (the worst loss from highest peak to lowest valley).

| Metric | Strategy | Simple S&P 500 Index (SPY) |
|---|---|---|
| **Total Return** | **+0.88%** | **+103.83%** |
| **CAGR (Annual Return)** | 0.17% | 14.46% |
| **Sharpe Ratio** | 0.12 | 0.89 |
| **Sortino Ratio** | 0.15 | 1.22 |
| **Ann. Volatility** | 1.49% | 16.88% |
| **Max Drawdown** | −4.30% | −24.50% |
| **Monthly Hit Rate vs SPY** | 36.5% | — |

### Observations:
- **Trade Hold Times:** The average trade was held for 4.9 days, which is within our target range. The signals are executing, but they are not profitable after subtracting slippage.
- **High Turnover:** The tournament selected different strategies 83% of the time each month. This suggests the strategy choices are highly unstable.
- **Low Volatility (Holding Cash):** The volatility was extremely low (1.49%) because the system kept most of the portfolio in cash. The safety gates rejected most stocks, and the ones that passed did not perform well in the subsequent month.

---

### Momentum Baseline Results

To compare, we ran a simple momentum strategy over the same 5-year period (ranking stocks by momentum and holding the top 50, always fully invested):

| Metric | Momentum Strategy (`xs_momentum`) | Simple S&P 500 Index (SPY) |
|---|---|---|
| **Total Return** | **+125.98%** | **+104.52%** |
| **CAGR (Annual Return)** | 16.69% | 14.51% |
| **Sharpe Ratio** | 0.82 | 0.89 |
| **Sortino Ratio** | 1.13 | 1.22 |
| **Ann. Volatility** | 21.83% | 16.89% |
| **Max Drawdown** | −22.38% | −24.50% |

- **Verdict:** The momentum strategy made more money than the S&P 500, but it did so by taking on much higher volatility. On a risk-adjusted basis (Sharpe/Sortino), it did not beat the market index.

---

## 5. Decision Rule

To approve any strategy for live deployment, it must beat the S&P 500 index (SPY) on risk-adjusted terms (higher Sharpe and Sortino ratios) during a full, realistic simulation across different market conditions (such as the 2021 market boom, the 2022 bear market, and the 2023-2025 recovery).

---

*Back to [README.md](../README.md).*
