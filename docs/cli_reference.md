# CLI Reference Guide

The `ggt` command-line tool is your command center. You will use it to run strategy simulations, view database statistics, download historical price data, and launch paper trading.

---

## Command Summary

| Command | Purpose (Plain English) |
|---|---|
| `ggt lab` | Run trading simulations (backtests) to see how strategies would have performed in the past. |
| `ggt paper` | Run virtual trading (paper trading) using real-time data and fake money. |
| `ggt ingest` | Download historical prices from crypto exchange APIs and save them to the database. |
| `ggt db` | Run diagnostics, clean up, or manage the database. |

---

## 1. ggt lab (Simulation Engine)

This command runs trading strategies through historical data to test their performance. It simulates a trader who rebalances their portfolio periodically (usually monthly) using historical prices.

### How to use it:
```bash
ggt lab --strategy <strategy_name> [options]
```

### Required Arguments:

- `--strategy`: The name of the trading strategy you want to evaluate (see choices below).

---

### Choosing a Strategy

Strategies are divided into two types:
- **Weight Strategies** (`target_kind="weights"`): These decide **how much** of your money to allocate to each stock (e.g., "put 5% of the portfolio into stock A, and 0% into stock B").
- **Signal Strategies** (`target_kind="signals"`): These decide **when** to buy and sell (e.g., "buy stock A now, sell it when indicator X triggers").

#### Weight Strategies
| Strategy Name | What it does in plain English |
|---|---|
| `xs_momentum` | **Cross-Sectional Momentum:** Ranks all stocks by their returns over a past lookback window (e.g., the last year) and splits money equally among the top-performing stocks. |
| `dual_momentum` | **Dual Momentum:** Same as `xs_momentum`, but checks if the stock market is actually going up. If the top stocks are losing money compared to cash, it moves your money to safe cash. |

#### Signal Strategies
| Strategy Name | What it does in plain English |
|---|---|
| `ema_cross` | **EMA Crossover:** Uses moving average trend lines. It buys when a fast moving average crosses above a slow moving average (indicating a rising trend) and sells when it crosses below. |
| `wfo_tournament` | **Moving Average Tournament:** Runs a mini-competition between 4 different EMA moving average settings on past data. It automatically picks the one that did best and uses it for the next month. |
| `bb_reversion` | **Bollinger Band Mean Reversion:** Assumes prices eventually return to their average. Buys when a stock's price falls below a lower statistical band (indicating it is temporarily "cheap") and sells when it returns to the middle band. |
| `rsi_reversion` | **RSI Mean Reversion:** Uses the Relative Strength Index (a speed scale from 0 to 100). Buys when RSI drops below an oversold threshold (like 30, meaning heavily dumped) and sells when it recovers. |
| `macd_divergence` | **MACD Divergence:** Looks for cases where the price is making new lows, but the MACD momentum tracker shows the downward selling pressure is slowing down—predicting a bounce. |
| `volume_bb_reversion` | **Volume-Confirmed BB Reversion:** Similar to Bollinger Bands, but only buys if the price drop is accompanied by a massive surge in trading volume (indicating buyers are actively stepping in). |
| `mtf_reversion` | **Multi-Timeframe Reversion:** A high-safety strategy. It only buys if the stock is oversold on both the long-term weekly chart AND the short-term daily chart. |
| `ensemble` | **Voting Strategy:** Combines up to 6 different indicators (RSI, Bollinger Bands, EMA, etc.). It only takes a trade if a minimum number of indicators agree and vote "yes". |
| `ensemble_conviction` | **Conviction Voting:** Same as the ensemble strategy, but it sizes positions based on how many indicators agreed. It invests more money when the vote is unanimous. |
| `conviction_bb` | **Band-Distance Sizing:** A Bollinger Band strategy that invests larger amounts of money the further the price plunges below the lower band (buying heavier when prices get cheaper). |

---

### Optional Arguments:

| Flag | Default | Description (Plain English) |
|---|---|---|
| `--market` | `equity` | Choose `equity` for stocks or `crypto` for cryptocurrencies. |
| `--universe` | `sp500` | The stock list to trade: `sp500` (S&P 500), `nasdaq100` (Nasdaq-100), or `russell2000` (Russell 2000). |
| `--eval-start` | `2021-01-31` | The starting date for the test simulation (YYYY-MM-DD). |
| `--eval-end` | Today's date | The ending date for the test simulation (YYYY-MM-DD). |
| `--top-n` | 50 | Limit the simulation to the top N most active/liquid stocks or coins. |
| `--lookback` | 252 | How many trading days of history the strategy looks back at to calculate momentum (252 days ≈ 1 calendar year). |
| `--skip` | 21 | How often the strategy updates its portfolio holdings (21 trading days ≈ 1 calendar month). |
| `--max-stocks` | None | Cap the total number of stocks loaded (useful for running super-fast test runs). |

### Simulation Modes (Choose One)

If you don't specify any of these flags, the lab runs a single simulation using default strategy parameters.

| Flag | Mode Name | What it does in plain English |
|---|---|---|
| `--sweep` | **Parameter Sweep (Grid Search)** | Tests a grid of different parameter combinations (e.g. testing RSI thresholds of 20, 25, 30, and 35) to find which setting was the most profitable. |
| `--wfo` | **Walk-Forward Optimization** | Simulates a realistic trading setup where the parameters are continuously re-optimized on a rolling training block of past data and then tested on a subsequent test block. |

#### Customizing your Sweep range
If you are running a `--sweep`, you can override the default parameter settings using the `--sweep-param` flag.
For example, to test custom RSI oversold levels:
`--sweep-param rsi_oversold=20,25,30,35`

---

### Lab Examples

Here are some common commands you can run:

```bash
# Run a single standard simulation using the voting (ensemble) strategy
ggt lab --strategy ensemble

# Run the voting strategy on Nasdaq-100 stocks instead of the S&P 500
ggt lab --strategy ensemble --universe nasdaq100

# Run a parameter sweep over Bollinger Band Reversion to find the best settings
ggt lab --strategy bb_reversion --sweep

# Run a parameter sweep testing custom RSI levels
ggt lab --strategy rsi_reversion --sweep --sweep-param rsi_oversold=20,25,30,35

# Run a realistic rolling Walk-Forward Optimization using the voting strategy
ggt lab --strategy ensemble --wfo

# Test cross-sectional momentum on a specific 2-year window
ggt lab --strategy xs_momentum --eval-start 2023-01-01 --eval-end 2024-12-31

# Run a quick diagnostic test on just 10 stocks capped at 5 total positions
ggt lab --strategy dual_momentum --top-n 10 --max-stocks 5
```

---

### Understanding the Simulation Output

When a simulation finishes, the results are stored in the database:
- **`lab_runs` table**: Stores one row for every simulation you run. It includes the strategy name, parameters, when it was run, and total metrics like average return, Sharpe ratio (risk-adjusted return), and maximum drawdown (worst peak-to-trough loss).
- **`lab_periods` table**: Stores how the strategy performed during each individual period/month of the simulation, allowing you to see month-by-month results.

You can inspect these tables using SQL:

```sql
-- View the 10 most recent simulation runs
SELECT run_id, strategy, config, created_at, mean_return, sharpe
FROM lab_runs ORDER BY created_at DESC LIMIT 10;

-- See the month-by-month details for a specific run
SELECT * FROM lab_periods WHERE run_id = '<run_id>' ORDER BY fold_start;
```

---

## 2. ggt paper (Paper Trading)

This command runs one cycle of virtual trading (trading with fake money using live price feeds). It is usually automated to run every weekday via a system scheduler (like cron) at 1:30 PM PT (just before the stock market closes).

### The Paper Trading Pipeline:
1. **Signal Generation**: Runs the voting (`ensemble`) strategy on today's latest market prices.
2. **Machine Learning Guard (ML Feature Gate)**: A LightGBM classification model reviews the proposed trades. If the model calculates that a trade has a low probability of success (precision < 50%), the trade is **dropped** to protect capital.
3. **Risk Guardrails**: Safety rules check the portfolio. They limit the total open positions to 30, cap any single stock at 5% of the portfolio, and freeze all trading if today's loss exceeds 3% or if the overall peak-to-trough loss (drawdown) reaches 15%.
4. **Order Execution**: Sends the approved orders to Alpaca (our paper trading broker) to be executed.
5. **Database Storage**: Records the executed trades and portfolio values to TimescaleDB.
6. **Telegram Notifications**: Sends a summary of trades and portfolio value straight to your chat app.

```bash
ggt paper
```

---

## 3. ggt ingest (Downloading Data)

Downloads historical candlestick price data (OHLCV: Open, High, Low, Close, Volume) from cryptocurrency exchanges and stores them in your local database so your simulations can run offline.

```bash
ggt ingest [options]
```

### Options:
- `--days`: How many days of past history to download. Defaults to 1 day.

### Examples:
```bash
ggt ingest              # Download the most recent 1 day of data
ggt ingest --days 180   # Download the last 6 months of data
ggt ingest --days 1095  # Download the last 3 years of data
```

---

## 4. ggt db (Database Management)

Commands for diagnosing and maintaining your local database.

```bash
ggt db <subcommand>
```

| Subcommand | What it does in plain English |
|---|---|
| `diag` | Prints table sizes and total row counts (lets you see how much data you have). |
| `clean` | Deletes broken, orphaned, or incomplete rows. |
| `truncate` | Wipes out database tables completely (asks for confirmation first). |
| `compression` | Turns on or off TimescaleDB's compression features (saves hard drive space). |
| `export` | Creates a database backup file. |

### Examples:
```bash
# Print database diagnostic info
ggt db diag

# Clean up database tables
ggt db clean

# Enable database compression to save disk space
ggt db compression --enable

# Back up the database to a backup.sql file
ggt db export > backup.sql
```

---

## 5. Machine Learning Signal Pre-Screen Script

Before running a full simulation, you can run a standalone machine learning script to pre-screen a signal and see if it has predictive power.

```bash
python scripts/ml_signal_screen.py --signal <strategy_name> [--start DATE] [--end DATE] [--universe UNIV]
```

This script evaluates how clean the buy/sell signals are. It prints out:
- **Precision**: How often the signals were correct.
- **Recall**: What percentage of good trades the strategy caught.
- **F1 Score**: A combined score of precision and recall.
- **Verdict**: A grade of `DROP`, `BORDERLINE`, or `STRONG` indicating signal quality.
- **Feature Importance**: Which market indicators (like RSI or volatility) were most helpful in predicting success.

Results are saved as a JSON file in the `results/` folder.

---

*See the [Architecture Guide](architecture.md) for how these components fit together, and [Developer Guidelines](../agents.md) for coding standards.*

*Back to [README.md](../README.md).*
