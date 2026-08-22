# CLI Reference Guide

The `ggt` command-line tool is your command center. You will use it to run strategy simulations, view database statistics, download historical price data, and launch paper trading.

---

## Command Summary

| Command | Purpose (Plain English) |
|---|---|
| `ggt lab` | Run trading simulations (backtests) to see how strategies would have performed in the past. |
| `ggt paper` | Run virtual trading (paper trading) using real-time data and fake money. |
| `ggt ingest` | ⚠️ **Parked — refuses to run, exits non-zero.** See §3. |
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

#### Research strategies — closed, kept for reproducibility

`STRATEGY_REGISTRY` holds **36** entries; the 12 above are the maintained
core. The remaining 24 are research candidates that were built, walk-forward
tested, and **closed NO-GO**. They still run, but none beat the deployed
config and they are not maintained. Check
`docs/research/RESEARCH_SNAPSHOT.md` for each verdict before spending a run
on one — and note the caveat there about event-date studies having been
measured on a day-shifted tape.

| Group | Strategies |
|---|---|
| Ensemble variants | `ensemble_ic`, `ensemble_kelly` |
| Alternative equity signals | `idio_vol`, `overnight_gap`, `max_effect`, `pairs_stat_arb` |
| Leveraged-ETF rotation | `leveraged_rotation_sp500`, `leveraged_rotation_nasdaq100`, `leveraged_rotation_russell2000` |
| Leveraged-ETF trend | `leveraged_trend_sp500`, `leveraged_trend_nasdaq100`, `leveraged_trend_russell2000` |
| Event-driven ⚠️ | `pead`, `fomc_drift`, `index_deletion_fade`, `insider_cluster_buy`, `congress_trades` |
| Positioning / flow | `short_interest`, `short_volume_ratio`, `retail_attention` |
| Cross-asset overlays | `fx_hedge_overlay`, `treasury_curve`, `commodity_trend` |
| Sentiment | `headline_sentiment` |

⚠️ The event-driven group joins bars to real-world calendar dates and was
therefore hit by the one-day-early timestamp bug fixed in `ef4e15f`. Their
NO-GO verdicts are **not trustworthy** and need re-running before being
cited as closed.

To get the authoritative list at any time, read the registry rather than
this table:

```bash
python -c "from ggTrader.lab.strategies import STRATEGY_REGISTRY as R; print(len(R)); print('\n'.join(sorted(R)))"
```

---

### Optional Arguments:

| Flag | Default | Description (Plain English) |
|---|---|---|
| `--market` | `equity` | Choose `equity` for stocks or `crypto` for cryptocurrencies. |
| `--universe` | `sp500` | The stock list to trade. Eight choices (`lab/cli.py:23-32`): `sp500`, `nasdaq100`, `russell2000`, `midcap400` (all four are real equity universes; `midcap400` is a **live production sleeve**), plus four narrow instrument baskets used only by their matching research strategies — `fx_hedge`, `fomc_treasury`, `commodity_trend`, `treasury_curve`. |
| `--eval-start` | `2021-01-31` | The starting date for the test simulation (YYYY-MM-DD). |
| `--eval-end` | Today's date | The ending date for the test simulation (YYYY-MM-DD). ⚠️ **Always set this explicitly on any run whose number you will quote.** The default is "now", so it moves between runs and two results become silently incomparable. This is how the once-headline 1.12/1.14 Sharpe figures became unreproducible — their window is gone. The tell is SPY's own Sharpe changing between runs, which cannot happen if the window is fixed. |
| `--top-n` | 50 | Limit the simulation to the top N most active/liquid stocks or coins. |
| `--lookback` | 252 | How many trading days of history the strategy looks back at to calculate momentum (252 days ≈ 1 calendar year). |
| `--skip` | 21 | How often the strategy updates its portfolio holdings (21 trading days ≈ 1 calendar month). |
| `--max-stocks` | None | Cap the total number of stocks loaded (useful for running super-fast test runs). |
| `--max-sector-count` | None | Cap the maximum number of stock holdings selected from any single GICS sector (risk management). |

### Simulation Modes (Choose One)

If you don't specify any of these flags, the lab runs a single simulation using default strategy parameters.

| Flag | Mode Name | What it does in plain English |
|---|---|---|
| `--sweep` | **Parameter Sweep (Grid Search)** | Tests a grid of different parameter combinations (e.g. testing RSI thresholds of 20, 25, 30, and 35) to find which setting was the most profitable. |
| `--wfo` | **Walk-Forward Optimization** | Simulates a realistic trading setup where the parameters are continuously re-optimized on a rolling training block of past data and then tested on a subsequent test block. |
| `--blend` | **Portfolio Sleeve Blending** | Blends multiple independent strategy@universe sleeves (e.g. `ensemble@sp500,ensemble@midcap400`) using rolling inverse-volatility to target-volatility scaling. |

#### Portfolio Blending Parameters
When running a `--blend` optimization, you can customize the portfolio combination overlay using:
- `--target-vol`: The annualized target volatility for the blended portfolio (default: `0.068` or 6.8%).
- `--blend-window`: The lookback window in days for computing sleeve covariance and returns volatility (default: `60` days).
- `--max-leverage`: Capping the portfolio leverage scale (default: `2.0` or 200%). ⚠️ **Production runs at `1.0`.** The default is deliberately left at 2.0 for unconstrained research comparisons, so a blend number computed at the default overstates what the live account can actually achieve. Pass `--max-leverage 1.0` for anything meant to describe the deployed config. This has caused an invalid run twice (2026-07-13 and 2026-08-18); see the comment at `src/ggTrader/lab/blend.py:84-87`.

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

# Run a Walk-Forward Optimization for Nasdaq-100 and MidCap 400 sleeves and blend them
ggt lab --strategy ensemble --blend "ensemble@nasdaq100,ensemble@midcap400" --eval-start 2021-01-31 --eval-end 2026-05-18

# Run cross-sectional momentum on the S&P 500 while capping holdings to at most 2 per GICS sector
ggt lab --strategy xs_momentum --max-sector-count 2
```

---

### Understanding the Simulation Output

When a simulation finishes, results are written to **seven** tables, all
created by `src/ggTrader/lab/persist.py`:

| Table | What it holds |
|---|---|
| `lab_runs` | One row per simulation — strategy, market, eval window, params, status. |
| `lab_summary` | The headline metrics per run (Sharpe, drawdown, etc.) plus the benchmark's, and gate diagnostics. |
| `lab_returns` | The daily return series for the run. |
| `lab_equity` | The daily equity curve, alongside the benchmark's. |
| `lab_plans` | The actual holdings plan at each rebalance date, with eligibility counts and coverage. |
| `lab_sweeps` / `lab_sweep_combos` | Parameter-sweep runs and the per-combination results. |

> **Note:** earlier versions of this document (and of `AGENTS.md`) referred to
> a `lab_periods` table holding per-fold metrics. **No such table exists or
> ever has.** Per-fold detail lives in `lab_returns` / `lab_equity`, and
> per-run metrics in `lab_summary`.

You can inspect these tables using SQL:

```sql
-- View the 10 most recent simulation runs with their headline metrics
SELECT r.run_id, r.strategy, r.eval_start, r.eval_end, r.created_at, s.metrics
FROM lab_runs r
LEFT JOIN lab_summary s USING (run_id)
ORDER BY r.created_at DESC LIMIT 10;

-- Pull the equity curve for a specific run
SELECT date, strategy_equity, benchmark_equity
FROM lab_equity WHERE run_id = '<run_id>' ORDER BY date;
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

## 3. ggt ingest (Downloading Data) — ⚠️ PARKED, REFUSES TO RUN

> **As of 2026-08-22 this command deliberately does nothing and exits
> non-zero.** It used to be worse: `src/ggTrader/cli/cmd_ingest.py:26-31`
> hardcoded a two-symbol crypto list with the actual
> `ingestor.sync_symbol_ohlcv(sym)` call **commented out**, then printed
> `Ingestion complete.` — reporting success having written nothing. Rather
> than continue lying about success, it now prints a parked message to
> stderr and exits `1`.
>
> **How data actually gets loaded:** equity OHLCV is fetched on demand and
> cached to TimescaleDB by `CachedYFinanceLoader`
> (`src/ggTrader/data/live/cached_yfinance_loader.py`), transparently, when
> a lab or paper run asks for symbols it does not already have. There is
> nothing to run by hand. Crypto ingestion is parked along with the rest of
> the crypto arc; `postgres_ingestor.py` is kept in place (unused by the CLI)
> in case crypto ingestion is revived.
>
> The documentation below describes the *intended* interface, kept for
> whenever the command is either implemented or removed.


Downloads historical candlestick price data (OHLCV: Open, High, Low, Close, Volume) from cryptocurrency exchanges and stores them in your local database so your simulations can run offline.

```bash
ggt ingest [options]
```

### Options:
- `--days`: How many days of past history to download. Defaults to 1 day.

### Examples (all currently refuse to run — see the warning above):
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

*See the [Architecture Guide](architecture.md) for how these components fit together, and [Developer Guidelines](../AGENTS.md) for coding standards.*

*Back to [README.md](../README.md).*
