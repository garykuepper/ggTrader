# Architecture Guide

This guide explains how the ggTrader codebase is structured and how its components work together. For commands and usage see the [CLI Reference](cli_reference.md); for developer-specific guidelines, see [Developer Guidelines](../agents.md).

---

## Technical Concepts & Terminology (Plain English)

Before diving into the code, here are the core concepts used throughout ggTrader:

- **Backtesting (Simulating)**: Running a trading strategy on historical price data to see how much money it would have made or lost in the past.
- **Walk-Forward Optimization (WFO)**: A realistic testing method. Instead of finding one rule that fits all of history, we split history into rolling time blocks. We use an earlier block (the **In-Sample** or train period) to find the best rules, and then test those rules on the immediate next block (the **Out-of-Sample** or test period). Then we shift forward and repeat.
- **Overfitting**: A common trap where a strategy is tuned so perfectly to past data that it looks amazing on paper, but fails completely on new, unseen data (because it "memorized" the past instead of finding a general pattern).
- **Walk-Forward Efficiency (WFE)**: A score that compares a strategy's performance during testing (out-of-sample) to its performance during training (in-sample). A high WFE suggests the strategy is robust; a low WFE suggests it is overfitted.
- **Vectorization**: Traditional simulators run day-by-day in a slow loop. ggTrader uses **vectorbt**, a library that performs backtesting calculations on entire arrays of data all at once using fast math. This allows you to simulate years of trading across hundreds of stocks in seconds.
- **Survivorship Bias**: Cheating in backtests by only testing on stocks that exist *today*. We prevent this by using a **Point-in-Time Universe**, which tracks exactly which companies were in the index on any specific day in the past (including those that later went bankrupt or were acquired).
- **OHLCV Data**: The basic building blocks of price charts—stands for **O**pen, **H**igh, **L**ow, **C**lose, and **V**olume for a given time period.

---

## ggTrader at a Glance

| Component | What it does in plain English |
|---|---|
| **Core Libraries** | **Python 3.10+** (language) · **vectorbt** (fast calculations) · **TimescaleDB** (database) · **yfinance** (free Yahoo Finance data). |
| **Data Sources** | Fetches live stock data from Yahoo Finance, uses Tiingo as a backup, loads crypto data from the database, and uses historical CSVs to track S&P 500 constituents over time. |
| **Lab Output** | Saves test records to `lab_runs` (overall results) and `lab_periods` (month-by-month results) in the database. |
| **Paper Trading** | Virtual trading on Alpaca. Includes a machine learning guard (ML feature gate) to filter bad trades, risk checks (guardrails), and Telegram notifications. |
| **Supported Markets** | US stock indices (S&P 500, Nasdaq-100, Russell 2000) and customizable cryptocurrency pairs. |
| **Vectorization** | All portfolio calculations run instantly using NumPy and Pandas vector operations. |

---

## Directory & File Structure

Here is where the source code lives inside `src/ggTrader/`:

```
src/ggTrader/
├── lab/                          # The Research & Simulation Engine
│   ├── cli.py                   # Command-line entry point for running simulations (ggt lab)
│   ├── data.py                  # Loads stock lists and historical prices (yfinance, TimescaleDB)
│   ├── harness.py               # Manages the walk-forward simulation loop and time periods
│   ├── metrics.py               # Calculates return, Sharpe ratio, and drawdowns
│   ├── persist.py               # Saves simulation results to database tables
│   ├── simulate.py              # Feeds data into vectorbt to run the portfolio math
│   ├── strategy.py              # Templates and rules for writing a custom strategy
│   ├── sweep.py                 # Generates grids of parameters to test (parameter sweeps)
│   ├── wfo.py                   # Handles rolling optimization, testing, and safety limits
│   ├── gates.py                 # Quality checks to filter out weak or noisy strategies
│   ├── train_gate.py            # Trains the machine learning model to filter trade signals
│   └── strategies/              # Individual Strategy Files
│       ├── momentum.py          # Weight strategies (e.g., buying top gainers)
│       ├── signals.py           # Signal strategies (e.g., buying moving average crosses)
│       ├── ensemble.py          # Voting strategies that combine multiple indicators
│       ├── conviction.py        # Position-sizing based on indicators
│       └── indicators.py        # Math for technical indicators (RSI, Bollinger Bands, EMA)
│
├── paper/                       # The Live Paper-Trading Deployment
│   ├── alpaca_broker.py         # Adapter to talk to Alpaca (our paper trading broker)
│   ├── signal_runner.py         # Generates the daily buy/sell votes from indicators
│   ├── trader.py                # Coordinates signals, risk checks, and submits orders
│   ├── feature_gate.py          # The machine learning model that blocks risky trades
│   ├── risk.py                  # Safety checks (caps position sizes, halts on losses)
│   ├── notifier.py              # Sends trade updates to Telegram
│   └── persist.py               # Saves daily paper trading logs and portfolio balances
│
├── data/                        # Data Downloader & Database Loader
│   ├── core/
│   │   ├── base_loader.py       # Basic instructions for downloading data
│   │   ├── stock_constants.py   # Yahoo Finance parameters and S&P 500 constants
│   │   └── constants.py         # Crypto mappings and quote currencies
│   ├── historical/
│   │   ├── timescaledb_loader.py # Fetches historical prices from your local database
│   │   └── postgres_ingestor.py  # Inserts newly downloaded data into the database
│   └── live/
│       └── yfinance_loader.py    # Downloads live stock data and saves it to a cache
│
├── utils/                       # Common Helpers
│   ├── config.py                # Global settings and defaults
│   ├── paths.py                 # Locates files relative to the project root directory
│   └── db_engine.py             # Manages connections to the database
│
└── cli/                         # Command-Line Subcommands
    ├── main.py                  # Root CLI setup (the main 'ggt' command)
    ├── cmd_ingest.py            # The 'ggt ingest' command to download data
    └── cmd_db.py                # The 'ggt db' command to manage the database
```

---

## Strategy Classification

Strategies in ggTrader belong to one of two categories:

### 1. Weight-Based Strategies (`target_kind="weights"`)
These strategies return a target percentage for each asset. For example, "allocate 5% of money to Apple, and 10% to Tesla." The simulator automatically generates buy/sell orders to match these weights at every rebalance.

| Strategy | Plain English Description |
|---|---|
| `xs_momentum` | **Cross-Sectional Momentum:** Ranks stocks by their past 12-month returns and divides portfolio money equally among the top N stocks. |
| `dual_momentum` | **Dual Momentum:** Same as above, but if the overall market trend is negative, it drops underperforming stocks and moves the money into cash for safety. |

### 2. Signal-Based Strategies (`target_kind="signals"`)
These strategies return binary flags: `True` for buy (entry) and `True` for sell (exit). The simulator executes trades based on these flags, allocating a fixed position size (default is 2% of cash) to each trade.

| Strategy | Plain English Description |
|---|---|
| `ema_cross` | **EMA Crossover:** Buys when a fast average price crosses above a slow average price (uptrend), and exits when it crosses below. |
| `wfo_tournament` | **Tournament:** Automatically selects the best performing EMA configuration from a list of combinations on recent past data. |
| `bb_reversion` | **Bollinger Band Reversion:** Buys when the price dips below the lower Bollinger Band (cheap) and sells when it returns to the middle average band. |
| `rsi_reversion` | **RSI Reversion:** Buys when the Relative Strength Index falls below 30 (oversold) and sells when it rebounds. |
| `macd_divergence` | **MACD Divergence:** Buys when price is falling but selling momentum is weakening (divergence). |
| `volume_bb_reversion` | **Volume Bollinger Band:** Reversion buy confirmed by a massive surge in volume (high activity at the bottom). |
| `mtf_reversion` | **Multi-Timeframe Reversion:** Buys only when the asset is oversold on both weekly (long-term) and daily (short-term) charts. |
| `ensemble` | **Voting Ensemble:** Combines multiple strategies and enters a trade only if a majority of them agree. |
| `ensemble_conviction` | **Conviction Voting:** Voting strategy that buys larger position sizes when more indicators agree. |
| `conviction_bb` | **Conviction Bollinger Band:** Buys larger positions the further the price plunges below the lower band. |

---

## Code Interfaces (Developer Reference)

All strategies must implement the `Strategy` protocol defined in [strategy.py](file:///home/flynn/ggTrader/src/ggTrader/lab/strategy.py):

```python
class Strategy(Protocol):
    name: str
    target_kind: str  # Must be "weights" or "signals"

    def select(self, asof, data, eligible) -> Plan:
        """
        Point-in-time selection.
        Must use only data available on or before the 'asof' date (no future peeking).
        """

    def to_targets(self, plans, data) -> DataFrame | SignalTargets:
        """
        Converts the plans generated during rebalancing into a target matrix
        (either weights or buy/sell signal matrices).
        """
```

Signal-based strategies also support parameter sweeps and walk-forward parameter updates:

```python
    def sweep_params(self) -> Dict[str, List[Any]]:
        """Returns the grid of settings to test (e.g. fast/slow EMA lengths)."""

    def sweep_signals(self, ohlcv, symbols) -> Tuple[entries, exits]:
        """Generates entry and exit signals for a specific parameter combination."""
```

---

## Core Workflows and Data Flows

### 1. Lab Simulation Run (`ggt lab`)

```
User runs command: ggt lab --strategy ensemble
    ↓
1. load_ohlcv()                 -> Downloads/loads price history from database or cache.
    ↓
2. equity_universe_between()    -> Filters which stocks were active in the index on each date (prevents survivorship bias).
    ↓
3. build_strategy()             -> Instantiates the requested strategy from the registry.
    ↓
4. walkforward()                -> Starts the simulation harness.
    ↓
5. For each Month (Fold):
   - Train Window: Identifies prior historical months (In-Sample).
   - Test Window: Evaluates the strategy on the current month (Out-of-Sample).
   - Runs vectorbt.Portfolio simulation on the test month.
   - Calculates Sharpe ratio, return, drawdowns, and trade counts.
    ↓
6. Saves results to `lab_runs` and `lab_periods` database tables.
    ↓
7. Prints a summary table to the terminal.
```

### 2. Parameter Sweep (`ggt lab --sweep`)

```
User runs command: ggt lab --strategy rsi_reversion --sweep
    ↓
1. strategy.sweep_params()      -> Retrieves the default grid of parameters (e.g. RSI thresholds = 20, 25, 30).
    ↓
2. build_grid()                 -> Creates every combination of parameters (Cartesian product).
    ↓
3. For each combination:
   - Sets up the strategy with those parameters.
   - Runs the walk-forward simulation.
   - Gathers out-of-sample metrics (returns, Sharpe ratio).
    ↓
4. Prints a summary table ranking all combinations from best to worst.
```

### 3. Walk-Forward Optimization (`ggt lab --wfo`)

```
User runs command: ggt lab --strategy ensemble --wfo
    ↓
1. generate_folds()             -> Divides history into rolling folds (e.g., 12 months training / 3 months testing).
    ↓
2. For each Fold:
   - Runs a parameter sweep on the 12-month training window.
   - Finds the parameter set that performed best.
   - Uses those parameters to trade during the 3-month test window.
   - Calculates Walk-Forward Efficiency (WFE) = Test Return / Train Return.
   - Circuit Breaker: Stops the test early if WFE drops below safety limits (indicating the strategy is failing).
    ↓
3. Calculates "Anchor Parameters" (settings that were stable across most folds).
    ↓
4. Prints the final test results, WFE, and recommended stable parameters.
```

### 4. Paper Trading Cycle (`ggt paper`)

```
System runs daily cron job: ggt paper
    ↓
1. signal_runner.generate_signals() -> Generates buy/sell flags on today's latest market prices.
    ↓
2. feature_gate.filter()            -> The LightGBM Machine Learning model acts as a security guard,
                                       dropping any signals with <50% calculated chance of success.
    ↓
3. risk.check_guardrails()          -> Safety checks: ensures max 30 positions, max 5% size per stock,
                                       and blocks trading if daily loss (>3%) or drawdown (>15%) limits are hit.
    ↓
4. trader.execute()                 -> Connects to Alpaca and submits buy/sell orders.
    ↓
5. persist.save()                   -> Saves order details and daily portfolio values to the database.
    ↓
6. notifier.send()                  -> Sends a Telegram message listing the executed trades.
```

---

## Vectorbt Portfolio Simulation

The module [simulate.py](file:///home/flynn/ggTrader/src/ggTrader/lab/simulate.py) handles the backtesting math. Instead of iterating day-by-day (which is slow), it batches everything.

### 1. Weight-Based Simulation (`simulate_weights()`)
Feeds target weights into vectorbt. Uses `cash_sharing=True` so all assets draw from the same money pool, and `group_by=strategy_index` to run multiple simulations in a single calculation.

```python
pf = vbt.Portfolio.from_orders(
    close=close,
    size=size,                       # Target weights (0.0 to 1.0)
    size_type="targetpercent",
    cash_sharing=True,               # Shared capital pool
    group_by=strategy_index,         # Grouped for speed
)
```

### 2. Signal-Based Simulation (`simulate_signals()`)
Feeds entry/exit flags into vectorbt. It purchases a fixed percentage size (default 2% of portfolio cash) per signal.

```python
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=SIGNAL_POSITION_SIZE,       # Fraction per entry (e.g. 0.02)
    size_type="percent",
    cash_sharing=True,               # Shared capital pool
    group_by=strategy_index,         # Grouped for speed
)
```

---

## Walk-Forward Fold Logic Details

The walk-forward driver in [harness.py](file:///home/flynn/ggTrader/src/ggTrader/lab/harness.py) splits historical data systematically:

1. **Universe Eligibility**: Filters out new stocks that do not have enough price history (e.g., needs 500+ trading days).
2. **Monthly Folds**: Runs simulations month-by-month.
3. **Warmup Period**: Automatically loads data from days prior to the start date (e.g., 252 days prior) so indicators (like moving averages) have enough history to calculate values on day one.
4. **Aggregation**: Calculates the mean and standard deviation of returns and Sharpe ratios across all test periods to gauge consistency.

---

## Technical Indicator Functions

The file [indicators.py](file:///home/flynn/ggTrader/src/ggTrader/lab/strategies/indicators.py) contains the math behind the trading signals:

- `bb_signals(close, period, std)`: Bollinger Band entry/exit flags.
- `bb_strength(close, period, std)`: Distance below the band (measures how cheap the stock is).
- `ema_signals(close, fast, slow)`: EMA Crossover entry/exit flags.
- `rsi_signals(close, period, oversold, exit)`: RSI entry/exit flags.
- `macd_signals(close, fast, slow, signal, window)`: MACD Divergence flags.
- `volume_bb_signals(...)`: Volume-confirmed Bollinger Band reversion.
- `mtf_signals(...)`: Multi-timeframe flags (weekly RSI + daily BB).

---

*Back to [README.md](../README.md).*
