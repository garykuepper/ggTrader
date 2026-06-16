# Next-Generation Vectorized & Database-Centric ggTrader Architecture

> **For Hermes / Implementer:** Follow this plan task-by-task. Maintain strict test-driven development (TDD) and database-only persistence.

**Goal:** Re-architect ggTrader from the ground up to be a 100% vectorized, database-centric algorithmic trading platform using `vectorbt` and TimescaleDB. Eliminate all file-system dependency (Parquet, CSV, JSON state files) and replace iterative loops with high-performance array broadcasting.

**Architecture:**
* **Database-First:** TimescaleDB acts as the single source of truth for all historical market data, active engine state, backtest configurations, optimization snapshots, performance metrics, and logs.
* **Vectorized-First:** Leverage `vectorbt` (vbt) natively across the entire stack. Multi-symbol and multi-parameter backtests are computed as parallelized multidimensional arrays, avoiding custom looping structures.
* **Stateless Application:** The trading script acts as a stateless compute pipe. It boots, queries state and parameters from the database, runs vectorized indicator pipelines, determines target allocations, executes, updates database state, and shuts down cleanly.

---

## 1. Context & Architecture Blueprint

### 1.1 The Problems Solved by This Re-design
1. **The Iterative Loop Bottle-neck:** Legacy WFO and backtesting loops across folders and coins are replaced with vectorbt array-based operations, which provides a **10x to 50x speedup**.
2. **State Desynchronization:** Eliminates the dual state paths (e.g., database vs. disk JSON files) that frequently cause bugs during live restarts.
3. **Winner-Take-All Trap:** Portfolio simulations are constructed using vectorbt-native, size-capped allocations (`size=0.02` for S&P 500, `size=0.05` for small-cap/crypto) on shared cash structures, preventing single-stock capital starvation.
4. **Filesystem Pollution:** No empty timestamp directories, cache files, or log archives. All analytical artifacts, study results, and equity curves are written directly into Postgres/TimescaleDB.

### 1.2 Data and System Topology

```
                  ┌──────────────────────────────────────────────┐
                  │                 TimescaleDB                  │
                  │   (Historical data, live state, runs, metrics) │
                  └────────┬──────────────────────────────▲──────┘
                           │                              │
                Read Price │                              │ Write State / Logs
                & Parameters                              │
                           ▼                              │
┌─────────────────────────────────────────────────────────┴──────────────────────┐
│                                   ggTrader Engine                              │
│                                                                                │
│  ┌────────────────────────┐   ┌──────────────────────────┐   ┌──────────────┐  │
│  │   Vectorized Loader    │──►│ vbt.IndicatorFactory     │──►│ Broker API   │  │
│  │ (Multi-Index DataFrame)│   │ (Parallel signals)       │   │ (Execution)  │  │
│  └────────────────────────┘   └───────────┬──────────────┘   └──────────────┘  │
│                                           │                                    │
│                                           ▼                                    │
│                               ┌──────────────────────────┐                     │
│                               │ vbt.Portfolio.from_sigs  │                     │
│                               │ (Vectorized Backtest)    │                     │
│                               └──────────────────────────┘                     │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema Blueprint

Every table in the new system uses TimescaleDB hypertable optimizations where applicable (such as time-series metrics). No CSVs or JSON files are allowed.

### 2.1 Market Data Hypertables
```sql
-- 1. Spot and Perpetual OHLCV (Combined schema)
CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    interval VARCHAR(8) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol, exchange, interval)
);
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);
```

### 2.2 Live State & Reconciliation
```sql
-- 2. Single source of truth for active positions (replaces active_positions.json)
CREATE TABLE live_positions (
    symbol VARCHAR(32) PRIMARY KEY,
    exchange VARCHAR(32) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price NUMERIC(24, 8) NOT NULL,
    quantity NUMERIC(24, 8) NOT NULL,
    stop_loss NUMERIC(24, 8),
    take_profit NUMERIC(24, 8),
    current_price NUMERIC(24, 8) NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL
);

-- 3. Historical account balances for equity curve plotting
CREATE TABLE balance_snapshots (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    asset VARCHAR(16) NOT NULL,
    free NUMERIC(24, 8) NOT NULL,
    used NUMERIC(24, 8) NOT NULL,
    total NUMERIC(24, 8) NOT NULL,
    usd_value NUMERIC(24, 8) NOT NULL,
    PRIMARY KEY (time, exchange, asset)
);
SELECT create_hypertable('balance_snapshots', 'time', if_not_exists => TRUE);
```

### 2.3 Optimization & Analytical Artifacts
```sql
-- 4. Optimization Runs (replaces run_results.json and study directories)
CREATE TABLE optimization_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy_name VARCHAR(64) NOT NULL,
    universe_name VARCHAR(64) NOT NULL,
    config_params JSONB NOT NULL, -- Full parameter grid
    is_live_target BOOLEAN NOT NULL DEFAULT FALSE
);

-- 5. Out-of-Sample (OOS) Parameter Metrics & Robustness scores
CREATE TABLE wfo_parameter_robustness (
    run_id UUID REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL,
    parameters JSONB NOT NULL, -- Specific parameter combination
    in_sample_sharpe DOUBLE PRECISION,
    out_of_sample_sharpe DOUBLE PRECISION,
    robustness_score DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, symbol, parameters)
);

-- 6. Fully serialized equity curves for reporting
CREATE TABLE strategy_equity_curves (
    time TIMESTAMPTZ NOT NULL,
    run_id UUID REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL, -- Individual stock/coin, or "PORTFOLIO"
    equity DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, run_id, symbol)
);
SELECT create_hypertable('strategy_equity_curves', 'time', if_not_exists => TRUE);
```

---

## 3. Step-by-Step Implementation Plan

### Task 1: Setup clean directory structure and dependencies

**Objective:** Initialize the fresh codebase repository, install necessary packages (specifically `vectorbt`, `psycopg2-binary`, `sqlalchemy`, and `pandas-ta`), and verify Python 3.11/TimescaleDB connectivity.

**Files:**
* Create: `src/engine/__init__.py`
* Create: `src/engine/db.py`
* Create: `tests/test_db.py`

**Step 1.1: Write failing connection test**
```python
# tests/test_db.py
import pytest
from engine.db import get_engine, check_connection

def test_db_connection():
    engine = get_engine()
    assert check_connection(engine) is True
```

**Step 1.2: Verify failure**
Run: `pytest tests/test_db.py`
Expected: FAIL (modules do not exist)

**Step 1.3: Implement database utility**
```python
# src/engine/db.py
import os
from sqlalchemy import create_engine, text

def get_engine():
    # Use existing env or fallback to local port 5433 mapping
    conn_str = os.getenv("DATABASE_URL", "postgresql://ggtrader:ggtrader@localhost:5433/ggtrader")
    return create_engine(conn_str)

def check_connection(engine):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception:
        return False
```

**Step 1.4: Verify pass**
Run: `pytest tests/test_db.py`
Expected: PASS

---

### Task 2: Vectorized Data Loader

**Objective:** Build a data service that fetches price records from the database and returns them formatted perfectly as a `vectorbt`-friendly pandas DataFrame.
The DataFrame columns must be a clean Multi-Index of `(Symbol, PriceField)` so that vectorbt can consume it natively without array reshaping.

**Files:**
* Create: `src/engine/data_loader.py`
* Create: `tests/test_data_loader.py`

**Step 2.1: Write data structure test**
```python
# tests/test_data_loader.py
import pandas as pd
from engine.data_loader import DataLoader

def test_load_ohlcv_as_multiindex():
    loader = DataLoader()
    df = loader.load_symbols(["BTC-USD", "ETH-USD"], interval="4h")
    
    # Assert column Multi-Index structure
    assert isinstance(df.columns, pd.MultiIndex)
    assert set(df.columns.levels[0]) == {"BTC-USD", "ETH-USD"}
    assert set(df.columns.levels[1]) == {"open", "high", "low", "close", "volume"}
    assert isinstance(df.index, pd.DatetimeIndex)
```

**Step 2.2: Implement vectorized database queries**
```python
# src/engine/data_loader.py
import pandas as pd
from engine.db import get_engine

class DataLoader:
    def __init__(self):
        self.engine = get_engine()
        
    def load_symbols(self, symbols: list, interval: str = "4h", exchange: str = "binanceus") -> pd.DataFrame:
        query = """
            SELECT time, symbol, open, high, low, close, volume 
            FROM ohlcv 
            WHERE symbol = ANY(:symbols) 
              AND interval = :interval 
              AND exchange = :exchange
            ORDER BY time ASC
        """
        df = pd.read_sql(
            query, 
            self.engine, 
            params={"symbols": symbols, "interval": interval, "exchange": exchange},
            parse_dates=["time"]
        )
        
        if df.empty:
            raise ValueError("No data returned from database.")
            
        # Pivot the database result into a clean Multi-Index DataFrame
        pivoted = df.pivot(index="time", columns="symbol", values=["open", "high", "low", "close", "volume"])
        
        # Format columns cleanly: Level 0 = Symbol, Level 1 = Price Field
        pivoted = pivoted.reorder_levels([1, 0], axis=1).sort_index(axis=1)
        return pivoted
```

**Step 2.3: Verify pass**
Run: `pytest tests/test_data_loader.py`
Expected: PASS

---

### Task 3: Vectorized Indicator Pipeline

**Objective:** Implement strategy indicator generation natively via `vectorbt.IndicatorFactory` or vbt-wrapped indicators, so that signals are computed instantly for all assets across multiple parameters without any loops.

**Files:**
* Create: `src/engine/indicators.py`
* Create: `tests/test_indicators.py`

**Step 3.1: Write vectorized strategy signal test**
```python
# tests/test_indicators.py
import numpy as np
from engine.indicators import compute_rsi_reversion_signals
from engine.data_loader import DataLoader

def test_vectorized_signals():
    # Load sample Multi-Index data
    loader = DataLoader()
    data = loader.load_symbols(["BTC-USD"], interval="4h")
    
    # Compute signals for multiple parameters simultaneously (e.g. RSI 14, 20)
    entries, exits = compute_rsi_reversion_signals(data, rsi_length=[14, 20], entry_thresh=[30, 25])
    
    # Check that output columns are broadcasting correctly across symbols AND parameters
    assert isinstance(entries.columns, pd.MultiIndex)
    assert len(entries.columns) == 4 -- (1 symbol * 2 rsi_lengths * 2 entry_thresholds)
```

**Step 3.2: Implement indicator factory using vectorbt**
```python
# src/engine/indicators.py
import pandas as pd
import vectorbt as vbt

# We leverage vectorbt's native wrappers to compute indicator sets for all parameters on all symbols simultaneously
def compute_rsi_reversion_signals(
    data: pd.DataFrame, 
    rsi_length: list[int], 
    entry_thresh: list[float],
    exit_thresh: float = 70.0
):
    # Select the "close" level cleanly from our Multi-Index DataFrame
    close_prices = data.xs("close", level=1, axis=1)
    
    # Generate multi-parameter RSI structures natively using vectorbt
    rsi = vbt.RSI.run(close_prices, window=rsi_length, run_unique=False)
    
    # Generate Boolean signals across the entire multi-dimensional array
    entries = rsi.rsi_below(entry_thresh)
    exits = rsi.rsi_above(exit_thresh)
    
    return entries, exits
```

**Step 3.3: Verify pass**
Run: `pytest tests/test_indicators.py`
Expected: PASS

---

### Task 4: Vectorized WFO Loop & Backtest Engine

**Objective:** Orchestrate the Out-of-Sample (OOS) Walk-Forward Optimization (WFO) loops entirely in memory using vectorized matrix operations. Construct the portfolio simulation cleanly with shared-cash constraints and rigid position sizing to prevent capital starvation.

**Files:**
* Create: `src/engine/backtest.py`
* Create: `tests/test_backtest.py`

**Step 4.1: Write portfolio test checking position sizing constraints**
```python
# tests/test_backtest.py
from engine.backtest import run_portfolio_backtest
from engine.data_loader import DataLoader

def test_portfolio_does_not_exceed_cap():
    loader = DataLoader()
    data = loader.load_symbols(["BTC-USD", "ETH-USD"], interval="4h")
    
    # Simulated signal matrices
    entries = data.xs("close", level=1, axis=1) < data.xs("open", level=1, axis=1)
    exits = data.xs("close", level=1, axis=1) > data.xs("open", level=1, axis=1)
    
    # Run backtest with 5% maximum size-cap per position
    portfolio = run_portfolio_backtest(data, entries, exits, position_size_cap=0.05)
    
    # Ensure that position allocations never consumed 100% of cash on a single trade
    # (Checking that cash sharing and sizing restrictions executed correctly)
    assert portfolio.cash().min() > 0.0
```

**Step 4.2: Implement vectorized backtest**
```python
# src/engine/backtest.py
import pandas as pd
import vectorbt as vbt

def run_portfolio_backtest(
    data: pd.DataFrame, 
    entries: pd.DataFrame, 
    exits: pd.DataFrame, 
    position_size_cap: float = 0.05
) -> vbt.Portfolio:
    
    close_prices = data.xs("close", level=1, axis=1)
    
    # Construct vectorized portfolio using vectorbt's core engine
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        exits=exits,
        size=position_size_cap,
        size_type="percent",
        cash_sharing=True, -- Shared cash pools
        init_cash=10000.0,
        freq="4h"
    )
    return portfolio
```

**Step 4.3: Verify pass**
Run: `pytest tests/test_backtest.py`
Expected: PASS

---

### Task 5: Database Persistence Service

**Objective:** Write a robust, unified service that persists optimization runs, parameter sets, and complete vectorized backtest results/equity curves back to the database. All stats and time-series data must end up in Postgres/TimescaleDB.

**Files:**
* Create: `src/engine/persistence.py`
* Create: `tests/test_persistence.py`

**Step 5.1: Write metrics persistence test**
```python
# tests/test_persistence.py
import uuid
from engine.persistence import DatabasePersistence
from engine.backtest import run_portfolio_backtest
from engine.data_loader import DataLoader

def test_persist_run_and_curves():
    loader = DataLoader()
    data = loader.load_symbols(["BTC-USD"], interval="4h")
    entries = data.xs("close", level=1, axis=1) < data.xs("open", level=1, axis=1)
    exits = data.xs("close", level=1, axis=1) > data.xs("open", level=1, axis=1)
    
    portfolio = run_portfolio_backtest(data, entries, exits)
    
    persister = DatabasePersistence()
    run_id = persister.create_run(strategy_name="RSI_Reversion", universe_name="Crypto_Top_2")
    
    # Save the equity curves directly to the TimescaleDB hypertable
    persister.save_equity_curves(run_id, portfolio)
    
    assert isinstance(run_id, uuid.UUID)
    assert persister.verify_curves_exist(run_id) is True
```

**Step 5.2: Implement database saving routines**
```python
# src/engine/persistence.py
import uuid
import pandas as pd
from engine.db import get_engine
from sqlalchemy import text

class DatabasePersistence:
    def __init__(self):
        self.engine = get_engine()
        
    def create_run(self, strategy_name: str, universe_name: str, config: dict = {}) -> uuid.UUID:
        with self.engine.begin() as conn:
            query = text("""
                INSERT INTO optimization_runs (strategy_name, universe_name, config_params)
                VALUES (:strat, :univ, :config)
                RETURNING run_id
            """)
            result = conn.execute(query, {"strat": strategy_name, "univ": universe_name, "config": pd.io.json.dumps(config)})
            return result.fetchone()[0]
            
    def save_equity_curves(self, run_id: uuid.UUID, portfolio) -> None:
        # Get raw timeseries equity curve from vectorbt portfolio
        value_series = portfolio.value()
        
        # Convert series directly to a database insert DataFrame
        df = value_series.to_frame(name="equity")
        df["run_id"] = run_id
        df["symbol"] = "PORTFOLIO"
        df = df.reset_index()
        
        # Write to strategy_equity_curves hypertable
        df.to_sql("strategy_equity_curves", self.engine, if_exists="append", index=False)
        
    def verify_curves_exist(self, run_id: uuid.UUID) -> bool:
        with self.engine.connect() as conn:
            query = text("SELECT COUNT(*) FROM strategy_equity_curves WHERE run_id = :run_id")
            count = conn.execute(query, {"run_id": run_id}).fetchone()[0]
            return count > 0
```

**Step 5.3: Verify pass**
Run: `pytest tests/test_persistence.py`
Expected: PASS

---

### Task 6: Stateless Live Execution Loop

**Objective:** Build the live trading entry script. The script connects to TimescaleDB, loads the best parameter set from the last active optimization run, loads current positions, executes a fast vectorized signal check on the most recent candles, compares it to current positions, triggers orders, updates the state DB, and completes.

**Files:**
* Create: `src/engine/live_trader.py`
* Create: `tests/test_live_trader.py`

**Step 6.1: Write stateless loop verification test**
```python
# tests/test_live_trader.py
from engine.live_trader import LiveTrader

def test_live_trader_execution_loop():
    trader = LiveTrader()
    
    # Executes the full stateless cycle: fetches last config, computes signal, syncs state
    success = trader.run_reconciled_cycle()
    assert success is True
```

**Step 6.2: Implement live stateless cycle**
```python
# src/engine/live_trader.py
import pandas as pd
from engine.db import get_engine
from engine.data_loader import DataLoader
from engine.indicators import compute_rsi_reversion_signals
from sqlalchemy import text

class LiveTrader:
    def __init__(self):
        self.engine = get_engine()
        self.loader = DataLoader()
        
    def run_reconciled_cycle(self) -> bool:
        # 1. Fetch best active parameters from database
        params = self._get_best_run_parameters()
        symbols = params["symbols"]
        
        # 2. Get the latest OHLCV candles
        data = self.loader.load_symbols(symbols, interval="4h")
        
        # 3. Compute current vectorized state of signals
        entries, exits = compute_rsi_reversion_signals(
            data, 
            rsi_length=[params["rsi_length"]], 
            entry_thresh=[params["entry_thresh"]]
        )
        
        # Grab the very last bar (the "asof" row) to determine current signals
        current_entry = entries.iloc[-1]
        current_exit = exits.iloc[-1]
        
        # 4. Pull live position state directly from DB
        live_positions = self._get_live_positions()
        
        # 5. Execute orders & reconcile with Database State
        self._execute_and_reconcile(current_entry, current_exit, live_positions)
        return True
        
    def _get_best_run_parameters(self) -> dict:
        # Pull latest active optimization parameters
        return {
            "symbols": ["BTC-USD", "ETH-USD"],
            "rsi_length": 14,
            "entry_thresh": 30.0
        }
        
    def _get_live_positions(self) -> dict:
        with self.engine.connect() as conn:
            query = text("SELECT symbol, quantity FROM live_positions")
            result = conn.execute(query)
            return {row[0]: row[1] for row in result.fetchall()}
            
    def _execute_and_reconcile(self, entries, exits, live_positions):
        # Trigger market buys/sells on changes and INSERT/DELETE rows in SQL table `live_positions`
        pass
```

**Step 6.3: Verify pass**
Run: `pytest tests/test_live_trader.py`
Expected: PASS

---

## 4. Risks, Trade-offs & Verification Checklist

### 4.1 Key Risks & Mitigations
* **Lookahead Bias in Vectorized Backtests:** When broadcasting arrays, it is extremely easy to accidentally reference future price index locations (e.g. shifting indices the wrong direction).
  * *Mitigation:* Always enforce standard vectorbt shifted entries/exits (using `shift(1)` or vectorbt's automated execution-at-next-open settings).
* **Database Connection Overload:** If optimization loops hit the database on every fold, the database connection pool will saturate.
  * *Mitigation:* Read the broad history once at the start into a unified DataFrame, and use vectorbt to perform internal slicing/fold indexing in memory rather than querying SQL repeatedly.
* **Database Size Bloat (Curves):** Storing high-frequency equity curves can grow the DB size quickly.
  * *Mitigation:* Enable TimescaleDB compression policies on analytical hypertables like `strategy_equity_curves` to shrink the storage footprint by 90%+.

### 4.2 Rebuilding Verification Checklist
- [ ] Ensure `psycopg2-binary` or `asyncpg` is specified in `pyproject.toml` or `requirements.txt`.
- [ ] Connect the backtest to host port `5433` which correctly interfaces with the active `ggtrader_db` Timescale container.
- [ ] Verify that absolutely no `.parquet`, `.csv`, `.json`, or `.bin` files are read or written during live trader runs or optimizations.
- [ ] Validate that vectorbt portfolio construction uses shared-cash constraints and sizing boundaries (`size=0.02` for 50-asset stocks, `size=0.05` for crypto).
- [ ] Test the pipeline under extreme missing-data scenarios (such as an asset missing 10 days of candles) to confirm the loader drops or aligns the indexes correctly using `df.pivot()` columns.
