# Stock Research & Trading Implementation Plan

This document describes how to extend ggTrader to research and trade US equities alongside its existing crypto capabilities. The goal is to reuse the asset-agnostic core (VectorBT backtesting, WFO, strategies, metrics, portfolio optimization) and add stock-specific modules for data, execution, regime filtering, and universe selection.

**Data Source:** yfinance (free, no API key, data back to 1980+, full market volume)  
**Broker:** Alpaca (execution only -- keys already configured in `.env`)  
**Universe:** S&P 500 stocks ranked by average daily dollar volume  
**Default Timeframe:** Daily (1d) bars  
**Scope:** Full pipeline -- research, production, and live trading  
**Execution Style:** Limit buy entries + trailing stop loss exits via Alpaca  

---

## Table of Contents

1. [Architecture Approach](#1-architecture-approach)
2. [File Map](#2-file-map)
3. [Step 1: Dependencies & Configuration](#step-1-dependencies--configuration)
4. [Step 2: Data Layer (YFinanceDataLoader)](#step-2-data-layer-yfinancedataloader)
5. [Step 3: Universe Selection](#step-3-universe-selection)
6. [Step 4: Regime Filtering](#step-4-regime-filtering)
7. [Step 5: Pipeline Integration (CLI + Setup)](#step-5-pipeline-integration-cli--setup)
8. [Step 6: Execution Engine](#step-6-execution-engine)
9. [Step 7: Verification & Testing](#step-7-verification--testing)
10. [Key Design Decisions](#key-design-decisions)

---

## 1. Architecture Approach

**Parallel asset-class modules** -- not a deep abstraction layer.

The crypto and stock domains differ in too many subtle ways for a shared interface to be worthwhile:

| Concern | Crypto | Stocks |
|---------|--------|--------|
| Market hours | 24/7 | 9:30 AM - 4:00 PM ET, holidays |
| Data API | CCXT (Kraken) | yfinance (daily OHLCV) |
| Execution API | CCXT market/OCO orders | Alpaca limit buy + trailing stop |
| Regime proxy | BTC EMA(50) vs EMA(200) | SPY EMA(50) vs EMA(200), VIX |
| Symbol format | `BTC-USD`, `ETH-USD` | `AAPL`, `MSFT` |
| Fees | 0.1% (Kraken) | 0% (Alpaca commission-free) |
| Polling cadence | Every 4h (aligned to candle close) | Once daily after market close |

The shared abstractions already exist:
- `BaseDataLoader` for data (`src/ggTrader/data/core/base_loader.py`)
- OHLCV MultiIndex DataFrames for everything downstream (backtesting, WFO, strategies, metrics, portfolio optimization)

An `--asset-class {crypto,stocks}` CLI flag selects the appropriate data loader, execution engine, regime filter, and universe builder. All downstream components are unchanged.

---

## 2. File Map

### New Files

| File | Purpose |
|------|---------|
| `src/ggTrader/data/live/yfinance_loader.py` | `YFinanceDataLoader(BaseDataLoader)` -- fetches daily OHLCV via yfinance, no API key needed |
| `src/ggTrader/data/live/cached_yfinance_loader.py` | `CachedYFinanceLoader(YFinanceDataLoader)` -- DB-first + yfinance fallback, caches all fetches to TimescaleDB |
| `src/ggTrader/data/core/stock_constants.py` | Market hours constants, S&P 500 constituent list |
| `src/ggTrader/core/stock_regime_filtering.py` | SPY EMA + VIX-based regime filters (replaces BTC regime for stocks) |
| `src/ggTrader/core/stock_execution_engine.py` | Alpaca-based live trading engine (limit buys + trailing stops), mirrors `execution_engine.py` structure |
| `src/ggTrader/core/market_hours.py` | Market hours utilities via Alpaca clock API |
| `scripts/update_universe_stocks.py` | S&P 500 volume-ranked universe builder using yfinance, mirrors `scripts/update_universe_ccxt.py` |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `alpaca-py` to dependencies (yfinance already present) |
| `src/ggTrader/utils/config.py` | Add `get_alpaca_credentials(paper=True)` helper |
| `src/ggTrader/utils/run_config.py` | Add `stock_pipeline_config()` with stock-appropriate defaults |
| `src/ggTrader/utils/setup.py` | Branch `load_data_and_setup()` on `config["ASSET_CLASS"]` to use `CachedYFinanceLoader` |
| `src/ggTrader/cli/cmd_research.py` | Add `--asset-class` flag, branch universe script and defaults |
| `src/ggTrader/cli/cmd_backtest.py` | Add `--asset-class` flag |
| `src/ggTrader/cli/cmd_production.py` | Add `--asset-class` flag |
| `src/ggTrader/cli/cmd_trade.py` | Add `--asset-class` flag, instantiate `StockExecutionEngine` |
| `src/ggTrader/core/benchmarking.py` | Skip BTC benchmark when `ASSET_CLASS == "stocks"` |

### Unchanged (Already Asset-Agnostic)

- `src/ggTrader/core/fast_backtest.py` -- VectorBT engine works on any OHLCV
- `src/ggTrader/core/wfo.py` -- Walk-Forward Optimization
- `src/ggTrader/core/sensitivity.py` -- Grid search
- `src/ggTrader/core/metrics.py` -- Sharpe, Sortino, Calmar
- `src/ggTrader/core/portfolio_optimizer.py` -- Allocation strategies
- `src/ggTrader/indicators/strategies.py` -- All 7 entry + 3 exit strategies
- `src/ggTrader/indicators/indicator_precompute.py` -- Indicator caching
- `src/ggTrader/pipeline/pipeline_runner.py` -- Pipeline orchestration
- `src/ggTrader/pipeline/param_grids.py` -- Parameter grids
- `src/ggTrader/data/historical/timescaledb_loader.py` -- DB reads (stocks coexist in same `ohlcv` table)
- `src/ggTrader/data/core/base_loader.py` -- Abstract interface

---

## Step 1: Dependencies & Configuration

### 1a. Add `alpaca-py` dependency

**File:** `pyproject.toml`

Add `alpaca-py` to the `dependencies` list (yfinance is already present):

```python
dependencies = [
    # ... existing deps ...
    "alpaca-py",   # for execution only (order placement, account, market clock)
    # "yfinance",  # already present -- used for stock OHLCV data
]
```

The `alpaca-py` package provides (execution only):

- `TradingClient` -- limit orders, trailing stop orders, account info, clock API

yfinance provides (data only):

- `yf.download()` -- historical daily OHLCV bars back to 1980+, full market volume, no API key

### 1b. Add Alpaca credential helper

**File:** `src/ggTrader/utils/config.py`

Add a `get_alpaca_credentials()` function following the existing `get_db_connection_string()` pattern:

```python
def get_alpaca_credentials(paper: bool = True) -> dict:
    """Return Alpaca API key, secret, and base URL from .env."""
    _load_env()
    if paper:
        return {
            "key_id": os.getenv("APCA_API_KEY_ID"),
            "secret_key": os.getenv("APCA_API_SECRET_KEY"),
            "base_url": os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        }
    return {
        "key_id": os.getenv("APCA_API_LIVE_KEY_ID"),
        "secret_key": os.getenv("APCA_API_LIVE_SECRET_KEY"),
        "base_url": "https://api.alpaca.markets",
    }
```

### 1c. Add stock pipeline config

**File:** `src/ggTrader/utils/run_config.py`

Add `stock_pipeline_config()` alongside the existing `full_pipeline_config()`. Key differences from crypto:

```python
def stock_pipeline_config() -> dict[str, Any]:
    """Defaults for stock research/trading pipeline."""
    return {
        "ASSET_CLASS": "stocks",
        "SYMBOLS_FILE": None,   # populated by universe script
        "MAX_SYMBOLS": 50,
        "START_DATE": os.getenv("GGTRADER_START_DATE", "2023-01-01"),
        "END_DATE": os.getenv("GGTRADER_END_DATE", "2025-12-31"),
        "INTERVAL": "1d",       # daily bars (vs 4h for crypto)
        "FREQ": "1d",
        "START_CASH": 10000,
        "PORTFOLIO_SHARE": 0.10,
        "FEES": 0.0,            # Alpaca is commission-free
        "SLIPPAGE": 0.001,      # tighter slippage for large-cap stocks
        # WFO settings (same structure as crypto)
        "N_SPLITS": 6,
        "TEST_RATIO": 3,
        "MIN_TRADES": 0,
        "MIN_CLOSED_TRADES_TRAIN": 3,
        "TRAIN_METRIC": "composite",
        "TRAIN_METRIC_COMPOSITE_WEIGHTS": {
            "sharpe": 0.25, "sortino": 0.25,
            "calmar": 0.25, "profit_factor": 0.25,
        },
        "MAX_TRAIN_DRAWDOWN_PCT": None,
        "CHUNK_SIZE": 500,
        "USE_VECTORIZED": True,
        "USE_VECTORIZED_SENSITIVITY": True,
        "USE_MOVERS": 0,
        "EXIT_TOURNAMENT": ["atr_trailing", "fixed_sl_tp", "trailing_stop"],
        "SENSITIVITY_EXIT_STRATEGY": "atr_trailing",
        "RECENT_VALIDATION_START_DATE": None,
        "RECENT_VALIDATION_END_DATE": None,
        "RECENT_VALIDATION_USE_CCXT_TAIL": False,  # stocks don't use CCXT
        # Stock-specific regime filtering
        "BTC_REGIME_FILTER": False,     # disable crypto regime
        "SPY_REGIME_FILTER": True,      # enable stock regime
        "SPY_REGIME_FILTER_SHORT_EMA": 50,
        "VIX_REGIME_FILTER": True,
        "VIX_REGIME_THRESHOLD": 25,
        "SPY_REGIME_FILTER_MIN_CORRELATION": 0.5,
        "ALTCOIN_REGIME_FILTER": False,  # N/A for stocks
        # Portfolio gates (same as crypto)
        "MAX_COINS_PER_STRATEGY": 10,
        "MAX_COIN_ALLOCATION": 0.25,
        "MIN_ROBUSTNESS_SCORE": 0.1,
        "MIN_VALID_TRAIN_FOLDS": 3,
        "MIN_FOLD_CONSISTENCY": 0.33,
        "EMA_WARMUP_BARS": 200,
        # Benchmark
        "BENCHMARK_SYMBOL": "SPY",
        # Anti-overfitting (same as crypto)
        "OOS_ROBUSTNESS_BLEND_ALPHA": 0.65,
        "TRAIN_METRIC_NORMALIZE_ZSCORE": True,
        "PARAM_STABILITY_WEIGHT": 0.3,
        "FOLD_CONSISTENCY_IN_GATE": True,
        "FOLD_CONSISTENCY_GATE_FLOOR": 0.25,
        "OOS_STABILITY_WEIGHT": 0.3,
        "WFO_CACHE_ENABLED": True,
    }
```

### 1d. Add stock constants

**File:** `src/ggTrader/data/core/stock_constants.py` (NEW)

```python
# yfinance interval mapping (project canonical -> yfinance accepted values)
# yfinance accepts: "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"
YFINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}

# NYSE market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
MARKET_TIMEZONE = "US/Eastern"

# S&P 500 constituents (static list, update quarterly)
# Can also be fetched dynamically from Wikipedia or a data provider.
SP500_SYMBOLS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
    "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
    # ... full list of ~503 tickers ...
]
```

> **Note:** The full S&P 500 list should either be maintained as a static list updated quarterly, or fetched dynamically from a data source at runtime. A hybrid approach (static fallback + optional dynamic refresh) is recommended.

---

## Step 2: Data Layer (YFinanceDataLoader)

### 2a. YFinanceDataLoader

**File:** `src/ggTrader/data/live/yfinance_loader.py` (NEW)

Implements `BaseDataLoader` using yfinance (already a project dependency). Mirrors `LiveExchangeLoader` (`exchange_loader.py`) in structure but much simpler -- no API keys, no rate limit concerns.

**Why yfinance instead of Alpaca for data:**

- Data back to **1980+** (vs 2016 for Alpaca) -- much deeper history for WFO
- **Full market volume** from all exchanges (Alpaca free tier only has IEX ~2% volume)
- **No API key needed**, no subscription tiers
- Already a dependency (`pyproject.toml`) and used for SPY benchmarks in `benchmarking.py`
- For daily bars there is no disadvantage vs Alpaca

**Key methods:**

```python
class YFinanceDataLoader(BaseDataLoader):
    """Stock data loader using yfinance. Free, no API key, daily bars back to 1980+."""

    def fetch_ohlcv(self, symbols, interval, start_date, end_date, limit=None):
        """Fetch OHLCV via yfinance and return MultiIndex DataFrame matching VectorBT format.

        Uses yf.download() which supports batch multi-symbol downloads in a single call.
        """
        import yfinance as yf

        df = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            interval=interval,    # "1d", "1h", "5m", etc.
            auto_adjust=True,     # adjusted OHLC (splits/dividends)
            group_by="ticker",    # returns MultiIndex (symbol, metric)
        )
        # Reshape to match project convention: MultiIndex (symbol, open/high/low/close/volume)
        # Lowercase metric names to match existing loaders
        # Localize index to UTC
        ...

    def list_symbols(self):
        """Return S&P 500 constituent symbols from stock_constants."""
        from ggTrader.data.core.stock_constants import SP500_SYMBOLS
        return SP500_SYMBOLS

    def get_top_by_volume(self, limit=50, window="30d", filter_symbols=None):
        """Rank stocks by average daily dollar volume using yfinance, return top N.

        Fetches recent daily bars for all candidates, computes avg(close * volume)
        over the window, and ranks descending.
        """
        ...
```

**Critical implementation detail:** The `fetch_ohlcv()` return format must exactly match what `TimescaleDBLoader` and `LiveExchangeLoader` produce:

```text
MultiIndex columns: (symbol, metric)
  - symbol: "AAPL", "MSFT", etc.
  - metric: "open", "high", "low", "close", "volume"
Index: DatetimeIndex (UTC-localized)
```

This is the contract that makes VectorBT, `FastBacktest`, and all strategies work without changes.

**yfinance notes:**

- `yf.download()` supports batch requests (all symbols in one call)
- `auto_adjust=True` handles stock splits and dividends automatically
- For daily bars, no market hours gaps to worry about
- Unofficial API -- occasionally breaks, but stable for years and widely used

### 2b. CachedYFinanceLoader

**File:** `src/ggTrader/data/live/cached_yfinance_loader.py` (NEW)

Follows the exact same pattern as `CachedExchangeLoader` (`cached_loader.py`). **All yfinance fetches are cached to TimescaleDB** so subsequent requests hit the DB first and only fetch new/missing bars from yfinance.

1. Inherit from `YFinanceDataLoader` (instead of `LiveExchangeLoader`)
2. Compose with `TimescaleDBLoader` for DB reads
3. `fetch_ohlcv()` flow:
   - Try DB first via `self.db_loader.fetch_ohlcv()`
   - If data is stale (last timestamp > 1.5 * interval old), fetch missing bars from yfinance
   - **Cache new yfinance data back to DB** via `_cache_to_db()`
   - Return combined, deduplicated result

```python
class CachedYFinanceLoader(YFinanceDataLoader):
    """Stock data loader with TimescaleDB caching. All yfinance fetches are persisted to DB."""

    def __init__(self, connection_string=None):
        super().__init__()
        self.db_loader = TimescaleDBLoader(connection_string=connection_string)
        self.connection_string = self.db_loader.connection_string

    def fetch_ohlcv(self, symbols, interval, start_date, end_date, limit=None, **kwargs):
        # 1. Fetch from DB
        db_df = self.db_loader.fetch_ohlcv(symbols, interval, start_date, end_date, limit)

        # 2. Check freshness (same logic as cached_loader.py lines 71-79)
        needs_fetch = db_df.empty or (now - db_df.index.max()) > (interval_td * 1.5)

        if not needs_fetch:
            return db_df

        # 3. Fetch missing bars from yfinance
        fetch_start = db_df.index.max() + timedelta(days=1) if not db_df.empty else start_date
        live_df = super().fetch_ohlcv(symbols, interval, fetch_start, end_date, limit)

        # 4. Cache to DB (reuse _cache_to_db pattern from cached_loader.py lines 143-201)
        self._cache_to_db(live_df, interval)

        # 5. Combine, deduplicate, return
        combined = pd.concat([db_df, live_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined

    def _cache_to_db(self, df, interval):
        """Save yfinance OHLCV DataFrame to TimescaleDB.

        Same INSERT ... ON CONFLICT upsert pattern as CachedExchangeLoader._cache_to_db().
        The 'trades' column is set to 0 for stock data (yfinance doesn't provide trade counts).
        """
        # Identical to cached_loader.py lines 143-201 except trades=0
        ...
```

**TimescaleDB compatibility:** Stock symbols (`AAPL`, `MSFT`) coexist with crypto symbols (`BTC-USD`, `ETH-USD`) in the same `ohlcv` table. The primary key `(timestamp, symbol, interval)` naturally separates them. No schema changes are needed. The `trades` column is set to `0` for stock data.

---

## Step 3: Universe Selection

**File:** `scripts/update_universe_stocks.py` (NEW)

Mirrors `scripts/update_universe_ccxt.py` -- fetches volume data and outputs a ranked JSON file.

**Flow:**

1. Load S&P 500 constituent list from `stock_constants.SP500_SYMBOLS`
2. Fetch recent daily bars via `yf.download()` for all constituents (single batch call)
3. Compute average daily dollar volume (`close * volume`) over the specified window (default: 30d)
4. Rank by volume descending, take top N
5. Output JSON matching crypto universe format:

```json
[
    {"rank": 1, "symbol": "NVDA", "average_notional_volume": 45200000000},
    {"rank": 2, "symbol": "TSLA", "average_notional_volume": 32100000000},
    ...
]
```

**CLI interface** (matching `update_universe_ccxt.py`):

```bash
python scripts/update_universe_stocks.py --limit 50 --out results/research/top_stocks.json --window 30d
```

**Caching:** Same-day universe files are cached at `results/research/universe_cache_stocks_YYYYMMDD_topN_WINDOW.json` to avoid redundant API calls (matching the crypto caching in `cmd_research.py` lines 122-155).

---

## Step 4: Regime Filtering

**File:** `src/ggTrader/core/stock_regime_filtering.py` (NEW)

Provides the stock equivalent of `regime_filtering.py`. Same function signature pattern so the orchestrator can call either module based on asset class.

### 4a. SPY Regime Filter

Directly analogous to `_compute_btc_regime_mask()` in `regime_filtering.py`:

```python
def _compute_spy_regime_mask(
    ohlcv: pd.DataFrame,
    config: dict,
) -> pd.Series | None:
    """Boolean mask: True when SPY EMA(50) > EMA(200) (bull market).

    Fetches SPY data with EMA_WARMUP_BARS warmup, then compares
    EMA(short) vs EMA(200). Same logic as BTC regime filter but
    using SPY as the benchmark.
    """
    # Try ohlcv first (if SPY is in the universe)
    # Fall back to yfinance (already used in benchmarking.py)
    # Return pd.Series[bool] aligned to ohlcv.index
```

### 4b. VIX Regime Filter

Complementary signal not available in crypto:

```python
def _compute_vix_regime_mask(
    ohlcv: pd.DataFrame,
    config: dict,
) -> pd.Series | None:
    """Boolean mask: True when VIX < threshold (low volatility = bullish).

    Fetches VIX data via yfinance (^VIX ticker).
    Default threshold: 25 (configurable via VIX_REGIME_THRESHOLD).
    """
```

### 4c. SPY Correlation Tiering

Analogous to `_compute_btc_correlations()`:

```python
def _compute_spy_correlations(
    ohlcv: pd.DataFrame,
    config: dict,
) -> dict[str, float]:
    """Per-stock daily log-return correlation with SPY.

    Used for tiered filtering:
    - High correlation (>= 0.5): SPY regime filter applies
    - Low correlation (< 0.5): stock trades freely (e.g., defensive sectors)
    """
```

### 4d. Integration Point

The orchestrator (`orchestrator.py`) currently calls `_compute_btc_regime_mask()` and `_apply_tiered_regime_mask()`. For stocks, it should call the SPY/VIX equivalents. This dispatch happens in `setup.py` or the orchestrator based on `config["ASSET_CLASS"]`:

```python
if config.get("ASSET_CLASS") == "stocks":
    if config.get("SPY_REGIME_FILTER"):
        regime_mask = _compute_spy_regime_mask(ohlcv, config)
    if config.get("VIX_REGIME_FILTER"):
        vix_mask = _compute_vix_regime_mask(ohlcv, config)
        regime_mask = regime_mask & vix_mask  # both must be bullish
else:
    if config.get("BTC_REGIME_FILTER"):
        regime_mask = _compute_btc_regime_mask(ohlcv, config)
```

---

## Step 5: Pipeline Integration (CLI + Setup)

### 5a. CLI: Add `--asset-class` flag

**Files:** `cmd_research.py`, `cmd_backtest.py`, `cmd_production.py`, `cmd_trade.py`

Add to each command's `register_*_parser()`:

```python
parser.add_argument(
    "--asset-class",
    type=str,
    default="crypto",
    choices=["crypto", "stocks"],
    help="Asset class to trade (default: crypto)",
)
```

### 5b. Research command changes

**File:** `src/ggTrader/cli/cmd_research.py`

In `run_research()`, branch on `args.asset_class`:

**Universe selection** (lines 122-155): When `stocks`, call `scripts/update_universe_stocks.py` instead of `scripts/update_universe_ccxt.py`. Cache key changes to `universe_cache_stocks_YYYYMMDD_topN_WINDOW.json`.

**Symbol format** (line 171): Stocks don't need `-USD` suffix. Skip the `s if "-" in s else f"{s}-USD"` transformation for stocks.

**Default interval**: Override to `1d` for stocks (vs `4h` for crypto).

**Config selection**: Use `stock_pipeline_config()` when `stocks`, `full_pipeline_config()` when `crypto`.

### 5c. Setup: Branch data loader

**File:** `src/ggTrader/utils/setup.py`

Modify `load_data_and_setup()` at line 45-47:

```python
# Current:
from ggTrader.data.live.cached_loader import CachedExchangeLoader
loader = CachedExchangeLoader()

# New:
if config.get("ASSET_CLASS") == "stocks":
    from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader
    loader = CachedYFinanceLoader()
else:
    from ggTrader.data.live.cached_loader import CachedExchangeLoader
    loader = CachedExchangeLoader()
```

Also update `load_hybrid_validation_ohlcv()` (line 272 onward) similarly -- when `ASSET_CLASS == "stocks"`, use `CachedYFinanceLoader` instead of `CachedExchangeLoader`, and skip the CCXT tail logic.

### 5d. Benchmarking changes

**File:** `src/ggTrader/core/benchmarking.py`

When `ASSET_CLASS == "stocks"`:
- Skip BTC buy-and-hold benchmark computation
- SPY becomes the primary (and only) benchmark
- The existing `_load_spy_close()` function already handles SPY data via yfinance

---

## Step 6: Execution Engine

### 6a. Market Hours Utility

**File:** `src/ggTrader/core/market_hours.py` (NEW)

Small utility module wrapping Alpaca's clock API:

```python
class MarketHours:
    """NYSE market hours awareness via Alpaca clock API."""

    def __init__(self, trading_client):
        self.client = trading_client

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.client.get_clock()
        return clock.is_open

    def next_market_close(self) -> datetime:
        """Next market close time (ET)."""
        clock = self.client.get_clock()
        return clock.next_close

    def next_market_open(self) -> datetime:
        """Next market open time (ET)."""
        clock = self.client.get_clock()
        return clock.next_open

    def seconds_until_close(self) -> float:
        """Seconds until next market close."""
        return (self.next_market_close() - datetime.now(tz=...).total_seconds()
```

### 6b. StockExecutionEngine

**File:** `src/ggTrader/core/stock_execution_engine.py` (NEW)

Mirrors `ExecutionEngine` (`execution_engine.py`) in structure but uses Alpaca SDK. Key differences:

**Initialization (compare execution_engine.py lines 85-120):**

```python
class StockExecutionEngine:
    """Live stock trading bot using Alpaca and WFO-optimized parameters."""

    # Same STRATEGY_MAP and EXIT_MAP as ExecutionEngine (imported from strategies.py)
    STRATEGY_MAP = { ... }  # identical
    EXIT_MAP = { ... }      # identical

    def __init__(self, config, results_path=None):
        self.config = config
        creds = get_alpaca_credentials(paper=config.get("PAPER", True))
        self.trading_client = TradingClient(
            api_key=creds["key_id"],
            secret_key=creds["secret_key"],
            paper=config.get("PAPER", True),
        )
        self.data_loader = CachedYFinanceLoader()  # yfinance for data, cached to DB
        self.market_hours = MarketHours(self.trading_client)
        self.tracker = TradeTracker(data_dir=config.get("TRACKER_DATA_DIR", "data/live_stocks"))
        # ... rest mirrors ExecutionEngine.__init__
```

**Data fetching:** Uses `self.data_loader.fetch_ohlcv()` via `CachedYFinanceLoader` (yfinance with DB caching).

**Regime filtering:** Calls `stock_regime_filtering._compute_spy_regime_mask()` instead of `_compute_btc_regime_mask()`.

**Order placement -- limit buys + trailing stops:**

The stock execution strategy uses **limit buy orders** for entries and **trailing stop loss orders** for exits. This avoids market order slippage and provides automatic downside protection:

1. **Entry:** When a signal fires, place a limit buy at or near the current ask price. If not filled within the next session, cancel and re-evaluate.
2. **Exit:** Once a position is opened, immediately attach a trailing stop loss order. The trail percentage/amount comes from the WFO-optimized exit strategy parameters (e.g., ATR trailing multiplier or fixed trailing %).

Alpaca SDK order methods replace CCXT:

| Crypto (Kraken/CCXT) | Stocks (Alpaca) |
|----------------------|-----------------|
| `exchange.create_market_buy_order()` | `trading_client.submit_order(LimitOrderRequest(...))` |
| `exchange.create_order(type='trailing-stop')` | `trading_client.submit_order(TrailingStopOrderRequest(...))` |
| Kraken OCO orders | Alpaca bracket orders (`OrderClass.OTO` -- one-triggers-other) |
| `exchange.fetch_balance()` | `trading_client.get_account()` |
| `exchange.fetch_order()` | `trading_client.get_order_by_id()` |
| `exchange.cancel_order()` | `trading_client.cancel_order_by_id()` |

**Position sizing:** Alpaca supports fractional shares, so the existing weight-based sizing from `portfolio_weights.json` works without changes. Total account value = `account.equity`.

**Event loop (compare execution_engine.py 4h polling loop):**

```python
def run_forever(self):
    """Daily trading loop -- runs once after each market close."""
    while True:
        if not self.market_hours.is_market_open():
            # Wait for market to open, then close
            sleep_until(self.market_hours.next_market_close() + timedelta(minutes=5))
        else:
            sleep_until(self.market_hours.next_market_close() + timedelta(minutes=5))

        # Market just closed -- daily bar is complete
        self.logger.info("Market closed. Evaluating signals on completed daily bar...")
        self._fetch_data()
        self._compute_signals()
        self._execute_trades()
        self._reconcile_positions()

        # Sleep until next day's close
        next_close = self.market_hours.next_market_close()
        self.logger.info(f"Next evaluation at {next_close}. Sleeping...")
```

**State persistence:** Same `active_positions.json` pattern as crypto, but stored at `data/active_positions_stocks.json` to keep them separate.

### 6c. Trade command changes

**File:** `src/ggTrader/cli/cmd_trade.py`

When `--asset-class stocks`:

```python
if args.asset_class == "stocks":
    from ggTrader.core.stock_execution_engine import StockExecutionEngine
    engine = StockExecutionEngine(config, results_path=args.results)
else:
    from ggTrader.core.execution_engine import ExecutionEngine
    engine = ExecutionEngine(config, results_path=args.results)
```

---

## Step 7: Verification & Testing

### Integration Tests

1. **Data layer:**
   ```bash
   # Verify YFinanceDataLoader returns correct DataFrame format
   python -c "
   from ggTrader.data.live.yfinance_loader import YFinanceDataLoader
   loader = YFinanceDataLoader()
   df = loader.fetch_ohlcv(['AAPL', 'MSFT'], '1d', start_date=..., end_date=...)
   assert df.columns.nlevels == 2  # MultiIndex (symbol, metric)
   assert set(df.columns.get_level_values(1).unique()) >= {'open','high','low','close','volume'}
   print('PASS: DataFrame format matches VectorBT expectations')
   "
   ```

2. **Cached loader (yfinance -> DB round-trip):**
   ```bash
   # Verify DB caching: first call fetches from yfinance and caches, second call hits DB
   python -c "
   from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader
   loader = CachedYFinanceLoader()
   df1 = loader.fetch_ohlcv(['AAPL'], '1d', ...)  # fetches from yfinance, caches to TimescaleDB
   df2 = loader.fetch_ohlcv(['AAPL'], '1d', ...)  # should hit DB first (no yfinance call)
   print('PASS: Cached loader works')
   "
   ```

3. **Universe selection:**
   ```bash
   python scripts/update_universe_stocks.py --limit 10 --out /tmp/test_universe.json --window 30d
   # Verify output is valid JSON with rank, symbol, average_notional_volume
   ```

4. **Full research pipeline (small scale):**
   ```bash
   ggt research --asset-class stocks --top 10 --days 730 --workers 2
   # Should complete WFO on 10 stocks, produce run_results.json and research_report.md
   # Report should show SPY benchmark (no BTC benchmark)
   ```

5. **Production pipeline:**
   ```bash
   ggt production --asset-class stocks --results-dir results/research/research_TIMESTAMP
   # Should produce portfolio_weights.json for stocks
   ```

6. **Live trading (paper, dry-run):**
   ```bash
   ggt trade --asset-class stocks --dry-run --paper
   # Should connect to Alpaca paper, respect market hours, log signal decisions
   # Should NOT place actual orders in dry-run mode
   ```

7. **Crypto regression:**
   ```bash
   ggt research --top 10 --days 365 --workers 2
   # Verify existing crypto pipeline still works unchanged (default --asset-class crypto)
   ```

### Unit Tests

Add test files mirroring existing test structure:

- `tests/test_yfinance_loader.py` -- verify DataFrame format matches VectorBT expectations
- `tests/test_cached_yfinance_loader.py` -- verify DB caching round-trip (yfinance -> DB -> DB hit)
- `tests/test_stock_regime_filtering.py` -- verify SPY/VIX regime mask computation
- `tests/test_stock_execution_engine.py` -- mock Alpaca orders, verify limit buy + trailing stop placement

---

## Key Design Decisions

### Why parallel modules instead of shared abstractions?

The existing `ExecutionEngine` has 460+ lines of Kraken-specific logic (CCXT market orders, Kraken trailing stop offset formatting, Kraken OCO orders, BTC regime filtering, 4-hour event loop). Making this work for both asset classes would require pervasive `if asset_class == "stocks"` branches. A parallel `StockExecutionEngine` shares the same conceptual structure but has clean, Alpaca-specific logic.

### Why not modify existing strategies?

The 7 entry strategies (EMA cross, RSI reversal, MACD cross, Bollinger Bands, PSAR+ADX, Donchian breakout, Supertrend) and 3 exit strategies (ATR trailing, fixed SL/TP, trailing stop) all operate on OHLCV data. They don't know or care whether the data represents crypto or stocks. The WFO process will automatically find which strategy and parameters work best for each stock. Stock-specific strategies (e.g., earnings-based signals, sector rotation) can be added to the strategy registry later without touching infrastructure.

### Why can the same TimescaleDB table hold both?

The `ohlcv` table has `(timestamp, symbol, interval)` as the primary key. Stock symbols (`AAPL`, `MSFT`) naturally occupy a different namespace from crypto symbols (`BTC-USD`, `ETH-USD`). No schema changes needed. Queries using `TimescaleDBLoader` will only return data for the requested symbols.

### Why yfinance for data and Alpaca for execution?

- **yfinance for data:** Free, no API key, data back to 1980+ (vs Alpaca 2016), full market volume (vs Alpaca free tier IEX-only ~2%), already a project dependency, batch downloads for 500+ symbols in one call. For daily bars there is no disadvantage.
- **Alpaca for execution:** Commission-free trading, limit orders + trailing stops, fractional shares, paper trading mode, market clock API. Keys already configured in `.env`.
- **Clean separation:** Data source and broker are independent concerns. If yfinance ever breaks, swapping in another data source (e.g., Alpaca, Alpha Vantage) only requires a new `BaseDataLoader` implementation -- execution is unaffected.

### Why daily bars as default?

- Avoids market hours complexity (overnight gaps, weekend gaps, half-days)
- Natural timeframe for swing trading on stocks
- Parallels the 4h crypto approach (both are multi-day holding periods)
- Hourly bars can be added later as a `--interval 1h` flag

---

*Back to [README.md](../README.md)*
