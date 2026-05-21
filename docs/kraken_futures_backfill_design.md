# Kraken Futures historical backfill — design notes

**Status:** Followup to Phase 3. Not blocking. Required before the
CashAndCarryBTC strategy can be backtested against real data.

## Goal

Make Kraken Futures (dated quarterly) OHLCV + derived basis available in
TimescaleDB so `SyntheticFeatureStore` can be swapped for a real
`features/derivatives.py` reading from the DB.

Concrete coverage target: 1h bars for BTC quarterly contracts from
2022-01-01 onward, plus enough of each contract's full lifecycle that the
backtest's roll logic sees realistic basis at every roll point.

## Data model

Two new TimescaleDB hypertables (do not touch existing `ohlcv` which is
spot-only):

### `futures_ohlcv`
```sql
CREATE TABLE futures_ohlcv (
    ts          TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,           -- "BTC-USD-260626"
    venue       TEXT NOT NULL,           -- "kraken_futures"
    expiry      DATE NOT NULL,
    open        NUMERIC,
    high        NUMERIC,
    low         NUMERIC,
    close       NUMERIC,
    volume      NUMERIC,
    PRIMARY KEY (ts, symbol)
);
SELECT create_hypertable('futures_ohlcv', 'ts');
CREATE INDEX ON futures_ohlcv (symbol, ts DESC);
CREATE INDEX ON futures_ohlcv (expiry);
```

### `futures_basis`
Derived (refreshed nightly) — annualized basis for each (contract, ts):
```sql
CREATE MATERIALIZED VIEW futures_basis AS
SELECT
    f.ts,
    f.symbol AS future_symbol,
    s.symbol AS spot_symbol,
    f.expiry,
    f.close AS future_price,
    s.close AS spot_price,
    EXTRACT(EPOCH FROM (f.expiry::timestamptz - f.ts)) / 86400.0 AS dte_days,
    (f.close - s.close) / s.close
        * 365.0
        / GREATEST(EXTRACT(EPOCH FROM (f.expiry::timestamptz - f.ts)) / 86400.0, 1) AS basis_apr
FROM futures_ohlcv f
JOIN ohlcv s
    ON s.ts = f.ts
   AND s.symbol = SPLIT_PART(f.symbol, '-', 1) || '-' || SPLIT_PART(f.symbol, '-', 2);
```

## Source

`ccxt.krakenfutures` — public endpoints, no auth needed for historical OHLCV.

```python
import ccxt
k = ccxt.krakenfutures({"enableRateLimit": True})
markets = k.load_markets()
# Find dated futures
dated = [m for m in markets.values() if m.get("type") == "future"]
# Per contract:
bars = k.fetch_ohlcv("BTC/USD:USD-260626", timeframe="1h", since=start_ms, limit=720)
# Paginate by advancing `since` until response is shorter than limit.
```

Rate-limit characteristics: ccxt's `enableRateLimit` respects Kraken's
500 req/min for public endpoints. A full backfill of 12 contracts × ~90 days
× 24 bars/day ÷ 720 bars/call ≈ ~40 paginated calls per contract = ~500
calls total. Comfortably under one minute of API time.

## Coverage challenges

1. **Contract availability.** Kraken's public history endpoints only return
   data for contracts the exchange still indexes. Matured contracts (>1
   year past expiry) may be absent. Verify by sampling
   `fetch_markets(params={"includeInactive": True})` and noting any gaps
   pre-2024.

2. **Roll-window gaps.** Two contracts trade simultaneously near a roll;
   both should be ingested. The strategy's roll logic chooses one but
   research/diagnostics will want both.

3. **First-day liquidity.** Quarterly contracts list ~6 months before
   expiry. Early days have wide spreads / sparse volume. Mark these in a
   `quality` column or filter at query time.

4. **USD vs BTC margined.** Both `BTC/USD:USD-NNNNNN` (linear) and
   `BTC/USD:BTC-NNNNNN` (inverse) exist. Cash-and-carry wants linear (P&L
   in USD). Ingest only the linear ones; ignore inverse for now.

## Ingester implementation

New module: `src/ggTrader/data/sources/kraken_futures.py` implementing the
`BarDataSource` Protocol from `data/sources/base.py` (also new — Phase 2
work for the formal Protocol; in the interim, ad-hoc).

```python
class KrakenFuturesBarSource:
    def __init__(self, exchange: ccxt.krakenfutures | None = None) -> None: ...
    def get_bars(self, instrument, start, end, timeframe="1h") -> pd.DataFrame: ...
    def backfill_all_dated_btc(self, start: datetime, end: datetime) -> None: ...
```

`backfill_all_dated_btc` paginates each contract from listing date to
expiry+1, INSERT-ON-CONFLICT-DO-NOTHING into `futures_ohlcv`.

## CLI

New ingest subcommand:
```bash
ggt ingest --venue kraken_futures --asset BTC --since 2022-01-01
```

Wire through `cli/cmd_ingest.py` (already exists — add a `--venue` arg).

## Continuous front-month series

For backtests that want a single price series rather than picking-and-rolling
themselves, build a materialized view that at each ts picks the contract with
the smallest positive `(expiry - ts)` beyond a configurable roll buffer
(default 24h). Materialize nightly.

```sql
CREATE MATERIALIZED VIEW front_month_futures AS ...
```

This is *not* required by `CashAndCarryBTC` (it manages its own active-future
selection via `CarryUniverse`), but cross-sectional or RV strategies will
need it.

## Verification plan

1. Run backfill end-to-end against 2022-2025.
2. Compare TimescaleDB row counts to expected: ~12 contracts × ~180 days ×
   24 bars/day ≈ 50k rows. Sanity-check.
3. Sample basis at known-contango months (e.g., late 2021 if data exists)
   and verify `basis_apr` lands in 5–20% range.
4. Swap `SyntheticFeatureStore` → `TimescaleDBFeatureStore` in
   `cmd_backtest_strategy.py`. Re-run the Phase 3 backtest. Compare metrics
   to synthetic baseline.
5. Re-run the Phase 3 integration test against real data; expect the
   `basis_apr >= 0.10` assertion still holds since we only enter trades that
   cleared the threshold.

## Effort estimate

- Ingester code + tests: 1 day
- Backfill run + verification: 0.5 day
- Wire real-data path into `cmd_backtest_strategy.py`: 0.5 day
- Total: ~2 days of focused work.

Defer until after the Phase 3 architectural feedback items (see
`phase3_architecture_feedback.md`) are addressed. Backfilling onto an
abstraction that's about to be reshaped is wasted churn.
