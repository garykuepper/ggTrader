-- Phase 4: Kraken Futures historical data backfill.
-- Two hypertables for raw data + two views derived from them.
--
-- Convention: timestamps are stored as ``timestamp without time zone`` (UTC),
-- matching the existing ``ohlcv`` table. The ingester writes UTC values.

-- ---------------------------------------------------------------------------
-- Funding rates (Kraken Futures perpetual contracts).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS funding_rates (
    "timestamp"              TIMESTAMP NOT NULL,
    symbol                   VARCHAR(32) NOT NULL,
    funding_rate             DOUBLE PRECISION NOT NULL,  -- absolute USD/contract per hour
    relative_funding_rate    DOUBLE PRECISION NOT NULL,  -- relative per-interval rate (decimal)
    PRIMARY KEY ("timestamp", symbol)
);
SELECT create_hypertable('funding_rates', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS funding_rates_symbol_ts_idx
    ON funding_rates (symbol, "timestamp" DESC);

-- ---------------------------------------------------------------------------
-- Perpetual OHLCV (Kraken Futures linear perps: PF_XBTUSD, PF_ETHUSD, ...).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS perp_ohlcv (
    "timestamp"  TIMESTAMP NOT NULL,
    symbol       VARCHAR(32) NOT NULL,
    "interval"   VARCHAR(5) NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION,
    volume       DOUBLE PRECISION,
    PRIMARY KEY ("timestamp", symbol, "interval")
);
SELECT create_hypertable('perp_ohlcv', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS perp_ohlcv_symbol_interval_ts_idx
    ON perp_ohlcv (symbol, "interval", "timestamp" DESC);

-- ---------------------------------------------------------------------------
-- funding_apr — realized funding annualized.
-- Kraken funds hourly on PF_*. Annual = rate * 24 * 365.
-- Also expose 24h and 7d rolling averages so the strategy can smooth.
-- ---------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS funding_apr CASCADE;
CREATE MATERIALIZED VIEW funding_apr AS
SELECT
    f."timestamp",
    f.symbol,
    f.relative_funding_rate * 24.0 * 365.0 AS funding_apr,
    AVG(f.relative_funding_rate * 24.0 * 365.0) OVER w_24h AS funding_apr_24h,
    AVG(f.relative_funding_rate * 24.0 * 365.0) OVER w_7d  AS funding_apr_7d,
    AVG(f.relative_funding_rate * 24.0 * 365.0) OVER w_30d AS funding_apr_30d
FROM funding_rates f
WINDOW
    w_24h AS (PARTITION BY f.symbol ORDER BY f."timestamp"
              ROWS BETWEEN 23  PRECEDING AND CURRENT ROW),
    w_7d  AS (PARTITION BY f.symbol ORDER BY f."timestamp"
              ROWS BETWEEN 167 PRECEDING AND CURRENT ROW),
    w_30d AS (PARTITION BY f.symbol ORDER BY f."timestamp"
              ROWS BETWEEN 719 PRECEDING AND CURRENT ROW);

CREATE INDEX IF NOT EXISTS funding_apr_symbol_ts_idx
    ON funding_apr (symbol, "timestamp" DESC);

-- ---------------------------------------------------------------------------
-- basis_series — spot vs perp premium, the funding proxy.
-- Joins perp_ohlcv (PF_XBTUSD 1h) against spot ohlcv (BTC-USD 1h) on timestamp.
-- premium = (perp_close - spot_close) / spot_close  (dimensionless, signed)
-- premium_apr = premium * 365 (NOT × 24×365). Kraken's funding-rate
-- coefficient is 24 — funding converges basis over ~24h, so per-hour
-- premium ≈ per-day funding fraction. ×365 lines up with realized funding
-- on the overlap window (correlation ≈ 0.91, mean bias ≈ +2.85% APR).
-- ---------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS basis_series CASCADE;
CREATE MATERIALIZED VIEW basis_series AS
SELECT
    p."timestamp",
    p.symbol AS perp_symbol,
    s.symbol AS spot_symbol,
    p.close  AS perp_close,
    s.close  AS spot_close,
    (p.close - s.close) / NULLIF(s.close, 0.0) AS premium,
    (p.close - s.close) / NULLIF(s.close, 0.0) * 365.0 AS premium_apr,
    AVG((p.close - s.close) / NULLIF(s.close, 0.0) * 365.0)
        OVER w_24h AS premium_apr_24h,
    AVG((p.close - s.close) / NULLIF(s.close, 0.0) * 365.0)
        OVER w_7d  AS premium_apr_7d,
    AVG((p.close - s.close) / NULLIF(s.close, 0.0) * 365.0)
        OVER w_30d AS premium_apr_30d
FROM perp_ohlcv p
JOIN ohlcv s
  ON s."timestamp" = p."timestamp"
 AND s."interval"  = p."interval"
 AND s.venue       = 'kraken_spot'
 AND s.symbol = CASE p.symbol
                    WHEN 'PF_XBTUSD' THEN 'BTC-USD'
                    WHEN 'PF_ETHUSD' THEN 'ETH-USD'
                    WHEN 'PF_SOLUSD' THEN 'SOL-USD'
                    ELSE NULL
                END
WHERE p."interval" = '1h'
WINDOW
    w_24h AS (PARTITION BY p.symbol ORDER BY p."timestamp"
              ROWS BETWEEN 23  PRECEDING AND CURRENT ROW),
    w_7d  AS (PARTITION BY p.symbol ORDER BY p."timestamp"
              ROWS BETWEEN 167 PRECEDING AND CURRENT ROW),
    w_30d AS (PARTITION BY p.symbol ORDER BY p."timestamp"
              ROWS BETWEEN 719 PRECEDING AND CURRENT ROW);

CREATE INDEX IF NOT EXISTS basis_series_perp_ts_idx
    ON basis_series (perp_symbol, "timestamp" DESC);
