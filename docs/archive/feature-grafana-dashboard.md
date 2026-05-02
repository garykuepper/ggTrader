# Implementation Plan: Grafana Dashboard Integration

## Objective
Enable real-time visualization of ggTrader's performance and market regimes using a Grafana dashboard accessible on the local network.

## Key Components
- **Grafana Service**: A new container in `docker-compose.yaml`.
- **Database Connection**: Use the existing TimescaleDB/Postgres instance as the data source.
- **Provisioning**: Pre-configure the PostgreSQL data source and an initial dashboard.
- **Data Ingestion**: Update `TradeTracker` or `ExecutionEngine` to periodically mirror live balance snapshots and closed trades to the PostgreSQL database (currently they are only in CSV).

## Implementation Steps

### 1. Infrastructure (Docker)
-   Update `docker-compose.yaml` to include the `grafana` service.
-   Map port `3001:3000` to avoid conflicts and allow local network access.
-   Configure volume persistence for Grafana data.
-   Add `host.docker.internal` extra host for DB access.

### 2. Grafana Provisioning
-   Create `grafana/provisioning/datasources/datasource.yaml` to automatically connect to the `ggtrader` Postgres DB.
-   (Optional) Create `grafana/provisioning/dashboards/dashboard.yaml` and a JSON definition for the initial dashboard.

### 3. Database Mirroring (Live Data)
-   Modify `TradeTracker` (or `ExecutionEngine`) to use `ResultDBManager` to mirror:
    -   **Balance Snapshots**: Every time a CSV row is written, also insert into the `equity_curves` table (using a special `run_id='LIVE'`).
    -   **Closed Trades**: Mirror to the `trades` table (using `run_id='LIVE'`).
-   This ensures Grafana has live data without needing a separate CSV exporter.

### 4. Dashboard Design (SQL-based)
-   **Equity Curve**: `SELECT timestamp, equity_value FROM equity_curves WHERE run_id = 'LIVE' ORDER BY timestamp ASC;`
-   **PnL Breakdown**: Aggregate from the `trades` table where `run_id = 'LIVE'`.
-   **Market Regime**: Query the `ohlcv` table for BTC price vs its EMA (or create a new table for explicit regime logging).

## Verification & Testing
1.  **Connectivity**: Access `http://<local-ip>:3001` and verify the data source is working.
2.  **Live Updates**: Open a position or wait for a balance snapshot and verify it appears in Grafana within 5 minutes.
3.  **Accuracy**: Compare Grafana stats with the `ggt pnl-daily` report output.
