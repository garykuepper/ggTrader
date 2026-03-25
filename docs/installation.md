## ⚙️ Database Setup (TimescaleDB)

`ggTrader` requires a TimescaleDB instance (PostgreSQL with the Timescale extension) to store OHLCV data and research results.

### 1. Installation

You can run TimescaleDB natively or via Docker (recommended for isolation):

```bash
docker run -d --name ggtrader_db -p 5432:5432 -e POSTGRES_PASSWORD=postgres timescale/timescaledb:latest-pg16
```

### 2. Connection String

Update your `.env` file to point to your database:

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ggtrader
```

### 3. Data Preparation

Once the DB is running, use the `ggt` CLI to initialize and sync data:

1. **Check Connectivity**: `python ggt.py db diag`
2. **Sync Universe**: `python ggt.py ingest --days 180`
3. **Optimize Storage**: `python ggt.py db compression --enable`

## 🐳 Docker Deployment (Live Bot)

If you wish to run the live trading engine in a container while keeping the database on the host:

1. **Configure Compose**: Ensure `docker-compose.yaml` uses `host.docker.internal` as the `DB_HOST`.
2. **Build & Run**:
   ```bash
   docker compose up --build -d
   ```

## 🚀 Running ggTrader

### Grand Research (Start Here)
Execute the parallel research pipeline to find the best assets and strategies:
```bash
python ggt.py research --top 50 --workers 5
```

### Portfolio Backtest
Replay the latest research results in a combined simulation:
```bash
python ggt.py backtest
```

### Live Trading
Start the execution loop for automated orders:
```bash
python ggt.py trade
```

---
*Back to [README.md](../readme.md)*
