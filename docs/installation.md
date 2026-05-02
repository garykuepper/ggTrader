## ⚙️ Database Setup (TimescaleDB)

`ggTrader` requires a TimescaleDB instance (PostgreSQL with the Timescale extension) to store OHLCV data and research results.

### 1. Installation

You can run TimescaleDB natively or via Docker (recommended for isolation):

```bash
docker run -d --name ggtrader_db -p 5432:5432 -e POSTGRES_PASSWORD=postgres timescale/timescaledb:latest-pg16
```

### 2. Environment Variables

Copy `.env.example` to `.env` and fill in your exchange credentials:

```text
# Kraken (Crypto)
KRAKEN_KEY=your_key
KRAKEN_SECRET=your_secret

# Alpaca (Stocks)
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets

# Database
POSTGRES_CONNECTION_STRING=postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader
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
# Crypto (default)
python ggt.py research --top 50 --workers 5

# Stocks
python ggt.py research --asset-class stocks --top 25 --workers 5
```

### Portfolio Backtest
Replay the latest research results in a combined simulation:
```bash
python ggt.py backtest
```

### Live Trading
Start the execution loop for automated orders:
```bash
# Crypto Live
python ggt.py trade --adaptive-sizing

# Stock Paper Trading
python ggt.py trade --asset-class stocks --paper
```

---
*Back to [README.md](../readme.md)*
