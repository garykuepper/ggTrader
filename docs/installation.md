# Installation Guide

Welcome! ggTrader is a research platform (or "lab") designed to test and tune trading strategies. It uses a method called **Walk-Forward Optimization**—which simulates trading by constantly adjusting strategy rules using past data and testing them on new data.

To run ggTrader, you will need:
1. **Python 3.10+**: The programming language used to run the software.
2. **TimescaleDB**: A specialized database built for handling time-series data (like price history over time).
3. **Exchange API Keys** (Optional): Digital credentials that allow ggTrader to download market data from exchanges like Kraken or Binance.

---

## 1. Setting Up Python

To start, you need to download the ggTrader source code and install its Python dependencies. We recommend using a **virtual environment** (`venv`), which is a private space on your computer that keeps these files separate from your other projects.

Run the following commands in your terminal:

```bash
# 1. Download the repository from GitHub
git clone https://github.com/garykuepper/ggTrader.git

# 2. Enter the project folder
cd ggTrader

# 3. Create a clean virtual environment named ".venv"
python -m venv .venv

# 4. Activate the virtual environment
# (On macOS/Linux use this command. On Windows, use: .venv\Scripts\activate)
source .venv/bin/activate

# 5. Install the project in "editable" mode (so changes to code take effect immediately)
pip install -e .
```

Verify that the command-line interface (CLI) is working by running:

```bash
ggt --help
```

You should see a list of the three main tools: `lab` (for simulations), `ingest` (for downloading data), and `db` (for database management).

---

## 2. Setting Up TimescaleDB

TimescaleDB is a popular database for storing price history. The simplest way to run it is through **Docker**, a tool that runs software in self-contained packages ("containers") without messing up your computer's settings.

If you have Docker installed, you can start the database with this command:

```bash
docker run -d \
  --name ggtrader_db \
  -p 5433:5432 \
  -e POSTGRES_USER=ggtrader \
  -e POSTGRES_PASSWORD=ggtrader \
  -e POSTGRES_DB=ggtrader \
  timescale/timescaledb:latest-pg16
```

> [!NOTE]
> The setting `-p 5433:5432` maps the database to port `5433` on your computer. This prevents conflicts if you already have standard PostgreSQL running on port `5432`.

---

## 3. Configuring the Environment

Settings and credentials (like passwords or API keys) are stored in a hidden file named `.env`. 

Copy the template file `.env.example` to a new file named `.env`:

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in the details:

```text
# Database connection settings
POSTGRES_CONNECTION_STRING=postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader

# Exchange API keys (only required if you use 'ggt ingest' to download raw data)
BINANCEUS_KEY=...
BINANCEUS_SECRET=...
KRAKEN_KEY=...
KRAKEN_SECRET=...

# Chat alerts (Optional: sends notifications to your phone/chat apps)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
```

> [!TIP]
> If you are just doing basic research (which downloads free stock data from Yahoo Finance), you **do not** need to provide exchange API keys.

---

## 4. Confirming Everything Works (First Run)

Let's test if ggTrader can connect to your database:

```bash
ggt db diag
```

If it successfully connects, it will print out table sizes and row counts (which will be empty at first).

Next, run a quick simulation to confirm that the entire backtesting pipeline works:

```bash
ggt lab --strategy wfo_tournament --top-n 10 --eval-start 2024-01-01 --eval-end 2024-12-31
```

If the command finishes without errors and prints out a "run ID" and summary table, your installation is complete!

---

## 5. Running with Docker (Optional)

If you prefer running everything in containerized environments, you can use the provided `docker-compose.yaml` file:

```bash
# Build the ggTrader container image
docker compose build --no-cache

# Start the database and services in the background
docker compose up -d
```

To run simulations inside the Docker container, prefix your commands like this:

```bash
docker compose run --rm ggtrader_live python ggt.py lab --strategy wfo_tournament
```

The container is pre-configured to reach the database running on your host machine automatically.

---

## What's Next?

- [CLI Reference](cli_reference.md) — How to use all the different command-line flags.
- [Architecture Guide](architecture.md) — How the system works under the hood.
- [Developer Guidelines](../AGENTS.md) — Coding standards for modifying the project.

---

*Back to [README.md](../README.md).*
