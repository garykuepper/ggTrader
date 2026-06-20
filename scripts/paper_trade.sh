#!/bin/bash
# Daily paper trading wrapper for cron.
# Runs the ensemble strategy on Alpaca paper trading.
#
# Schedule via crontab (run daily at 4:30 PM ET / 1:30 PM PT):
#   30 13 * * 1-5 /home/flynn/ggTrader/scripts/paper_trade.sh
#
# Logs to ~/logs/paper_trade_YYYYMMDD.log

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -euo pipefail

PROJECT_DIR="/home/flynn/ggTrader"
LOG_DIR="/home/flynn/logs"
LOG_FILE="${LOG_DIR}/paper_trade_$(date +%Y%m%d).log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

echo "[$(date)] Starting paper trading run..." >> "${LOG_FILE}"

docker compose run --rm ggtrader_live python -u ggt.py paper \
    >> "${LOG_FILE}" 2>&1

echo "[$(date)] paper_trade.sh complete" >> "${LOG_FILE}"
