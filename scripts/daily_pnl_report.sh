#!/bin/bash
# Daily PnL report wrapper for cron.
#
# PATH must be set explicitly so cron's minimal environment finds docker/date.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
#
# Generates a daily PnL report from the ggTrader live trading data and pushes
# it to all configured notification channels (Telegram / Discord) via env vars.
#
# Schedule via crontab (run daily at 6am local):
#   0 6 * * * /home/flynn/ggTrader/scripts/daily_pnl_report.sh
#
# Logs to ~/logs/daily_pnl_report_YYYYMMDD.log

set -euo pipefail

PROJECT_DIR="/home/flynn/ggTrader"
LOG_DIR="/home/flynn/logs"
LOG_FILE="${LOG_DIR}/daily_pnl_report_$(date +%Y%m%d).log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

# Use the running ggtrader_live container if available; otherwise spin up a
# one-shot container. The container has the .env file mounted with credentials.
if docker ps --format '{{.Names}}' | grep -q '^ggtrader_live$'; then
    echo "[$(date)] Running pnl-daily inside ggtrader_live container..." >> "${LOG_FILE}"
    docker exec ggtrader_live python -u ggt.py pnl-daily \
        --since "$(date -u -d 'yesterday' +%Y-%m-%d)" \
        >> "${LOG_FILE}" 2>&1
else
    echo "[$(date)] Container not running — using one-shot run..." >> "${LOG_FILE}"
    docker compose run --rm ggtrader_live python -u ggt.py pnl-daily \
        --since "$(date -u -d 'yesterday' +%Y-%m-%d)" \
        >> "${LOG_FILE}" 2>&1
fi

echo "[$(date)] daily_pnl_report.sh complete" >> "${LOG_FILE}"
