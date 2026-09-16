#!/bin/bash
# Daily paper trading wrapper for cron.
# Runs the ensemble strategy on Alpaca paper trading.
#
# Schedule via crontab (run daily at 3:45 PM ET / 12:45 PM PT — 15 min before
# the US close so DAY market orders fill same session instead of queuing to the
# next open, which otherwise adds overnight-gap slippage vs the backtest):
#   45 12 * * 1-5 /home/flynn/ggTrader/scripts/paper_trade.sh
#
# Logs to ~/logs/paper_trade_YYYYMMDD.log

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -euo pipefail

PROJECT_DIR="/home/flynn/ggTrader"
LOG_DIR="/home/flynn/logs"
LOG_FILE="${LOG_DIR}/paper_trade_$(date +%Y%m%d).log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

# Page on any failure. The trader's own Telegram notifier lives inside the
# Python run, so a crash before its first send (signal generation, the
# leverage guard, a DB outage) previously paged nobody -- the 7/29-7/30
# leverage-guard halt went unnoticed for a day. This trap is the outer
# net: nonzero exit => Telegram with the tail of today's log.
alert_on_failure() {
    local rc=$?
    set +e
    if [ "${rc}" -ne 0 ]; then
        local tail_text
        tail_text=$(tail -n 20 "${LOG_FILE}" 2>/dev/null | sed 's/[`*_\[]/ /g' || true)
        { set -a; . "${PROJECT_DIR}/.env"; set +a; } 2>/dev/null || true
        python3 - "${rc}" "${tail_text}" <<'PY'
import sys
sys.path.insert(0, "/home/flynn/scripts")
from notify import load_telegram_credentials, send_telegram
rc, tail = sys.argv[1], sys.argv[2]
token, chat_id = load_telegram_credentials()
send_telegram(f"🚨 ggTrader paper_trade.sh FAILED (exit {rc})\n\n{tail}", token, chat_id)
PY
    fi
    exit "${rc}"
}
trap alert_on_failure EXIT

echo "[$(date)] Starting paper trading run..." >> "${LOG_FILE}"

docker compose run --rm ggtrader_live python -u ggt.py paper --live \
    >> "${LOG_FILE}" 2>&1

echo "[$(date)] paper_trade.sh complete" >> "${LOG_FILE}"
