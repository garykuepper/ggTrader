#!/usr/bin/env bash
# Sleeps until 13:15 PT (15 min after the 12:45 PT paper_trade.sh cron job),
# then runs the reduced-budget gated-WFO position-sizing regime comparison.
# Launched fully detached (setsid + disown) so it survives the parent shell
# ending; see docs/research/2026-08-19-position-sizing-regimes.md and
# scripts/position_sizing_wfo.py for context/method.
set -euo pipefail

cd /home/flynn/ggTrader

TARGET_EPOCH=$(TZ=America/Los_Angeles python3 -c "
import datetime, zoneinfo
tz = zoneinfo.ZoneInfo('America/Los_Angeles')
now = datetime.datetime.now(tz)
target = now.replace(hour=13, minute=15, second=0, microsecond=0)
if target <= now:
    target += datetime.timedelta(days=1)
print(int(target.timestamp()))
")

NOW_EPOCH=$(date +%s)
SLEEP_SECS=$((TARGET_EPOCH - NOW_EPOCH))
if [ "$SLEEP_SECS" -gt 0 ]; then
    echo "[scheduler] sleeping ${SLEEP_SECS}s until 13:15 PT ($(date -d "@${TARGET_EPOCH}"))"
    sleep "$SLEEP_SECS"
fi

echo "[scheduler] launching position_sizing_wfo.py at $(date)"
exec env PYTHONPATH=src .venv/bin/python scripts/position_sizing_wfo.py \
    --mode full \
    --eval-start 2023-10-30 \
    --eval-end 2026-04-30 \
    --n-jobs 3 \
    --out docs/research/_position_sizing_wfo_results.json
