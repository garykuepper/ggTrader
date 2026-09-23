#!/bin/bash
# Daily paper-trading PnL report -> Telegram.
#
# Reads the last two `paper_snapshots` rows and yesterday's `paper_trades`
# straight from the ggtrader_db container (no ggt subcommand; the old
# `pnl-daily` command was deleted in 82931a4 and this script sat broken
# from 2026-05-06 until 2026-09). Equities-only: the crypto book is parked.
#
# Schedule: 0 6 * * 2-6 /home/flynn/ggTrader/scripts/daily_pnl_report.sh
# (Tue-Sat 06:00 PT, i.e. the morning after each Mon-Fri session.)
# Logs to ~/logs/daily_pnl_report_YYYYMMDD.log

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -euo pipefail

PROJECT_DIR="/home/flynn/ggTrader"
LOG_DIR="/home/flynn/logs"
LOG_FILE="${LOG_DIR}/daily_pnl_report_$(date +%Y%m%d).log"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"
set -a; . "${PROJECT_DIR}/.env"; set +a

SNAP=$(docker exec ggtrader_db psql -U ggtrader -d ggtrader -At -F '|' -c "
  SELECT run_date, round(portfolio_value::numeric, 2), round(cash::numeric, 2),
         (SELECT count(*) FROM jsonb_object_keys(positions))
  FROM paper_snapshots ORDER BY run_date DESC LIMIT 2;")

TRADES=$(docker exec ggtrader_db psql -U ggtrader -d ggtrader -At -F '|' -c "
  SELECT side || ' ' || symbol || ' \$' || round(amount::numeric, 0)
         || CASE WHEN reason = 'cash_sweep' THEN ' (sweep)' ELSE '' END
  FROM paper_trades
  WHERE run_date = (SELECT max(run_date) FROM paper_snapshots)
  ORDER BY side, symbol;")

FIRST=$(docker exec ggtrader_db psql -U ggtrader -d ggtrader -At -F '|' -c "SELECT round(portfolio_value::numeric, 2) FROM paper_snapshots ORDER BY run_date ASC LIMIT 1;")

python3 - "${SNAP}" "${TRADES}" "${FIRST}" <<'PY' >> "${LOG_FILE}" 2>&1
import sys
sys.path.insert(0, "/home/flynn/scripts")
from notify import load_telegram_credentials, send_telegram

snap_raw, trades_raw, first_raw = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r.split("|") for r in snap_raw.strip().splitlines() if r.strip()]
if not rows:
    print("no snapshots; nothing to report")
    sys.exit(0)
today = rows[0]
pv, cash, npos = float(today[1]), float(today[2]), int(today[3])
first = float(first_raw.strip()) if first_raw.strip() else None
lines = [f"📊 ggTrader paper — {today[0]}", f"Value: ${pv:,.2f}"]
if len(rows) > 1:
    prev = float(rows[1][1])
    delta = pv - prev
    lines.append(f"Day: {delta:+,.2f} ({delta / prev:+.2%}) vs {rows[1][0]}")
if first:
    lines.append(f"Since inception: {pv - first:+,.2f} ({pv / first - 1:+.2%})")
lines.append(f"Cash: ${cash:,.2f} ({cash / pv:.1%}) · Positions: {npos}")
trades = [t for t in trades_raw.strip().splitlines() if t.strip()]
lines.append(f"Orders: {len(trades)}" + (("\n  " + "\n  ".join(trades)) if trades else ""))
msg = "\n".join(lines)
print(msg)
token, chat_id = load_telegram_credentials()
ok = send_telegram(msg, token, chat_id)
print("telegram:", "sent" if ok else "FAILED")
PY

echo "[$(date)] daily_pnl_report.sh complete" >> "${LOG_FILE}"
