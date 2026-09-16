# Ops Track and Tape Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the five live-trader fixes from the spec's §4 in the spec's
order (core revert first), plus the benchmark-tape restore from §6.1 that
blocks every research run, so the research track can start while the
paper/ refactor waits its turn.

**Architecture:** Six sequential deploys to `src/ggTrader/paper/` and
`scripts/`, each its own commit, each verified on the next 12:45 PT live
run before the next begins. Tasks 1–4 reuse the already-written,
fully-specified tasks in
`docs/superpowers/plans/2026-09-11-paper-trading-remediation.md` (referred
to below as **REM**), re-sequenced; Tasks 5–6 are new. The paper/ refactor
(spec §5), research runs (§6.2–6.5), and prune (§7) get their own plans
after this one lands.

**Tech Stack:** Python 3.12, pytest + `unittest.mock`, SQLAlchemy /
TimescaleDB (`ggTrader.lab.persist.get_engine`), Alpaca paper broker,
bash cron wrappers, `~/scripts/notify.py` for Telegram.

**Spec:** `docs/superpowers/specs/2026-09-16-cleanup-and-strategy-program-design.md`
(§4 ops track, §6.1 tape restore, §8 success criteria).

## Global Constraints

- One live change per deploy cycle; verify on the next 12:45 PT run before
  starting the next task (spec §3).
- Deploy loop for anything under `src/ggTrader/paper/`: `git push origin
  main` → `.github/workflows/docker-build.yml` publishes
  `ghcr.io/garykuepper/ggtrader:latest` (~2.5 min) → `docker compose pull
  ggtrader_live && docker compose up -d ggtrader_live` in
  `/home/flynn/ggTrader`. The container is not bind-mounted; a commit has
  zero live effect until this runs.
- `ruff check .` and `ruff format .` clean before every commit
  (`line-length = 100`, rules `E`/`F`/`W`/`I`).
- Absolute imports rooted at `src` (`from ggTrader.<package> import ...`).
- Every DB call in `src/ggTrader/paper/` fails soft (`try/except` → log
  warning → safe default), never raises out of the run loop.
- Tests never hit a real DB or send a real Telegram message: patch
  `_get_engine` / the notifier as `tests/paper/` already does.
- `ohlcv.timestamp` is naive UTC; never write a tz-aware datetime into it.
- Never touch `data/` or DB config except through the explicit one-off
  script in REM Task 2.
- Commit trailer for every commit in this plan:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01Y3VzwB87KQ1o2MwazR5XH8`
  (replace the `Claude Sonnet 5` trailer shown inside REM's commit
  messages with these two lines).

---

### Task 0: Commit the pending doc state this plan builds on

**Files:**
- Add: `docs/superpowers/plans/2026-09-11-paper-trading-remediation.md` (untracked)
- Modify (already edited, uncommitted): `docs/next_steps.md`,
  `docs/research/RESEARCH_SNAPSHOT.md`,
  `docs/research/prompts/local-implementation-prompt-TEMPLATE.md`,
  `docs/research/prompts/web-strategy-research-prompt.md`

**Interfaces:** none.

- [ ] **Step 1: Confirm the working tree contains only those docs**

Run: `git status --short`
Expected: exactly the four `M` docs lines and the one `??` plan line above.
If anything else appears, stop and ask.

- [ ] **Step 2: Commit**

```bash
git add docs/next_steps.md docs/research/RESEARCH_SNAPSHOT.md \
  docs/research/prompts/local-implementation-prompt-TEMPLATE.md \
  docs/research/prompts/web-strategy-research-prompt.md \
  docs/superpowers/plans/2026-09-11-paper-trading-remediation.md
git commit -m "docs: 2026-09-10 active step, snapshot regen, and the 09-11 remediation plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3VzwB87KQ1o2MwazR5XH8"
```

---

### Task 1: Core revert (REM Task 4, moved first)

**Files:**
- Modify: `src/ggTrader/paper/signal_runner.py` (add `generate_core_signals`)
- Modify: `src/ggTrader/paper/trader.py:33` (import), `:412` (call site)
- Test: `tests/paper/test_signal_runner.py`, `tests/paper/test_trader.py`

**Interfaces:**
- Consumes: `generate_signals(universe="sp500") -> dict` (existing).
- Produces: `generate_core_signals() -> dict` with keys `sleeves`
  (`{"sp500": <signals dict>}`), `weights` (`{"sp500": 1.0}`), `scale`
  (`1.0`), `rebalanced_today` (`False`), `fallback_used` (`False`). Same
  shape as `generate_blended_signals()`, so `trader.py`'s sleeve loop and
  `RiskGuard.sleeve_position_notional(pv, 1.0, 1.0)` resolve to
  `pv * 0.033` with no further change.

- [ ] **Step 1: Execute REM Task 4, Steps 1–7 exactly as written**

Open `docs/superpowers/plans/2026-09-11-paper-trading-remediation.md`,
section "Task 4: Revert live trading from the 3-sleeve blend to the
standalone SP500 core", and run its Steps 1 through 7 (failing test →
implement `generate_core_signals` → switch the call site → `sed` rename of
the patch target in `tests/paper/test_trader.py` → full paper suite →
lint → commit). Use this plan's commit trailer.

- [ ] **Step 2: Add the sizing acceptance test the spec requires**

The spec's acceptance for this step is "new buys size near 3.3% of PV".
REM Task 4 does not assert that, so add to
`tests/paper/test_signal_runner.py`, inside `TestGenerateCoreSignals`:

```python
    @patch("ggTrader.paper.signal_runner.generate_signals")
    def test_core_weights_restore_full_slot_size(self, mock_generate):
        """weights=1.0, scale=1.0 must give sleeve_position_notional ==
        position_notional (3.3% of PV), i.e. no blend-overlay shrink."""
        from ggTrader.paper.risk import RiskConfig, RiskGuard

        mock_generate.return_value = {
            "buys": [], "sells": [], "as_of": "2026-09-16",
            "universe_size": 503, "gate": {"gate_enabled": False},
        }
        core = generate_core_signals()
        guard = RiskGuard(RiskConfig())
        pv = 102_000.0
        sleeve = guard.sleeve_position_notional(pv, core["weights"]["sp500"], core["scale"])
        assert sleeve == guard.position_notional(pv)
        assert sleeve == pytest.approx(pv * 0.033, abs=0.01)
```

Add `import pytest` at the top of the file if it is not already imported
(it is — line 9).

- [ ] **Step 3: Run the test**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_signal_runner.py::TestGenerateCoreSignals -v`
Expected: PASS (both tests). If `RiskGuard`'s constructor signature differs
from `RiskGuard(RiskConfig())`, read `src/ggTrader/paper/risk.py:1-60` and
match it; do not change `risk.py`.

- [ ] **Step 4: Lint, amend into the revert commit, deploy**

```bash
ruff check --fix tests/paper/test_signal_runner.py && ruff format tests/paper/test_signal_runner.py
git add tests/paper/test_signal_runner.py
git commit --amend --no-edit
git push origin main
# wait ~2.5 min for the CI image, then:
docker image inspect ghcr.io/garykuepper/ggtrader:latest --format '{{.Created}}'   # before
docker compose pull ggtrader_live && docker compose up -d ggtrader_live
docker image inspect ghcr.io/garykuepper/ggtrader:latest --format '{{.Created}}'   # after: newer
```

- [ ] **Step 5: Verify on the next 12:45 PT run**

After the run, check all three:

```bash
grep -E "Scale: |Weights: " ~/logs/paper_trade_$(date +%Y%m%d).log
```
Expected: `Scale: 1.00x | Weights: sp500=100%`.

```sql
-- via the postgres MCP or: docker exec ggtrader_db psql -U postgres -d ggtrader -c "..."
SELECT run_date, side, symbol, amount FROM paper_trades
WHERE run_date = CURRENT_DATE AND side = 'BUY' AND reason IS DISTINCT FROM 'cash_sweep';
```
Expected: every strategy BUY amount within ±5% of `0.033 * portfolio_value`
(≈ $3,400 on a $102K account), and every symbol an SP500 member. If no
buys fired that day, wait for the first day with one before starting
Task 2.

Existing midcap/nasdaq positions stay until their own exit signals fire.
Do not force-sell them.

---

### Task 2: Persistent split state and NAV restatement (REM Tasks 1–2)

**Files:**
- Modify: `src/ggTrader/paper/persist.py`, `src/ggTrader/paper/trader.py`
- Create: `scripts/restate_split_snapshots.py` (REM Task 2 Step 7)
- Test: `tests/paper/test_paper_persist.py`, `tests/paper/test_trader.py`

**Interfaces:**
- Produces (REM Task 1): `get_open_split_corrections() -> dict[str, float]`,
  `save_split_correction(symbol: str, ex_date: str, factor: float) -> None`,
  `clear_split_correction(symbol: str) -> None`, table `paper_split_state`.
- Produces (REM Task 2): `_compute_split_corrections` reads persisted
  state first; the `_SPLIT_LOOKBACK_DAYS` constant is removed.

- [ ] **Step 1: Execute REM Task 1, Steps 1–5 exactly as written**

Failing tests for the three persist functions → implement schema and
functions → tests pass → lint → commit (this plan's trailer).

- [ ] **Step 2: Execute REM Task 2, Steps 1–6 exactly as written**

Failing tests → rewrite `_compute_split_corrections` to persist-first
semantics → tests pass → lint → commit → deploy (push, CI, pull,
recreate).

- [ ] **Step 3: Execute REM Task 2, Step 7 (one-off restatement)**

Run the restatement script it specifies against `paper_snapshots` for
2026-08-26 through the current date (REM says "→ 2026-09-08"; extend the
end date to today). Back up first:

```bash
docker exec ggtrader_db psql -U postgres -d ggtrader -c \
  "CREATE TABLE paper_snapshots_backup_$(date +%Y%m%d) AS SELECT * FROM paper_snapshots;"
```

- [ ] **Step 4: Verify on the next 12:45 PT run**

```sql
SELECT run_date,
       positions->'MNST'->>'unrealized_plpc' AS mnst_pct,
       positions->'MNST'->>'market_value'     AS mnst_mv
FROM paper_snapshots ORDER BY run_date DESC LIMIT 1;
```
Expected: `mnst_pct` between -0.10 and 0.00 (true loss ≈ -5%), `mnst_mv`
≈ 2 × the broker's raw market value. The Telegram daily summary shows a
split-correction line for MNST. Only then start Task 3.

---

### Task 3: Sweep hysteresis (REM Task 5)

**Files:**
- Modify: `src/ggTrader/paper/cash_sweep.py`, `src/ggTrader/paper/trader.py` (~795–803)
- Test: `tests/paper/test_cash_sweep.py`

**Interfaces:**
- Produces: `buy_trigger_pct() -> float` (env `SWEEP_BUY_TRIGGER_PCT`,
  default `0.08`), `DEFAULT_BUY_TRIGGER_PCT = 0.08`;
  `compute_sweep_buy(..., buy_trigger_pct: float = DEFAULT_BUY_TRIGGER_PCT)`.

**Spec deviation, recorded here:** the spec (§4.3) names both a buy
trigger (cash > 8% of PV) and a sell trigger (cash < 2%). REM Task 5
implements the buy side only and explains why the sell side is unnecessary:
`compute_sweep_sell_for_funding` already sells only to cover a genuine
shortfall for the day's strategy buys, never on a schedule. The observed
same-session round-trip (9/15: sweep sell $673 to fund AVGO, then sweep
buy $786 of leftover) is eliminated by the buy-side band alone, because
post-buy cash (~5% of PV) no longer clears the 8% trigger. If sweep
round-trips persist a week after deploy, open a follow-up task for the
sell side; do not add it here.

- [ ] **Step 1: Execute REM Task 5, Steps 1–7 exactly as written**

Failing tests → env reader + dead-band → call-site keyword → fix any
pre-existing fixture in the new dead-band → lint → commit (this plan's
trailer) → deploy.

- [ ] **Step 2: Verify over the next three 12:45 PT runs**

```sql
SELECT run_date, count(*) FILTER (WHERE side='SELL') AS sweep_sells,
       count(*) FILTER (WHERE side='BUY') AS sweep_buys
FROM paper_trades
WHERE reason = 'cash_sweep' AND run_date >= CURRENT_DATE - 3
GROUP BY 1 ORDER BY 1;
```
Expected: no `run_date` with both a sweep sell and a sweep buy.

---

### Task 4: Arm the catastrophe stop (REM Task 3)

**Files:** `.env` only (not tracked).

**Interfaces:** none; `catastrophe_stop.py` already reads
`CATASTROPHE_STOP_ENABLED`.

**Precondition (hard):** Task 2 Step 4 verified live. Arming before that
force-sells MNST on a fictitious -50% loss.

- [ ] **Step 1: Confirm nothing in the corrected book is near the -25% floor**

```sql
SELECT key AS symbol, (value->>'unrealized_plpc')::float AS pct
FROM paper_snapshots, jsonb_each(positions)
WHERE run_date = (SELECT max(run_date) FROM paper_snapshots)
ORDER BY 2 LIMIT 5;
```
Expected: worst position better than -0.20. If anything is at or below
-0.20, stop and report before arming.

- [ ] **Step 2: Execute REM Task 3, Steps 2–4 exactly as written**

Flip the flag, recreate the container (`docker compose up -d
ggtrader_live` re-reads `.env`; no image rebuild needed), verify the
"armed" log line, watch one week, add the changelog entry.

---

### Task 5: Crash alerting and a working daily PnL report

**Files:**
- Modify: `scripts/paper_trade.sh`
- Rewrite: `scripts/daily_pnl_report.sh`
- Modify: crontab (add the 06:00 entry back)
- Modify: `/home/flynn/AGENTS.md:94` (cron table row)
- Modify: `docs/changelog.md`

**Interfaces:**
- Consumes: `send_telegram(message, bot_token, chat_id, format_markdown=False)`
  and `load_telegram_credentials() -> (token, chat_id)` from
  `/home/flynn/scripts/notify.py` (reads `TELEGRAM_BOT_TOKEN` /
  `TELEGRAM_CHAT_ID` from the environment, falling back to Hermes' `.env`).
- Consumes: `paper_snapshots(run_date, portfolio_value, cash, positions)`
  and `paper_trades(run_date, side, symbol, amount, reason)`.

- [ ] **Step 1: Add a failure trap to `scripts/paper_trade.sh`**

Replace the file body after the `mkdir -p "${LOG_DIR}"` line with:

```bash
cd "${PROJECT_DIR}"

# Page on any failure. The trader's own Telegram notifier lives inside the
# Python run, so a crash before its first send (signal generation, the
# leverage guard, a DB outage) previously paged nobody -- the 7/29-7/30
# leverage-guard halt went unnoticed for a day. This trap is the outer
# net: nonzero exit => Telegram with the tail of today's log.
alert_on_failure() {
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
        local tail_text
        tail_text=$(tail -n 20 "${LOG_FILE}" 2>/dev/null | sed 's/[`*_\[]/ /g')
        set -a; . "${PROJECT_DIR}/.env"; set +a
        python3 - "${rc}" "${tail_text}" <<'PY'
import sys
sys.path.insert(0, "/home/flynn/scripts")
from notify import load_telegram_credentials, send_telegram
rc, tail = sys.argv[1], sys.argv[2]
token, chat_id = load_telegram_credentials()
send_telegram(f"🚨 ggTrader paper_trade.sh FAILED (exit {rc})\n\n{tail}", token, chat_id)
PY
    fi
}
trap alert_on_failure EXIT

echo "[$(date)] Starting paper trading run..." >> "${LOG_FILE}"

docker compose run --rm ggtrader_live python -u ggt.py paper --live \
    >> "${LOG_FILE}" 2>&1

echo "[$(date)] paper_trade.sh complete" >> "${LOG_FILE}"
```

Keep the existing header comment, `export PATH=...`, `set -euo pipefail`,
and the three variable definitions above it unchanged.

- [ ] **Step 2: Test the trap without trading**

```bash
cd /home/flynn/ggTrader
LOG_FILE=/tmp/claude-1000/paper_trade_traptest.log
bash -c 'set -euo pipefail; PROJECT_DIR=/home/flynn/ggTrader; LOG_FILE='"$LOG_FILE"'
  echo "simulated failure line" > "$LOG_FILE"
  source <(sed -n "/^alert_on_failure()/,/^trap alert_on_failure EXIT/p" scripts/paper_trade.sh)
  false'
```
Expected: a Telegram message "🚨 ggTrader paper_trade.sh FAILED (exit 1)"
containing "simulated failure line". Also run `bash -n
scripts/paper_trade.sh` (expected: no output).

- [ ] **Step 3: Rewrite `scripts/daily_pnl_report.sh`**

Replace the entire file with:

```bash
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

PSQL="docker exec ggtrader_db psql -U postgres -d ggtrader -At -F '|'"

SNAP=$(${PSQL} -c "
  SELECT run_date, round(portfolio_value::numeric, 2), round(cash::numeric, 2),
         (SELECT count(*) FROM jsonb_object_keys(positions))
  FROM paper_snapshots ORDER BY run_date DESC LIMIT 2;")

TRADES=$(${PSQL} -c "
  SELECT side || ' ' || symbol || ' \$' || round(amount::numeric, 0)
         || CASE WHEN reason = 'cash_sweep' THEN ' (sweep)' ELSE '' END
  FROM paper_trades
  WHERE run_date = (SELECT max(run_date) FROM paper_snapshots)
  ORDER BY side, symbol;")

FIRST=$(${PSQL} -c "SELECT round(portfolio_value::numeric, 2) FROM paper_snapshots ORDER BY run_date ASC LIMIT 1;")

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
```

If `docker exec ggtrader_db psql -U postgres -d ggtrader -c 'select 1'`
fails, read the user/db out of `POSTGRES_CONNECTION_STRING` in `.env` and
substitute them in `PSQL`.

- [ ] **Step 4: Run the report once by hand**

Run: `bash scripts/daily_pnl_report.sh && tail -12 ~/logs/daily_pnl_report_$(date +%Y%m%d).log`
Expected: the log shows the composed message with today's value, day
delta, inception delta, cash share, and order list, then `telegram: sent`,
and the same message arrives on Telegram.

- [ ] **Step 5: Schedule it and fix the docs**

```bash
(crontab -l; echo "0 6 * * 2-6 /home/flynn/ggTrader/scripts/daily_pnl_report.sh") | crontab -
crontab -l | grep daily_pnl_report
```
Expected: the new line present (leave the old `# DISABLED 2026-05-06` line
as is).

Edit `/home/flynn/AGENTS.md` line 94 to read:

```
| `06:00 Tue–Sat` | `ggTrader/scripts/daily_pnl_report.sh` | ggTrader **equities paper** PnL → Telegram (rewired 2026-09; reads `paper_snapshots` directly) |
```

Add to `docs/changelog.md` under a `## 2026-09-16` heading (create it if
absent, above older entries):

```
- **Ops alerting:** `scripts/paper_trade.sh` now traps any nonzero exit and
  sends a Telegram alert with the log tail (closes the 7/29–7/30 silent-halt
  gap). `scripts/daily_pnl_report.sh` rewritten against `paper_snapshots` /
  `paper_trades` and rescheduled 06:00 Tue–Sat; it had been broken since
  2026-05-06.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/paper_trade.sh scripts/daily_pnl_report.sh docs/changelog.md
git commit -m "ops: page on paper_trade.sh failure; rewire daily PnL report to paper_snapshots

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3VzwB87KQ1o2MwazR5XH8"
```

`/home/flynn/AGENTS.md` is outside this repo; commit it in `/home/flynn`
with `git -C /home/flynn add AGENTS.md && git -C /home/flynn commit -m
"docs: daily_pnl_report cron row"` if that directory is a git repo
(`git -C /home/flynn rev-parse` succeeds); otherwise the edit alone is
enough. No container rebuild is needed for this task: nothing under
`src/` changed.

---

### Task 6: Benchmark and ETF tape keepalive plus one-off backfill

**Files:**
- Modify: `src/ggTrader/paper/signal_runner.py` (constant + function)
- Modify: `src/ggTrader/paper/trader.py` (one call after signal generation)
- Test: `tests/paper/test_signal_runner.py`, `tests/paper/test_trader.py`
- Modify: `docs/changelog.md`

**Interfaces:**
- Consumes: `fetch_stock_ohlcv(symbols, start, end=None, interval="1d",
  use_db_cache=True, ...)` from `ggTrader.lab.data`. With `use_db_cache`
  the `CachedYFinanceLoader` does an incremental yfinance fetch for any
  symbol whose last cached bar is stale and persists it via `_cache_to_db`
  (naive-UTC write, `cached_yfinance_loader.py:215-226`), so calling it
  daily keeps the tape current with no new writer.
- Produces: `BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY", "TLT", "GLD",
  "DBC", "IEF")` and `refresh_benchmark_tape(lookback_days: int = 30) ->
  list[str]` (returns the symbols that came back with data; never raises).

**Why here and not in the loader:** the loader already does the right
thing when asked. SPY's tape died on 8/21 because nothing asks for SPY
any more (it is not a sleeve constituent, and the old benchmark-only
writer was retired in the 8/20 migration). The paper run is the one
process that runs every trading day, so it is the natural keepalive.

- [ ] **Step 1: Write the failing tests**

Add to `tests/paper/test_signal_runner.py`:

```python
class TestRefreshBenchmarkTape:
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_fetches_all_benchmark_symbols_with_db_cache(self, mock_fetch):
        from ggTrader.paper.signal_runner import BENCHMARK_SYMBOLS, refresh_benchmark_tape

        mock_fetch.return_value = _mock_ohlcv(list(BENCHMARK_SYMBOLS), n_days=30)

        got = refresh_benchmark_tape(lookback_days=30)

        args, kwargs = mock_fetch.call_args
        assert sorted(args[0]) == sorted(BENCHMARK_SYMBOLS)
        assert kwargs.get("use_db_cache", True) is True
        assert sorted(got) == sorted(BENCHMARK_SYMBOLS)

    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv", side_effect=RuntimeError("yf down"))
    def test_never_raises(self, mock_fetch):
        from ggTrader.paper.signal_runner import refresh_benchmark_tape

        assert refresh_benchmark_tape() == []

    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_reports_only_symbols_that_returned_data(self, mock_fetch):
        from ggTrader.paper.signal_runner import refresh_benchmark_tape

        mock_fetch.return_value = _mock_ohlcv(["SPY", "TLT"], n_days=30)

        assert sorted(refresh_benchmark_tape()) == ["SPY", "TLT"]
```

Add to `tests/paper/test_trader.py` a new class after `TestSellExits`,
using the module's existing helpers `_blend(buys, sells, as_of=...)`
(line 89, wraps flat lists in the sleeve shape) and
`_make_trader(positions=None, ...)` (line 112, returns
`(trader, broker, notifier)` with a fully mocked broker). The class-level
`@patch` stack copies `TestSellExits`'s exactly:

```python
@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestBenchmarkTapeKeepalive:
    @patch("ggTrader.paper.trader.refresh_benchmark_tape", return_value=[])
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_refresh_returning_nothing_does_not_abort_run(
        self, mock_signals, mock_refresh, *_
    ):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, broker, _notifier = _make_trader(positions={})

        result = trader.run()

        mock_refresh.assert_called_once()
        assert result["errors"] == []
        broker.submit_buy.assert_not_called()

    @patch("ggTrader.paper.trader.refresh_benchmark_tape", return_value=["SPY", "TLT"])
    @patch("ggTrader.paper.trader.generate_core_signals")
    def test_refresh_is_called_after_signals(self, mock_signals, mock_refresh, *_):
        mock_signals.return_value = _blend(buys=[], sells=[], as_of="2026-06-19")
        trader, _broker, _notifier = _make_trader(positions={})

        trader.run()

        assert mock_signals.call_count == 1
        mock_refresh.assert_called_once_with()
```

(`refresh_benchmark_tape` itself never raises, so the "exception path" is
covered by `TestRefreshBenchmarkTape.test_never_raises` above; the trader
tests only need to prove the call happens and an empty result is
harmless.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/paper/test_signal_runner.py::TestRefreshBenchmarkTape tests/paper/test_trader.py::TestBenchmarkTapeKeepalive -v`
Expected: FAIL — `ImportError: cannot import name 'refresh_benchmark_tape'`
/ `AttributeError: ... has no attribute 'refresh_benchmark_tape'`.

- [ ] **Step 3: Implement**

In `src/ggTrader/paper/signal_runner.py`, after `SLEEVE_UNIVERSES` is
imported and before `generate_signals`, add:

```python
import logging

_log = logging.getLogger(__name__)

#: Symbols no sleeve trades but the lab benchmarks against or builds
#: sleeves from. Nothing else on the box fetches them daily, so the
#: paper run keeps their tape alive (SPY's went dead 2026-08-21 when the
#: old benchmark-only writer was retired; TLT/GLD/DBC stalled 2026-07-20).
BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY", "TLT", "GLD", "DBC", "IEF")


def refresh_benchmark_tape(lookback_days: int = 30) -> list[str]:
    """Touch the benchmark/ETF tape so the DB cache stays current.

    `fetch_stock_ohlcv(use_db_cache=True)` incrementally fetches and
    persists any symbol whose last cached bar is stale, so a daily call
    over a short window is enough. Never raises: this is a side job of the
    live run and must not block trading. Returns the symbols that came
    back with data (empty on any failure).
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    start = today - pd.Timedelta(days=lookback_days)
    try:
        df = fetch_stock_ohlcv(
            list(BENCHMARK_SYMBOLS),
            start=str(start.date()),
            end=str(today.date()),
            use_db_cache=True,
        )
    except Exception as exc:
        _log.warning("benchmark tape refresh failed (non-fatal): %s", exc)
        return []
    if df.empty:
        return []
    return sorted(df.columns.get_level_values(0).unique().tolist())
```

In `src/ggTrader/paper/trader.py`:
- Line 33 import: `from ggTrader.paper.signal_runner import generate_core_signals, refresh_benchmark_tape`
- Immediately after the `blend = generate_core_signals()` try/except block
  (after the `raise`, before the `if blend["fallback_used"]:` check), add:

```python
        # Side job: keep SPY and the macro ETFs' tape current for the lab.
        # Wrapped inside the function; a failure here logs and moves on.
        refreshed = refresh_benchmark_tape()
        if not refreshed:
            _log.warning("benchmark tape refresh returned no symbols")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/paper/ -v`
Expected: PASS, full directory.

- [ ] **Step 5: One-off backfill from the host, before deploying**

The live run only fetches a 30-day window; the gap is longer (SPY since
8/21, ETFs since 7/20). Backfill once from the host venv, which reaches
the DB at `localhost:5433`:

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/python -c "
from ggTrader.paper.signal_runner import refresh_benchmark_tape
print(refresh_benchmark_tape(lookback_days=120))"
```
Expected: `['DBC', 'GLD', 'IEF', 'SPY', 'TLT']`.

Acceptance (spec §6.1):

```sql
SELECT symbol, max(timestamp)::date AS newest,
       count(*) FILTER (WHERE extract(dow FROM timestamp) IN (0, 6)) AS weekend_bars
FROM ohlcv
WHERE venue = 'yfinance' AND interval = '1d'
  AND symbol IN ('SPY','TLT','GLD','DBC','IEF')
GROUP BY 1 ORDER BY 1;
```
Expected: `newest` = the most recent completed trading day for all five,
`weekend_bars` = 0 for all five. If `weekend_bars` > 0 for any symbol,
stop: the day-shift bug is back; do not deploy.

Also confirm SPY did not regain duplicate days:

```sql
SELECT timestamp::date, count(*) FROM ohlcv
WHERE venue='yfinance' AND interval='1d' AND symbol='SPY'
GROUP BY 1 HAVING count(*) > 1;
```
Expected: zero rows.

- [ ] **Step 6: Lint, changelog, commit, deploy**

Add to `docs/changelog.md` under `## 2026-09-16`:

```
- **Benchmark tape keepalive:** the paper run now refreshes SPY/TLT/GLD/DBC/IEF
  daily via `refresh_benchmark_tape()` (`paper/signal_runner.py`). SPY had
  no bars after 2026-08-21 and the macro ETFs none after 2026-07-20, which
  made every "vs SPY" lab run uncitable. One-off backfill run the same day.
```

```bash
ruff check --fix src/ggTrader/paper/signal_runner.py src/ggTrader/paper/trader.py \
  tests/paper/test_signal_runner.py tests/paper/test_trader.py
ruff format src/ggTrader/paper/signal_runner.py src/ggTrader/paper/trader.py \
  tests/paper/test_signal_runner.py tests/paper/test_trader.py
git add src/ggTrader/paper/signal_runner.py src/ggTrader/paper/trader.py \
  tests/paper/test_signal_runner.py tests/paper/test_trader.py docs/changelog.md
git commit -m "feat(paper): daily benchmark/ETF tape keepalive (SPY, TLT, GLD, DBC, IEF)

SPY's tape died 2026-08-21 when the benchmark-only writer was retired;
nothing else requested it. The paper run is the one daily process, so it
now touches these symbols through the cached loader's incremental path.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3VzwB87KQ1o2MwazR5XH8"
git push origin main
# wait for CI, then:
docker compose pull ggtrader_live && docker compose up -d ggtrader_live
```

- [ ] **Step 7: Verify on the next 12:45 PT run**

Re-run the acceptance query from Step 5. Expected: `newest` for all five
symbols equals that session's date. Then update `docs/next_steps.md`'s
ACTIVE STEP to point at the research-track plan as the next thing to
write, and delete the ops steps that are now done.

---

## Self-Review

**Spec coverage.**
- §4.1 core revert → Task 1 (with the sizing acceptance the spec adds
  beyond REM). ✅
- §4.2 split-state persistence + restatement → Task 2. ✅
- §4.3 sweep hysteresis → Task 3; sell-side trigger deliberately not
  built, deviation recorded with the reason. ✅
- §4.4 catastrophe stop → Task 4, gated on Task 2's live verification. ✅
- §4.5 alerting (crash trap + PnL report) → Task 5. ✅
- §6.1 tape restore → Task 6 (keepalive + one-off backfill + weekend-bar
  and SPY-duplicate checks). ✅
- §5 refactor, §6.2–6.5 research runs, §7 prune: out of this plan by
  design; each gets its own plan once this one is verified live.

**Placeholder scan.** Task 6's trader tests use the module's real
helpers (`_blend` at `test_trader.py:89`, `_make_trader` at `:112`) and
copy `TestSellExits`'s patch stack verbatim. Task 1 Step 2 uses
`RiskGuard(RiskConfig())`, matching `risk.py:22` (`cfg: RiskConfig | None`).
Every code step carries the full diff or command. REM references point to
a tracked file (after Task 0) with its own complete steps, not to
"similar" work.

**Type consistency.** `generate_core_signals()` (Task 1) and
`refresh_benchmark_tape()` (Task 6) are imported from the same module on
the same `trader.py:33` line; Task 6's import line includes both.
`compute_sweep_buy`'s `buy_trigger_pct` keyword (Task 3) matches REM Task
5's definition and call site. `paper_trades.reason = 'cash_sweep'` used in
Tasks 3 and 5 matches `cash_sweep.SWEEP_TRADE_REASON`.
