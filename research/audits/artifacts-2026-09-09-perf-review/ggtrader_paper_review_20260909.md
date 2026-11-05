# ggTrader Paper-Trading Performance & Profitability Review

**Date:** 2026-09-09 · **Task:** t_f8285900 · **Mode:** READ-ONLY advisory review
**Deployment reviewed:** `ensemble` 3-sleeve blend (sp500/midcap400/nasdaq100) + inverse-vol/target-vol overlay, Alpaca paper, cron 12:45 PT Mon-Fri, cash sweep ENABLED (since 2026-08-24), catastrophe stop OFF, ML gate OFF.

---

## 1. Executive summary

The paper account returned **+1.15% as-reported / +2.03% split-fair** over its 11-week live window (2026-06-23 → 2026-09-08, 52 return days) versus **SPY +4.49%** on the identical window. The strategy book itself is roughly flat (+$133 unrealized across 30 names ex-SPY); the return that exists comes mostly from the SPY cash-sweep position bought 8/24. Annualized Sharpe is 1.71 (split-fair) but with a standard error of **±2.21** — statistically indistinguishable from zero, and from SPY's 2.12, on this sample size. Eleven weeks cannot validate or invalidate anything; the honest read is "no evidence of edge, no evidence of disaster, benchmark still ahead."

Three operational findings matter more than the P&L number:

1. **A hard-constraint incident already happened.** On 7/29-7/30 the account's multiplier was 4.0x; the 1x-cash guard in `trader.py:378-382` correctly refused to trade, costing one full session (7/29, no snapshot) and pushing 7/30's trades to a manual 6:35 PM after-close rerun (16 buys filled at the next open — uncontrolled overnight-gap exposure).
2. **The MNST 2-for-1 split (ex 2026-08-11) is still unapplied by the broker, and the display-side fix expires.** `_SPLIT_LOOKBACK_DAYS = 14` in `trader.py` means split corrections stopped applying 8/25 while the position is still held: reported NAV is understated ~$898 (0.87%) today, the daily return series contains a fictitious -0.69% day (8/26), and — critically — **enabling `CATASTROPHE_STOP_ENABLED` right now would force-sell MNST on a fictitious -52% unrealized loss**. The rollout note in `docs/next_steps.md` ("nothing in the current book would trigger") is wrong as of today's book.
3. **The planned core revert (week of 8/31) has not happened.** `paper_rebalance_state` shows the 3-sleeve blend still live as of 9/1 (weights .31/.30/.39, scale 0.695). Combined with the sweep, the account is now 61% passive SPY / 34% active strategy / 5% cash — the blend overlay is governing a shrinking minority of the book.

---

## 2. Performance snapshot

Window pinned: **2026-06-23 (inception, $102,459.43 all cash — `paper_snapshots_preshift_backup_20260822`) → 2026-09-08 ($103,639.75 as reported)**. 52 daily return observations. All figures from `paper_snapshots`/`paper_trades` (psql dumps in `/tmp/t_f8285900/`, metrics in `metrics.json`).

| Metric | As-reported | Split-fair* | SPY (same window) |
|---|---|---|---|
| Cumulative return | **+1.15%** | **+2.03%** | **+4.49%** |
| End value | $103,639.75 | $104,537.97 | — |
| Ann. Sharpe (daily, 252d) | 0.94 | 1.71 (SE ±2.21) | 2.12 |
| Max drawdown | -1.97% (9/1) | -1.34% (9/1) | shallower |
| Daily win rate | 46.2% | 48.1% | — |
| Peak value | $105,029 (8/21) | $105,309 (8/25) | — |

\* Split-fair = reported PV + the MNST display understatement (2× the shown market value on dates where the broker's unapplied split is visible: 8/11-8/19 and 8/26-9/8). This is the strategy's economic P&L view. The as-reported column is what the account can actually realize today — the broker's paper book genuinely only has 20.804 MNST shares (sellable for ~$898), so the ~$950 delta is a real loss inside the paper simulation caused by Alpaca's failure to process the corp action, not by the strategy. Both views are honest; they answer different questions.

**Sub-windows (split-fair):**

| Window | Cum | Ann. Sharpe | SPY same window | Note |
|---|---|---|---|---|
| Pre-sweep 6/23 → 8/20 | +2.40% | 2.75 | +3.96% | avg **62.5% of PV idle in cash** |
| Post-sweep 8/21 → 9/8 | -0.47% | -1.48 | +0.10% | 11 days; SPY stub + weak active book |

Monthly (split-fair): Jun (6/23-30) -0.01% · Jul +1.64% · Aug +0.74% · Sep (through 9/8) -0.35%.

**Composition drift (9/8 snapshot):** 61.1% SPY (sweep, $63,388) · 33.8% active strategy (30 names, $35,047) · 5.0% cash. The sweep has made the account majority-passive; the blend overlay's scale (0.695) now applies to the active sleeve capital only.

**Turnover (112 trades, 6/24-9/4):** total two-sided notional $253.9K → 11.8x annualized; strategy-only $171.5K → **8.0x annualized**; sweep churn $82.4K in 15 trades over 12 sessions (~$5.5K/day round-tripping against the 5% reserve). Unrealized P&L ex-SPY: **+$133** across 30 names (best: HAS +$548, VZ +$517; worst: PNR -$277, KNF -$162). Dividend accruals booked: $105.50 (5 rows, `paper_dividend_accruals`).

**Benchmark caveats (stated plainly):** SPY daily bars come from the corrected `ohlcv` tape, which **ends 2026-08-21** — the old benchmark writer is retired and nothing else writes SPY (it is not a sleeve constituent). For 8/24-9/8 I used the account's own 12:45 PT SPY marks from snapshots, which is internally consistent with how PV is marked (also ~12:45 PT), but it is not a close-to-close series. Both strategy and benchmark marks share the same timestamp regime, so the comparison is fair; the number is not exchange-official. Any future "vs SPY" lab run needs this tape gap fixed first.

**Honest framing:** this is true out-of-sample (live signals, live paper fills, no lookahead), but 52 return days cannot support a Sharpe estimate tighter than ±2.2. Do not quote the 1.71 as evidence of anything.

---

## 3. Observed failure modes (from logs and DB, not speculation)

**F1 — Leverage-guard halt, missed session (7/29-7/30). SEVERITY: HIGH.**
`paper_trade_20260729.log` and `...20260730.log` (12:45 runs) both show: `RuntimeError: Account multiplier is 4.0x (margin-enabled); the blend overlay assumes an unlevered (1.0x) account.` The guard worked as designed. Cost: 7/29 has no snapshot and no trades (fully missed session); 7/30 was rescued by a manual 18:35 rerun that submitted 16 buys + 1 sell after the close — those filled at the 7/31 open, taking overnight-gap exposure the 12:45 PT schedule was specifically chosen to avoid (`paper_trade.sh` header). The crash also bypassed the Telegram notifier (it dies inside the failed run), so the only trace is a traceback in a log file nobody is paged on. The multiplier was reset sometime before 7/31 12:45; nothing in the repo records who/when.

**F2 — MNST split: unapplied by broker, display correction expired (ongoing). SEVERITY: HIGH (data integrity).**
MNST 2-for-1, ex 2026-08-11 (`split_check.py` docstring; tape shows close 91.21 on 8/10 → 45.74 on 8/11, qty never doubled from 20.804). Snapshot history: displayed MV halves 8/11; the deployed snapshot-primary guard (`c39ae1d`) shows corrected $1,980-$2,030 values 8/20-8/25; then corrections silently stop from 8/26 — `_SPLIT_LOOKBACK_DAYS = 14` (`trader.py:46`) expired while the position is still held. Effects today: NAV understated $898 (0.87%); 8/26 daily return shows -0.69% (re-halving, not markets); `paper_risk_state.peak_value` ($104,518) is tracked on the distorted series; and `unrealized_pl` for MNST reads -52% when the true economic figure is -4.8% (cost basis $1,887.11 is split-invariant; true MV $1,796). The broker-side qty was never fixed — `apply_corrections_to_positions` is display-only by design (`split_check.py:165-177`).

**F3 — Catastrophe-stop landmine (latent, would fire on corrupted data). SEVERITY: HIGH if enabled.**
`docs/next_steps.md` step 3 says "consider `CATASTROPHE_STOP_ENABLED=true` (-25% floor; nothing in the current book would trigger — NXPI sits at -19%)". Both halves are stale: NXPI was sold 9/3 ($2,728, -18%), and MNST currently displays -52% unrealized because of F2. `catastrophe_stop.unrealized_pct` consumes positions *after* `apply_corrections_to_positions` — with the lookback expired the corrections dict is empty, so it would see the raw broker numbers and force-sell 20.8 MNST shares at ~$43, realizing a fictitious -$950. **Do not enable the flag until F2 is fixed.**

**F4 — Missed/irregular sessions.** 52 snapshots vs 54 weekday sessions 6/24-9/8: the gaps are 7/3 (July 4th observed — run executed, market closed, benign) and 7/29 (F1, not benign). 9/7 Labor Day excluded correctly. Cron timing is clean at 12:45:01-:02 PT from 6/29 onward; the first five June runs (6/22-6/26) ran 13:30 PT under the earlier schedule. The one 18:35 run is the F1 manual rerun.

**F5 — Universe data noise, daily.** Every run logs yfinance failures for delisted/renamed tickers (e.g. 9/8: SATS 404, JHG/BLD delisted, EA no data). The loader continues, but those symbols silently drop from the sleeve universe that day — small, recurring signal-input drift with zero alerting. `[data] 82 rows x 503/400/101 symbols` per run.

**F6 — No other halts found.** No ML-gate blocks (gate disabled; 2026-06-27 ablation showed it anti-predictive), no regime blocks, no stale pending orders (`paper_pending_orders` empty), no rejected orders, no drawdown or daily-loss halts (both far from thresholds: max DD -1.97% vs 15% limit).

---

## 4. Top 5 recommendations (ranked)

### Execution / config (act now)

**R1. Fix MNST accounting and make split corrections persistent — before any other config change.**
- *Rationale:* F2/F3. The 14-day rolling lookback is structurally wrong for a position held 50+ days past a split the broker never applied. Known-unapplied splits need durable per-symbol state (e.g. a `paper_split_state` table or "until broker applies or position closes" semantics), plus a one-off restatement of the 8/26-9/8 snapshots, plus a decision on the broker-side qty (manually restore the 20.8 missing shares in the paper account, or accept and document the ~$950 venue loss).
- *Expected impact:* removes a 0.87% NAV understatement, a fictitious -0.69% daily return, and unblocks R4. *Risk:* low — display/state change, no signal path. *Effort:* small code change + one-off data fix + container rebuild (the standing ~2.5 min CI → pull → recreate loop).

**R2. Execute the core revert (blend → SP500 core) — it is a week overdue and the evidence has not softened.**
- *Rationale:* the pinned 17-fold re-baseline on the corrected tape (`docs/research/_rebaseline_corrected_tape_20260822.json`, window 2021-01-31 → 2026-04-30): core Sharpe 0.99 / CAGR 8.0% / MaxDD -7.7% vs 3-sleeve blend @ lev 1.0 0.69 / 4.78% / -6.7%; SPY window-matched Sharpe 0.78 but CAGR 12.79%. Third independent core-beats-blend result. Live evidence is directionally consistent: 11 weeks of the blend delivered +2.03% split-fair vs SPY +4.49%, with the active book flat (+$133 ex-SPY). With the sweep on, the account is already 61% SPY — the overlay is complexity governing a minority of capital.
- *Expected impact:* per re-baseline, +3.2pts CAGR at similar-or-better MaxDD; deletes the overlay/rebalance machinery from the live path. *Risk:* live-vs-backtest drift (the standing counter-argument — but drift argues for the *simpler* config, per `next_steps.md`). *Effort:* config/env change + container recreate. Note: 11 weeks of live data is NOT independent evidence for or against this (SE ±2.2); the re-baseline is the evidence.

**R3. Restore a working daily PnL/ops report with failure alerting.**
- *Rationale:* `scripts/daily_pnl_report.sh` is a documented fossil (invokes the deleted `pnl-daily` CLI subcommand; cron line disabled 2026-05-06). During the incident window, the only signals were tracebacks in per-day log files. A run that crashes before its first broker call (F1) currently notifies nobody.
- *Expected impact:* incidents surface in minutes (Telegram) instead of at review time. *Risk:* none (read-only reporting). *Effort:* small — a few SQL lines against `paper_snapshots`/`paper_trades` + the existing notifier; plus a cron-level "if the run exits non-zero, alert" wrapper so pre-notifier crashes still page.

**R4. Enable the catastrophe stop only after R1, with the MNST case as the acceptance test.**
- *Rationale:* the backstop is sound (NXPI sat ~-19% for eight weeks with no RSI exit — `catastrophe_stop.py` docstring) but F3 shows it would today fire on corrupted data. After R1, MNST's true unrealized is -4.8% (no trigger), and the book's worst true position is PNR at -8.1% — so enabling it post-fix is genuinely inert, matching the intent of the rollout note.
- *Expected impact:* caps tail loss per position at -25% real. *Risk:* low once R1 lands; without R1 it is an active hazard. *Effort:* env flag + recreate.

**R5. Add hysteresis to the cash sweep to kill the daily round-trip churn.**
- *Rationale:* 15 sweep trades / $82.4K notional across 8 sessions, oscillating around the 5% reserve (8/25: sell $752 then buy $763; 8/26: sell $761/buy $759; 9/3: sell $2,232/buy $4,267). Each round-trip pays the SPY spread for zero expected return. A dead-band (e.g. act only when cash < 2% or > 8% of PV) cuts this ~70% with identical average exposure.
- *Expected impact:* small but pure friction savings; fewer ledger rows polluting `paper_trades` attribution. *Risk:* trivial. *Effort:* small.

### New research (NO-GO discipline applies — none of this is justified by the live sample)

The queued items in `next_steps.md` remain the right queue and the right order: re-run `ensemble_ic` (1.01) and `ensemble_kelly` (0.98) against the real 0.99 core baseline (their NO-GOs were issued against the phantom 1.12), then the TLT/GLD/DBC cross-asset sleeve. Treat all of it as lower priority than R1-R4 — the operation has data-integrity debt that makes any new live comparison less trustworthy than it looks. Also note for any future benchmarked run: **the DB has no SPY bars after 2026-08-21** (F2/benchmark caveat) — restore a SPY writer or pull from yfinance before quoting another "vs SPY" number.

---

## 5. 1x-cash constraint audit

- **Guard:** `trader.py:378-382` refuses to run `--live` if account multiplier > 1.0. It fired correctly on 7/29-7/30 (F1). Current runs succeed, so the account is back at 1.0x.
- **No leverage in the live config:** overlay `max_leverage` is 1.0 (re-baseline overlay params; `paper_rebalance_state.scale` = 0.695, i.e. de-risking, not leveraging). The sweep buys SPY with cash only and holds a 5% reserve. Nothing in the current book violates the constraint.
- **The violation risk is operational, not configurational:** someone flipped the paper account to 4.0x margin once already, and the failure mode it produced (silent missed session, after-hours manual rerun) is worse than the guard itself. Recommend: (a) the R3 failure alerting, and (b) if Alpaca paper allows locking the account's margin setting, lock it.
- **Latent interaction to keep in mind:** with F2 unfixed, any risk feature that consumes broker `unrealized_pl`/`market_value` (catastrophe stop, drawdown halt inputs) is reading distorted data on MNST. Not a leverage issue, but the same class of "guard fed bad numbers" problem.

---

## 6. Methodology & artifacts

- **Data pulled (read-only):** `paper_snapshots` (52 rows), `paper_trades` (112 rows), `paper_snapshots_preshift_backup_20260822` (inception baseline 6/23), `paper_risk_state`, `paper_rebalance_state`, `paper_pending_orders`, `paper_dividend_accruals`, `ohlcv` (SPY, MNST tapes), all via `docker exec ggtrader_db psql` COPY/SELECT — no writes. All 57 `/home/flynn/logs/paper_trade_*.log` scanned. Code read: `paper/trader.py`, `overlay.py`, `risk.py`, `cash_sweep.py`, `catastrophe_stop.py`, `split_check.py`, `feature_gate.py`, `scripts/paper_trade.sh`, `scripts/daily_pnl_report.sh`. Canon: `AGENTS.md`, `docs/next_steps.md`, `docs/research/_rebaseline_corrected_tape_20260822.json`.
- **Raw artifacts:** `/tmp/t_f8285900/` — `snapshots.csv`, `snapshots0.csv`, `trades_full.csv`, `mnst.csv`, `spy.csv`, `spy_marks.csv`, `metrics.json`, `analyze.py` (reproduces every number above), `extra.py`. A bundle of these is attached alongside this report.
- **Metric conventions:** daily returns from consecutive snapshot PV; Sharpe = mean/std × √252 (sample std, n-1); no risk-free deduction (paper, short window); turnover annualized as two-sided notional / avg PV × (252/52). Survivorship/lookahead: live forward-run data only; the inception row predates all trading; SPY benchmark uses the corrected tape (post-migration family) — the pre-fix tape's SPY rows were excluded by the 2026-08-20 migration.
- **Limitations:** 52 return days (Sharpe SE ±2.21); paper fills (Alpaca paper execution quality is optimistic vs real money); benchmark post-8/21 uses 12:45 PT account marks, not official closes; split-fair view assumes the broker *would* apply the split — the realizable view (as-reported) is 0.87% lower.

## 7. Key uncertainties

- Whether the ~$950 MNST shortfall gets restored by Alpaca or stands as a permanent venue loss in the paper account (changes which cumulative-return column is "real").
- Who/what flipped the account to 4.0x on 7/29 and whether it can recur (no repo/DB record).
- Whether the 8/17-8/21 SPY rows at 00:00 (15 rows, single family, ends 8/21) came from the migration work or a still-undeployed writer — i.e., whether SPY tape maintenance is intentional or accidentally dead. Either way it is dead *now*.
- Live-vs-backtest execution drift remains unquantified (no slippage ledger; `amount` is filled notional, fills are market orders at 12:45 PT).
