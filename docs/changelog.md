# Changelog

## 2026-09-16

- **Ops alerting:** `scripts/paper_trade.sh` now traps any nonzero exit and
  sends a Telegram alert with the log tail (closes the 7/29–7/30 silent-halt
  gap). `scripts/daily_pnl_report.sh` rewritten against `paper_snapshots` /
  `paper_trades` and rescheduled 06:00 Tue–Sat; it had been broken since
  2026-05-06.

## 2026-09-11

### Persistent split-state tracking, re-verified against snapshot evidence indefinitely

Replaced the 14-day rolling `_SPLIT_LOOKBACK_DAYS` window as the sole memory
for an unapplied broker split with a durable `paper_split_state` table
(`699203a`, `72ac60c`): a detected-unapplied split now stays corrected for as
long as the position is held, not just for 14 days after its ex-date — this
is the MNST fix (see the catastrophe-stop entry below for the incident this
closes). A final whole-branch review found that the persisted correction had
no re-verification/expiry path: once the broker's live corporate-actions feed
stopped mentioning a symbol (its ex_date aged out of the feed's own lookback
window), the persisted correction could never be confirmed applied or
cleared, so a belated broker-side split apply would have double-corrected the
position forever. Fixed this session: `persist.get_open_split_states()`
(alongside, not replacing, the existing `get_open_split_corrections()`)
surfaces each persisted correction's own `ex_date`, and
`trader._compute_split_corrections` now builds a synthetic corporate-action
entry from any persisted, currently-held symbol the broker feed no longer
reports, merging it back into the same `find_split_applied_symbols` check
used for live feed data — so persisted state keeps getting re-verified against
`paper_snapshots` history every run, indefinitely, and is cleared the moment
that evidence confirms the broker applied it.

**Status: code-complete on `worktree-paper-trading-remediation`, not yet
deployed to the live container.** The one-off NAV restatement for
2026-08-26→09-08 still needs the deploy to land before it can be run against
live data (see the ACTIVE STEP in `docs/next_steps.md`).
