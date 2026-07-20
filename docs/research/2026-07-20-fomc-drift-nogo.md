# Pre-FOMC Long-Treasury Drift (Candidate A7): NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-20
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/fomc_drift.py`
(`FomcDriftStrategy`): candidate A7 from `WEB_RESEARCH_CANDIDATES.md`'s
2026-07-19 cross-asset register, promoted out of parked status after its
citation was replaced with a verified, current, directly-on-point paper —
Jun Pan & Qing Peng, *"The Pre-FOMC Drift in Long-Term Treasury Bonds"*
(current draft, June 2026). Mechanism: long TLT/IEF/EDV the trading day
before a scheduled FOMC announcement, exit at/around the announcement —
a pure event-driven calendar anomaly, not a cross-sectional signal.

**New infrastructure built:** `src/ggTrader/lab/fomc_calendar.py` — a
scraper for the Federal Reserve's own public calendar pages (free, no API
key). Two source formats stitched together: `fomchistorical{year}.htm`
(2011-2020, parses "Month D1-D2 Meeting" text, handling same-month,
cross-month, and abbreviated-month variants — verified against known
dates like the December 2015 "liftoff" meeting) and `fomccalendars.htm`
(2021+, parses embedded `fomcpresconfYYYYMMDD` anchor IDs, a more
reliable structured source). Deliberately excludes unscheduled/emergency
actions (e.g. the March 2020 inter-meeting cuts) — the mechanism requires
the meeting to have been publicly pre-announced, which is what
distinguishes "day before a known event" from "day before a surprise."
122 scheduled announcement dates recovered, 2011-01-26 through
2026-06-17, cached to `data/universe/fomc_meeting_dates.csv`
(`scripts/fomc_calendar_backfill.py`).

Used the `target_kind="signals"` protocol (exact entry/exit bars) rather
than this lab's monthly-rebalance `"weights"` protocol, since a 0-3
trading-day hold can't be expressed at monthly granularity — matches the
pattern used by `overnight_gap`, another daily-event signal strategy.

**Found and fixed a real bug during first-run debugging:** this lab's
OHLCV bars carry a non-midnight time-of-day (`16:00:00+00:00`, the
yfinance loader's close-time convention), while FOMC event dates from
`fomc_calendar.py` are plain midnight-normalized Timestamps. An
exact-timestamp equality check between the two silently matched *zero*
events on the very first live run (all-zero entries/exits, a giveaway
that something was structurally broken rather than genuinely showing no
signal) — the initial unit tests used synthetic midnight-aligned bars and
didn't catch it. Fixed by comparing on calendar date alone
(`.normalize()`), with a regression test using a non-midnight-bar fixture
added before the fix.

**Result: clean NO-GO.** Full honest walk-forward, `fomc_treasury`
universe (TLT/IEF/EDV), `eval_start 2012-01-01` (54 folds), sweeping
`entry_offset_days` ∈ {1, 2} × `hold_days` ∈ {0, 1, 2}.

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | Total Return (13.5yr) | CAGR | Max Drawdown | WFE | Regime Halt |
|---|---|---|---|---|---|---|
| **fomc_drift (this report)** | 0.10 | 0.45% | 0.03% | -1.2% | 0.44 (below 0.50 floor) | Active |
| SPY (reference — not the real comparison, see below) | 0.80 | — | 14.8% | -33.7% | N/A | N/A |

The strategy is only in position for a small fraction of the calendar
(352 nonzero-return bars out of 4,220 in the OOS window, consistent with
~121 events × 1-3 day holds), so a naive comparison to always-invested
SPY isn't the right framing — the informative number here is the
strategy's own total return over 13.5 years of doing exactly what the
paper describes: **0.45% cumulative, essentially flat.** This is not a
"wrong benchmark" situation like the previous candidate (A1) — the WFO's
own aggregate return, measured only over the days the strategy is
actually exposed, directly tests the paper's claim (is there a positive
drift on these specific days), and the answer is no, at a magnitude
indistinguishable from noise.

Two independent corroborating signals, not just the flat return:
**aggregate WFE (0.44) sits just below this project's 0.50 overfitting
floor**, and **the regime-halt mechanism was active** — the best-scoring
combo per fold was unstable, selected in only 14 of 54 folds (26%),
consistent with a signal that doesn't generalize rather than one that's
being genuinely captured and then washed out by costs.

## 3. Actionable Research Directions

None ranked — this is a closure report. The underlying FOMC calendar
infrastructure (`fomc_calendar.py`, the cached 122-date list) is reusable
for any future candidate needing scheduled-macro-event timing (e.g. a
different pre/post-FOMC instrument or window).

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Pre-FOMC long-Treasury drift (TLT/IEF/EDV, day-before entry,
day-of/after exit) — REJECTED.** Total OOS return 0.45% over 13.5 years
(effectively flat), Sharpe 0.10, WFE 0.44 (below the 0.50 gate floor),
regime halt active (winning combo unstable, 26% fold-selection rate).
Not an eval-window-drift pattern (this is the strategy's full available
history, not a truncated or mismatched window) and not a data-quality
issue (the event calendar was verified against known dates and the
timestamp-matching bug was caught and fixed before this result was
generated) — a clean "no exploitable drift in this implementation"
result.

**Known implementation limitations, for the record:**
- Sample restricted to 2012-2026 (Fed press-conference era onward,
  ~14.5 years, 122 events) — the paper's own academic sample may go back
  further (pre-1994 FOMC decisions weren't publicly announced same-day at
  all, so the mechanism's economically sound window starts around 1994
  regardless; this implementation's 2011-2020 historical-page scraper
  could in principle be extended earlier, but wasn't attempted given the
  already-clean null result).
- Entry/exit timing (day-before close to announcement-day close, per the
  swept `entry_offset_days`/`hold_days` grid) is an approximation of
  "the day before the announcement" — the paper's own precise event
  window (previous close to announcement open, or another interval) was
  not replicated exactly.
- Equal-weight across TLT/IEF/EDV rather than a duration-tilted
  construction — the paper documents the effect concentrated at longer
  maturities, which an equal-weight 3-ETF blend only partially captures.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `fomc_drift` in any configuration.** No live-config
changes. Move to the next candidate in the register's home-lab priority
order: **A3 (commodity medium-term trend)** — a standalone candidate
split out of the prior draft's combined "carry+trend+basis reversal"
idea, free/cheap ETF proxies (commodity ETFs), practitioner-grade
sourcing (Bloomberg BERY index commentary).

**Infrastructure kept regardless of this NO-GO:** `fomc_calendar.py` and
the cached FOMC date list remain reusable for any future macro-calendar-
timed candidate.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** could a duration-tilted construction (e.g.
long-only EDV, the longest-duration ETF, rather than an equal-weight
TLT/IEF/EDV blend) recover the effect the paper documents as
"concentrated at longer maturities" — i.e., is the null result diluted
by including shorter-duration IEF, or is the underlying drift itself just
not present in this sample?

**Resolution:** Plausible but likely second-order — IEF (7-10yr) and TLT
(20yr+) are both long-duration relative to the front end of the curve,
and EDV (25yr+ zero-coupon, the most duration-concentrated of the three)
is already one-third of the equal-weight blend. A cheap, low-effort
follow-up would be re-running this exact WFO on EDV alone before
concluding the mechanism doesn't exist at all — but given the aggregate
result is not marginally negative but essentially zero, and the
regime-halt/WFE evidence independently points to instability rather than
a diluted-but-present signal, this isn't ranked as an active next step;
noted for the record rather than queued.
