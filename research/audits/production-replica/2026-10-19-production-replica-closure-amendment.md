# Amendment: close the daily-bar model on the primary failure

**Status: approved by Flynn's “approval” instruction in this thread on
2026-10-19, before resuming the independent studies.**

Original: [production-replica audit](2026-10-19-production-replica-audit.md).
Earlier approved amendments: [daily-bar scope](2026-10-19-production-replica-parity-amendment.md)
and [event-return timing](2026-10-19-production-replica-fomc-timing-amendment.md).
Those files remain unchanged.

## Evidence and reason for this amendment

The primary window, 2021-01-31 through 2026-04-30, passed the registered data
checks: 98.60% member-day coverage and no detected tainted trades. This means it
passed the specified detector, not that every underlying price is independently
verified. The primary arm returned 14.1871% CAGR versus SPY's 14.8294%, an excess
of -0.6423 percentage points/year, with Newey-West t = -0.6898. Both registered
pre-tax bars fail. All four primary sensitivity arms also trail SPY.

The holdout passed the coverage floor at 87.39% but stopped on 21 trade/artifact
matches representing 11 unique trades in TIE, MEE and RHT. Its remaining four
arms were not run. The registered rule classifies a missing fresh comparison as
an artifact; this is not proof that every flagged move is economically false.

Source: `../_production_replica_results.json`, keys `windows.primary`,
`windows.holdout.coverage_member_days`, `windows.holdout.mean_coverage`,
`windows.holdout.arms.primary.tainted_trades`, and `verdict`.

A fresh Claude Opus 5.5 read-only review inspected the artifacts and raw database
prices. It found incompatible price levels in TIE and MEE, and suspicious stale,
low-volume CPWR prices that the same-provider comparison did not reject. Mixed
security identities are a hypothesis requiring independent identification.
RHT's jump may instead reflect a genuine corporate event. The report must
distinguish observed cached rows, inferred causes and independently verified
facts. No estimated performance contribution from these names is established.

## Approved decision rule and scope

Allow a **NO-GO for the specified historical daily-bar model** when the primary
window passes the registered data checks and fails either pre-tax bar, even
though the separate holdout remains stopped. Both windows must pass for GO, so
the holdout cannot rescue this primary failure. This change permits only the
negative model conclusion on the existing primary evidence; it does not permit
GO with unresolved data stops.

Keep the holdout **INCONCLUSIVE** and its performance non-citable. Preserve all
stop flags, original results, logs and curves. Do not drop symbols, replace
prices, filter events or run the remaining holdout arms. No holdout repair is
required to close this study. Document TIE, MEE, RHT, CPWR and the detector's
limitations. Exact historical deployed-trader replication remains **UNVERIFIED**.

Keep the raw driver's `verdict` as `INCONCLUSIVE/stopped`. Record the approved
closure decision separately, with this amendment as its authority, and explain
the distinction in the report. Do not relabel the stopped run as a clean result.

## Work permitted after approval

1. Preserve and commit Task 6's reviewed implementation and stopped artifacts.
2. Finish the independent Task 8 studies on their original windows, grids,
   costs and bars. Retain the completed FOMC and xs_momentum checkpoints.
   Dual momentum must restart its strategy run: its 14 completed folds were
   logged, but the driver checkpoints only after an entire strategy completes.
3. Run Task 9's original A23 distribution-tax bounds. Owner entry terms remain
   unchanged. Do not run Task 7: the pre-tax bar failed.
4. Complete Task 10's report, record corrections and regenerated snapshot.
   State separately: daily-bar model NO-GO, holdout INCONCLUSIVE, exact live
   replication UNVERIFIED. Apply all review, test, lint and per-task commit gates.

The stock sleeve's future remains the owner's decision at the freeze lift. This
amendment authorizes no live change now or later. Keep the freeze, restricted
imports, SELECT-only database rule, permitted automatic loader-cache writes,
no-deploy/no-push/no-merge restrictions, and all other registered stop conditions.

## Approval

Flynn approved this amendment explicitly in this thread on 2026-10-19.
