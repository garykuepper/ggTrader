# Prospective protocol: Opus review and parent resolutions

Opus 5.5 returned **NEEDS FIXES**, not an approval, on the first draft. Parent
checked its findings against the brief/plan and resolved them in the proposed
[concrete protocol](2026-10-25-intraday-prospective-observation.md).
No second reviewer verdict is claimed. No code or observation was started.

## Findings and resolutions

1. Anchor identity: use the same canonical first valid prior-session close as the
   previous exit. Parent found a conflict in the suggested 15:20 anchor recovery
   versus a valuation grace ending before 15:20. Resolved by conditional 15:15
   recovery; the 15:20 decision-input stage only consumes its completed canonical
   anchor. Bootstrap is separate because it has no prior hypothetical position.
2. Fault severity: one-session input/decision faults do not halt known-equity
   operation. Unknown traded-day accounting beyond grace, conflicting decisions
   and ledger integrity faults halt. Preserve signal decisions when possible;
   no-position policy outcomes are not fixed-rule zeros. Resume needs an approved
   future amendment and cannot repair the original evidence retroactively.
3. Late valuations: first valid capture before the next official session's 15:20
   may value the position, but remains late for operational acceptance. Preserve
   all original snapshots and never replace a valid canonical price by preference.
4. Closures: freeze 252 scheduled slots and their endpoint. Unexpected cancelled
   or shortened slots do not extend it. Report actual observation N separately.
5. Descriptive output: no comparison to performance thresholds, passing flags or
   GO/NO-GO fields. Specify NW-5 computation and unresolved variance semantics.
6. Dates: propose third full session strictly after activation date. Hash exact
   dates before activation completes. This replaces the proposed November 2 deadline.

Also clarified inclusive 09:59 request end, SIP/minute timestamp semantics,
issuer amount rounding, terminal issuer completeness, ex-distribution-flat
definition, separate after-cost tax paths, local-only publication, claim/rename
semantics and host-clock status. No independent timestamp service is introduced.

Not adopted: optional 10:20 input stage and automatic whole-window revision pulls.
They add operations and requests beyond this minimal observer. Stage deadline
remains 30 seconds **per request**, bounded by the stage's wall-clock cutoff;
the review's alternate per-stage interpretation is not adopted. Daily budget
includes conditional recovery, pagination and retries, shared across processes.

## Owner decisions presented together

- Activation-relative inception and fixed 252 slots, including exceptional closures.
- 15:15 conditional recovery, 15:20 inputs, 15:25 decision, 16:20 archive; late
  valuation grace strictly before the next official session's 15:20.
- One-session versus halting faults, failed 20-slot operational checkpoint,
  owner-controlled prospective resume and no erased days.
- $100,000 hypothetical starting wealth; 1/2 bp per-side base/stress costs;
  24% annual-positive-after-cost-P&L tax proxy; zero idle-cash yield.
- Local timestamps only; operational/descriptive evidence with no profitability pass.

Approval of this exact brief authorizes offline implementation and preparation
of bounded source/readiness evidence. Scheduler activation is a later concrete
decision under the approved plan. No orders or production changes follow.

Task: `node:delegated-task:command%3Amcp%3Addcb1fe4-adb1-43af-b8a1-2705ab37e909%3Adelegate-task%3Aintraday-prospective-opus-protocol-2026-10-25-r1`.
